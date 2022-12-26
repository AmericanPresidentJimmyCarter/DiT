import base64
import io
from threading import Lock
from typing import Dict, List, Union
import ujson
from fastapi import FastAPI, HTTPException, Response
# from fastapi.responses import Response

import argparse
import sys

import torch

from pydantic import BaseModel

import open_clip
from open_clip import tokenizer
from t5 import FrozenT5Embedder
from unclip_prior import UnCLIPPriorPipeline
from copy import deepcopy

from data import tensor_to_b64_string


CACHE_DIR = '/fsx/hlky/.cache'
CONDITIONING_DEVICE = 'cuda:7'

# CACHE_DIR = '/home/user/.cache'
# CONDITIONING_DEVICE = 'cuda:0'

mtx_prior = Lock()

class ConditioningRequest(BaseModel):
    captions: List[str]


class ConditioningResponse(BaseModel):
    flat: str
    full: str
    flat_uncond: str
    full_uncond: str
    prior_flat: str
    prior_flat_uncond: str


def spawn_clip_model():
    _clip_model, _, _ = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained="laion2b_s32b_b79k", cache_dir=CACHE_DIR,
    )
    del _clip_model.visual
    _clip_model = _clip_model.to(CONDITIONING_DEVICE).eval().requires_grad_(False)
    return _clip_model


def spawn_t5_model():
    _t5_model = FrozenT5Embedder(
        device=CONDITIONING_DEVICE,
        cache_dir=CACHE_DIR,
    ).to(CONDITIONING_DEVICE)
    return _t5_model


def spawn_prior_model():
    pl = UnCLIPPriorPipeline.from_pretrained("kakaobrain/karlo-v1-alpha",
        torch_dtype=torch.float32)
    return pl


clip_model = spawn_clip_model()
t5_model = spawn_t5_model()
prior_model = spawn_prior_model()

app = FastAPI()


def generate_clip_embeddings(model, text_tokens) -> torch.Tensor:
    """
    Get the CLIP embedding before feature extraction/normalization.

    TODO Alter the unet to use this instead of the final squished embedding.
    """
    cast_dtype = model.transformer.get_cast_dtype()

    x = model.token_embedding(text_tokens).to(
        CONDITIONING_DEVICE
    )  # [batch_size, n_ctx, d_model]

    x = x + model.positional_embedding.to(cast_dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = model.transformer(x, attn_mask=model.attn_mask)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = model.ln_final(x)  # [batch_size, n_ctx, transformer.width]
    return x


def captions_to_conditioning_tensors(clip_model, t5_model, captions):
    device = t5_model.device

    text_tokens = tokenizer.tokenize(captions)
    text_tokens = text_tokens.to(device)
    clip_embeddings = clip_model.encode_text(text_tokens).float().to(device)
    clip_embeddings_full = (
        generate_clip_embeddings(clip_model, text_tokens).float().to(device)
    )
    t5_embeddings_full = t5_model(captions).to(device)
    text_embeddings = torch.cat(
        [clip_embeddings, torch.mean(t5_embeddings_full, dim=1)], 1
    )
    text_embeddings_full = torch.cat(
        [clip_embeddings_full, t5_embeddings_full], 2
    )

    text_tokens_uncond = tokenizer.tokenize([''] * len(captions))
    text_tokens_uncond = text_tokens_uncond.to(device)
    clip_embeddings_uncond = clip_model.encode_text(text_tokens_uncond).float().to(device)
    clip_embeddings_full_uncond = generate_clip_embeddings(clip_model, text_tokens_uncond).float().to(device)
    t5_embeddings_full_uncond = t5_model([''] * len(captions)).to(device)
    text_embeddings_uncond = torch.cat(
        [clip_embeddings_uncond, torch.mean(t5_embeddings_full_uncond, dim=1)], 1
    )
    text_embeddings_full_uncond = torch.cat(
        [clip_embeddings_full_uncond, t5_embeddings_full_uncond], 2
    )

    return (
        text_embeddings,
        text_embeddings_full,
        text_embeddings_uncond,
        text_embeddings_full_uncond,
    )


def captions_to_prior_tensors_thread_safe(prior_model, captions):
    components = prior_model.components

    components.pop("prior")
    components.pop("prior_scheduler")
    # components.pop("text_proj")
    # components.pop("text_encoder")
    # components.pop("tokenizer")

    prior = deepcopy(prior_model.prior)
    scheduler = deepcopy(prior_model.prior_scheduler)
    # text_proj = deepcopy(prior_model.text_proj)
    # text_encoder = deepcopy(prior_model.text_encoder)
    # tokenizer = deepcopy(prior_model.tokenizer)

    _prior_model = UnCLIPPriorPipeline(
        **components,
        prior=prior,
        prior_scheduler=scheduler,
        # text_encoder,
        # tokenizer,
        # text_proj,
    )
    prior_flat = _prior_model(captions)
    prior_flat_uncond = _prior_model([''] * len(captions))
    return (prior_flat, prior_flat_uncond)


def captions_to_prior_tensors(_prior_model, captions):
    prior_flat = None
    prior_flat_uncond = None
    with mtx_prior:
        prior_flat = _prior_model(captions)
        prior_flat_uncond = _prior_model([''] * len(captions))
    return (prior_flat, prior_flat_uncond)


@app.post("/conditionings")
def conditionings(req: ConditioningRequest) -> Response:
    global clip_model
    global t5_model
    global prior_model

    try:
        flat = None
        full = None
        flat_uncond = None
        full_uncond = None
        prior_flat = None
        prior_flat_uncond = None
        with torch.no_grad():
            flat, full, flat_uncond, full_uncond = \
                captions_to_conditioning_tensors(clip_model, t5_model,
                    req.captions)
        prior_flat, prior_flat_uncond = \
            captions_to_prior_tensors(prior_model, req.captions)
        # resp = ConditioningResponse(
        #     flat=tensor_to_b64_string(flat),
        #     full=tensor_to_b64_string(full),
        #     flat_uncond=tensor_to_b64_string(flat_uncond),
        #     full_uncond=tensor_to_b64_string(full_uncond),
        # )
        resp = {
            'flat': tensor_to_b64_string(flat),
            'full': tensor_to_b64_string(full),
            'flat_uncond': tensor_to_b64_string(flat_uncond),
            'full_uncond': tensor_to_b64_string(full_uncond),
            'prior_flat': tensor_to_b64_string(prior_flat),
            'prior_flat_uncond': tensor_to_b64_string(prior_flat_uncond),
        }
        return Response(content=ujson.dumps(resp), media_type="application/json")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
