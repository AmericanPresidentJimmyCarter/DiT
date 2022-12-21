import base64
import io
from typing import Dict, List, Union

from fastapi import FastAPI, HTTPException

import argparse
import sys

import torch
from pydantic import BaseModel

import open_clip
from open_clip import tokenizer
from t5 import FrozenT5Embedder

from data import tensor_to_b64_string, get_dataloader_laion_coco


CONDITIONING_DEVICE = 'cuda:7'


class ConditioningRequest(BaseModel):
    captions: List[str]


class BatchResponse(BaseModel):
    images: str
    captions: List[str]
    conditioning_flat: str
    conditioning_full: str


class ConditioningResponse(BaseModel):
    flat: str
    full: str


class Arguments:
    batch_size = 16
    num_workers = 32
    dataset_path = "laion/laion-coco"
    # cache_dir = "/home/user/.cache"  # cache_dir for models


def spawn_clip_model():
    _clip_model, _, _ = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained="laion2b_s32b_b79k", # cache_dir=args.cache_dir
    )
    del _clip_model.visual
    _clip_model = _clip_model.to(CONDITIONING_DEVICE).eval().requires_grad_(False)
    return _clip_model


def spawn_t5_model():
    _t5_model = FrozenT5Embedder(
        device=CONDITIONING_DEVICE,
        # cache_dir=args.cache_dir,
    ).to(CONDITIONING_DEVICE)
    return _t5_model


dataset = get_dataloader_laion_coco(Arguments(), spawn_clip_model,
    spawn_t5_model)


clip_model = spawn_clip_model()
t5_model = spawn_t5_model()


batch_iterator = iter(dataset)
epoch = 0

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

    return (
        text_embeddings,
        text_embeddings_full,
    )


@app.post("/batch")
def batch() -> BatchResponse:
    global batch_iterator
    global epoch

    try:
        images, captions = next(batch_iterator)
        flat = captions.get('flat')
        full = captions.get('full')
        captions = captions.get('captions')
        resp = BatchResponse(
            captions=captions,
            images=tensor_to_b64_string(images),
            conditioning_flat=tensor_to_b64_string(flat),
            conditioning_full=tensor_to_b64_string(full),
        )
        return resp
    except StopIteration:
        epoch += 1
        print(f"Hit stop iteration, welcome to your next epoch: {epoch + 1}")
        batch_iterator = iter(dataset)
        images, captions = next(batch_iterator)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/conditionings")
def conditionings(req: ConditioningRequest) -> ConditioningResponse:
    global clip_model
    global t5_model

    try:
        flat = None
        full = None
        with torch.no_grad():
            flat, full = \
                captions_to_conditioning_tensors(clip_model, t5_model,
                    req.captions)
        resp = ConditioningResponse(
            flat=tensor_to_b64_string(flat),
            full=tensor_to_b64_string(full),
        )
        return resp
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
