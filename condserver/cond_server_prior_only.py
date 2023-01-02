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

from oclip_prior import load_prior_model, image_embeddings_for_text

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


app = FastAPI()
prior = load_prior_model()


def captions_to_conditioning_tensors(captions):
    return (
        torch.zeros(len(captions), 1024),
        torch.zeros(len(captions), 77, 1024),
        torch.zeros(len(captions), 1024),
        torch.zeros(len(captions), 77, 1024),
    )


def captions_to_prior_tensors(_prior, captions):
    prior_flat = None
    prior_flat_uncond = None
    with mtx_prior:
        prior_flat = image_embeddings_for_text(_prior, captions)
        prior_flat_uncond = image_embeddings_for_text(_prior, [''] * len(captions))
    return (prior_flat, prior_flat_uncond)


@app.post("/conditionings")
def conditionings(req: ConditioningRequest) -> Response:
    global prior

    try:
        flat = None
        full = None
        flat_uncond = None
        full_uncond = None
        prior_flat = None
        prior_flat_uncond = None
        with torch.no_grad():
            flat, full, flat_uncond, full_uncond = \
                captions_to_conditioning_tensors(req.captions)
        prior_flat, prior_flat_uncond = \
            captions_to_prior_tensors(prior, req.captions)
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
