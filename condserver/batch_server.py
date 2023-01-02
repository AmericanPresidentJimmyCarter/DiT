import base64
import io
from typing import Dict, List, Union
import ujson
from fastapi import FastAPI, HTTPException, Response

import argparse
import sys

import torch
from pydantic import BaseModel


from data import tensor_to_b64_string, get_dataloader


class BatchRequest(BaseModel):
    is_main: bool


class BatchResponse(BaseModel):
    images: str
    captions: List[str]
    conditioning_flat: str
    conditioning_full: str
    unconditioning_flat: str
    unconditioning_full: str
    prior_flat: str
    prior_flat_uncond: str


class Arguments:
    small_batch_size = 16
    batch_size = 18
    num_workers = 4

    # small_batch_size = 2
    # batch_size = 2
    # num_workers = 1
    # dataset_path = "laion/laion-coco"
    dataset_path = 'gigant/oldbookillustrations_2'
    cache_dir = '/scratch/hfcache'
    # cache_dir = "/home/user/.cache"  # cache_dir for models


dataset = get_dataloader(Arguments())


batch_iterator = iter(dataset)
epoch = 0

app = FastAPI()

@app.post("/batch")
def batch(req: BatchRequest) -> Response:
    global batch_iterator
    global epoch

    batch_size = Arguments().batch_size

    try:
        images, captions = next(batch_iterator)

        flat = captions.get('flat')
        full = captions.get('full')
        flat_uncond = captions.get('flat_uncond')
        # print('flat', flat_uncond)
        full_uncond = captions.get('full_uncond')
        # print('full', full_uncond)
        prior_flat = captions.get('prior_flat')
        prior_flat_uncond = captions.get('prior_flat_uncond')
        captions = captions.get('captions')
        # resp = BatchResponse(
        #     captions=captions,
        #     images=tensor_to_b64_string(images),
        #     conditioning_flat=tensor_to_b64_string(flat),
        #     conditioning_full=tensor_to_b64_string(full),
        #     unconditioning_flat=tensor_to_b64_string(flat_uncond),
        #     unconditioning_full=tensor_to_b64_string(full_uncond),
        # )
        print(f'Distributing new batch, size {len(images)}')
        resp = {
            'captions': captions,
            'images': tensor_to_b64_string(images),
            'conditioning_flat': tensor_to_b64_string(flat),
            'conditioning_full': tensor_to_b64_string(full),
            'unconditioning_flat': tensor_to_b64_string(flat_uncond),
            'unconditioning_full': tensor_to_b64_string(full_uncond),
            'prior_flat': tensor_to_b64_string(prior_flat),
            'prior_flat_uncond': tensor_to_b64_string(prior_flat_uncond),
        }
        # if req.is_main:
        #    resp['captions'] = resp['captions'][0:Arguments().small_batch_size]
        #    resp['images'] = resp['images'][0:Arguments().small_batch_size]
        #    resp['conditioning_flat'] = resp['conditioning_flat'][0:Arguments().small_batch_size]
        #    resp['conditioning_full'] = resp['conditioning_full'][0:Arguments().small_batch_size]
        #    resp['unconditioning_flat'] = resp['unconditioning_flat'][0:Arguments().small_batch_size]
        #    resp['unconditioning_full'] = resp['unconditioning_full'][0:Arguments().small_batch_size]

        return Response(content=ujson.dumps(resp), media_type="application/json")
    except StopIteration:
        epoch += 1
        print(f"Hit stop iteration, welcome to your next epoch: {epoch + 1}")
        batch_iterator = iter(dataset)
        images, captions = next(batch_iterator)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
