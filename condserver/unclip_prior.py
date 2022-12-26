# Copyright 2022 Kakao Brain and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Union

import torch
from torch.nn import functional as F

from diffusers import PriorTransformer
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import UnCLIPScheduler
from transformers import CLIPTextModelWithProjection, CLIPTokenizer

from diffusers.utils import is_accelerate_available, logging
from diffusers.pipelines.unclip.text_proj import UnCLIPTextProjModel

DEVICE = 'cuda:0'
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class UnCLIPPriorPipeline(DiffusionPipeline):
    """
    Pipeline for text-to-image generation using unCLIP

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        text_encoder ([`CLIPTextModelWithProjection`]):
            Frozen text-encoder.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        prior ([`PriorTransformer`]):
            The canonincal unCLIP prior to approximate the image embedding from the text embedding.
    """

    prior: PriorTransformer
    prior_scheduler: UnCLIPScheduler
    text_proj: UnCLIPTextProjModel
    text_encoder: CLIPTextModelWithProjection
    tokenizer: CLIPTokenizer

    def __init__(
        self,
        prior: PriorTransformer,
        prior_scheduler: UnCLIPScheduler,
        text_encoder: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        text_proj: UnCLIPTextProjModel,
    ):
        super().__init__()

        self.register_modules(
            prior=prior,
            prior_scheduler=prior_scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_proj=text_proj,
        )


    def prepare_latents(self, shape, dtype, device, generator, latents, scheduler):
        if latents is None:
            latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        latents = latents * scheduler.init_noise_sigma
        return latents

    def _encode_prompt(self, prompt, device, num_images_per_prompt, do_classifier_free_guidance):
        device = 'cpu'
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        # get prompt text embeddings
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        text_mask = text_inputs.attention_mask.bool().to(device)

        if text_input_ids.shape[-1] > self.tokenizer.model_max_length:
            removed_text = self.tokenizer.batch_decode(text_input_ids[:, self.tokenizer.model_max_length :])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )
            text_input_ids = text_input_ids[:, : self.tokenizer.model_max_length]

        # text_encoder_output = self.text_encoder(text_input_ids.to(device))
        text_encoder_output = self.text_encoder(text_input_ids)
        text_encoder_output.text_embeds.to(device)
        text_encoder_output.last_hidden_state.to(device)

        text_embeddings = text_encoder_output.text_embeds
        text_encoder_hidden_states = text_encoder_output.last_hidden_state

        text_embeddings = text_embeddings.repeat_interleave(num_images_per_prompt, dim=0)
        text_encoder_hidden_states = text_encoder_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
        text_mask = text_mask.repeat_interleave(num_images_per_prompt, dim=0)

        if do_classifier_free_guidance:
            uncond_tokens = [""] * batch_size

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_text_mask = uncond_input.attention_mask.bool().to(device)
            uncond_embeddings_text_encoder_output = self.text_encoder(uncond_input.input_ids.to(device))

            uncond_embeddings = uncond_embeddings_text_encoder_output.text_embeds
            uncond_text_encoder_hidden_states = uncond_embeddings_text_encoder_output.last_hidden_state

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method

            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_images_per_prompt)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_images_per_prompt, seq_len)

            seq_len = uncond_text_encoder_hidden_states.shape[1]
            uncond_text_encoder_hidden_states = uncond_text_encoder_hidden_states.repeat(1, num_images_per_prompt, 1)
            uncond_text_encoder_hidden_states = uncond_text_encoder_hidden_states.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )
            uncond_text_mask = uncond_text_mask.repeat_interleave(num_images_per_prompt, dim=0)

            # done duplicates

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
            text_encoder_hidden_states = torch.cat([uncond_text_encoder_hidden_states, text_encoder_hidden_states])

            text_mask = torch.cat([uncond_text_mask, text_mask])

        return text_embeddings, text_encoder_hidden_states, text_mask

    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, the pipeline's
        models have their state dicts saved to CPU and then are moved to a `torch.device('meta') and loaded to GPU only
        when their specific submodule has its `forward` method called.
        """
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        # TODO: self.prior.post_process_latents is not covered by the offload hooks, so it fails if added to the list
        models = [
            self.decoder,
            self.text_proj,
            self.text_encoder,
        ]
        for cpu_offloaded_model in models:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if self.device != torch.device("meta") or not hasattr(self.decoder, "_hf_hook"):
            return self.device
        for module in self.decoder.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        device: str = DEVICE,
        generator: Optional[torch.Generator] = torch.Generator(device=DEVICE),
        prior_guidance_scale: float = 4.0,
        prior_num_inference_steps: int = 25,
    ):
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            prior_guidance_scale (`float`, *optional*, defaults to 4.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
        """
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        do_classifier_free_guidance = prior_guidance_scale > 1.0

        text_embeddings, text_encoder_hidden_states, text_mask = self._encode_prompt(
            prompt, device, 1, do_classifier_free_guidance
        )

        # prior

        self.prior_scheduler.set_timesteps(prior_num_inference_steps, device=device)
        prior_timesteps_tensor = self.prior_scheduler.timesteps

        embedding_dim = self.prior.config.embedding_dim
        prior_latents = self.prepare_latents(
            (batch_size, embedding_dim),
            text_embeddings.dtype,
            device,
            generator,
            None,
            self.prior_scheduler,
        )
        prior_latents.to(device)

        for i, t in enumerate(prior_timesteps_tensor):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([prior_latents] * 2) if do_classifier_free_guidance else prior_latents
            latent_model_input = latent_model_input.to(device)
            text_embeddings = text_embeddings.to(device)
            text_encoder_hidden_states = text_encoder_hidden_states.to(device)
            text_mask = text_mask.to(device)
            self.prior = self.prior.to(device)

            predicted_image_embedding = self.prior(
                latent_model_input,
                timestep=t,
                proj_embedding=text_embeddings,
                encoder_hidden_states=text_encoder_hidden_states,
                attention_mask=text_mask,
            ).predicted_image_embedding

            if do_classifier_free_guidance:
                predicted_image_embedding_uncond, predicted_image_embedding_text = predicted_image_embedding.chunk(2)
                predicted_image_embedding = predicted_image_embedding_uncond + prior_guidance_scale * (
                    predicted_image_embedding_text - predicted_image_embedding_uncond
                )

            if i + 1 == prior_timesteps_tensor.shape[0]:
                prev_timestep = None
            else:
                prev_timestep = prior_timesteps_tensor[i + 1]

            prior_latents = self.prior_scheduler.step(
                predicted_image_embedding,
                timestep=t,
                sample=prior_latents,
                generator=generator,
                prev_timestep=prev_timestep,
            ).prev_sample

        prior_latents = self.prior.post_process_latents(prior_latents)

        image_embeddings = prior_latents

        return image_embeddings
