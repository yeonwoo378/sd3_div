import os
import torch
import numpy as np
import argparse
from tqdm import tqdm
from diffusers import StableDiffusion3Pipeline
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
import json
import torch.nn.functional as F

from utils import *
from pipelines.sd3 import sd3_timestep_pipe, update, decode_latent
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]
image.save("flux-dev.png")

def flux_timestep_pipe(pipe, latents, prompt_embeds, pooled_prompt_embeds, text_ids, t):
    latents, latent_image_ids = pipe.prepare_latents(
        1*1,
        pipe.num_channels_latents,
        512,
        512,
        prompt_embeds.dtype,
        'cuda',
        None,
        latents,
    )

    guidance = 3.5
    true_cfg_scale = 1.0
    _current_timestep= t
    timestep = t.expand(latents.shape[0]).to(latents.dtype)
    noise_pred = pipe.transformer(
        hidden_states=latents,
        timestep=timestep / 1000,
        guidance=guidance,
        pooled_projections=pooled_prompt_embeds,
        encoder_hidden_states=prompt_embeds,
        txt_ids=text_ids,
        img_ids=latent_image_ids,
        joint_attention_kwargs=pipe.joint_attention_kwargs,
        return_dict=False,
    )[0]

    return noise_pred



def sd3_timestep_pipe(pipe, latents, prompt_embeds, pooled_prompt_embeds, t,
          guidance_scale=7.0, device='cuda'):

    # latents = latents.detach().to(device)
    # latents.requires_grad_(True)
            
    latent_model_input = torch.cat([latents, latents], dim=0)
                
    timestep = t.expand(latent_model_input.shape[0])
    noise_pred = pipe.transformer(
        hidden_states=latent_model_input,
        timestep=timestep,
        encoder_hidden_states=prompt_embeds,
        pooled_projections=pooled_prompt_embeds,
        return_dict=False,
    )[0]

    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred_guided = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond) # is actually v

    return noise_pred_guided