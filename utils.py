import torch
import inspect
from typing import Any, Callable, Dict, List, Optional, Union
from tqdm import tqdm
from PIL import Image
import clip
import hpsv2


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def parse_prompts(csv_file):
    import csv

    prompt_list = []
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            if i == 0:
                continue  # Skip header
            prompt_list.append(row[0])
    return prompt_list


def get_ir(model, prompt, img_path, device='cuda'):
    scores = model.score(prompt, [img_path])
    return scores

def get_pickscore(model, processor, prompt, img_path, device='cuda'):
    image = Image.open(img_path).convert("RGB")
    image_inputs = processor(
        images = [image],
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt"
    ).to(device)

    text_inputs = processor(
        text = prompt,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        # embed
        image_embs = model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
    
        text_embs = model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
    
        # score
        scores = model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
        
    return scores.item()

def get_clipscore(model, preprocess, prompt, img_path, device='cuda'):
    img = Image.open(img_path).convert("RGB")
    image_input = preprocess(img).unsqueeze(0).to(device)
    text_input = clip.tokenize([prompt], truncate=True).to(device)

    with torch.no_grad():
        image_feats = model.encode_image(image_input)
        text_feats = model.encode_text(text_input)

        image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    similarity = (text_feats @ image_feats.T).squeeze().item()
    return similarity


def get_hpsv2(prompt, img_path, device='cuda'):
    scores = hpsv2.score(img_path, prompt, hps_version='v2.1')[0]
    return scores.item()