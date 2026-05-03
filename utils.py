import torch
import inspect
from typing import Any, Callable, Dict, List, Optional, Union
from tqdm import tqdm
from PIL import Image
import clip
import hpsv2
import numpy as np
from einops import rearrange, repeat, reduce
from transformers import CLIPProcessor, CLIPVisionModelWithProjection
from typing import Optional, Tuple, Dict
import torch.nn as nn

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


@torch.no_grad()
def divergence_stepper( v_func,
                        v_func_kwargs,
                        x_key='z',
                        t_key='t',
                        stop_t=1.0,
                        num_updates=1,
                        num_delta=1,
                        num_eps=1,
                        delta_scale=1,
                        delta_scheduler=lambda t: 2 ** (-t),
                        seed_delta=None,
                        seed_eps=None,
                        delta=None,
                        improved=None,
                        resample_delta=False,
                        resample_eps=False,
                        sequential_vjp=True,
                        sequential_hutchinson=True,
                        eta=0.0,
                        sync_over_time=False
                        ):

    # import ipdb; ipdb.set_trace()
    t = v_func_kwargs[t_key]
    if isinstance(t, torch.Tensor):
        assert (t == t.mean()).all().item(), "All timesteps in the batch must be the same for divergence_stepper."
        t = 1. - t.mean().item() / 1000

    if num_updates <= 0 or t > stop_t:
        return v_func_kwargs[x_key], v_func(**v_func_kwargs), improved, delta
    # import ipdb; ipdb.set_trace()
    z = v_func_kwargs[x_key]        
    B = z.shape[0]
    D = np.prod(z.shape[1:])  # C * H * W
    
    delta_generator = None
    eps_generator = None
    
    if seed_delta is not None:
        if sync_over_time:
            delta_generator = torch.Generator(device=z.device).manual_seed(seed_delta) # + int(t * 1000))
        else:
            delta_generator = torch.Generator(device=z.device).manual_seed(seed_delta + int(t * 1000))
    if seed_eps is not None:
        eps_generator = torch.Generator(device=z.device).manual_seed(seed_eps)
    sync_eps_with_delta = num_eps == 1 and seed_eps == seed_delta

    # import ipdb; ipdb.set_trace()
    
    for update_idx in range(num_updates):
        require_sample_delta = (update_idx == 0) or resample_delta
        require_sample_eps = (update_idx == 0) or resample_eps

        # compute divergence and find the best perturbation
        if sequential_vjp:
            assert (not resample_delta) or (num_delta==1)
            for delta_idx in range(num_delta+1):

                # pass if no need to get the divergence of original z
                # if delta_idx == 0 and update_idx !=0: # buggy
                #     continue
                
                if delta is None or improved is None:
                    assert improved is None and t == 0.0 and delta_idx <= 1
                    delta = torch.randn(z.shape, generator=delta_generator, device=z.device) # if delta_idx != 0 else torch.zeros_like(z, device=z.device)
                elif update_idx > 0:
                    temp_delta_generator = torch.Generator(device=z.device).manual_seed(seed_delta + update_idx + int(t * 1000))
                    temp_delta = torch.randn(z.shape, generator=temp_delta_generator, device=z.device)
                    pass
                elif delta_idx > 0:
                    new_delta = torch.randn(z.shape, generator=delta_generator, device=z.device) #if delta_idx != 0 else torch.zeros_like(z, device=z.device)
                    delta = torch.where(
                        improved.reshape(-1, *([1]*(z.ndim-1))), # hard-coded shape
                        delta, # True
                        new_delta # False
                    )
                # no update delta when delta_idx=0
                assert seed_delta != seed_eps, "Is a Biased Estimator"

                if sync_eps_with_delta and delta_idx != 0:
                    eps = delta.detach()
                    raise NotImplementedError # not using anymore!

                else:
                    eps = torch.randn(z.shape, generator=eps_generator, device=z.device) 

                if delta_idx == 0:
                    perturbed_z = z 
                elif update_idx == 0:
                    perturbed_z = z + delta_scale * delta_scheduler(update_idx) * delta # TODO: clarify
                else:
                    perturbed_z = z + delta_scale * delta_scheduler(update_idx) * temp_delta # TODO: clarify
                with torch.enable_grad():
                    perturbed_z = perturbed_z.detach().requires_grad_(True)
                    v_func_kwargs[x_key] = perturbed_z
                    
                    v_pred = v_func(**v_func_kwargs)  # [B, C, H, W]
                    v_pred_eps = (v_pred * eps).flatten(1).sum(1)  # [B]
                    grad_v = torch.autograd.grad(
                        outputs=v_pred_eps,          # [B]
                        inputs=perturbed_z,                      # [B, C, H, W]
                        grad_outputs=torch.ones_like(v_pred_eps),  # [B]
                        create_graph=False,
                        retain_graph=False,         
                    )[0].detach()  # [B, C, H, W]
                    divergence = -(grad_v * eps).flatten(1).sum(1) / D  # [B]
                
                threshold = - (1 / (1 - t))

                if delta_idx == 0:
                    best_divergence = divergence.detach()
                    best_v_pred = v_pred.detach()
                    best_perturbed_z = perturbed_z.detach()
                elif update_idx == 0:
                    improved = (divergence < (best_divergence - eta)) #& (best_divergence >= threshold)
                    improved_shape = (B,) + (1,) * (len(z.shape) - 1)
                    best_divergence = torch.where(improved, divergence, best_divergence)
                    best_v_pred = torch.where(
                        improved.view(improved_shape),
                        v_pred,
                        best_v_pred,
                    )
                    # print(improved.view(improved_shape).shape)
                    best_perturbed_z = torch.where(
                        improved.view(improved_shape),
                        perturbed_z.detach(),
                        best_perturbed_z,
                    )
                else:
                    temp_improved = (divergence < (best_divergence - eta)) #& (best_divergence >= threshold)
                    improved_shape = (B,) + (1,) * (len(z.shape) - 1)
                    best_divergence = torch.where(temp_improved, divergence, best_divergence)
                    best_v_pred = torch.where(
                        temp_improved.view(improved_shape),
                        v_pred,
                        best_v_pred,
                    )
                    # print(improved.view(improved_shape).shape)
                    best_perturbed_z = torch.where(
                        temp_improved.view(improved_shape),
                        perturbed_z.detach(),
                        best_perturbed_z,
                    )

                    # improved = improved & temp_improved

            # update iteration-wise
            z = best_perturbed_z # update z
            v_pred = best_v_pred
        
        # currently not using hereafter
        else:
            # build delta
            raise NotImplementedError
            if require_sample_delta:            
                delta_shape = (num_delta+1, ) + z.shape  # [num_delta, B, C, H, W]
                delta = torch.randn(delta_shape, generator=delta_generator, device=z.device, dtype=z.dtype)  # [num_delta, B, C, H, W]
                delta[0] = 0.0  # no perturbation for the first sample
            
            # build eps.
            if require_sample_eps:
                if sync_eps_with_delta:
                    eps = delta.unsqueeze(0)  # [1, num_delta * B, C, H, W]
                    # eps = repeat(eps, '1 nd b ... -> (ne nd b) ...', ne=num_eps)
                else:
                    eps_shape = (num_eps,) + z.shape  # [num_eps, B, C, H, W]
                    eps = torch.randn(eps_shape, generator=eps_generator, device=z.device, dtype=z.dtype)  # [num_eps, B, C, H, W]
                    # eps_shape = (num_eps, num_delta+1,) + z.shape # Tip: sample more and rearrange for independent eps.
            
            # expand v_func_kwargs
            perturbed_z = z.unsqueeze(0) + delta_scale * delta_scheduler(t) * delta
            perturbed_z = rearrange(perturbed_z, 'nd b ... -> (nd b) ...', nd=num_delta+1, b=B)
            perturbed_z = perturbed_z.detach().requires_grad_(True)
            
            # compute v
            with torch.enable_grad():            
                v_func_kwargs[x_key] = perturbed_z
                v_func_kwargs_expanded = expand_v_func_kwargs(v_func_kwargs, batch_size=B, expand_size=num_delta+1)
                v_pred_expanded = v_func(**v_func_kwargs_expanded)  # [(num_delta+1) * B, C, H, W]

                divergence = []
                if sequential_hutchinson:
                    for eps_idx, eps_i in enumerate(eps):  # [B, C, H, W]
                        retain_graph = (eps_idx < eps.shape[0] - 1) # retain graph except for the last one
                        v_pred_eps = (v_pred_expanded * eps_i.unsqueeze(0)).flatten(1).sum(1)  # [(num_delta+1) * B]
                        grad_v = torch.autograd.grad(
                            outputs=v_pred_eps,          # [(num_delta+1) * B]
                            inputs=perturbed_z,                      # [(num_delta+1) * B, C, H, W]
                            grad_outputs=torch.ones_like(v_pred_eps),  # [(num_delta+1) * B]
                            create_graph=False,
                            retain_graph=retain_graph,         
                        )[0].detach()  # [(num_delta+1) * B, C, H, W]
                        
                        divergence_i = (grad_v * eps_i.unsqueeze(0)).flatten(1).sum(1) / D  # [(num_delta+1) * B]
                        divergence.append(divergence_i)
                    divergence = torch.stack(divergence, dim=0)  # [num_eps, (num_delta+1) * B]
                    divergence = divergence.mean(0)  # [(num_delta+1) * B]
                else:
                    v_pred_eps = (v_pred_expanded.unsqueeze(0) * eps.flatten(1).unsqueeze(1)).flatten(2).sum(2)  # [num_eps, (num_delta+1) * B]
                    grad_v = torch.autograd.grad(
                        outputs=v_pred_eps,          # [num_eps, (num_delta+1) * B]
                        inputs=perturbed_z,                      # [(num_delta+1) * B, C, H, W]
                        grad_outputs=torch.ones_like(v_pred_eps),  # [num_eps, (num_delta+1) * B]
                        create_graph=False,
                        retain_graph=False,         
                    )[0].detach()  # [(num_delta+1) * B, C, H, W]
                    
                    divergence = (grad_v.unsqueeze(0) * eps.flatten(1).unsqueeze(1)).flatten(2).sum(2) / D  # [num_eps, (num_delta+1) * B]
                    divergence = divergence.mean(0)  # [(num_delta+1) * B]
            # select the best perturbation based on divergence
            divergence = divergence.view(num_delta+1, B)  # [num_delta+1, B]
            best_divergence, best_idx = torch.min(divergence, dim=0)
            best_perturbed_z = rearrange(perturbed_z, '(nd b) ... -> nd b ...', nd=num_delta+1, b=B)[best_idx, torch.arange(B)]  # [B, C, H, W]
            best_v_pred = rearrange(v_pred_expanded, '(nd b) ... -> nd b ...', nd=num_delta+1, b=B)[best_idx, torch.arange(B)]  # [B, C, H, W]
            z = best_perturbed_z.detach()
    return best_perturbed_z, best_v_pred, improved, delta


_AESTHETIC_URLS: Dict[str, str] = {
    "openai/clip-vit-large-patch14": "https://github.com/LAION-AI/aesthetic-predictor/raw/main/sa_0_4_vit_l_14_linear.pth",
    "openai/clip-vit-base-patch16":  "https://github.com/LAION-AI/aesthetic-predictor/raw/main/sa_0_4_vit_b_16_linear.pth",
    "openai/clip-vit-base-patch32":  "https://github.com/LAION-AI/aesthetic-predictor/raw/main/sa_0_4_vit_b_32_linear.pth",
}

# simple in-process cache to avoid re-loading every call
_CACHE: Dict[Tuple[str, str], Tuple[CLIPProcessor, CLIPVisionModelWithProjection, nn.Linear]] = {}


def _get_device(device: Optional[str]) -> torch.device:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    return torch.device(device)


def load_aesthetic_models(
    clip_model_name: str,
    device: torch.device,
    cache_dir: Optional[str] = None,
) -> Tuple[CLIPProcessor, CLIPVisionModelWithProjection, nn.Linear]:
    key = (clip_model_name, str(device))
    if key in _CACHE:
        return _CACHE[key]

    if clip_model_name not in _AESTHETIC_URLS:
        raise ValueError(f"Unsupported clip_model_name: {clip_model_name}. "
                         f"Choose one of {list(_AESTHETIC_URLS.keys())}")

    processor = CLIPProcessor.from_pretrained(clip_model_name)
    vision = CLIPVisionModelWithProjection.from_pretrained(clip_model_name).to(device).eval()

    proj_dim = vision.config.projection_dim  # e.g., 768 for ViT-L/14
    head = nn.Linear(proj_dim, 1).to(device).eval()

    url = _AESTHETIC_URLS[clip_model_name]
    # torch.hub handles caching; you can also pass a custom cache_dir if you want
    state_dict = torch.hub.load_state_dict_from_url(
        url,
        model_dir=cache_dir,
        map_location="cpu",
        check_hash=False,
    )
    head.load_state_dict(state_dict)

    _CACHE[key] = (processor, vision, head)
    return processor, vision, head


@torch.inference_mode()
def aesthetic_score(processor, vision, head, prompt: str, image_path: str, device: Optional[str] = None) -> float:
    """
    Returns LAION aesthetic score (roughly 0..10).
    - prompt: accepted for API compatibility (NOT used in v1 aesthetic predictor)
    - image_path: path to image file
    """
    # dev = _get_device(device)
    # clip_model_name = "openai/clip-vit-large-patch14"

    # processor, vision, head = _load_aesthetic_models(clip_model_name, dev)

    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to('cuda')

    outputs = vision(pixel_values=pixel_values)
    image_embeds = outputs.image_embeds if hasattr(outputs, "image_embeds") else outputs[0]

    # IMPORTANT: L2-normalize CLIP image embedding before linear head (as in reference implementations)
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

    score = head(image_embeds).squeeze().item()
    return float(score)
