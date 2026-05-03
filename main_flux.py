import os
import torch
import argparse
from tqdm import tqdm
from diffusers import FluxPipeline
from diffusers.utils.torch_utils import randn_tensor
import numpy as np
from diffusers.pipelines.flux.pipeline_flux import calculate_shift
import json
import math

from utils import *
from pipelines.flux1_dev import flux_timestep_pipe, update, decode_latent

BACKBONE_MODEL_IDS = {
    "flux1-dev":     "black-forest-labs/FLUX.1-dev",
    "flux1-schnell": "black-forest-labs/FLUX.1-schnell",
}

BACKBONE_DEFAULT_RES = {
    "flux1-dev":     1024,
    "flux1-schnell": 1024,
}

BACKBONE_DEFAULT_GUIDANCE = {
    "flux1-dev":     3.5,
    "flux1-schnell": 1.0,
}

BACKBONE_DEFAULT_STEPS = {
    "flux1-dev":     50,
    "flux1-schnell": 4,
}


def get_scheduled_value(total, cur, schedule_type):
    if schedule_type == 'constant':
        return total

    elif schedule_type == 'linear':  # 1 - (t/T)
        return total * (1. - cur / total)

    elif schedule_type == 'cosine':
        return total * 0.5 * (1. + math.cos(math.pi * cur / total))

    elif schedule_type == 'sqrt':
        return total * math.sqrt(1. - cur / total)

    elif schedule_type == 'concave':
        return (1. - cur / total) ** 2

    elif schedule_type == 'convex':
        return 1. - (cur / total) ** 2

    else:
        raise ValueError(f"Unknown schedule_type: {schedule_type}")


def main(args):
    print(f"Running with args: {args}")
    os.makedirs(os.path.join("results", args.exp_name), exist_ok=True)
    with open(os.path.join("results", args.exp_name, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    model_id = BACKBONE_MODEL_IDS[args.backbone]

    height = args.height if args.height is not None else BACKBONE_DEFAULT_RES[args.backbone]
    width  = args.width  if args.width  is not None else BACKBONE_DEFAULT_RES[args.backbone]
    h_lat, w_lat = height // 8, width // 8
    D = 16 * h_lat * w_lat

    print(f"Backbone: {args.backbone}  ({model_id})")
    print(f"Resolution: {height}x{width}  ->  latent {h_lat}x{w_lat}  D={D}")

    pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    pipe.transformer.eval()
    pipe.vae.eval()
    pipe.transformer.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
        pipe.text_encoder.requires_grad_(False)
        pipe.text_encoder.eval()
    if hasattr(pipe, "text_encoder_2") and pipe.text_encoder_2 is not None:
        pipe.text_encoder_2.requires_grad_(False)
        pipe.text_encoder_2.eval()

    guidance_scale = args.guidance_scale if args.guidance_scale is not None else BACKBONE_DEFAULT_GUIDANCE[args.backbone]
    num_inference_steps = args.num_inference_steps if args.num_inference_steps is not None else BACKBONE_DEFAULT_STEPS[args.backbone]

    prompts = parse_prompts(args.prompt_path)
    print(f"Parsed {len(prompts)} prompts from {args.prompt_path}")

    result_path = os.path.join("results", args.exp_name)
    os.makedirs(result_path, exist_ok=True)

    # latent_image_ids are the same for all prompts at fixed resolution.
    # Must pass h_lat//2, w_lat//2: _prepare_latent_image_ids returns (h*w, 3) IDs,
    # and the packed latent sequence length is (h_lat//2) * (w_lat//2).
    latent_image_ids = pipe._prepare_latent_image_ids(
        batch_size=1,
        height=h_lat // 2,
        width=w_lat // 2,
        device="cuda",
        dtype=torch.bfloat16,
    )

    for prompt_idx, prompt in enumerate(tqdm(prompts)):
        # FLUX encode_prompt returns (prompt_embeds, pooled_prompt_embeds, text_ids)
        # No negative prompts — guidance is handled inside the transformer for FLUX.1-dev
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
                prompt=prompt,
                prompt_2=None,
                device="cuda",
                num_images_per_prompt=1,
                max_sequence_length=args.max_sequence_length,
            )

        # Compute resolution-dependent time-shift (mu) to match official FluxPipeline behavior
        image_seq_len = (h_lat // 2) * (w_lat // 2)
        mu = calculate_shift(
            image_seq_len,
            pipe.scheduler.config.get("base_image_seq_len", 256),
            pipe.scheduler.config.get("max_image_seq_len", 4096),
            pipe.scheduler.config.get("base_shift", 0.5),
            pipe.scheduler.config.get("max_shift", 1.15),
        )
        sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
        timesteps, num_inference_steps_ = retrieve_timesteps(
            pipe.scheduler,
            num_inference_steps=num_inference_steps,
            device="cuda",
            sigmas=sigmas,
            mu=mu,
        )

        shape = (1, 16, h_lat, w_lat)
        init_latent = randn_tensor(
            shape,
            dtype=torch.bfloat16,
            device="cuda",
            generator=torch.Generator("cuda").manual_seed(args.seed),
        )

        latents = init_latent.clone()
        improved = None
        delta = None

        for i, t in enumerate(timesteps):
            best_latents = latents.detach()

            flow_t = 1. - t.item() / 1000
            noise_scale = get_scheduled_value(1., flow_t, schedule_type=args.perturb_schedule)

            v_func_kwargs = {
                'pipe': pipe,
                'latents': latents,
                'prompt_embeds': prompt_embeds,
                'pooled_prompt_embeds': pooled_prompt_embeds,
                'text_ids': text_ids,
                'latent_image_ids': latent_image_ids,
                't': t,
                'guidance_scale': guidance_scale,
            }

            best_latents, best_pred, improved, delta = divergence_stepper(
                v_func=flux_timestep_pipe,
                v_func_kwargs=v_func_kwargs,
                x_key='latents',
                seed_delta=1234,
                seed_eps=42,
                num_updates=args.num_iters,
                stop_t=0.5,
                delta_scale=args.noise_scale * noise_scale,
                improved=improved,
                delta=delta,
            )

            with torch.no_grad():
                latents = update(pipe, best_latents, t, best_pred, device="cuda")

        with torch.no_grad():
            image = decode_latent(pipe, latents, ret_type="pil", device="cuda")
            image.save(os.path.join(result_path, f"{prompt_idx:05d}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Divergence-based corrector sampling for FLUX.1 latents")
    parser.add_argument("--exp_name", type=str, default="debug")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt_path", type=str, default="data.csv")
    parser.add_argument(
        "--backbone",
        type=str,
        default="flux1-dev",
        choices=list(BACKBONE_MODEL_IDS.keys()),
        help="flux1-dev | flux1-schnell",
    )
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width",  type=int, default=512)
    parser.add_argument("--guidance_scale", type=float, default=None,
                        help="Guidance scale (default: backbone-specific; 3.5 for dev, 1.0 for schnell)")
    parser.add_argument("--max_sequence_length", type=int, default=512,
                        help="Max T5 sequence length for prompt encoding")

    parser.add_argument("--num_iters", type=int, default=1,
                        help="Number of extra proposals per timestep (baseline always included)")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of Hutchinson probe vectors per divergence estimate")
    parser.add_argument("--noise_scale", type=float, default=1e-3,
                        help="Base scale for proposal perturbation")
    parser.add_argument("--perturb_schedule", type=str, default="linear",
                        help="Perturbation scale schedule: constant | linear | cosine | sqrt | concave | convex")
    parser.add_argument("--t_until", type=int, default=-1,
                        help="Apply corrector until this timestep index (inclusive); -1 means all")
    parser.add_argument("--optimize_target", type=str, default="divergence",
                        help="divergence | abs_divergence | raw_divergence")

    parser.add_argument("--num_inference_steps", type=int, default=None,
                        help="Number of diffusion steps (default: backbone-specific; 50 for dev, 4 for schnell)")
    parser.add_argument("--hutchinson_dist", type=str, default="gaussian",
                        help="gaussian | rademacher")

    args = parser.parse_args()
    main(args)
