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

D = 16 * 64 * 64  # latent dimension (per sample) SD3


def _make_hutchinson_eps_like(x: torch.Tensor, generator: torch.Generator, dist: str = "gaussian"):
    """
    Hutchinson probe vector eps with E[eps eps^T] = I.
    gaussian: eps ~ N(0, I)
    rademacher: eps_i in {-1, +1} (lower variance often; optional)
    """
    if dist == "gaussian":
        return torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
    elif dist == "rademacher":
        # {0,1} -> {-1,+1}
        r = torch.randint(0, 2, x.shape, device=x.device, generator=generator, dtype=torch.int8)
        return (r * 2 - 1).to(dtype=x.dtype)
    else:
        raise ValueError(f"Unknown Hutchinson dist: {dist}")


def estimate_divergence_hutchinson(
    pipe,
    latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    t,
    eps_list,
    guidance_scale: float = 7.0,
    device: str = "cuda",
    return_pred = True
):
    """
    Returns scalar divergence estimate:
      div(u)(x) = Tr(du/dx)
    using Hutchinson estimator: Tr(J) = E[eps^T J eps].

    Implementation uses vJP:
      eps^T J eps = d/dx <u(x), eps> dotted with eps.
    """
    # Make latents a leaf so autograd.grad works reliably and does not keep old graphs.
    x = latents.detach().requires_grad_(True)

    # Forward: u(x) (here: "pred" from sd3_timestep_pipe)
    pred = sd3_timestep_pipe(
        pipe,
        x,
        prompt_embeds,
        pooled_prompt_embeds,
        t,
        guidance_scale=guidance_scale,
        device=device,
    )

    # Accumulate multiple Hutchinson probes (common across candidates if eps_list is shared)
    div_total = torch.zeros((), device=x.device, dtype=torch.float32)

    n = len(eps_list)

    assert n == 1
    with torch.enable_grad():
        for s, eps in enumerate(eps_list):
            # Important: eps should not require grad
            eps = eps.detach()

            # scalar: <pred, eps> / D  (divide-by-D is optional; keeps scale tame)
            vp = torch.sum(pred * eps) / D

            # vjp = d/dx vp = (J^T eps) / D
            vjp = torch.autograd.grad(
                vp,
                x,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )[0]

            # eps^T J eps / D
            div_s = torch.sum(vjp * eps).to(torch.float32)
            div_total = div_total + div_s

    div_avg = div_total / float(n)

    # Return scalar tensor (float32)
    if return_pred:
        return div_avg, pred
    return div_avg


def divergence_objective(div_val: torch.Tensor, optimize_target: str):
    """
    We generally want "small magnitude" divergence, not large negative divergence.
    Default: squared magnitude.
    """
    if optimize_target == "divergence":
        return div_val * div_val  # minimize |div| via square
    elif optimize_target == "abs_divergence":
        return torch.abs(div_val)
    elif optimize_target == "raw_divergence":
        return div_val
    else:
        raise ValueError(f"Unknown optimize_target: {optimize_target}")


def main(args):
    print(f"Running with args: {args}")
    os.makedirs(os.path.join("results", args.exp_name), exist_ok=True)
    with open(os.path.join("results", args.exp_name, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # load pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=torch.float16,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    pipe.transformer.eval()
    pipe.vae.eval()
    # Freeze params so autograd only computes grads w.r.t. latents (saves compute/memory)
    pipe.transformer.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
        pipe.text_encoder.requires_grad_(False)
        pipe.text_encoder.eval()
    if hasattr(pipe, "text_encoder_2") and pipe.text_encoder_2 is not None:
        pipe.text_encoder_2.requires_grad_(False)
        pipe.text_encoder_2.eval()
    if hasattr(pipe, "text_encoder_3") and pipe.text_encoder_3 is not None:
        pipe.text_encoder_3.requires_grad_(False)
        pipe.text_encoder_3.eval()

    prompts = parse_prompts(args.prompt_path)
    print(f"Parsed {len(prompts)} prompts from {args.prompt_path}")

    result_path = os.path.join("results", args.exp_name)
    os.makedirs(result_path, exist_ok=True)

    for prompt_idx, prompt in enumerate(tqdm(prompts)): # batchsize=1
        # Encode prompt (no grads needed)
        with torch.no_grad():
            # encode_prompt: if prompt_2/prompt_3 is None, it falls back to prompt internally :contentReference[oaicite:2]{index=2}
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = (
                pipe.encode_prompt(prompt, None, None)
            )
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        # preprocess timestep
        timesteps, num_inference_steps = retrieve_timesteps(
            pipe.scheduler,
            num_inference_steps=args.num_inference_steps,
            device="cuda",
        )

        shape = (1, 16, 64, 64)
        init_latent = randn_tensor(
            shape,
            dtype=torch.float16,
            device="cuda",
            generator=torch.Generator("cuda").manual_seed(args.seed),
        )

        latents = init_latent.clone()


        for i, t in enumerate(timesteps):
            # lr schedule (keep your original logic)
            if args.lr_schedule == "linear":
                lr = args.lr * (1 - i / len(timesteps))
            elif args.lr_schedule == "sqrt":
                lr = args.lr * (1 - (i / len(timesteps)) ** 0.5)
            elif args.lr_schedule == "cbrt":
                lr = args.lr * (1 - (i / len(timesteps)) ** (1.0 / 3.0))
            elif args.lr_schedule == "constant":
                lr = args.lr
 
            else:
                raise ValueError(f"Unknown lr_schedule: {args.lr_schedule}")

            
            best_latents = latents.detach()
            best_div = None
            best_pred = None

            # Generators for reproducibility per (prompt, timestep)
            eps_gen = torch.Generator("cuda").manual_seed(args.seed + 10 * i + 1)

            # Hutchinson probes (shared across all candidate latents at this timestep)
            eps_list = [
                _make_hutchinson_eps_like(latents, generator=eps_gen, dist=args.hutchinson_dist) # gaussain
                for _ in range(args.num_samples)
            ]

            sigma = float(lr) * float(args.noise_scale)

            for iter in range(args.num_iters+1):
                if args.t_until != -1 and i > args.t_until: # currently doing half
                    break  # skip corrector after t_until
                if iter == 0:
                    proposal = best_latents
                else:
                    proposal = best_latents + sigma * torch.randn(best_latents.shape,  device=best_latents.device, dtype=best_latents.dtype, generator=eps_gen)

                div_val, pred = estimate_divergence_hutchinson(
                    pipe,
                    proposal,
                    prompt_embeds,
                    pooled_prompt_embeds,
                    t,
                    eps_list,
                    guidance_scale=7.0,
                    device="cuda",
                )
                # obj, pred = divergence_objective(div_val, args.optimize_target).detach()

                if div_val.item() < best_div.item():
                    best_div = div_val.detach()
                    best_latents = proposal.detach()
                    best_pred = pred.detach()

            # update latents (predictor)
            with torch.no_grad():
                final_pred = best_pred

                latents = update(
                    pipe,
                    best_latents,
                    t,
                    final_pred,
                    device="cuda",
                )

        # decode + save
        with torch.no_grad():
            image = decode_latent(pipe, latents, ret_type="pil", device="cuda")
            image.save(os.path.join(result_path, f"{prompt_idx:05d}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Divergence-based corrector sampling for SD3 latents")
    parser.add_argument("--exp_name", type=str, default="debug", help="Name of the experiment")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--prompt_path", type=str, default="data.csv", help="Path to the prompt file")

    parser.add_argument("--lr", type=float, default=0.1, help="Scale for corrector proposal radius (used with noise_scale)")
    parser.add_argument("--num_iters", type=int, default=1, help="Number of extra proposals per timestep (baseline always included)")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of Hutchinson probe vectors per divergence estimate")
    parser.add_argument("--noise_scale", type=float, default=1e-3, help="Base scale for proposal perturbation")
    parser.add_argument("--lr_schedule", type=str, default="constant", help="Learning rate schedule")
    parser.add_argument("--t_until", type=int, default=-1, help="Apply corrector until this timestep index (inclusive); -1 means all timesteps")
    parser.add_argument("--optimize_target", type=str, default="divergence", help="divergence | abs_divergence | raw_divergence")

    parser.add_argument("--num_inference_steps", type=int, default=25, help="Number of diffusion steps")

    # Added: choose Hutchinson probe distribution (kept default as gaussian to avoid big behavior change)
    parser.add_argument("--hutchinson_dist", type=str, default="gaussian", help="gaussian | rademacher")

    args = parser.parse_args()
    main(args)
