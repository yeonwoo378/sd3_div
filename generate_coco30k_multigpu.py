#!/usr/bin/env python3
"""
Generate COCO-30K images with 8-GPU parallelism using the SD3 divergence corrector.

Each GPU processes an independent shard (stride-based) of the prompt list,
so no inter-process communication or DDP is needed.

Example:
    python generate_coco30k_multigpu.py \
        --metadata-jsonl coco_val2014_30k_seed42/metadata.jsonl \
        --results-dir results/coco30k_sd3_div \
        --num-gpus 8 \
        --num-inference-steps 50 \
        --num-iters 1 \
        --noise-scale 1e-3 \
        --perturb-schedule linear \
        --seed 0 \
        --skip-existing
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import inspect
from typing import List, Optional, Union

import numpy as np
import torch
import torch.multiprocessing as mp
from diffusers import StableDiffusion3Pipeline
from diffusers.utils.torch_utils import randn_tensor
from tqdm import tqdm

from coco30k_loader_and_generate import Coco30KPromptDataset
from pipelines.sd3 import decode_latent, sd3_timestep_pipe, update


# ── helpers inlined from utils.py to avoid its heavy optional imports ──────────

def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")
    if timesteps is not None:
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
    elif sigmas is not None:
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
    return scheduler.timesteps, len(scheduler.timesteps)


@torch.no_grad()
def divergence_stepper(
    v_func,
    v_func_kwargs,
    x_key="z",
    t_key="t",
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
    sync_over_time=False,
):
    t = v_func_kwargs[t_key]
    if isinstance(t, torch.Tensor):
        t = 1.0 - t.mean().item() / 1000

    if num_updates <= 0 or t > stop_t:
        return v_func_kwargs[x_key], v_func(**v_func_kwargs), improved, delta

    z = v_func_kwargs[x_key]
    B = z.shape[0]
    D = int(np.prod(z.shape[1:]))

    delta_generator = (
        torch.Generator(device=z.device).manual_seed(seed_delta + int(t * 1000))
        if seed_delta is not None and not sync_over_time
        else (
            torch.Generator(device=z.device).manual_seed(seed_delta)
            if seed_delta is not None
            else None
        )
    )
    eps_generator = (
        torch.Generator(device=z.device).manual_seed(seed_eps)
        if seed_eps is not None
        else None
    )
    sync_eps_with_delta = num_eps == 1 and seed_eps == seed_delta

    for update_idx in range(num_updates):
        assert sequential_vjp, "Only sequential_vjp=True is supported here."
        assert (not resample_delta) or (num_delta == 1)

        for delta_idx in range(num_delta + 1):
            if delta is None or improved is None:
                delta = torch.randn(z.shape, generator=delta_generator, device=z.device)
            elif update_idx > 0:
                temp_delta_generator = torch.Generator(device=z.device).manual_seed(
                    seed_delta + update_idx + int(t * 1000)
                )
                temp_delta = torch.randn(z.shape, generator=temp_delta_generator, device=z.device)
            elif delta_idx > 0:
                new_delta = torch.randn(z.shape, generator=delta_generator, device=z.device)
                delta = torch.where(
                    improved.reshape(-1, *([1] * (z.ndim - 1))),
                    delta,
                    new_delta,
                )

            assert seed_delta != seed_eps, "Is a Biased Estimator"
            eps = torch.randn(z.shape, generator=eps_generator, device=z.device)

            if delta_idx == 0:
                perturbed_z = z
            elif update_idx == 0:
                perturbed_z = z + delta_scale * delta_scheduler(update_idx) * delta
            else:
                perturbed_z = z + delta_scale * delta_scheduler(update_idx) * temp_delta

            with torch.enable_grad():
                perturbed_z = perturbed_z.detach().requires_grad_(True)
                v_func_kwargs[x_key] = perturbed_z
                v_pred = v_func(**v_func_kwargs)
                v_pred_eps = (v_pred * eps).flatten(1).sum(1)
                grad_v = torch.autograd.grad(
                    outputs=v_pred_eps,
                    inputs=perturbed_z,
                    grad_outputs=torch.ones_like(v_pred_eps),
                    create_graph=False,
                    retain_graph=False,
                )[0].detach()
                divergence = -(grad_v * eps).flatten(1).sum(1) / D

            if delta_idx == 0:
                best_divergence = divergence.detach()
                best_v_pred = v_pred.detach()
                best_perturbed_z = perturbed_z.detach()
            elif update_idx == 0:
                improved = divergence < (best_divergence - eta)
                improved_shape = (B,) + (1,) * (len(z.shape) - 1)
                best_divergence = torch.where(improved, divergence, best_divergence)
                best_v_pred = torch.where(improved.view(improved_shape), v_pred, best_v_pred)
                best_perturbed_z = torch.where(
                    improved.view(improved_shape), perturbed_z.detach(), best_perturbed_z
                )
            else:
                temp_improved = divergence < (best_divergence - eta)
                improved_shape = (B,) + (1,) * (len(z.shape) - 1)
                best_divergence = torch.where(temp_improved, divergence, best_divergence)
                best_v_pred = torch.where(
                    temp_improved.view(improved_shape), v_pred, best_v_pred
                )
                best_perturbed_z = torch.where(
                    temp_improved.view(improved_shape), perturbed_z.detach(), best_perturbed_z
                )

        z = best_perturbed_z
        v_pred = best_v_pred

    return best_perturbed_z, best_v_pred, improved, delta


def get_scheduled_value(total: float, cur: float, schedule_type: str) -> float:
    if schedule_type == "constant":
        return total
    elif schedule_type == "linear":
        return total * (1.0 - cur / total)
    elif schedule_type == "cosine":
        return total * 0.5 * (1.0 + math.cos(math.pi * cur / total))
    elif schedule_type == "sqrt":
        return total * math.sqrt(1.0 - cur / total)
    elif schedule_type == "concave":
        return (1.0 - cur / total) ** 2
    elif schedule_type == "convex":
        return 1.0 - (cur / total) ** 2
    else:
        raise ValueError(f"Unknown schedule_type: {schedule_type}")


def generate_one(
    pipe: StableDiffusion3Pipeline,
    prompt: str,
    sample_id: int,
    device: str,
    args: argparse.Namespace,
) -> "PIL.Image.Image":
    """Run the divergence-corrector denoising loop for a single prompt."""
    with torch.no_grad():
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(prompt, None, None)
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat(
            [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
        )

    timesteps, _ = retrieve_timesteps(
        pipe.scheduler,
        num_inference_steps=args.num_inference_steps,
        device=device,
    )

    latents = randn_tensor(
        (1, 16, 64, 64),
        dtype=torch.float16,
        device=device,
        generator=torch.Generator(device).manual_seed(args.seed + sample_id),
    )

    improved = None
    delta = None

    for t in timesteps:
        flow_t = 1.0 - t.item() / 1000.0
        noise_scale = get_scheduled_value(1.0, flow_t, args.perturb_schedule)

        v_func_kwargs = {
            "pipe": pipe,
            "latents": latents,
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "t": t,
        }

        best_latents, best_pred, improved, delta = divergence_stepper(
            v_func=sd3_timestep_pipe,
            v_func_kwargs=v_func_kwargs,
            x_key="latents",
            seed_delta=0,
            seed_eps=42,
            num_updates=args.num_iters,
            stop_t=0.5,
            delta_scale=args.noise_scale * noise_scale,
            improved=improved,
            delta=delta,
        )

        with torch.no_grad():
            latents = update(pipe, best_latents, t, best_pred, device=device)

    with torch.no_grad():
        image = decode_latent(pipe, latents, ret_type="pil", device=device)

    return image


def worker(rank: int, args: argparse.Namespace) -> None:
    """Per-GPU worker: loads the pipeline and generates its shard of prompts."""
    device = f"cuda:{rank}"
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Build the full sorted dataset (no skip_existing here — we check per-item below)
    dataset = Coco30KPromptDataset(
        metadata_jsonl=args.metadata_jsonl,
        results_dir=results_dir,
        skip_existing=False,
        sort_by_sample_id=True,
    )

    # Stride-based sharding: rank 0 gets [0, 8, 16, ...], rank 1 gets [1, 9, 17, ...], etc.
    shard = [dataset.rows[i] for i in range(rank, len(dataset), args.num_gpus)]

    if args.skip_existing:
        shard = [r for r in shard if not (results_dir / str(r["output_file"])).exists()]

    print(f"[GPU {rank}] {len(shard)} prompts to generate", flush=True)

    if not shard:
        return

    # Load pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=torch.float16,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.transformer.eval()
    pipe.vae.eval()
    pipe.transformer.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    for enc_name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
        enc = getattr(pipe, enc_name, None)
        if enc is not None:
            enc.requires_grad_(False)
            enc.eval()

    manifest_path = results_dir / f"manifest_gpu{rank}.jsonl"

    for row in tqdm(shard, desc=f"GPU {rank}", position=rank, leave=True):
        save_path = results_dir / str(row["output_file"])

        if args.skip_existing and save_path.exists():
            continue

        image = generate_one(
            pipe=pipe,
            prompt=str(row["caption"]),
            sample_id=int(row["sample_id"]),
            device=device,
            args=args,
        )

        tmp = save_path.with_name(save_path.stem + ".tmp" + save_path.suffix)
        image.save(tmp, format="PNG")
        os.replace(tmp, save_path)

        record = {
            "sample_id": int(row["sample_id"]),
            "caption": str(row["caption"]),
            "output_file": str(row["output_file"]),
            "save_path": str(save_path),
            "ann_id": row.get("ann_id"),
            "image_id": row.get("image_id"),
            "gpu": rank,
            "num_inference_steps": args.num_inference_steps,
            "num_iters": args.num_iters,
            "noise_scale": args.noise_scale,
            "perturb_schedule": args.perturb_schedule,
            "seed": args.seed #+ int(row["sample_id"]),
        }
        with manifest_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[GPU {rank}] Done.", flush=True)


def merge_manifests(results_dir: Path, num_gpus: int) -> None:
    merged = results_dir / "manifest.jsonl"
    with merged.open("w", encoding="utf-8") as out:
        for rank in range(num_gpus):
            part = results_dir / f"manifest_gpu{rank}.jsonl"
            if part.exists():
                out.write(part.read_text(encoding="utf-8"))
    print(f"Merged manifest written to: {merged}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate COCO-30K images across 8 GPUs with SD3 divergence corrector"
    )
    parser.add_argument("--metadata-jsonl", type=Path, required=True,
                        help="Path to metadata.jsonl from sample_coco30k_val.py")
    parser.add_argument("--results-dir", type=Path, default=Path("results"),
                        help="Root output directory; images go to results-dir/exp-name/")
    parser.add_argument("--exp-name", type=str, required=True,
                        help="Experiment name; images saved to results-dir/exp-name/")
    parser.add_argument("--num-gpus", type=int, default=8,
                        help="Number of GPUs to use")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip prompts whose output file already exists")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base seed; per-sample seed = seed + sample_id")

    # Diffusion parameters
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--num-iters", type=int, default=1,
                        help="Number of divergence corrector proposals per timestep")
    parser.add_argument("--noise-scale", type=float, default=1e-3,
                        help="Base scale for corrector perturbation")
    parser.add_argument("--perturb-schedule", type=str, default="linear",
                        choices=["constant", "linear", "cosine", "sqrt", "concave", "convex"])

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results_dir = Path(args.results_dir) / args.exp_name
    args.results_dir = results_dir  # workers read from args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=4, default=str)

    n_available = torch.cuda.device_count()
    if args.num_gpus > n_available:
        raise RuntimeError(
            f"Requested {args.num_gpus} GPUs but only {n_available} are available."
        )

    print(f"Launching {args.num_gpus} workers → {results_dir} (exp: {args.exp_name})")

    mp.set_start_method("spawn", force=True)
    mp.spawn(worker, args=(args,), nprocs=args.num_gpus, join=True)

    merge_manifests(results_dir, args.num_gpus)
    print("All done.")


if __name__ == "__main__":
    main()
