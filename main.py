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

D = 16 * 64 * 64  # latent dimension

def main(args):
    print(f"Running with args: {args}")
    # save current args as config file
    os.makedirs(os.path.join("results", args.exp_name), exist_ok=True)
    with open(os.path.join("results", args.exp_name, "config.json"), 'w') as f:
        json.dump(vars(args), f, indent=4)


    # load pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    # get prompt
    prompts = parse_prompts(args.prompt_path)
    print(f"Parsed {len(prompts)} prompts from {args.prompt_path}")

    result_path = os.path.join("results", args.exp_name)
    os.makedirs(result_path, exist_ok=True)

    ## loop
    for prompt_idx, prompt in enumerate(tqdm(prompts)):
       # Encode prompt
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(prompt, None, None)
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        # preprocess timestep
        timesteps, num_inference_steps = retrieve_timesteps(
            pipe.scheduler, 
            num_inference_steps=args.num_inference_steps,
            device='cuda')
        shape = (1, 16, 64, 64)
        init_latent = randn_tensor(
            shape,
            dtype=torch.float16,
            device="cuda",
            generator=torch.Generator("cuda").manual_seed(args.seed),
        )

        latents = init_latent.clone()
        for i, t in enumerate(timesteps):
            # predict noise
            loss_total = 0.0

            if args.lr_schedule == "linear":
                lr = args.lr * (1 - i / len(timesteps))
            elif args.lr_schedule == "sqrt":
                lr = args.lr * (1 - (i / len(timesteps))**0.5)
            elif args.lr_schedule == "cbrt":
                lr = args.lr * (1 - (i / len(timesteps))**(1./3.))
            elif args.lr_schedule == "constant":
                lr = args.lr
            elif args.lr_schedule == "custom1":
                if i < len(timesteps) * 0.2:
                    lr = args.lr * 10
                else:
                    lr = args.lr * (1 - (i / len(timesteps))**(1./3.))
            elif args.lr_schedule == "custom2":
                if i < len(timesteps) * 0.2:
                    lr = args.lr * 10
                elif i < len(timesteps) * 0.5:
                    lr = args.lr
                else:
                    lr = 0.0
            elif 't_only' in args.lr_schedule:
                target_t = int(args.lr_schedule.split('_')[-1])
                if i == target_t:
                    lr = args.lr * 50
                else:
                    lr = 0.0


            for iter in range(args.num_iters):
                latents = latents.detach().requires_grad_(True)
                pred = sd3_timestep_pipe(
                    pipe,
                    latents,
                    prompt_embeds,
                    pooled_prompt_embeds,
                    t,
                    guidance_scale=7.0,
                    device='cuda'
                )
                for s in range(args.num_samples): # estimate divergence
  
                    with torch.no_grad():
                        noise = torch.randn_like(latents).to("cuda").to(latents.dtype) 
                        perturbed_pred = sd3_timestep_pipe(
                            pipe,
                            latents + noise* args.noise_scale,
                            prompt_embeds,
                            pooled_prompt_embeds,
                            t,
                            guidance_scale=7.0,
                            device='cuda'
                        )

                    

                    if args.optimize_target == "divergence":
                        diff = perturbed_pred.detach() - pred
                        loss = torch.sum(diff * noise) / (D * args.noise_scale)
                    elif args.optimize_target == "exact":
                        vp = torch.sum(noise * pred) / D
                        vjp = torch.autograd.grad(vp, latents, retain_graph=False)[0]
                        loss = torch.sum(vjp * noise)
  


                    else:
                        loss = F.mse_loss(perturbed_pred, pred)
                    # loss = torch.sum(diff*noise) / (D * args.noise_scale ** 2)

                    loss_total += loss

                loss_avg = loss_total / args.num_samples
                if args.optimize_target != 'exact':
                    grad = torch.autograd.grad(loss_avg, latents)[0]
                else:
                    pass
                # latent update
                latents = latents - lr * grad
                
            # update latents
            with torch.no_grad():
                final_pred = sd3_timestep_pipe(
                    pipe,
                    latents,
                    prompt_embeds,
                    pooled_prompt_embeds,
                    t,
                    guidance_scale=7.0,
                    device='cuda'
                )


                latents = update(
                    pipe,
                    latents,
                    t,
                    final_pred,
                    device='cuda'
                )
            
        with torch.no_grad():
            image = decode_latent(
                pipe,
                latents,
                ret_type='pil',
                device='cuda')
            image.save(os.path.join(result_path, f"{prompt_idx:05d}.png"))

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script with argparse")
    parser.add_argument("--exp_name", type=str, default="debug", help="Name of the experiment")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--prompt_path", type=str, default="data.csv", help="Path to the prompt file")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate for update")
    parser.add_argument("--num_iters", type=int, default=1, help="Number of optimization iterations per timestep")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of samples for divergence estimation")
    parser.add_argument("--noise_scale", type=float, default=1e-3, help="Scale of the noise added during perturbation")
    parser.add_argument("--lr_schedule", type=str, default="constant", help="Learning rate schedule (constant or linear)")
    parser.add_argument("--optimize_target", type=str, default="divergence", help="Target to optimize")


    parser.add_argument("--num_inference_steps", type=int, default=25, help="Number of samples for divergence estimation")
    args = parser.parse_args()
    main(args)