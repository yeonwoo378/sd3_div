import torch
import torch.nn.functional as F
# from utils import retrieve_timesteps

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
    noise_pred_guided = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    return noise_pred_guided

@torch.no_grad()
def update(pipe, latent, t, noise_pred, device='cuda'):
    latents = latent.detach().to(device)
    latents.requires_grad_(True)

    # Perform the update step
    ret_latents = pipe.scheduler.step(
        noise_pred, 
        t,
        latents.detach(),
        return_dict=False)[0]

    return ret_latents

@torch.no_grad()    
def decode_latent(pipe, latents, ret_type='pil',device='cuda'):
    
    assert latents.shape[0] == 1, "Only batch size of 1 is supported for decoding."
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    image = pipe.vae.decode(latents.to(device), return_dict=False)[0]
    image = pipe.image_processor.postprocess(image, output_type=ret_type)[0]

    return image