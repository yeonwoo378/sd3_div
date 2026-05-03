import torch

def flux_timestep_pipe(pipe, latents, prompt_embeds, pooled_prompt_embeds,
                       text_ids, latent_image_ids, t,
                       guidance_scale=3.5, device='cuda'):
    # divergence_stepper creates delta/eps as float32, so perturbed_z may be float32.
    # Cast to match the transformer dtype (same pattern as sd3_timestep_pipe).
    latents = latents.to(prompt_embeds.dtype)
    B, C, H, W = latents.shape

    # Pack (B, C, H, W) -> (B, H//2 * W//2, C*4)
    packed = pipe._pack_latents(latents, B, C, H, W)

    timestep = t.expand(packed.shape[0]).to(packed.dtype)

    # FLUX.1-dev uses distilled guidance; schnell ignores it
    if pipe.transformer.config.guidance_embeds:
        guidance = torch.full([1], guidance_scale, device=latents.device, dtype=torch.float32)
        guidance = guidance.expand(packed.shape[0])
    else:
        guidance = None

    noise_pred = pipe.transformer(
        hidden_states=packed,
        timestep=timestep / 1000,
        guidance=guidance,
        pooled_projections=pooled_prompt_embeds,
        encoder_hidden_states=prompt_embeds,
        txt_ids=text_ids,
        img_ids=latent_image_ids,
        joint_attention_kwargs=getattr(pipe, 'joint_attention_kwargs', None),
        return_dict=False,
    )[0]

    # Unpack (B, seqlen, C*4) -> (B, C, H, W)
    noise_pred = pipe._unpack_latents(
        noise_pred,
        height=H * pipe.vae_scale_factor,
        width=W * pipe.vae_scale_factor,
        vae_scale_factor=pipe.vae_scale_factor,
    )

    return noise_pred


@torch.no_grad()
def update(pipe, latent, t, noise_pred, device='cuda'):
    latents = latent.detach().to(device)
    latents.requires_grad_(True)

    ret_latents = pipe.scheduler.step(
        noise_pred,
        t,
        latents.detach(),
        return_dict=False)[0]

    return ret_latents


@torch.no_grad()
def decode_latent(pipe, latents, ret_type='pil', device='cuda'):
    assert latents.shape[0] == 1, "Only batch size of 1 is supported for decoding."
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    image = pipe.vae.decode(latents.to(device), return_dict=False)[0]
    image = pipe.image_processor.postprocess(image, output_type=ret_type)[0]
    return image
