# -*- coding: UTF-8 -*-
"""
@Time : 07/04/2025 17:31
@Author : xiaoguangliang
@File : consistency.py
@Project : code
"""
# Deep learning framework

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

# Hugging Face
from diffusers import LDMPipeline, VQModel

# Configuration
from utils.metrics import evaluate, calculate_fid_score, calculate_inception_score
from utils.loggers import WandBLogger
from utils.training import setup_accelerator
from models.vqmodel import create_vqmodel
from models.vae import create_vae

selected_pipeline = LDMPipeline

vqmodel_path = "runs/vqvae-vqvae-ddpm-face-500-32/checkpoints/model_vqvae.pth"
# vae_path = "runs/vae-vae-ddpm-face-500/checkpoints/model_vae.pth"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# url = "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors"  # can also be a local file


def train_loop(
        config,
        model,
        noise_scheduler,
        optimizer,
        train_dataloader,
        lr_scheduler,
        test_dataloader=None,
):
    accelerator, repo = setup_accelerator(config)

    # Initialize wandb
    WandBLogger.login()
    wandb_logger = WandBLogger(config, accelerator)
    wandb_logger.setup(model)

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the objects in the same order you gave them to the prepare method.
    if test_dataloader is not None:
        model, optimizer, train_dataloader, lr_scheduler, test_dataloader = (
            accelerator.prepare(
                model, optimizer, train_dataloader, lr_scheduler, test_dataloader
            )
        )
    else:
        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler
        )

    global_step = 0

    # vae = AutoencoderKL.from_single_file(url)
    # vae.eval().requires_grad_(False)

    vqvae = create_vqmodel(config)
    vqvae = vqvae.to(device)
    vqvae.load_state_dict(torch.load(vqmodel_path, map_location=device)['model_state_dict'])
    vqvae.eval().requires_grad_(False)

    # vae = create_vae(config)
    # vae = vae.to(device)
    # vae.load_state_dict(torch.load(vae_path, map_location=device)['model_state_dict'])
    # vae.eval().requires_grad_(False)

    # vqvae = VQModel.from_pretrained("CompVis/ldm-celebahq-256", subfolder="vqvae")
    # vqvae = vqvae.to(device)
    # vqvae.eval().requires_grad_(False)

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(
            total=len(train_dataloader), disable=not accelerator.is_local_main_process
        )
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["images"]
            # Sample noise to add to the images
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bs,),
                device=clean_images.device,
            ).long()

            # Encode image to latent space
            latents = vqvae.encode(clean_images).latents
            latents = latents.detach().clone()
            latents = latents * vqvae.config.scaling_factor

            # latents = vae.encode(clean_images).latent_dist.sample()

            latents = latents * noise_scheduler.init_noise_sigma
            # # Add noise (diffusion process)
            noise = torch.randn(latents.shape).to(clean_images.device)
            # # Add noise to the clean images according to the noise magnitude at each timestep
            # # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_latents, timesteps, return_dict=False)[0]

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(clean_images, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(noise_pred, target)
                # loss = F.l1_loss(noise_pred, target)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            wandb_logger.log_step(logs, global_step)

            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = selected_pipeline(
                vqvae=accelerator.unwrap_model(vqvae),
                # vqvae=accelerator.unwrap_model(vae),
                unet=accelerator.unwrap_model(model),
                scheduler=noise_scheduler
            )

            generate_samples = (
                                       epoch + 1
                               ) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1
            save_model = (
                                 epoch + 1
                         ) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1

            if generate_samples:
                evaluate(config, epoch, pipeline)
            if save_model:
                if config.push_to_hub:
                    repo.push_to_hub(commit_message=f"Epoch {epoch}", blocking=True)
                else:
                    pipeline.save_pretrained(config.output_dir)

            progress_bar.close()

    # Now we evaluate the model on the test set
    if (
            accelerator.is_main_process
            and config.calculate_fid
            and test_dataloader is not None
    ):
        pipeline = selected_pipeline(
            vqvae=accelerator.unwrap_model(vqvae),
            # vqvae=accelerator.unwrap_model(vae),
            unet=accelerator.unwrap_model(model),
            scheduler=noise_scheduler
        )
        fid_score = calculate_fid_score(config, pipeline, test_dataloader)

        wandb_logger.log_fid_score(fid_score)

    if (
            accelerator.is_main_process
            and config.calculate_is
            and test_dataloader is not None
    ):
        inception_score = calculate_inception_score(
            config, pipeline, test_dataloader, device=accelerator.device
        )
        wandb_logger.log_inception_score(inception_score)
    wandb_logger.finish()
