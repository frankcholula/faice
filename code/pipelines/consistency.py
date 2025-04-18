# -*- coding: UTF-8 -*-
"""
@Time : 07/04/2025 17:31
@Author : xiaoguangliang
@File : consistency.py
@Project : code
"""
# Deep learning framework
from typing import Optional

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch import Tensor

# Hugging Face
from diffusers import ConsistencyModelPipeline
from diffusers.schedulers import CMStochasticIterativeScheduler
from diffusers.utils.torch_utils import randn_tensor

# Configuration
from utils.metrics import evaluate, calculate_fid_score, calculate_inception_score
from utils.loggers import WandBLogger
from utils.training import setup_accelerator

selected_pipeline = ConsistencyModelPipeline


def train_loop(
        config,
        model,
        noise_scheduler,
        optimizer,
        train_dataloader,
        lr_scheduler,
        test_dataloader=None):
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

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(
            total=len(train_dataloader), disable=not accelerator.is_local_main_process
        )
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["images"]
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            if isinstance(noise_scheduler, CMStochasticIterativeScheduler):
                # timesteps_idx = torch.randint(
                #     0,
                #     noise_scheduler.config.num_train_timesteps,
                #     (bs,),
                #     dtype=torch.int64,
                # )
                timesteps_idx = torch.linspace(0, noise_scheduler.config.num_train_timesteps - 1, steps=bs,
                                               dtype=torch.int64)
                timesteps_idx = torch.flip(timesteps_idx, dims=[0])
                init_timesteps = torch.take(noise_scheduler.timesteps, timesteps_idx)
                init_timesteps = init_timesteps.to(clean_images.device)

                noise_scheduler.set_timesteps(timesteps=timesteps_idx, device=clean_images.device)
                # timesteps = noise_scheduler.timesteps

                noisy_images = noise_scheduler.add_noise(clean_images, noise, init_timesteps)
            else:
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bs,),
                    device=clean_images.device,
                ).long()
                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                if isinstance(noise_scheduler, CMStochasticIterativeScheduler):
                    # sigma = convert_sigma(noise_scheduler, clean_images, init_timesteps)
                    # model_kwargs = {"return_dict": False}
                    # model_output, denoised = denoise(model, noisy_images, sigma, noise_scheduler,
                    #                                  **model_kwargs)

                    # noise_scheduler.set_timesteps(timesteps=timesteps_idx, device=clean_images.device)
                    # timesteps_denoise = noise_scheduler.timesteps

                    scaled_sample = noise_scheduler.scale_model_input(noisy_images, init_timesteps)
                    model_output = model(scaled_sample, init_timesteps, return_dict=False)[0]

                    denoised = noise_scheduler.step(model_output, init_timesteps, noisy_images,
                                                    generator=torch.manual_seed(0))[0]

                    loss = F.mse_loss(denoised, clean_images)
                else:
                    noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                    loss = F.mse_loss(noise_pred, noise)
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
                unet=accelerator.unwrap_model(model), scheduler=noise_scheduler
            )

            generate_samples = (
                                       epoch + 1
                               ) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1
            save_model = (
                                 epoch + 1
                         ) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1

            if generate_samples:
                evaluate(config, epoch, pipeline)
                # After inference, reset the parameters of scheduler
                # noise_scheduler = CMStochasticIterativeScheduler(
                #     num_train_timesteps=config.num_train_timesteps
                # )
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
            unet=accelerator.unwrap_model(model), scheduler=noise_scheduler
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


def denoise(model, x_t, sigma, noise_scheduler, **model_kwargs):
    distillation = False
    if not distillation:
        c_skip, c_out, c_in = [
            append_dims(x, x_t.ndim) for x in get_scalings(noise_scheduler, sigma)
        ]
    else:
        c_skip, c_out, c_in = [
            append_dims(x, x_t.ndim)
            for x in get_scalings_for_boundary_condition(noise_scheduler, sigma)
        ]
    rescaled_t = 1000 * 0.25 * torch.log(sigma + 1e-44)
    rescaled_t = torch.flatten(rescaled_t)
    m_input = c_in * x_t
    model_output = model(m_input, rescaled_t, **model_kwargs)[0]
    denoised = c_out * model_output + c_skip * x_t

    return model_output, denoised


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


def get_scalings(noise_scheduler, sigma):
    sigma_data = noise_scheduler.config.sigma_data
    c_skip = sigma_data ** 2 / (sigma ** 2 + sigma_data ** 2)
    c_out = sigma * sigma_data / (sigma ** 2 + sigma_data ** 2) ** 0.5
    c_in = 1 / (sigma ** 2 + sigma_data ** 2) ** 0.5
    return c_skip, c_out, c_in


def get_scalings_for_boundary_condition(noise_scheduler, sigma):
    sigma_data = noise_scheduler.config.sigma_data
    c_skip = sigma_data ** 2 / (
            (sigma - noise_scheduler.sigma_min) ** 2 + sigma_data ** 2
    )
    c_out = (
            (sigma - noise_scheduler.sigma_min)
            * sigma_data
            / (sigma ** 2 + sigma_data ** 2) ** 0.5
    )
    c_in = 1 / (sigma ** 2 + sigma_data ** 2) ** 0.5
    return c_skip, c_out, c_in


def convert_sigma(noise_scheduler, original_samples, timesteps):
    sigmas = noise_scheduler.sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
    if original_samples.device.type == "mps" and torch.is_floating_point(timesteps):
        # mps does not support float64
        schedule_timesteps = noise_scheduler.timesteps.to(original_samples.device, dtype=torch.float32)
        timesteps = timesteps.to(original_samples.device, dtype=torch.float32)
    else:
        schedule_timesteps = noise_scheduler.timesteps.to(original_samples.device)
        timesteps = timesteps.to(original_samples.device)

    # self.begin_index is None when scheduler is used for training, or pipeline does not implement set_begin_index
    if noise_scheduler.begin_index is None:
        step_indices = [noise_scheduler.index_for_timestep(t, schedule_timesteps) for t in timesteps]
    elif noise_scheduler.step_index is not None:
        # add_noise is called after first denoising step (for inpainting)
        step_indices = [noise_scheduler.step_index] * timesteps.shape[0]
    else:
        # add noise is called before first denoising step to create initial latent(img2img)
        step_indices = [noise_scheduler.begin_index] * timesteps.shape[0]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < len(original_samples.shape):
        sigma = sigma.unsqueeze(-1)

    return sigma


def get_weightings(weight_schedule, snrs, sigma_data):
    if weight_schedule == "snr":
        weightings = snrs
    elif weight_schedule == "snr+1":
        weightings = snrs + 1
    elif weight_schedule == "karras":
        weightings = snrs + 1.0 / sigma_data ** 2
    elif weight_schedule == "truncated-snr":
        weightings = torch.clamp(snrs, min=1.0)
    elif weight_schedule == "uniform":
        weightings = torch.ones_like(snrs)
    else:
        raise NotImplementedError()
    return weightings


def get_snr(sigmas):
    return sigmas ** -2
