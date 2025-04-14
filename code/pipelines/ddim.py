# Deep learning framework

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

# Hugging Face
from diffusers import DDIMPipeline, DDPMScheduler

# Configuration
from utils.metrics import calculate_fid_score, calculate_inception_score
from utils.metrics import evaluate
from utils.loggers import WandBLogger
from utils.training import setup_accelerator


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
    
    # Use ddpm scheduler for training, set prediction type to "v_prediction", apply rescale_betas_zero_snr
    train_noise_scheduler = DDPMScheduler.from_config(noise_scheduler.config, prediction_type="v_prediction", rescale_betas_zero_snr=True)
    #noise_scheduler(prediction_type="v_prediction", rescale_betas_zero_snr=True)
    noise_scheduler.config.prediction_type = "v_prediction"

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
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bs,),
                device=clean_images.device,
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = train_noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                # noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                # loss = F.mse_loss(noise_pred, noise)
                
                # v = α_t * ε + β_t * x_0, x_0 = clean_images, ε = noise, t = timesteps
                # v = train_noise_scheduler.get_velocity(clean_images, timesteps, noise)
                
                # manually calculate velocity
                alphas_cumprod = train_noise_scheduler.alphas_cumprod.to(clean_images.device)
                sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
                sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
                # reshape to match the shape of clean_images
                sqrt_alpha_prod = sqrt_alpha_prod.view(-1, 1, 1, 1)
                sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.view(-1, 1, 1, 1)
                v = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * clean_images

                # Predict velocity
                v_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(v_pred, v)

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
            pipeline = DDIMPipeline(
                unet=accelerator.unwrap_model(model), scheduler=noise_scheduler
            )

            generate_samples = (
                epoch + 1
            ) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1
            save_model = (
                epoch + 1
            ) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1
            save_to_wandb = epoch == config.num_epochs - 1

            if generate_samples:
                evaluate(config, epoch, pipeline)
            if save_model:
                if config.push_to_hub:
                    repo.push_to_hub(commit_message=f"Epoch {epoch}", blocking=True)
                else:
                    pipeline.save_pretrained(config.output_dir)
                    if save_to_wandb:
                        wandb_logger.save_model()

            progress_bar.close()

    # Now we evaluate the model on the test set
    if (
        accelerator.is_main_process
        and config.calculate_fid
        and test_dataloader is not None
    ):
        pipeline = DDIMPipeline(
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
