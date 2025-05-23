# Deep learning framework
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import numpy as np

# Hugging Face
from diffusers import DDPMPipeline, DDIMPipeline
from pipelines.ccddpm_pipeline import CCDDPMPipeline

# Configuration
from utils.metrics import calculate_fid_score, calculate_inception_score
from utils.metrics import evaluate
from utils.loggers import WandBLogger
from utils.training import setup_accelerator
from utils.loss import get_loss
from utils.model_tools import name_to_label

import lpips

AVAILABLE_PIPELINES = {
    "ddpm": DDPMPipeline,
    "ddim": DDIMPipeline,
    "pndm": DDIMPipeline,
    "cond": CCDDPMPipeline,
}


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

    # Initialize lpips
    lpips_fn = None
    if config.use_lpips_regularization:
        lpips_fn = lpips.LPIPS(net=config.lpips_net).to(config.device)
        lpips_fn.eval()

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
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                if noise_scheduler.config.prediction_type == "epsilon":
                    # Predict the noise residual
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    # Predict velocity
                    target = noise_scheduler.get_velocity(
                        clean_images, noise, timesteps
                    )
                # Predict the target (noise or velocity)
                if config.pipeline == "cond":
                    # Extract class labels for conditioning
                    # class_labels = batch["labels"]

                    image_names = batch["image_names"]
                    image_labels = [name_to_label(img_name) for img_name in image_names]
                    image_labels = np.array(image_labels)
                    # Convert the name in image_names to int number
                    class_labels = torch.tensor(image_labels, dtype=torch.int).reshape(
                        -1
                    )

                    # the encoder_hidden_states are really just a placeholder since we're only using labels.
                    encoder_hidden_states = torch.zeros(
                        bs,
                        1,  # random sequence length
                        model.config.cross_attention_dim,
                        device=clean_images.device,
                    )
                    pred = model(
                        noisy_images,
                        timesteps,
                        class_labels=class_labels,
                        encoder_hidden_states=encoder_hidden_states,
                        return_dict=False,
                    )[0]

                else:
                    pred = model(noisy_images, timesteps, return_dict=False)[0]

                loss = get_loss(pred, target, config, lpips_fn=lpips_fn)
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
            pipeline = AVAILABLE_PIPELINES[config.pipeline](
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
        pipeline = AVAILABLE_PIPELINES[config.pipeline](
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
