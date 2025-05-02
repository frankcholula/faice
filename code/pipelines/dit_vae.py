# Deep learning framework
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import numpy as np
from torch import nn

# Hugging Face
from diffusers import DiTPipeline, AutoencoderKL

# Configuration
from utils.metrics import calculate_fid_score, calculate_inception_score
from utils.metrics import evaluate
from utils.loggers import WandBLogger
from utils.training import setup_accelerator
from models.vae import create_vae
from pipelines.dit import name_to_label

selected_pipeline = DiTPipeline
vae_path = "runs/vae-vae-ddpm-face-500/checkpoints/model_vae.pth"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
num_class = 2


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

    # url = "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors"  # can also be a local file
    # vae = AutoencoderKL.from_single_file(url)
    # vae.eval().requires_grad_(False)

    vae = create_vae(config)
    vae = vae.to(device)
    vae.load_state_dict(torch.load(vae_path, map_location=device)['model_state_dict'])
    vae.eval().requires_grad_(False)

    # vae = AutoencoderKL.from_pretrained("facebook/DiT-XL-2-256", subfolder="vae")
    # vae = vae.to(device)
    # vae.eval().requires_grad_(False)

    model.train()
    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(
            total=len(train_dataloader), disable=not accelerator.is_local_main_process
        )
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["images"]
            image_names = batch["image_names"]
            bs = clean_images.shape[0]

            image_labels = [name_to_label(img_name) for img_name in image_names]

            image_labels = np.array(image_labels)
            # Convert the name in image_names to int number
            image_labels = image_labels.astype(int)
            map_ids = torch.tensor(image_labels, dtype=torch.int)
            map_ids = map_ids.to(device)

            vae.to(device)

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bs,),
                device=clean_images.device,
            ).long()

            # Encode image to latent space
            latents = vae.encode(clean_images).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            latents = latents * noise_scheduler.init_noise_sigma
            # # Add noise (diffusion process)
            noise = torch.randn_like(latents).to(clean_images.device)
            # # Add noise to the clean images according to the noise magnitude at each timestep
            # # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(latents, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images,
                                   timestep=timesteps,
                                   class_labels=map_ids,
                                   return_dict=False)[0]

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(clean_images, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # loss = F.mse_loss(noise_pred, target)
                loss = F.l1_loss(noise_pred, target)
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
                accelerator.unwrap_model(model),
                accelerator.unwrap_model(vae),
                noise_scheduler
            )

            generate_samples = (
                                       epoch + 1
                               ) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1
            save_model = (
                                 epoch + 1
                         ) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1
            save_to_wandb = epoch == config.num_epochs - 1

            if generate_samples:
                class_labels = torch.randint(
                    0,
                    num_class,
                    (config.train_batch_size,),
                    device=device,
                ).int()
                evaluate(config, epoch, pipeline, class_labels=class_labels)
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
        pipeline = selected_pipeline(
            accelerator.unwrap_model(model),
            accelerator.unwrap_model(vae),
            noise_scheduler
        )
        class_labels = torch.randint(
            0,
            num_class,
            (config.train_batch_size,),
            device=device,
        ).int()
        fid_score = calculate_fid_score(config, pipeline, test_dataloader, class_labels=class_labels)

        wandb_logger.log_fid_score(fid_score)

    if (
            accelerator.is_main_process
            and config.calculate_is
            and test_dataloader is not None
    ):
        inception_score = calculate_inception_score(
            config, pipeline, test_dataloader, device=accelerator.device, class_labels=class_labels
        )
        wandb_logger.log_inception_score(inception_score)
    wandb_logger.finish()
