# -*- coding: UTF-8 -*-
"""
@Time : 23/04/2025 16:31
@Author : xiaoguangliang
@File : vae_train.py
@Project : code
"""
import os
import gc
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from torchvision.utils import save_image
import wandb

# Configuration
from utils.loggers import WandBLogger
from utils.training import setup_accelerator
from models.vae import create_vae
from utils.plot import plot_images
from utils.metrics import calculate_clean_fid, make_grid

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


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

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(
            total=len(train_dataloader), disable=not accelerator.is_local_main_process
        )
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["images"]

            with accelerator.accumulate(model):
                # Predict the noise residual
                encoded = model.encode(clean_images)
                z = encoded.latent_dist.sample()
                decoded = model.decode(z)[0]

                # 计算 loss
                rec_loss = F.mse_loss(clean_images, decoded)
                kl_loss = encoded.latent_dist.kl().mean()
                loss = rec_loss + kl_loss * 0.0025

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

            generate_samples = (
                                       epoch + 1
                               ) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1
            save_model = (
                                 epoch + 1
                         ) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1
            save_to_wandb = epoch == config.num_epochs - 1

            if generate_samples:
                evaluate(config, epoch, decoded)
            if save_model:
                if config.push_to_hub:
                    repo.push_to_hub(commit_message=f"Epoch {epoch}", blocking=True)
                else:
                    model_path = f"{config.output_dir}/checkpoints"
                    if not os.path.exists(model_path):
                        os.makedirs(model_path)
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                    }, model_path + '/model_vae.pth')
                    if save_to_wandb:
                        wandb_logger.save_model()

            progress_bar.close()

    # Now we evaluate the model on the test set
    if (
            accelerator.is_main_process
            and config.calculate_fid
            and test_dataloader is not None
    ):
        model_path = f"{config.output_dir}/checkpoints/model_vae.pth"
        vae_inference(model_path, config, test_dataloader)

    wandb_logger.finish()


def vae_inference(model_path, config, test_dataloader):
    checkpoint = torch.load(model_path)
    vae = create_vae(config)
    vae = vae.to(device)
    vae.load_state_dict(checkpoint['model_state_dict'])

    vae.eval()

    real_dir = os.path.join(config.output_dir, "fid", "real")
    fake_dir = os.path.join(config.output_dir, "fid", "fake")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    fake_count = 0

    print(">" * 10, "Evaluate the vae model ...")
    # with torch.no_grad():
    #     noise = torch.randn(81, 16, 16, 16).to(device)
    #     generated_images = vae.decode(noise).sample
    #     generated_images = (generated_images / 2 + 0.5).clamp(0, 1)
    #     img_dir = f"{config.output_dir}/samples"
    #     if not os.path.exists(img_dir):
    #         os.makedirs(img_dir)
    #     # Plot images
    #     plot_images(generated_images, save_dir=img_dir, save_title="vae_decode", cols=9)

    for batch in test_dataloader:
        real_images = batch["images"].to(device)
        encoded = vae.encode(real_images)
        z = encoded.latent_dist.sample()

        del encoded
        gc.collect()

        img_dir = f"{config.output_dir}/samples"
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        # Plot images
        generated_images = (z / 2 + 0.5).clamp(0, 1)
        plot_images(generated_images, save_dir=img_dir, save_title="z", cols=9)

        decoded = vae.decode(z)[0]

        del z
        gc.collect()

        generated_images = (decoded / 2 + 0.5).clamp(0, 1)
        plot_images(generated_images, save_dir=img_dir, save_title="decoded", cols=9)

        # Calculate FID
        real_image_names = batch["image_names"]
        for i, image in enumerate(real_images):
            img_name = real_image_names[i]
            save_image(image, os.path.join(real_dir, f"{img_name}.jpg"))

        del real_image_names
        gc.collect()

        for image in generated_images:
            save_image(
                image,
                os.path.join(fake_dir, f"{fake_count:03d}.jpg"),
            )
            fake_count += 1

        del generated_images
        gc.collect()

    _ = calculate_clean_fid(real_dir, fake_dir)


def evaluate(config, epoch, decoded):
    generated_images = (decoded / 2 + 0.5).clamp(0, 1)
    # Make a grid out of the images
    # Convert the image size (b, c, h, w) to (b, w, h)
    generated_images = generated_images.permute(0, 3, 2, 1)
    generated_images = generated_images[:16]
    image_grid = make_grid(generated_images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid_path = f"{test_dir}/{epoch:04d}.png"
    image_grid.save(image_grid_path)

    if config.use_wandb:
        wandb.log(
            {
                "generated_images": wandb.Image(
                    image_grid_path, caption=f"Epoch {epoch}"
                ),
                "epoch": epoch,
            }
        )

