# -*- coding: UTF-8 -*-
"""
@Time : 23/04/2025 16:31
@Author : xiaoguangliang
@File : vqvae_train.py
@Project : code
"""
import os
import wandb
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from torchvision.utils import save_image
from diffusers.utils.pil_utils import numpy_to_pil
from torchvision import transforms

# Configuration
from utils.loggers import WandBLogger
from utils.training import setup_accelerator
from models.vqmodel import vqvae_b_3, vqvae_b_16, vqvae_b_32, vqvae_b_64
from utils.plot import plot_images
from utils.metrics import calculate_clean_fid, make_grid, calculate_inception_score_vae

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
                encoded = model.encode(clean_images)
                z = encoded.latents
                # quantized_z, vq_loss, _ = model.quantize(z)
                # decoded = model.decode(z, force_not_quantize=True)[0]
                decoded, commit_loss = model.decode(z, return_dict=False)

                # Calculate loss
                rec_loss = F.mse_loss(clean_images, decoded)

                # loss_weight = 0.0025
                # loss_weight = 0.05
                # loss_weight = 0.1
                # loss_weight = 0.15
                # loss_weight = 0.25
                # loss_weight = 0.3
                # loss_weight = 0.35
                loss_weight = 0.4
                # loss_weight = 0.5
                loss = rec_loss + commit_loss * loss_weight

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
                evaluate(config, epoch, model, test_dataloader)
            if save_model:
                if config.push_to_hub:
                    repo.push_to_hub(commit_message=f"Epoch {epoch}", blocking=True)
                else:
                    model_path = f"{config.output_dir}/checkpoints"
                    if not os.path.exists(model_path):
                        os.makedirs(model_path)
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": loss,
                        },
                        model_path + "/model_vqvae.pth",
                    )
                    if save_to_wandb:
                        wandb_logger.save_model()

            progress_bar.close()

    # Now we evaluate the model on the test set
    if (
        accelerator.is_main_process
        and config.calculate_fid
        and test_dataloader is not None
    ):
        # model_path = f"{config.output_dir}/checkpoints/model_vqvae.pth"
        vqvae_inference(model, config, test_dataloader, wandb_logger)

    wandb_logger.finish()


def vqvae_inference(vqvae, config, test_dataloader, wandb_logger):
    # checkpoint = torch.load(model_path, map_location=device)
    # vqvae = vqvae_b_3(config)
    # vqvae = vqvae.to(device)
    # vqvae.load_state_dict(checkpoint['model_state_dict'])

    vqvae.eval()

    real_dir = os.path.join(config.output_dir, "fid", "real")
    fake_dir = os.path.join(config.output_dir, "fid", "fake")
    reconstruction_dir = os.path.join(config.output_dir, "fid", "reconstruction")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)
    os.makedirs(reconstruction_dir, exist_ok=True)

    fake_count = 0

    print(">" * 10, "Evaluate the vqvae model ...")
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_dataloader)):
            real_images = batch["images"].to(device)

            # Normalize the test dataset
            transform = transforms.Normalize([0.5], [0.5])
            real_images_norm = transform(real_images)

            encoded = vqvae.encode(real_images_norm)
            z = encoded.latents
            noise = torch.randn_like(z).to(device)

            img_dir = f"{config.output_dir}/samples"
            if not os.path.exists(img_dir):
                os.makedirs(img_dir)

            # Plot images
            # generated_images = (z / 2 + 0.5).clamp(0, 1)
            # plot_images(generated_images, save_dir=img_dir, save_title="z", cols=9)

            quantized_z, _, _ = vqvae.quantize(z)

            # generated_images = (quantized_z / 2 + 0.5).clamp(0, 1)
            # plot_images(generated_images, save_dir=img_dir, save_title="quantized_z", cols=9)

            decoded = vqvae.decode(quantized_z, force_not_quantize=True)[0]
            decoded_fake = vqvae.decode(noise)[0]

            generated_images = (decoded / 2 + 0.5).clamp(0, 1)
            # plot_images(generated_images, save_dir=img_dir, save_title="decoded", cols=9)
            generated_images_fake = (decoded_fake / 2 + 0.5).clamp(0, 1)

            # Calculate FID
            real_image_names = batch["image_names"]
            for i, image in enumerate(real_images):
                img_name = real_image_names[i]
                save_image(image, os.path.join(real_dir, f"{img_name}.jpg"))

            for image in generated_images_fake:
                save_image(
                    image,
                    os.path.join(fake_dir, f"{fake_count:03d}.jpg"),
                )
                fake_count += 1

            for image in generated_images:
                save_image(
                    image,
                    os.path.join(reconstruction_dir, f"{fake_count:03d}.jpg"),
                )
                fake_count += 1

        fid_score = calculate_clean_fid(
            real_dir, fake_dir, msg="Clean FID score for sampling"
        )
        fid_score_rec = calculate_clean_fid(
            real_dir, reconstruction_dir, msg="Clean FID score for reconstruction"
        )
        wandb_logger.log_fid_score(fid_score)
        wandb_logger.log_fid_score_rec(fid_score_rec)


def evaluate(config, epoch, vqvae, test_dataloader):
    print("Evaluate training ...")
    with torch.no_grad():
        to_generate_images = []
        for batch in tqdm(test_dataloader):
            real_images = batch["images"].to(device)

            # Normalize the test dataset
            transform = transforms.Normalize([0.5], [0.5])
            real_images_norm = transform(real_images)

            encoded = vqvae.encode(real_images_norm)
            z = encoded.latents

            quantized_z, _, _ = vqvae.quantize(z)

            decoded = vqvae.decode(quantized_z, force_not_quantize=True)[0]

            generated_images = (decoded / 2 + 0.5).clamp(0, 1)

            to_generate_images.append(generated_images)

            to_generate_images = torch.cat(to_generate_images, dim=0)
            if to_generate_images.shape[0] >= 16:
                break

        to_generate_images = to_generate_images[:16]
        # Make a grid out of the images
        # Convert the image size (b, c, h, w) to (b, w, h)
        to_generate_images = to_generate_images.cpu().permute(0, 2, 3, 1).numpy()
        to_generate_images = numpy_to_pil(to_generate_images)

        # generated_images = generated_images.permute(0, 3, 2, 1)
        image_grid = make_grid(to_generate_images, rows=4, cols=4)

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
