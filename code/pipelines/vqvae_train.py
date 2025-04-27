# -*- coding: UTF-8 -*-
"""
@Time : 23/04/2025 16:31
@Author : xiaoguangliang
@File : vqvae_train.py
@Project : code
"""
import os
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

# Configuration
from utils.loggers import WandBLogger
from utils.training import setup_accelerator
from models.vqmodel import create_vqmodel
from utils.plot import plot_images


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
                z = encoded.latents
                quantized_z, loss, _ = model.quantize(z)
                decoded = model.decode(quantized_z, force_not_quantize=True)[0]

                # 计算 loss
                rec_loss = F.mse_loss(clean_images, decoded)
                quant_loss = loss
                loss = rec_loss + quant_loss * 0.0025

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

            save_model = (
                                 epoch + 1
                         ) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1

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
                    }, model_path + '/model_vqvae.pth')

            progress_bar.close()

    # Now we evaluate the model on the test set
    if (
            accelerator.is_main_process
            and config.calculate_fid
            and test_dataloader is not None
    ):
        model_path = config.output_dir + '/model_vqvae.pth'
        vqvae_inference(model_path, config, test_dataloader)

    wandb_logger.finish()


def vqvae_inference(model_path, config, test_dataloader):
    checkpoint = torch.load(model_path)
    vqvae = create_vqmodel(config)
    vqvae.load_state_dict(checkpoint['model_state_dict'])

    vqvae.eval()
    print(">"*10, "Evaluate the vqvae model ...")
    for batch in test_dataloader:
        encoded = vqvae.encode(batch)
        z = encoded.latents

        img_dir = f"{config.output_dir}/samples"
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        generated_images = (z / 2 + 0.5).clamp(0, 1)
        plot_images(generated_images, save_dir=img_dir, save_title="z", cols=9)

        quantized_z, _, _ = vqvae.quantize(z)
        generated_images = (quantized_z / 2 + 0.5).clamp(0, 1)
        plot_images(generated_images, save_dir=img_dir, save_title="quantized_z", cols=9)

        decoded = vqvae.decode(quantized_z, force_not_quantize=True)[0]
        generated_images = (decoded / 2 + 0.5).clamp(0, 1)

        plot_images(generated_images, save_dir=img_dir, save_title="decoded", cols=9)
