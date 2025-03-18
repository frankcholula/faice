# -*- coding: UTF-8 -*-
"""
@Time : 09/03/2025 11:01
@Author : xiaoguangliang
@File : U_Net2D.py
@Project : faice
"""
import os

import torch
from PIL import Image
from diffusers import UNet2DModel, UNet2DConditionModel, DDPMPipeline, DDIMPipeline
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMScheduler, CosineDPMSolverMultistepScheduler
from accelerate import Accelerator
from huggingface_hub import HfFolder, Repository, whoami
from tqdm.auto import tqdm
from pathlib import Path
from accelerate import notebook_launcher, launchers
from loguru import logger
from peft import LoraConfig, get_peft_model

from codes.data_exploration.preprocess_data import get_data
from codes.conf.global_setting import BASE_DIR, config


def unet2d_model():
    # model = UNet2DModel(
    #     sample_size=config.image_size,  # the target image resolution
    #     in_channels=3,  # the number of input channels, 3 for RGB images
    #     out_channels=3,  # the number of output channels
    #     layers_per_block=2,  # how many ResNet layers to use per UNet block
    #     block_out_channels=(128, 128, 256, 256, 512, 512),
    #     # the number of output channels for each UNet block
    #     down_block_types=(
    #         "DownBlock2D",  # a regular ResNet downsampling block
    #         "DownBlock2D",
    #         "DownBlock2D",
    #         "DownBlock2D",
    #         "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
    #         "DownBlock2D",
    #     ),
    #     up_block_types=(
    #         "UpBlock2D",  # a regular ResNet upsampling block
    #         "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
    #         "UpBlock2D",
    #         "UpBlock2D",
    #         "UpBlock2D",
    #         "UpBlock2D",
    #     ),
    # )
    model = UNet2DConditionModel(
        sample_size=config.image_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D",),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D")
    )

    # Define LoRA setting
    # lora_config = LoraConfig(
    #     r=8,  # rank
    #     lora_alpha=32,
    #     # target_modules=["conv1", "conv1"],
    #     target_modules=["down_blocks.0.resnets.0.conv1", "down_blocks.0.resnets.0.conv2"],
    #     lora_dropout=0.1,
    #     bias="none",
    #     task_type="FEATURE_EXTRACTION"
    # )
    lora_config = LoraConfig(
        task_type="UNet",  # Specify the task type for UNet-based models.
        inference_mode=False,
        r=8,  # Low-rank dimension.
        lora_alpha=32,  # Scaling factor.
        target_modules=["conv_in", "conv_out", "time_embedding"]  # Example target modules.
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)

    return model


def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.manual_seed(config.seed),
    ).images

    # Make a grid out of the images
    image_grid = make_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")


def get_full_repo_name(model_id: str, organization: str = None, token: str = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler, device):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if config.push_to_hub:
            repo_name = get_full_repo_name(Path(config.output_dir).name)
            repo = Repository(config.output_dir, clone_from=repo_name)
        elif config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["images"]
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:

            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
            # pipeline = DDIMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                if config.push_to_hub:
                    repo.push_to_hub(commit_message=f"Epoch {epoch}", blocking=True)
                else:
                    pipeline.save_pretrained(config.output_dir)


def main_train(data_dir):
    # 1. Make train dataset
    dataset = get_data(data_dir)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

    # 2. Make model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # if torch.backends.mps.is_available():
    #     device = torch.device("mps")
    model = unet2d_model()
    # Check the shape of input and output
    sample_image = dataset[0]["images"].unsqueeze(0)
    logger.info(f"Input shape: {sample_image.shape}")
    logger.info(f"Output shape: {model(sample_image, timestep=0).sample.shape}")

    model.to(device)

    # 4. Set up the optimizer, the learning rate scheduler and the loss scaling for AMP
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler, device)

    notebook_launcher(train_loop, args, num_processes=1)


if __name__ == "__main__":
    data_path = BASE_DIR + "/data/celeba_hq_256/"
    main_train(data_path)
