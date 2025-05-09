# -*- coding: UTF-8 -*-
"""
@Time : 05/05/2025 18:05
@Author : xiaoguangliang
@File : stable_diffusion.py
@Project : code
"""
import os
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import pandas as pd
import json
from collections import defaultdict

# Hugging Face
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel
from diffusers.training_utils import (
    EMAModel,
    compute_snr,
)
from diffusers.utils.import_utils import is_xformers_available
from packaging import version
import accelerate
from accelerate.state import AcceleratorState
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers
from torch.utils.data import Dataset, DataLoader

# Configuration
from utils.metrics import calculate_fid_score, calculate_inception_score
from utils.metrics import evaluate
from utils.loggers import WandBLogger
from utils.training import setup_accelerator
from models.vae import vae_b_4, vae_b_16, vae_l_4, vae_l_16
from utils.model_tools import name_to_label

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
num_class = 2
pretrained_model_name_or_path = "stable-diffusion-v1-5/stable-diffusion-v1-5"
vae_path = "runs/vae_l_4-vae-ddpm-face-500-0.05-16/checkpoints/model_vae.pth"

selected_pipeline = StableDiffusionPipeline


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

    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder"
        )
        vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path, subfolder="vae"
        )
        vae.eval().requires_grad_(False)

        # vae = vae_l_4(config)
        # # vae = vae_b_16(config)
        # vae = vae.to(device)
        # vae.load_state_dict(torch.load(vae_path, map_location=device)['model_state_dict'])
        # vae.eval().requires_grad_(False)

    # model = UNet2DConditionModel.from_pretrained(
    #     pretrained_model_name_or_path, subfolder="unet",
    # )

    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path, subfolder="tokenizer"
    )

    # Create EMA for the unet.
    if config.use_ema:
        ema_unet = EMAModel(
            model.parameters(),
            model_cls=UNet2DConditionModel,
            model_config=model.config,
            foreach=config.foreach_ema,
        )

    model.train()  # important! This enables embedding dropout for classifier-free guidance

    if config.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                print(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            print("Start using xformers ...")
            model.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if config.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            if config.use_ema:
                load_model = EMAModel.from_pretrained(
                    os.path.join(input_dir, "unet_ema"),
                    UNet2DConditionModel,
                    foreach=config.foreach_ema,
                )
                ema_unet.load_state_dict(load_model.state_dict())
                if config.offload_ema:
                    ema_unet.pin_memory()
                else:
                    ema_unet.to(accelerator.device)
                del load_model

            for _ in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(
                    input_dir, subfolder="unet"
                )
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if config.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if config.scale_lr:
        config.learning_rate = (
            config.learning_rate
            * config.gradient_accumulation_steps
            * config.train_batch_size
            * accelerator.num_processes
        )

    train_prompts = load_prompts(config.stable_diffusion_prompt_dir)

    test_prompt_dict = load_request_prompt(config.stable_diffusion_request_prompt_dir)
    test_prompts = []
    for t_batch in tqdm(test_dataloader):
        image_names = t_batch["image_names"]
        t_prompts = [test_prompt_dict[int(x)] for x in image_names]
        test_prompts.extend(t_prompts)

    # For evaluation
    evaluate_batch_size = 16
    evaluation_prompts = test_prompts[:evaluate_batch_size]

    if config.use_ema:
        if config.offload_ema:
            ema_unet.pin_memory()
        else:
            ema_unet.to(accelerator.device)

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device)
    vae.to(accelerator.device)
    model.to(accelerator.device)

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(
            total=len(train_dataloader), disable=not accelerator.is_local_main_process
        )
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["images"]
            bs = clean_images.shape[0]

            image_names = batch["image_names"]
            batch_prompts = [train_prompts[x] for x in image_names]
            batch["input_ids"] = tokenize_captions(batch_prompts, tokenizer)
            batch["input_ids"] = batch["input_ids"].to(accelerator.device)

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bs,),
                device=device,
            ).long()

            # Convert images to latent space
            latents = vae.encode(batch["images"]).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            # # Add noise (diffusion process)
            noise = torch.randn_like(latents).to(device)
            # # Add noise to the clean images according to the noise magnitude at each timestep
            # # (this is the forward diffusion process)

            noisy_latent = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[
                0
            ]

            with accelerator.accumulate(model):
                # Predict the noise residual
                model_pred = model(
                    noisy_latent, timesteps, encoder_hidden_states, return_dict=False
                )[0]

                # Get the target for loss depending on the prediction type
                if config.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(
                        prediction_type=config.prediction_type
                    )
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(
                        clean_images, noise, timesteps
                    )
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                    )

                if config.snr_gamma is None:
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="mean"
                    )
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack(
                        [snr, config.snr_gamma * torch.ones_like(timesteps)], dim=1
                    ).min(dim=1)[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)

                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="none"
                    )
                    loss = (
                        loss.mean(dim=list(range(1, len(loss.shape))))
                        * mse_loss_weights
                    )
                    loss = loss.mean()

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
            pipeline = StableDiffusionPipeline.from_pretrained(
                pretrained_model_name_or_path,
                text_encoder=accelerator.unwrap_model(text_encoder),
                vae=accelerator.unwrap_model(vae),
                unet=accelerator.unwrap_model(model),
                tokenizer=tokenizer,
                # revision=config.revision,
            )
            pipeline = pipeline.to(accelerator.device)

            if config.enable_xformers_memory_efficient_attention:
                pipeline.enable_xformers_memory_efficient_attention()

            generate_samples = (
                epoch + 1
            ) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1
            save_model = (
                epoch + 1
            ) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1
            save_to_wandb = epoch == config.num_epochs - 1

            if generate_samples:
                if config.use_ema:
                    # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                    ema_unet.store(model.parameters())
                    ema_unet.copy_to(model.parameters())

                evaluate(config, epoch, pipeline, prompt=evaluation_prompts)
                if config.use_ema:
                    # Switch back to the original UNet parameters.
                    ema_unet.restore(model.parameters())
            if save_model:
                if config.push_to_hub:
                    repo.push_to_hub(commit_message=f"Epoch {epoch}", blocking=True)
                else:
                    if config.use_ema:
                        ema_unet.copy_to(model.parameters())
                    pipeline.save_pretrained(config.output_dir)
                    if save_to_wandb:
                        wandb_logger.save_model()

            progress_bar.close()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    # Now we evaluate the model on the test set
    if (
        accelerator.is_main_process
        and config.calculate_fid
        and test_dataloader is not None
    ):
        if config.use_ema:
            ema_unet.copy_to(model.parameters())

        pipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path,
            text_encoder=accelerator.unwrap_model(text_encoder),
            vae=accelerator.unwrap_model(vae),
            unet=accelerator.unwrap_model(model),
            tokenizer=tokenizer,
            # revision=config.revision,
        )
        pipeline = pipeline.to(accelerator.device)

        if config.enable_xformers_memory_efficient_attention:
            pipeline.enable_xformers_memory_efficient_attention()

        fid_score = calculate_fid_score(
            config, pipeline, test_dataloader, prompt_dict=test_prompt_dict
        )

        wandb_logger.log_fid_score(fid_score)

    if (
        accelerator.is_main_process
        and config.calculate_is
        and test_dataloader is not None
    ):
        inception_score = calculate_inception_score(
            config,
            pipeline,
            test_dataloader,
            device=accelerator.device,
            prompt_dict=test_prompt_dict,
        )
        wandb_logger.log_inception_score(inception_score)
    wandb_logger.finish()


def deepspeed_zero_init_disabled_context_manager():
    """
    returns either a context list that includes one that will disable zero.Init or an empty context list
    """
    deepspeed_plugin = (
        AcceleratorState().deepspeed_plugin
        if accelerate.state.is_initialized()
        else None
    )
    if deepspeed_plugin is None:
        return []

    return [deepspeed_plugin.zero3_init_context_manager(enable=False)]


def tokenize_captions(prompts, tokenizer):
    captions = []
    for caption in prompts:
        captions.append(caption)
    inputs = tokenizer(
        captions,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs.input_ids


def collate_fn(examples):
    pixel_values = torch.stack([example["images"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([example["input_ids"] for example in examples])
    return {"images": pixel_values, "input_ids": input_ids}


def load_prompts(prompt_path):
    f = open(prompt_path, "r")
    data = json.load(f)
    prompts_data = defaultdict()
    for k, v in data.items():
        image_name = k.split(".")[0]
        prompts_data[image_name] = v["overall_caption"]
    return prompts_data


def load_request_prompt(prompt_path):
    # Get first batch_size rows prompts
    data = pd.read_table(prompt_path, header=None)
    data.columns = ["image", "prompt"]
    data["image"] = data["image"].apply(lambda x: int(x.split(".")[0]))
    prompt_dict = dict(zip(data["image"], data["prompt"]))
    return prompt_dict


if __name__ == "__main__":
    stable_diffusion_prompt_dir: str = (
        "../datasets/celeba_hq_stable_diffusion/captions_hq.json"
    )
    stable_diffusion_request_prompt_dir: str = (
        "../datasets/celeba_hq_stable_diffusion/request_hq.txt"
    )
    # train_prompts = load_prompts(stable_diffusion_prompt_dir)
    test_prompts = load_request_prompt(stable_diffusion_request_prompt_dir)
