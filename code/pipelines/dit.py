# Deep learning framework
from copy import deepcopy
import os
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import numpy as np

import accelerate
from diffusers import Transformer2DModel
from diffusers.utils.import_utils import is_xformers_available
from diffusers.training_utils import EMAModel
from packaging import version

# Configuration
from utils.metrics import calculate_fid_score, calculate_inception_score
from utils.metrics import evaluate
from utils.loggers import WandBLogger
from utils.training import setup_accelerator
from utils.model_tools import name_to_label, update_ema, requires_grad
from pipelines.custom_pipelines import CustomTransformer2DPipeline

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
num_class = 2

selected_pipeline = CustomTransformer2DPipeline


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

    # Create EMA for the unet.
    if config.use_ema:
        ema_model = EMAModel(
            model.parameters(),
            model_cls=Transformer2DModel,
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
            print('Start using xformers ...')
            model.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if config.use_ema:
                    ema_model.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            if config.use_ema:
                load_model = EMAModel.from_pretrained(
                    os.path.join(input_dir, "unet_ema"), Transformer2DModel, foreach=config.foreach_ema
                )
                ema_model.load_state_dict(load_model.state_dict())
                if config.offload_ema:
                    ema_model.pin_memory()
                else:
                    ema_model.to(accelerator.device)
                del load_model

            for _ in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = Transformer2DModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if config.use_ema:
        if config.offload_ema:
            ema_model.pin_memory()
        else:
            ema_model.to(accelerator.device)

    if config.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                print(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            print('Start using xformers ...')
            model.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

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
            map_ids = torch.tensor(image_labels, dtype=torch.int, device=device).reshape(-1)

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bs,),
                device=clean_images.device,
            ).long()

            # # Add noise (diffusion process)
            noise = torch.randn_like(clean_images).to(clean_images.device)
            # # Add noise to the clean images according to the noise magnitude at each timestep
            # # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images,
                                   timestep=timesteps,
                                   class_labels=map_ids).sample
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
                accelerator.unwrap_model(model),
                # dit=accelerator.unwrap_model(ema),
                scheduler=noise_scheduler
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
                    ema_model.store(model.parameters())
                    ema_model.copy_to(model.parameters())
                if config.enable_xformers_memory_efficient_attention:
                    pipeline.enable_xformers_memory_efficient_attention()

                evaluate_batch_size = 16
                class_labels = torch.randint(
                    0,
                    num_class,
                    (evaluate_batch_size,),
                    device=device,
                ).int()
                evaluate(config, epoch, pipeline, class_labels=class_labels)

                if config.use_ema:
                    # Switch back to the original UNet parameters.
                    ema_model.restore(model.parameters())

            if save_model:
                if config.push_to_hub:
                    repo.push_to_hub(commit_message=f"Epoch {epoch}", blocking=True)
                else:
                    if config.use_ema:
                        ema_model.copy_to(model.parameters())

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
            ema_model.copy_to(model.parameters())

        pipeline = selected_pipeline(
            dit=accelerator.unwrap_model(model),
            # dit=accelerator.unwrap_model(ema),
            scheduler=noise_scheduler
        )
        pipeline = pipeline.to(accelerator.device)

        if config.enable_xformers_memory_efficient_attention:
            pipeline.enable_xformers_memory_efficient_attention()

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
