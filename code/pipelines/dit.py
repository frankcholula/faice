# Deep learning framework
import os
from typing import List, Optional, Tuple, Union
from copy import deepcopy

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import numpy as np
from torch import nn
import pandas as pd

from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput

# Configuration
from utils.metrics import calculate_fid_score, calculate_inception_score
from utils.metrics import evaluate
from utils.loggers import WandBLogger
from utils.training import setup_accelerator
from utils.model_tools import name_to_label, update_ema, requires_grad

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
num_class = 2


class CustomTransformer2DPipeline(DiffusionPipeline):
    r"""
    Pipeline for image generation.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        dit: Transformer2DModel
    """

    def __init__(self, dit, scheduler):
        super().__init__()
        self.register_modules(dit=dit, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
            self,
            batch_size: int = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            num_inference_steps: int = 1000,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Example:


        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        # Sample gaussian noise to begin loop
        if isinstance(self.dit.config.sample_size, int):
            image_shape = (
                batch_size,
                self.dit.config.in_channels,
                self.dit.config.sample_size,
                self.dit.config.sample_size,
            )
        else:
            image_shape = (batch_size, self.dit.config.in_channels, *self.dit.config.sample_size)

        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            image = randn_tensor(image_shape, generator=generator, dtype=self.dit.dtype)
            image = image.to(self.device)
        else:
            image = randn_tensor(image_shape, generator=generator, device=self.device, dtype=self.dit.dtype)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        class_labels = torch.randint(
            0,
            num_class,
            (batch_size,),
            device=device,
        ).int()

        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            # Convert one number t to 1d-array
            t = t.cpu().numpy()
            t = np.array([t])
            t = torch.from_numpy(t).to(device)
            model_output = self.dit(image, timestep=t, class_labels=class_labels).sample

            # 2. compute previous image: x_t -> x_t-1
            t = t.cpu()
            image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)


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

    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

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

            # label_num = 2700
            # emb_size = 64
            # label_emb = nn.Embedding(num_embeddings=label_num, embedding_dim=emb_size)
            #
            # map_ids = label_emb(map_ids)

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
                update_ema(ema, model)

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
                # accelerator.unwrap_model(model),
                dit=accelerator.unwrap_model(ema),
                scheduler=noise_scheduler
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

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    # Now we evaluate the model on the test set
    if (
            accelerator.is_main_process
            and config.calculate_fid
            and test_dataloader is not None
    ):
        pipeline = selected_pipeline(
            # dit=accelerator.unwrap_model(model),
            dit=accelerator.unwrap_model(ema),
            scheduler=noise_scheduler
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



