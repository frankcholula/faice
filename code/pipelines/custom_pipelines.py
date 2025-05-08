# -*- coding: UTF-8 -*-
"""
@Time : 07/05/2025 17:51
@Author : xiaoguangliang
@File : custom_pipelines.py
@Project : code
"""
import inspect
from typing import List, Optional, Tuple, Union

import torch
import numpy as np
from diffusers.utils.torch_utils import randn_tensor
from diffusers import DiffusionPipeline
from diffusers import AutoencoderKL, Transformer2DModel
from diffusers import DDIMScheduler
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


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
        class_labels: List[int],
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
        batch_size = class_labels.shape[0]
        if isinstance(self.dit.config.sample_size, int):
            image_shape = (
                batch_size,
                self.dit.config.in_channels,
                self.dit.config.sample_size,
                self.dit.config.sample_size,
            )
        else:
            image_shape = (
                batch_size,
                self.dit.config.in_channels,
                *self.dit.config.sample_size,
            )

        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            image = randn_tensor(image_shape, generator=generator, dtype=self.dit.dtype)
            image = image.to(self.device)
        else:
            image = randn_tensor(
                image_shape,
                generator=generator,
                device=self.device,
                dtype=self.dit.dtype,
            )

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            # Convert one number t to 1d-array
            t = t.cpu().numpy()
            t = np.array([t])
            t = torch.from_numpy(t).to(device)
            model_output = self.dit(image, timestep=t, class_labels=class_labels).sample

            # 2. compute previous image: x_t -> x_t-1
            t = t.cpu()
            image = self.scheduler.step(
                model_output, t, image, generator=generator
            ).prev_sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)


class CustomTransformerVAEPipeline(DiffusionPipeline):
    r"""
    Pipeline for unconditional image generation using latent diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        vae ():
        scheduler ([`SchedulerMixin`]):
            [`DDIMScheduler`] is used in combination with `unet` to denoise the encoded image latents.
    """

    def __init__(
        self, vae: AutoencoderKL, dit: Transformer2DModel, scheduler: DDIMScheduler
    ):
        super().__init__()
        self.register_modules(vae=vae, dit=dit, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        class_labels: List[int],
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float = 0.0,
        num_inference_steps: int = 50,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        **kwargs,
    ) -> Union[Tuple, ImagePipelineOutput]:
        r"""
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`, *optional*, defaults to 1):
                Number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Example:

        ```py

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        batch_size = class_labels.shape[0]
        latents = randn_tensor(
            (
                batch_size,
                self.dit.config.in_channels,
                self.dit.config.sample_size,
                self.dit.config.sample_size,
            ),
            generator=generator,
        )
        latents = latents.to(self.device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        self.scheduler.set_timesteps(num_inference_steps)

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )

        extra_kwargs = {}
        if accepts_eta:
            extra_kwargs["eta"] = eta

        for t in self.progress_bar(self.scheduler.timesteps):
            latent_model_input = self.scheduler.scale_model_input(latents, t)

            # Convert one number t to 1d-array
            t = t.cpu().numpy()
            t = np.array([t])
            t = torch.from_numpy(t).to(device)
            noise_prediction = self.dit(
                latent_model_input, timestep=t, class_labels=class_labels
            ).sample

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(
                noise_prediction, t, latents, **extra_kwargs
            ).prev_sample

        # adjust latents with inverse of vae scale
        latents = latents / self.vae.config.scaling_factor
        # decode the image latents with the VAE
        image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
