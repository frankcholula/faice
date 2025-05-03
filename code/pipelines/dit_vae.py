# Deep learning framework
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import numpy as np
from torch import nn
from typing import List, Optional, Tuple, Union
import inspect
from collections import OrderedDict
from copy import deepcopy

# Hugging Face
from diffusers import DiffusionPipeline, DiTPipeline, AutoencoderKL, DDIMScheduler, Transformer2DModel
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
from diffusers.utils.torch_utils import randn_tensor

# Configuration
from utils.metrics import calculate_fid_score, calculate_inception_score
from utils.metrics import evaluate
from utils.loggers import WandBLogger
from utils.training import setup_accelerator
from models.vae import create_vae, create_vae_xl
from pipelines.dit import name_to_label

# vae_path = "runs/vae-vae-ddpm-face-500-16/checkpoints/model_vae.pth"
vae_path = "runs/vae_xl-vae-ddpm-face-500-4/checkpoints/model_vae.pth"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
num_class = 2


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

    def __init__(self, vae: AutoencoderKL, dit: Transformer2DModel, scheduler: DDIMScheduler):
        super().__init__()
        self.register_modules(vae=vae, dit=dit, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
            self,
            batch_size: int = 1,
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

        latents = randn_tensor(
            (batch_size, self.dit.config.in_channels, self.dit.config.sample_size,
             self.dit.config.sample_size),
            generator=generator,
        )
        latents = latents.to(self.device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        self.scheduler.set_timesteps(num_inference_steps)

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())

        extra_kwargs = {}
        if accepts_eta:
            extra_kwargs["eta"] = eta

        class_labels = torch.randint(
            0,
            num_class,
            (batch_size,),
            device=device,
        ).int()

        for t in self.progress_bar(self.scheduler.timesteps):
            latent_model_input = self.scheduler.scale_model_input(latents, t)

            # Convert one number t to 1d-array
            t = t.cpu().numpy()
            t = np.array([t])
            t = torch.from_numpy(t).to(device)
            noise_prediction = self.dit(latent_model_input, timestep=t, class_labels=class_labels).sample

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_prediction, t, latents, **extra_kwargs).prev_sample

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


selected_pipeline = DiTPipeline


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

    # vae = AutoencoderKL.from_pretrained("facebook/DiT-XL-2-256", subfolder="vae")
    # vae = vae.to(device)
    # vae.eval().requires_grad_(False)

    # vae = create_vae(config)
    vae = create_vae_xl(config)
    vae = vae.to(device)
    vae.load_state_dict(torch.load(vae_path, map_location=device)['model_state_dict'])
    vae.eval().requires_grad_(False)

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
            class_labels = torch.tensor(image_labels, dtype=torch.int, device=device).reshape(-1)
            # class_labels = class_labels.to(device)
            # class_labels = torch.tensor(class_labels, device=device).reshape(-1)
            class_null = torch.tensor([1000] * bs, device=device)
            class_labels_input = torch.cat([class_labels, class_null], 0)

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

            latent_model_input = torch.cat([latents] * 2)

            half = latent_model_input[: len(latent_model_input) // 2]
            latent_model_input = torch.cat([half, half], dim=0)

            # # Add noise (diffusion process)
            noise = torch.randn_like(latent_model_input).to(clean_images.device)
            # # Add noise to the clean images according to the noise magnitude at each timestep
            # # (this is the forward diffusion process)
            timesteps = torch.cat([timesteps, timesteps], dim=0)
            noisy_latent = noise_scheduler.add_noise(latent_model_input, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_latent,
                                   timestep=timesteps,
                                   class_labels=class_labels_input,
                                   return_dict=False)[0]

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(clean_images, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(noise_pred, target)
                # loss = F.l1_loss(noise_pred, target)
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
                accelerator.unwrap_model(ema),
                accelerator.unwrap_model(vae),
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
                    # pipeline.save_pretrained(config.output_dir)
                    checkpoint = {
                        "model": model.state_dict(),
                        "ema_model": ema.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    }
                    model_path = f"{config.output_dir}/checkpoints/model_dit.pth"
                    torch.save(checkpoint, model_path)
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
            # accelerator.unwrap_model(model),
            accelerator.unwrap_model(ema),
            accelerator.unwrap_model(vae),
            scheduler=noise_scheduler
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


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    # ema_params = OrderedDict(ema_model.named_parameters())
    # model_params = OrderedDict(model.named_parameters())
    #
    # for name, param in model_params.items():
    #     # TO-DO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
    #     ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

    for ema_param, model_param in zip(ema_model.parameters(), model.parameters()):
        ema_param.copy_(decay * ema_param + (1 - decay) * model_param)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag
