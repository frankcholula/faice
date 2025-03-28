# -*- coding: UTF-8 -*-
"""
@Time : 20/03/2025 13:28
@Author : xiaoguangliang
@File : train.py
@Project : faice
"""
import os
from dataclasses import asdict

import numpy as np
import torch
from PIL import Image
from diffusers import DDPMPipeline, DDIMPipeline, PNDMPipeline, ConsistencyModelPipeline, ScoreSdeVePipeline, \
    KarrasVePipeline, LDMPipeline, UniDiffuserPipeline
from diffusers import DDPMScheduler, DDIMScheduler, PNDMScheduler, ScoreSdeVeScheduler, KarrasVeScheduler, \
    UniPCMultistepScheduler
from diffusers.schedulers import ConsistencyDecoderScheduler, CMStochasticIterativeScheduler
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from huggingface_hub import HfFolder, Repository, whoami
from tqdm.auto import tqdm, trange
from pathlib import Path
from accelerate import notebook_launcher
import sentry_sdk
from sentry_sdk import capture_exception
import wandb

from codes.conf.log_conf import logger
from codes.core.data_exploration.preprocess_data import get_data
from codes.conf.global_setting import BASE_DIR, SETTINGS
from codes.conf.model_config import model_config
from codes.conf.model_config import wandb_config
from codes.core.FID_score import calculate_fid, make_fid_input_images
# from codes.core.models.U_Net2D_with_pretrain import unet2d_model
from codes.core.models.U_Net2D import unet2d_model

# from codes.core.models.VQModels import vqvae

# Capture the error with Sentry
sentry_sdk.init(SETTINGS.SENTRY_URL)

pipeline_selector = {
    # "DDPM": {"pipeline": DDPMPipeline, "scheduler": DDPMScheduler},
    # "PNDM": {"pipeline": PNDMPipeline, "scheduler": PNDMScheduler},
    # "Consistency_DDPM": {"pipeline": ConsistencyModelPipeline,
    #                      "scheduler": DDPMScheduler},

    "DDIM": {"pipeline": DDIMPipeline, "scheduler": DDIMScheduler},
    "DDIM_DDPM": {"pipeline": DDIMPipeline, "scheduler": DDPMScheduler},
    # "ScoreSdeVe": {"pipeline": ScoreSdeVePipeline, "scheduler": ScoreSdeVeScheduler},

    # unexpected keyword argument num_train_timesteps
    # "Karras": {"pipeline": KarrasVePipeline, "scheduler": KarrasVeScheduler},

    # "LDMP_DDIM": {"pipeline": LDMPipeline, "scheduler": DDIMScheduler}, # TypeError: LDMPipeline.__init__() missing 1 required positional argument: 'vqvae'
    # "LDMP_PNDM": {"pipeline": LDMPipeline, "scheduler": PNDMScheduler},

    # "Consistency": {"pipeline": ConsistencyModelPipeline,
    #                 "scheduler": CMStochasticIterativeScheduler},

}


def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        if i >= 16:
            break
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`

    # Get the name of the pipeline
    pipeline_name = pipeline.__class__.__name__
    logger.info(f"Evaluating {pipeline_name}")
    if 'DDIM' in pipeline_name:
        images = pipeline(
            eta=0.5,
            batch_size=config.eval_batch_size,
            generator=torch.manual_seed(config.seed),
            num_inference_steps=1000
        ).images
    else:
        images = pipeline(
            batch_size=config.eval_batch_size,
            generator=torch.manual_seed(config.seed),
            num_inference_steps=1000
        ).images

    # Make a grid out of the images
    image_grid = make_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")


def generate_images_for_test(config, pipeline, num_images=model_config.num_images):
    logger.info("Generate fake images")
    batch_size = config.eval_batch_size
    num_batches = (num_images + batch_size - 1) // batch_size  # Ceiling division

    all_fake_images = []
    pipeline_name = pipeline.__class__.__name__
    logger.info(f"Generating fake images with {pipeline_name}")

    for i in trange(num_batches):
        batch_seed = config.seed + i  # Use a different seed for each batch to ensure diversity
        if 'DDIM' in pipeline_name:
            images = pipeline(
                eta=0.5,
                batch_size=batch_size,
                generator=torch.manual_seed(batch_seed),
                output_type="np",
                num_inference_steps=1000
            ).images
        else:
            images = pipeline(
                batch_size=batch_size,
                generator=torch.manual_seed(batch_seed),
                output_type="np",
                num_inference_steps=1000
            ).images

        # Convert images from float32 to uint8
        images_uint8 = (images * 255).astype(np.uint8)

        # Save images
        for j, image in enumerate(images_uint8):
            k = i * batch_size + j
            if k >= num_images:
                break
            test_dir = os.path.join(config.output_dir, "test_samples")
            os.makedirs(test_dir, exist_ok=True)
            # Save image
            image = Image.fromarray(image)
            image.save(f"{test_dir}/{k:04d}.png")

        fake_images = torch.tensor(images)
        fake_images = fake_images.permute(0, 3, 1, 2)
        all_fake_images.append(fake_images)

    # Concatenate all batches into a single tensor
    fake_images = torch.cat(all_fake_images)[:num_images]  # Ensure exactly 300 images
    return fake_images


def get_full_repo_name(model_id: str, organization: str = None, token: str = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler, device,
               selected_pipeline, wandb_run):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )

    if wandb_config.use_wandb and wandb_config.wandb_watch_model:
        wandb_run.watch(model, log="all", log_freq=wandb_config.log_freq)

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

    # Get the name of selected_pipeline
    pipeline_name = selected_pipeline.__name__
    scheduler_name = noise_scheduler.__class__.__name__

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["images"]

            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            if 'Karras' in scheduler_name:
                num_train_timesteps = 1000
            else:
                num_train_timesteps = noise_scheduler.config.num_train_timesteps
            timesteps = torch.randint(
                0, num_train_timesteps, (bs,), device=device
            ).long()

            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(device)

            if 'LDMP' in pipeline_name:
                pass
                # Encode image to latent space
                # latents = vqvae.encode(clean_images).latents
                # # Add noise (diffusion process)
                # noise = torch.randn_like(latents)
                # # Add noise to the clean images according to the noise magnitude at each timestep
                # # (this is the forward diffusion process)
                # noisy_images = noise_scheduler.add_noise(latents, noise, timesteps)
            elif 'Karras' in scheduler_name:
                noisy_images = noise_scheduler.add_noise_to_input(clean_images, sigma=0.02,
                                                                  generator=torch.manual_seed(config.seed))
            else:
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

            if wandb_config.use_wandb:
                wandb_run.log(logs, step=global_step)

            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:

            if 'LDMP' in pipeline_name:
                pass
                # pipeline = selected_pipeline(vqvae=accelerator.unwrap_model(vqvae),
                #                              unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
            else:
                pipeline = selected_pipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline)

                # Calculate FID
                if epoch == config.num_epochs - 1:
                    fake_images = generate_images_for_test(config, pipeline)
                    real_images = make_fid_input_images(config.test_dir)

                    calculate_fid(real_images, fake_images, device)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                if config.push_to_hub:
                    repo.push_to_hub(commit_message=f"Epoch {epoch}", blocking=True)
                else:
                    pipeline.save_pretrained(config.output_dir)


def main(data_dir):
    # 1. Make train dataset
    dataset = get_data(data_dir)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=model_config.train_batch_size,
                                                   shuffle=True)

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

    # 3. Set up the optimizer, the learning rate scheduler and the loss scaling for AMP
    optimizer = torch.optim.AdamW(model.parameters(), lr=model_config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=model_config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * model_config.num_epochs),
    )

    # 4. Select pipline and scheduler
    for k, v in tqdm(pipeline_selector.items()):

        # Update the wandb_run_name
        origin_name = wandb_config.wandb_run_name
        run_name = f"{k}_{origin_name}"
        wandb_run = wandb.init(entity=wandb_config.wandb_entity,
                               project=wandb_config.wandb_project,
                               name=run_name,
                               config=asdict(model_config))

        logger.info(f"Select {k} pipeline and scheduler")
        selected_scheduler = v['scheduler']
        selected_pipeline = v['pipeline']

        # Update output_dir
        model_config.output_dir = os.path.join(model_config.base_output_dir, f"{k}")

        # Get the name of scheduler
        scheduler_name = selected_scheduler.__name__
        if 'DDPM' in scheduler_name:
            noise_scheduler = selected_scheduler(
                num_train_timesteps=1000,
                beta_start=0.0001,
                beta_end=0.02,
                beta_schedule="linear",
            )
        elif 'DDIM' in scheduler_name:
            noise_scheduler = selected_scheduler(
                num_train_timesteps=1000,
                beta_start=0.0001,
                beta_end=0.02,
                beta_schedule="linear",
                # clip_sample=False,
                timestep_spacing="trailing"
            )
        elif 'PDNM' in scheduler_name:
            noise_scheduler = selected_scheduler(
                num_train_timesteps=1000,
                beta_start=0.0001,
                beta_end=0.02,
                beta_schedule="linear",
            )
        elif 'ScoreSdeVe' in scheduler_name:
            noise_scheduler = selected_scheduler(
                num_train_timesteps=1000,
                snr=0.15,
                sigma_min=0.001,
                sigma_max=2348.0,
                sampling_eps=1e-3,
                correct_steps=3,
            )
        elif 'Karras' in scheduler_name:
            noise_scheduler = selected_scheduler(
                # sigma_min=0.02,
                # sigma_max=100,
                # s_churn=80.0,
                # s_min=0.01,
                # s_max=100,
            )
        elif 'CMS' in scheduler_name:
            noise_scheduler = selected_scheduler(
                num_train_timesteps=1000,
                sigma_min=0.001,
                sigma_max=200,
                # s_noise=1.0,
                # rho=0.7,
            )
        else:
            noise_scheduler = selected_scheduler(num_train_timesteps=1000)

        args = (model_config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler, device,
                selected_pipeline, wandb_run)

        notebook_launcher(train_loop, args, num_processes=1)

        if wandb_config.use_wandb:
            wandb_run.alert(title="Finish", text=f"The {k} training is done")

        if wandb_config.use_wandb:
            wandb_run.finish()


if __name__ == "__main__":
    # data_path = BASE_DIR + "/data/celeba_hq_256/"
    # data_path = BASE_DIR + "/data/celebahq256_3000/train"
    data_path = BASE_DIR + "/data/celeba_hq_split/train"

    try:
        main(data_path)
    except Exception as exc:
        capture_exception(exc)
        print(exc)
        raise exc
