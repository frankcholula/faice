# Standard library imports
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
from dotenv import load_dotenv

# Image handling
from PIL import Image

# Deep learning framework
import torch
import torch.nn.functional as F
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm.auto import tqdm

# Hugging Face
from diffusers import DDPMPipeline
from huggingface_hub import HfFolder, Repository, whoami
from accelerate import Accelerator

# Monitoring and logging
import wandb


def setup_wandb():
    wandb_entity = os.getenv("WANDB_ENTITY")
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if not wandb_api_key:
        raise ValueError("WANDB_API_KEY not found in .env file")

    if not wandb_entity:
        raise ValueError("WANDB_ENTITY not found in .env file")
    wandb.login(key=wandb_api_key)


load_dotenv()
setup_wandb()


@dataclass
class TrainingConfig:
    """
    Base configuration for diffuser training.
    """

    # image params
    image_size = 128  # the generated image resolution

    # training params
    train_batch_size = 16
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs: int = 20
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    seed = 0

    # saving params
    save_image_epochs = 5
    save_model_epochs = 10
    output_dir: str = field(default=None)  # the model name locally and on the HF Hub
    overwrite_output_dir: bool = (
        True  # overwrite the old model when re-running the notebook
    )

    # hugging face hub params
    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_private_repo = False

    # wandb params
    use_wandb: bool = True  # use wandb for logging
    wandb_entity: str = os.getenv("WANDB_ENTITY")

    wandb_project: str = field(default=None)

    wandb_run_name: Optional[str] = None
    wandb_watch_model: bool = True

    # dataset
    dataset_name: str = field(default=None)

    def __post_init__(self):
        if self.output_dir is None:
            raise NotImplementedError("output_dir must be specified")
        if self.dataset_name is None:
            raise NotImplementedError("dataset_name must be specified")
        if self.wandb_project is None:
            raise NotImplementedError("wandb_project must be specified")
        if self.wandb_run_name is None:
            self.wandb_run_name = f"ddpm-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


@dataclass
class ButterflyConfig(TrainingConfig):
    output_dir: str = "ddpm-butterflies-128"
    dataset_name: str = "huggan/smithsonian_butterflies_subset"
    wandb_project: str = "ddpm-butterflies-128"


@dataclass
class FaceConfig(TrainingConfig):
    output_dir: str = "ddpm-celebahq-256"
    dataset_name: str = "uos-celebahq-256x256"
    wandb_project: str = "ddpm-celebahq-256"
    num_epochs: int = 1
    save_image_epochs: int = 1
    save_model_epochs: int = 1
    train_dir: str = "datasets/celeba_hq_split/train"
    test_dir: str = "datasets/celeba_hq_split/test"
    calculate_fid: bool = True


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
    image_grid_path = f"{test_dir}/{epoch:04d}.png"
    image_grid.save(image_grid_path)
    
    if config.use_wandb:
        wandb.log({
            "generated_images": wandb.Image(
                image_grid_path, caption=f"Epoch {epoch}"
            ),
            "epoch": epoch,
        })


def calculate_fid_score(config, pipeline, test_dataloader, device=None):
    """Calculate FID score between generated images and test dataset"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create FID instance with normalize=True since we'll provide images in [0,1] range
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    # Process real images, no need to convert to BCHW format
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Calculating FID (real images)"):
            real_images = batch["images"].to(device)
            # Convert from [-1, 1] to [0, 1] range
            real_images = (real_images + 1.0) / 2.0
            fid.update(real_images, real=True)

    # Process fake images, need to convert to BCHW but no need to rescale.
    with torch.no_grad():
        for i in tqdm(
            range(0, len(test_dataloader.dataset), config.eval_batch_size),
            desc="Calculating FID (generated images)",
        ):
            # Generate images as numpy arrays
            output = pipeline(
                batch_size=min(
                    config.eval_batch_size, len(test_dataloader.dataset) - i
                ),
                generator=torch.manual_seed(config.seed + i),
                output_type="np.array",
            ).images
            
            # Convert numpy arrays to tensors
            fake_images = torch.tensor(output)
            # Rearrange from BHWC to BCHW format
            fake_images = fake_images.permute(0, 3, 1, 2)
            fake_images = fake_images.to(device)
            fid.update(fake_images, real=False)
    
    # Compute final FID score
    fid_score = fid.compute().item()
    print(f"FID Score: {fid_score}")
    return fid_score

def get_full_repo_name(model_id: str, organization: str = None, token: str = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def train_loop(
    config,
    model,
    noise_scheduler,
    optimizer,
    train_dataloader,
    lr_scheduler,
    test_dataloader,
):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )

    # Initialize wandb
    if config.use_wandb and accelerator.is_main_process:
        wandb.init(
            entity=config.wandb_entity,
            project=config.wandb_project,
            name=config.wandb_run_name,
            config={
                "learning_rate": config.learning_rate,
                "epochs": config.num_epochs,
                "train_batch_size": config.train_batch_size,
                "image_size": config.image_size,
                "seed": config.seed,
                "dataset": config.dataset_name,
                "model_architecture": "UNet2D",
                "scheduler": "DDPM",
            },
        )
        if config.wandb_watch_model:
            wandb.watch(model, log="all", log_freq=10)

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
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bs,),
                device=clean_images.device,
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
            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if config.use_wandb and accelerator.is_main_process:
                wandb.log(logs, step=global_step)

            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = DDPMPipeline(
                unet=accelerator.unwrap_model(model), scheduler=noise_scheduler
            )

            generate_samples = (
                epoch + 1
            ) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1
            save_model = (
                epoch + 1
            ) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1

            if generate_samples:
                evaluate(config, epoch, pipeline)
            if save_model:
                if config.push_to_hub:
                    repo.push_to_hub(commit_message=f"Epoch {epoch}", blocking=True)
                else:
                    pipeline.save_pretrained(config.output_dir)
        if accelerator.is_main_process and config.calculate_fid:

            progress_bar.close()

    # Now we evaluate the model on the test set
    if accelerator.is_main_process and config.calculate_fid:
        pipeline = DDPMPipeline(
            unet=accelerator.unwrap_model(model), scheduler=noise_scheduler
        )
        fid_score = calculate_fid_score(config, pipeline, test_dataloader)

        if config.use_wandb and fid_score is not None:
            wandb.run.summary["fid_score"] = fid_score

    if config.use_wandb and accelerator.is_main_process:
        wandb.finish()
