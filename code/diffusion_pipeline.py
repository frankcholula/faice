# Standard library imports
import os
import glob
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List

# Deep learning and related imports
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm.auto import tqdm

# Diffusers and Hugging Face imports
from datasets import load_dataset
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator, notebook_launcher
from huggingface_hub import HfFolder, Repository, whoami

# Image handling
from PIL import Image

# wandb integration
import wandb
from datetime import datetime


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
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    seed = 0
    
    # saving params
    save_image_epochs = 10
    save_model_epochs = 25
    output_dir: str = field(default=None)  # the model name locally and on the HF Hub
    overwrite_output_dir:bool = True  # overwrite the old model when re-running the notebook


    # hugging face hub params    
    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_private_repo = False

    # wandb params
    use_wandb: bool= True # use wandb for logging
    wandb_entity: str = "tsufanglu"
    wandb_project: str = field(default=None)

    wandb_run_name: Optional[str] = None
    wandb_watch_model: bool = True

    # dataset
    dataset_name: str = field(default=None)

    def __post__init__(self):
        if self.output_dir is None:
            raise NotImplementedError("output_dir must be specified")
        if self.dataset_name is None:
            raise NotImplementedError("dataset_name must be specified")
        if self.wandb_project is None:
            raise NotImplementedError("wandb_project must be specified")
        if self.wandb.run_name is None:
            self.wanadb_run_name = f"ddpm-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        

@dataclass
class ButterflyConfig(TrainingConfig):
    output_dir: str = "ddpm-butterflies-128"
    dataset_name: str = "huggan/smithsonian_butterflies_subset"
    wandb_project: str = "ddpm-butterflies-128"

@dataclass
class FaceConfig(TrainingConfig):
    output_dir: str = "ddpm-celeba-hq-256"
    dataset_name: str = "korexyz/celeba-hq-256x256"
    wandb_project: str = "ddpm-celeba-hq-256"
