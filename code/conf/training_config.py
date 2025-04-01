from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import os


@dataclass
class BaseConfig:
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
    wandb_project: str = "faice"
    wandb_run_name: Optional[str] = None
    wandb_watch_model: bool = True

    # dataset
    dataset_name: str = field(default=None)

    # evaluation
    calculate_fid: bool = False

    def __post_init__(self):
        if self.output_dir is None:
            raise NotImplementedError("output_dir must be specified")
        if self.dataset_name is None:
            raise NotImplementedError("dataset_name must be specified")
        if self.wandb_run_name is None:
            self.wandb_run_name = f"ddpm-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


@dataclass
class ButterflyConfig(BaseConfig):
    num_epochs: int = 1
    output_dir: str = "runs/ddpm-butterflies-128"
    dataset_name: str = "huggan/smithsonian_butterflies_subset"
    wandb_run_name: str = f"ddpm-butterflies-128-{num_epochs}"


@dataclass
class FaceConfig(BaseConfig):
    output_dir: str = "runs/ddpm-celebahq-256"
    dataset_name: str = "uos-celebahq-256x256"
    num_epochs: int = 1
    save_image_epochs: int = 1
    save_model_epochs: int = 1
    train_dir: str = "datasets/celeba_hq_split/train"
    test_dir: str = "datasets/celeba_hq_split/test"
    calculate_fid: bool = False
    wandb_run_name: str = f"ddpm-celebahq-256-{num_epochs}"
