from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv
import os

load_dotenv()


@dataclass
class BaseConfig:
    """
    Base configuration for diffuser training.
    """

    # model params
    model_type: str = "unet2d"
    scheduler_type: str = "ddpm"
    pipeline_type: str = "ddpm"

    # dataset params need to be set by subclass
    dataset_type: str = field(default=None)
    dataset_name: str = field(default=None)
    image_size: int = 128

    # training params
    train_batch_size: int = 16
    eval_batch_size: int = 16
    num_epochs: int = 20
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 500
    mixed_precision: str = "fp16"
    seed: int = 0

    # saving params
    save_image_epochs: int = 5
    save_model_epochs: int = 10
    output_dir: str = field(default=None)  # need to be set by subclass
    overwrite_output_dir: bool = True

    # hugging face hub params
    push_to_hub = False
    hub_private_repo = False

    # wandb params
    use_wandb: bool = True  # use wandb for logging
    wandb_entity: str = os.environ.get("WANDB_ENTITY")
    wandb_project: str = "faice"
    wandb_run_name: Optional[str] = None
    wandb_watch_model: bool = True

    # evaluation
    calculate_fid: bool = False

    def __post_init__(self):
        if self.output_dir is None:
            raise NotImplementedError("output_dir must be specified")
        if self.dataset_name is None:
            raise NotImplementedError("dataset_name must be specified")
        if self.wandb_run_name is None:
            self.wandb_run_name = (
                f"{self.scheduler_type}-{self.datset_type}-{self.num_epochs}"
            )


@dataclass
class ButterflyConfig(BaseConfig):
    num_epochs: int = 1
    output_dir: str = "runs/ddpm-butterflies-128"
    dataset_name: str = "huggan/smithsonian_butterflies_subset"


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
