from dataclasses import dataclass
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
    model: str = "unet"
    scheduler: str = "ddpm"
    beta_schedule: str = "linear"
    pipeline: str = "ddpm"
    prediction_type: str = "epsilon"
    rescale_betas_zero_snr: bool = False

    # dataset params need to be set by subclass
    dataset: str = None
    dataset_name: str = None
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
    num_train_timesteps: int = 1000
    num_inference_steps: int = 1000

    # saving params
    save_image_epochs: int = 50
    save_model_epochs: int = 50
    output_dir: str = None
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
    calculate_is: bool = False

    def __post_init__(self):
        if self.dataset_name is None:
            raise NotImplementedError("dataset_name must be specified")


@dataclass
class ButterflyConfig(BaseConfig):
    dataset: str = "butterfly"
    dataset_name: str = "huggan/smithsonian_butterflies_subset"
    num_epochs: int = 1
    image_size: int = 128


@dataclass
class FaceConfig(BaseConfig):
    dataset: str = "face"
    dataset_name: str = "uos-celebahq-256x256"
    num_epochs: int = 1
    save_image_epochs: int = 50
    save_model_epochs: int = 50
    train_dir: str = "datasets/celeba_hq_split/train"
    test_dir: str = "datasets/celeba_hq_split/test"


CONFIG_REGISTRY = {
    "butterfly": ButterflyConfig,
    "face": FaceConfig,
}


def get_config(dataset: str):
    if dataset not in CONFIG_REGISTRY:
        raise ValueError(f"Dataset {dataset} not recognized.")
    return CONFIG_REGISTRY[dataset]()


def get_all_datasets():
    return list(CONFIG_REGISTRY.keys())
