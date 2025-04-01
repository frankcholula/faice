import argparse
from typing import Dict, List
from conf.training_config import ButterflyConfig, FaceConfig


def get_available_scheduelrs() -> List:
    return ["ddpm", "ddim", "pndm", "lms"]


def get_available_modles() -> List:
    return ["unet", "unet3d", "controlnet"]


def get_available_pipelines() -> List:
    return ["ddpm", "ddim"]


def get_available_datasets() -> Dict:
    return {
        "butterfly": ButterflyConfig,
        "face": FaceConfig,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Training arguments")
    parser.add_argument(
        "--dataset",
        type=str,
        default="face",
        choices=get_available_datasets(),
        help="Dataset to use for training",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="unet",
        choices=get_available_modles(),
        help="Model to use for training",
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        default="ddpm",
        choices=get_available_pipelines(),
        help="Pipeline to use for training",
    )

    parser.add_argument(
        "--scheduler",
        type=str,
        default="ddpm",
        choices=get_available_scheduelrs(),
        help="Scheduler to use for training",
    )

    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=16, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate for training"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=20, help="Number of epochs for training"
    )
    parser.add_argument(
        "--image_size", type=int, default=256, help="Image size for training"
    )
    parser.add_argument(
        "--output_dir", type=str, default="runs", help="Output directory for training"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for training")
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable wandb logging",
    )
