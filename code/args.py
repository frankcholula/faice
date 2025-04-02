import argparse
import inspect
import sys
from pipelines import ddpm
from diffusers import DDPMScheduler, DDIMScheduler, PNDMScheduler
from models.unet import create_unet
from conf.training_config import get_config, get_all_datasets


def create_scheduler(scheduler: str, num_train_timesteps: int = 1000):
    if scheduler.lower() == "ddpm":
        return DDPMScheduler(num_train_timesteps=num_train_timesteps)
    elif scheduler.lower() == "ddim":
        return DDIMScheduler(num_train_timesteps=num_train_timesteps)
    elif scheduler.lower() == "pndm":
        return PNDMScheduler(num_train_timesteps=num_train_timesteps)
    else:
        raise ValueError(f"Scheduler type '{scheduler}' is not supported.")


def create_model(model: str, config):
    if model.lower() == "unet2d":
        return create_unet(config)
    else:
        raise ValueError(f"Model type '{model}' is not supported.")


def create_pipeline(pipeline: str):
    if pipeline.lower() == "ddpm":
        return ddpm.train_loop
    else:
        raise ValueError(f"Pipeline type '{pipeline}' is not supported.")


def parse_args():
    parser = argparse.ArgumentParser(description="Diffusion Model Training")

    parser.add_argument(
        "--dataset", choices=get_all_datasets(), help="Dataset to use"
    )
    parser.add_argument("--model", help="Model architecture")
    parser.add_argument("--scheduler", help="Noise scheduler")
    parser.add_argument("--pipeline", help="Training pipeline")

    parser.add_argument("--train_batch_size", type=int, help="Batch size for training")
    parser.add_argument("--eval_batch_size", type=int, help="Batch size for evaluation")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--image_size", type=int, help="Image size for training")
    parser.add_argument("--seed", type=int, help="Random seed")

    parser.add_argument("--output_dir", help="Directory to save models and results")
    parser.add_argument(
        "--train_dir", help="Directory with training images (for face dataset)"
    )
    parser.add_argument(
        "--test_dir", help="Directory with test images (for face dataset)"
    )

    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument(
        "--calculate_fid", action="store_true", help="Calculate FID score"
    )

    args = parser.parse_args()
    dataset = args.dataset
    config = get_config(dataset)
    args_dict = vars(args)

    # update config with command line arguments
    for key, value in args_dict.items():
        if value is not None:
            if key == "no_wandb":
                setattr(config, "use_wandb", not value)  # Handle special case
            elif key != "dataset":
                setattr(config, key, value)

    return config


def get_config_and_components():
    """Get the config and create all necessary components."""
    config = parse_args()

    print(f"Selected dataset: {config.dataset} ({config.dataset_name})")
    print(f"Selected model: {config.model}")
    print(f"Selected scheduler: {config.scheduler}")
    print(f"Selected pipeline: {config.pipeline}")

    model = create_model(config.model, config)
    scheduler = create_scheduler(config.scheduler)
    pipeline = create_pipeline(config.pipeline)

    return config, model, scheduler, pipeline


def print_config(config):
    """Print configuration parameters."""
    print("\nConfiguration:")
    print("=" * 50)
    for key, value in inspect.getmembers(config):
        if not key.startswith("__") and not inspect.ismethod(value):
            print(f"{key}: {value}")


if __name__ == "__main__":
    config, model, scheduler, pipeline = get_config_and_components()
    print_config(config)
