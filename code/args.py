import argparse
import inspect
import sys
from pipelines import ddpm
from diffusers import DDPMScheduler, DDIMScheduler, PNDMScheduler
from models.unet import create_unet
from conf.training_config import get_config, get_all_dataset_types


def create_scheduler(scheduler_type: str, num_train_timesteps: int = 1000):
    if scheduler_type.lower() == "ddpm":
        return DDPMScheduler(num_train_timesteps=num_train_timesteps)
    elif scheduler_type.lower() == "ddim":
        return DDIMScheduler(num_train_timesteps=num_train_timesteps)
    elif scheduler_type.lower() == "pndm":
        return PNDMScheduler(num_train_timesteps=num_train_timesteps)
    else:
        raise ValueError(f"Scheduler type '{scheduler_type}' is not supported.")


def create_model(model_type: str, config):
    if model_type.lower() == "unet2d":
        return create_unet(config)
    else:
        raise ValueError(f"Model type '{model_type}' is not supported.")


def create_pipeline(pipeline_type: str):
    if pipeline_type.lower() == "ddpm":
        return ddpm.train_loop
    else:
        raise ValueError(f"Pipeline type '{pipeline_type}' is not supported.")


def parse_args():
    parser = argparse.ArgumentParser(description="Diffusion Model Training")

    parser.add_argument(
        "--dataset_type", choices=get_all_dataset_types(), help="Dataset to use"
    )
    parser.add_argument("--model_type", help="Model architecture")
    parser.add_argument("--scheduler_type", help="Noise scheduler")
    parser.add_argument("--pipeline_type", help="Training pipeline")

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
    dataset_type = args.dataset_type
    config = get_config(dataset_type)
    args_dict = vars(args)

    # update config with command line arguments
    for key, value in args_dict.items():
        if value is not None:
            if key == "no_wandb":
                setattr(config, "use_wandb", not value)  # Handle special case
            elif key != "dataset_type":
                setattr(config, key, value)

    return config


def get_config_and_components():
    """Get the config and create all necessary components."""
    config = parse_args()
    
    print(f"Selected dataset: {config.dataset_type} ({config.dataset_name})")
    print(f"Selected model: {config.model_type}")
    print(f"Selected scheduler: {config.scheduler_type}")
    print(f"Selected pipeline: {config.pipeline_type}")
    
    model = create_model(config.model_type, config)
    scheduler = create_scheduler(config.scheduler_type)
    pipeline = create_pipeline(config.pipeline_type)
    
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