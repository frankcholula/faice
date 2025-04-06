import argparse
import inspect
import sys
from pipelines import ddpm
from diffusers import DDPMScheduler, DDIMScheduler, PNDMScheduler
from models.unet import create_unet2d
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
        return create_unet2d(config)
    else:
        raise ValueError(f"Model type '{model}' is not supported.")


def create_pipeline(pipeline: str):
    if pipeline.lower() == "ddpm":
        return ddpm.train_loop
    else:
        raise ValueError(f"Pipeline type '{pipeline}' is not supported.")


def parse_args():
    parser = argparse.ArgumentParser(description="Diffusion Model Training")

    # define argument groups
    dataset_group = parser.add_argument_group("Dataset and Augmentation")
    # TODO: tweak hyperparameters for training in training_group
    training_group = parser.add_argument_group("Training and Evaluation")
    logging_group = parser.add_argument_group("Logging and Output")  # Don't touch this
    # TODO: implement more models, schedulers, and pipelines in model_group
    model_group = parser.add_argument_group("Model, Scheduler, and Pipeline")

    dataset_group.add_argument(
        "--dataset", choices=get_all_datasets(), help="Dataset to use"
    )
    dataset_group.add_argument(
        "--train_dir", help="Directory with training images (for face dataset)"
    )
    dataset_group.add_argument(
        "--test_dir", help="Directory with test images (for face dataset)"
    )
    dataset_group.add_argument(
        "--glur", help="Gaussian blurring augmentation", action="store_true"
    )
    dataset_group.add_argument(
        "--RHFlip", help="Random horizontal flip augmentation", action="store_true"
    )

    training_group.add_argument(
        "--train_batch_size", type=int, help="Batch size for training"
    )
    training_group.add_argument(
        "--eval_batch_size", type=int, help="Batch size for evaluation"
    )
    training_group.add_argument(
        "--num_epochs", type=int, help="Number of training epochs"
    )
    training_group.add_argument("--learning_rate", type=float, help="Learning rate")
    training_group.add_argument(
        "--image_size", type=int, help="Image size for training"
    )
    training_group.add_argument(
        "--calculate_fid", action="store_true", help="Calculate FID score"
    )
    training_group.add_argument(
        "--calculate_is", action="store_true", help="Calculate Inception score"
    )

    model_group.add_argument("--model", help="Model architecture")
    model_group.add_argument("--scheduler", help="Noise scheduler")
    model_group.add_argument("--pipeline", help="Training pipeline")

    logging_group.add_argument(
        "--output_dir", help="Directory to save models and results"
    )
    # expose some wandb arguments
    logging_group.add_argument(
        "--no_wandb", action="store_true", help="Disable W&B logging"
    )
    logging_group.add_argument("--wandb_run_name", type=str, help="W&B run name")

    parser.add_argument("--verbose", action="store_true", help="Print detailed config")
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
    if config.wandb_run_name is None:
        config.wandb_run_name = f"{config.scheduler}-{dataset}-{config.num_epochs}"
    if config.output_dir is None:
        config.output_dir = f"runs/{config.scheduler}-{dataset}-{config.num_epochs}"
    return config


def get_config_and_components():
    """Get the config and create all necessary components."""
    config = parse_args()

    print(f"Selected dataset: {config.dataset} ({config.dataset_name})")
    print(f"Selected model: {config.model}")
    print(f"Selected scheduler: {config.scheduler}")
    print(f"Selected pipeline: {config.pipeline}")
    print(f"W&B run name: {config.wandb_run_name}")
    print(f"Local output directory: {config.output_dir}")
    print(f"Gaussion Blur? : {config.gblur}")
    print(f"Random Horizontal Flip? : {config.RHFlip}")

    verbose = hasattr(config, "verbose") and config.verbose
    if verbose:
        print_config(config)
    confirmation = (
        input("\nDo you want to proceed with this configuration? (y/n): ")
        .strip()
        .lower()
    )
    if confirmation != "y" and confirmation != "yes":
        print("Training aborted by user.")
        sys.exit(0)
    model = create_model(config.model, config)
    scheduler = create_scheduler(config.scheduler)
    pipeline = create_pipeline(config.pipeline)

    return config, model, scheduler, pipeline


def print_config(config):
    """Print configuration parameters."""
    print("\nDetailed Configuration:")
    print("=" * 50)
    for key, value in inspect.getmembers(config):
        if not key.startswith("__") and not inspect.ismethod(value):
            print(f"{key}: {value}")
