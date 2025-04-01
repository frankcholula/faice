import argparse
import inspect
from typing import Dict, List


def get_available_schedulers() -> List:
    return ["ddpm", "ddim", "pndm", "lms"]

def create_scheduler(scheduler_type: str, num_train_timesteps: int = 1000):
    from diffusers import DDPMScheduler
    scheduler_map = {
        "ddpm": lambda: DDPMScheduler(num_train_timesteps=num_train_timesteps),
    }
    return scheduler_map.get(scheduler_type.lower(), scheduler_map["ddpm"])()


def get_available_models() -> List:
    return ["unet"]


def create_model(model_type: str, config):
    if model_type.lower() == "unet":
        from models.unet import create_unet
        return create_unet(config)
    

def get_available_pipelines() -> List:
    return ["ddpm"]


def create_pipeline(pipeline_type: str = "ddpm"):
    if pipeline_type.lower() == "ddpm":
        from pipelines.ddpm import train_loop
    return train_loop

def get_available_datasets() -> Dict:
    from conf.training_config import ButterflyConfig, FaceConfig
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
        choices=get_available_models(),
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
        choices=get_available_schedulers(),
        help="Scheduler to use for training",
    )

    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size for training"
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

    # Add additional parameters
    parser.add_argument(
        "--param",
        action="append",
        nargs=2,
        metavar=("key", "value"),
        help="Additional parameters for training",
    )

    args = parser.parse_args()

    # Convert the args to a config object
    config_class = get_available_datasets()[args.dataset]
    config = config_class()

    if args.train_batch_size is not None:
        config.train_batch_size = args.train_batch_size
    if args.eval_batch_size is not None:
        config.eval_batch_size = args.eval_batch_size
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.num_epochs is not None:
        config.num_epochs = args.num_epochs
    if args.image_size is not None:
        config.image_size = args.image_size
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.seed is not None:
        config.seed = args.seed
    if args.no_wandb:
        config.use_wandb = False

    if args.param:
        for key, value in args.param:
            if hasattr(config, key):
                # Try to convert to the right type
                attr_type = type(getattr(config, key))
                try:
                    if attr_type == bool:
                        # Special handling for booleans
                        value = value.lower() in ("yes", "true", "t", "1")
                    else:
                        value = attr_type(value)
                    setattr(config, key, value)
                except ValueError:
                    print(f"WARNING: Couldn't convert {value} to {attr_type} for {key}")
            else:
                print(f"WARNING: Configuration has no attribute '{key}'")

    return config, args.model, args.scheduler, args.pipeline


def get_config_and_components():
    config, model_type, scheduler_type, pipeline_type = parse_args()
    print(f"Selected dataset: {config.dataset_name}")
    print(f"Selected model: {model_type}")
    print(f"Selected scheduler: {scheduler_type}")
    print(f"Selected pipeline: {pipeline_type}")
    model = create_model(model_type, config)
    scheduler = create_scheduler(scheduler_type)
    pipeline = create_pipeline(pipeline_type)
    return config, model, scheduler, pipeline

if __name__ == "__main__":
    config, model, scheduler, pipeline = get_config_and_components()
    print("\nDetailed Configuration Params")
    print("=" * 50)
    for k, v in inspect.getmembers(config):
        if not k.startswith("__") and not inspect.ismethod(v):
            print(f"{k}: {v}")
