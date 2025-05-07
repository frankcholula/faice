import argparse
import inspect
import sys
from pipelines import (
    consistency,
    ldmp,
    vae_train,
    vqvae_train,
    dit,
    dit_vae,
    base_pipeline,
)
from diffusers import (
    DDPMScheduler,
    DDIMScheduler,
    PNDMScheduler,
    DPMSolverMultistepScheduler,
)
from diffusers.schedulers import CMStochasticIterativeScheduler
import models
from models.unet import create_unet
from conf.training_config import get_config, get_all_datasets


def create_scheduler(
        scheduler: str,
        beta_schedule: str,
        num_train_timesteps: int,
        prediction_type: str,
        rescale_betas_zero_snr: bool,
):
    if scheduler.lower() == "ddpm":
        return DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
            rescale_betas_zero_snr=rescale_betas_zero_snr,
        )
    elif scheduler.lower() == "ddim":
        return DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
            rescale_betas_zero_snr=rescale_betas_zero_snr,
        )
    elif scheduler.lower() == "pndm":
        return PNDMScheduler(
            num_train_timesteps=num_train_timesteps, beta_schedule=beta_schedule
        )
    elif scheduler.lower() == "cmstochastic":
        return CMStochasticIterativeScheduler(num_train_timesteps=num_train_timesteps)
    elif scheduler.lower() == "dpmsolvermultistep":
        return DPMSolverMultistepScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            solver_order=2,
        )
    else:
        raise ValueError(
            f"Scheduler type '{scheduler}' or noise scheduler type '{beta_schedule}' is not supported."
        )


def create_model(model_name: str, config):
    # TODO: refactor this to use the same init_modlel function in the future.
    if model_name == 'unet' and config.unet_variant in ["base", "ddpm", "adm", "cond"]:
        model = create_unet(config)
    else:
        model = models.init_model(config.model, config)

    return model


def create_pipeline(pipeline: str):
    if pipeline.lower() in ["ddim", "ddpm", "pndm", "cond"]:
        return base_pipeline.train_loop
    elif pipeline.lower() == "consistency":
        return consistency.train_loop
    elif pipeline.lower() == "ldmp":
        return ldmp.train_loop
    elif pipeline.lower() == "vqvae":
        return vqvae_train.train_loop
    elif pipeline.lower() == "vae":
        return vae_train.train_loop
    elif pipeline.lower() == "dit":
        return dit.train_loop
    elif pipeline.lower() == "dit_vae":
        return dit_vae.train_loop
    else:
        raise ValueError(f"Pipeline type '{pipeline}' is not supported.")


def parse_args():
    parser = argparse.ArgumentParser(description="Diffusion Model Training")

    # define argument groups
    dataset_group = parser.add_argument_group("Dataset and Augmentation")
    training_group = parser.add_argument_group("Training and Evaluation")
    logging_group = parser.add_argument_group("Logging and Output")  # Don't touch this
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
        "--gblur", help="Gaussian blurring augmentation", action="store_true"
    )
    dataset_group.add_argument(
        "--RHFlip", help="Random horizontal flip augmentation", action="store_true"
    )
    dataset_group.add_argument(
        "--center_crop_arr",
        help="Random horizontal flip augmentation",
        action="store_true",
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
    training_group.add_argument(
        "--num_train_timesteps", type=int, default=1000, help="Number of training steps"
    )
    training_group.add_argument(
        "--num_inference_steps",
        type=int,
        default=1000,
        help="Number of inference steps",
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
    training_group.add_argument("--use_ema", action="store_true", help="Use ema")
    training_group.add_argument(
        "--foreach_ema", action="store_true", help="For each ema"
    )
    training_group.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Enable xformers memory efficient attention",
    )
    training_group.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Enable TF32 for faster training on Ampere GPUs,",
    )
    training_group.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Use gradient checkpointing",
    )
    training_group.add_argument(
        "--scale_lr", action="store_true", help="Scale learning rate"
    )
    training_group.add_argument(
        "--gradient_accumulation_steps", type=int, help="Gradient accumulation steps"
    )

    model_group.add_argument("--model", help="Model architecture")
    model_group.add_argument(
        "--unet_variant",
        choices=["base", "ddpm", "adm", "cond"],
        default="base",
        help="Which UNet variant to use when --model==unet",
    )
    model_group.add_argument(
        "--layers_per_block",
        type=int,
        default=2,
        help="Number of layers per block for UNet (if applicable)",
    )
    model_group.add_argument(
        "--base_channels",
        type=int,
        default=128,
        help="Base channels for UNet (if applicable)",
    )
    model_group.add_argument(
        "--multi_res",
        action="store_true",
        default=False,
        help="Use multi-resolution attention for DDPMUNet (if applicable)",
    )
    model_group.add_argument(
        "--attention_head_dim",
        type=int,
        help="Attention head dimension for DDPMUNet (if applicable)",
    )
    model_group.add_argument(
        "--downsample_type",
        choices=["conv", "resnet"],
        default="conv",
        help="Downsample type for DDPMUNet (if applicable)",
    )
    model_group.add_argument(
        "--upsample_type",
        choices=["conv", "resnet"],
        default="conv",
        help="Upsample type for DDPMUNet (if applicable)",
    )
    model_group.add_argument(
        "--scheduler",
        choices=["ddpm", "ddim", "pndm", "cmstochastic"],
        default="ddpm",
        help="Sampling scheduler",
    )
    model_group.add_argument(
        "--beta_schedule",
        choices=["linear", "scaled_linear", "squaredcos_cap_v2"],
        default="linear",
        help="Beta schedule",
    )
    model_group.add_argument(
        "--pipeline",
        choices=["ddpm", "ddim", "pndm", "consistency", "cond", 'ldmp', 'dit', 'dit_vae', 'vae', 'vqvae'],
        default="ddpm",
        help="Training pipeline",
    )

    model_group.add_argument(
        "--condition_on",
        choices=["male", "female"],
        default="male",
        help="Condition on male or female on inference (if cond pipeline is selected)",
    )
    model_group.add_argument(
        "--rescale_betas_zero_snr",
        action="store_true",
        default=False,
        help="Rescale betas to zero at the end of the training",
    )
    model_group.add_argument(
        "--prediction_type",
        choices=["epsilon", "v_prediction"],
        default="epsilon",
        help="Prediction type for sampling (epsilon or v)",
    )
    model_group.add_argument(
        "--eta", type=float, default=1.0, help="eta value for DDIM sampling"
    )
    logging_group.add_argument(
        "--output_dir", help="Directory to save models and results"
    )
    # expose some wandb arguments
    logging_group.add_argument(
        "--no_wandb", action="store_true", help="Disable W&B logging"
    )
    logging_group.add_argument("--wandb_run_name", type=str, help="W&B run name")

    parser.add_argument("--verbose", action="store_true", help="Print detailed config")
    parser.add_argument(
        "--no_confirm", action="store_true", help="Skip confirmation prompt"
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
    if config.wandb_run_name is None:
        config.wandb_run_name = f"{config.model}-{config.pipeline}-{config.scheduler}-{dataset}-{config.num_epochs}"
    if config.output_dir is None:
        config.output_dir = f"runs/{config.model}-{config.pipeline}-{config.scheduler}-{dataset}-{config.num_epochs}"
    return config


def get_config_and_components():
    """Get the config and create all necessary components."""
    config = parse_args()

    print(f"Selected dataset: {config.dataset} ({config.dataset_name})")
    print(f"Selected model: {config.model}")
    if config.model == "unet":
        print(f"Selected UNet variant: {config.unet_variant}")
    print(f"Selected scheduler: {config.scheduler}")
    print(f"Selected pipeline: {config.pipeline}")
    print(f"W&B run name: {config.wandb_run_name}")
    print(f"Local output directory: {config.output_dir}")
    print(f"Gaussian Blur? : {config.gblur}")
    print(f"Random Horizontal Flip? : {config.RHFlip}")
    print(f"center crop arr? : {config.center_crop_arr}")
    print(f"Prediction_type: {config.prediction_type}")
    print(f"Rescale_betas_zero_snr?: {config.rescale_betas_zero_snr}")

    verbose = hasattr(config, "verbose") and config.verbose
    if verbose:
        print_config(config)

    if not hasattr(config, "no_confirm") or not config.no_confirm:
        confirmation = (
            input("\nDo you want to proceed with this configuration? (y/n): ")
            .strip()
            .lower()
        )
        if confirmation != "y" and confirmation != "yes":
            print("Training aborted by user.")
            sys.exit(0)
    else:
        print("\nSkipping confirmation as --no_confirm flag is set.")
    model = create_model(config.model, config)
    scheduler = create_scheduler(
        config.scheduler,
        config.beta_schedule,
        config.num_train_timesteps,
        config.prediction_type,
        config.rescale_betas_zero_snr,
    )
    pipeline = create_pipeline(config.pipeline)

    return config, model, scheduler, pipeline


def print_config(config):
    """Print configuration parameters."""
    print("\nDetailed Configuration:")
    print("=" * 50)
    for key, value in inspect.getmembers(config):
        if not key.startswith("__") and not inspect.ismethod(value):
            print(f"{key}: {value}")
