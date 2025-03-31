# Standard library imports
import os
from pathlib import Path

# Deep learning framework
from accelerate import Accelerator
from huggingface_hub import Repository

# Local imports
from utils.metrics import get_full_repo_name


def setup_accelerator(config):
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )

    # Set up the repository for pushing to hub if requested
    repo = None
    if accelerator.is_main_process:
        if config.push_to_hub:
            repo_name = get_full_repo_name(Path(config.output_dir).name)
            repo = Repository(config.output_dir, clone_from=repo_name)
        elif config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)

        accelerator.init_trackers("train_example")

    return accelerator, repo
