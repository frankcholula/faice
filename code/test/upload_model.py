# -*- coding: UTF-8 -*-
"""
@Time : 28/05/2025 10:20
@Author : xiaoguangliang
@File : upload_model.py
@Project : code
"""
import os
from pathlib import Path
from utils.metrics import get_full_repo_name
from huggingface_hub import Repository


class Config():
    use_wandb = False
    num_inference_steps = 100
    seed = 0
    output_dir = "runs/stable_diffusion_inference"


config = Config()

model_name = Path(config.output_dir).name
repo_name = get_full_repo_name(model_name)
repo = Repository(config.output_dir, clone_from=repo_name)

repo.push_to_hub(commit_message=f"upload model: {model_name}", blocking=True)

