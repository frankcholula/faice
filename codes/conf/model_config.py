# -*- coding: UTF-8 -*-
"""
@Time : 19/03/2025 16:03
@Author : xiaoguangliang
@File : model_config.py
@Project : faice
"""
from typing import Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime

import wandb

from codes.conf.global_setting import BASE_DIR


# ********************************************* MODEL SETTING ********************************************* #
@dataclass
class TrainingConfig:
    image_size = 256  # the generated image resolution
    train_batch_size = 36
    eval_batch_size = 36  # how many images to sample during evaluation
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 30
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    base_output_dir = BASE_DIR + "/output/celeba_hq_split_training"
    output_dir = base_output_dir  # the model name locally and on the HF Hub
    # test_dir = BASE_DIR + "/data/celeba_hq_split/test"
    test_dir = BASE_DIR + "/data/celeba_hq_split/test30"
    # The number of generating images
    num_images = 30

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0


model_config = TrainingConfig()


@dataclass
class WandbConfig:
    project = "celebahq-256-splitted"
    use_wandb: bool = True  # use wandb for logging
    wandb_entity: str = "ngene"
    wandb_project: str = field(default=None)
    wandb_watch_model: bool = True
    wandb_run_name: str = f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log_freq: int = 10


wandb_config = WandbConfig()

