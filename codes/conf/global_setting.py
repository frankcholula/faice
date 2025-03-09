# -*- coding: UTF-8 -*-
"""
@Time : 09/03/2025 11:20
@Author : xiaoguangliang
@File : global_setting.py
@Project : faice
"""
import os
import json

from dataclasses import dataclass

# ********************************************* PATH SETTING ********************************************* #

# 项目基础路径
BASE_DIR = os.path.dirname(os.path.dirname(__file__))


# ********************************************* MODEL SETTING ********************************************* #
@dataclass
class TrainingConfig:
    image_size = 128  # the generated image resolution
    train_batch_size = 8
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 2
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 30
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = BASE_DIR + "/output/celeba_hq_256_training"  # the model name locally and on the HF Hub

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0


config = TrainingConfig()
