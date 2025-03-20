# -*- coding: UTF-8 -*-
"""
@Time : 19/03/2025 16:03
@Author : xiaoguangliang
@File : model_config.py
@Project : faice
"""
from dataclasses import dataclass

from codes.conf.global_setting import BASE_DIR


# ********************************************* MODEL SETTING ********************************************* #
@dataclass
class TrainingConfig:
    image_size = 256  # the generated image resolution
    train_batch_size = 32
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 1
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 30
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = BASE_DIR + "/output/celeba_hq_256_training"  # the model name locally and on the HF Hub
    test_dir = BASE_DIR + "/data/celebahq256_3000/valid"

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0


config = TrainingConfig()
