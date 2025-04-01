# -*- coding: UTF-8 -*-
"""
@Time : 20/03/2025 13:30
@Author : xiaoguangliang
@File : U_Net_with_LoRA.py
@Project : faice
"""
from diffusers import UNet2DModel

from conf.log_conf import logger
from conf.model_config import model_config
from peft import LoraConfig, get_peft_model, PeftModel


def unet2d_model():
    model = UNet2DModel(
        sample_size=model_config.image_size,  # the target image resolution
        in_channels=3,  # the number of input channels, 3 for RGB images
        out_channels=3,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(128, 128, 256, 256, 512, 512),
        # the number of output channels for each UNet block
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",  # a regular ResNet upsampling block
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )

    # Define LoRA setting
    lora_config = LoraConfig(
        r=8,  # rank
        lora_alpha=32,
        # target_modules=["conv1", "conv1"],
        target_modules=["down_blocks.0.resnets.0.conv1", "down_blocks.0.resnets.0.conv2"],
        lora_dropout=0.1,
        bias="none",
        task_type="FEATURE_EXTRACTION",
        inference_mode=False
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    # Load LoRA weights from the pretrained repository "sassad/face-lora".
    logger.info("LoRA weights loaded into UNet2DModel.")
    model = PeftModel.from_pretrained(model, "sassad/face-lora")

    return model
