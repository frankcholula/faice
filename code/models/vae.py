# -*- coding: UTF-8 -*-
"""
@Time : 22/04/2025 19:19
@Author : xiaoguangliang
@File : vae.py
@Project : code
"""
from diffusers import AutoencoderKL


def create_vae(config):
    vae = AutoencoderKL(
        sample_size=config.image_size,  # the target image resolution
        in_channels=3,  # the number of input channels, 3 for RGB images
        out_channels=3,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        latent_channels=16,
        block_out_channels=(128, 256, 512),
        down_block_types=(
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
        ),
        up_block_types=(
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
        ),

    )
    return vae
