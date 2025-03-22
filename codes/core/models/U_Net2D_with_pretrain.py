# -*- coding: UTF-8 -*-
"""
@Time : 20/03/2025 13:30
@Author : xiaoguangliang
@File : U_Net2D_with_pretrain.py
@Project : faice
"""
from diffusers import UNet2DModel

from codes.conf.log_conf import logger
from codes.conf.model_config import config


def freeze_layers(model, freeze_until_layer):
    """
    Freeze layers until the specified layer index.
    """
    layers = 0
    for name, param in model.named_parameters():
        # Split the parameter name by '.'
        parts = name.split('.')

        # Check if the second part is a digit (e.g., '0', '1')
        if len(parts) > 1 and parts[1].isdigit():
            layer_index = int(parts[1])
            layers += 1
            if layer_index < freeze_until_layer:
                param.requires_grad = False
            else:
                param.requires_grad = True
        else:
            # Skip parameters that do not match the expected format
            continue
    logger.info(f"The model has {layers} layers and freeze the front {freeze_until_layer} layers")


def unet2d_model():
    model = UNet2DModel(
        sample_size=config.image_size,  # the target image resolution
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

    # Initialize pretrained model
    model = model.from_pretrained(
        "google/ddpm-celebahq-256",  # Base model
    )

    # Freeze some layers
    frozen_layers = 3
    # frozen_layers = 409
    # frozen_layers = 205
    freeze_layers(model, freeze_until_layer=frozen_layers)

    return model


if __name__ == '__main__':
    model = unet2d_model()
