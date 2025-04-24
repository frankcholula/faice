# -*- coding: UTF-8 -*-
"""
@Time : 09/04/2025 19:26
@Author : xiaoguangliang
@File : unet_resnet.py
@Project : code
"""
from diffusers import UNet2DModel


def create_unet_resnet512(config):
    model = UNet2DModel(
        sample_size=config.image_size,  # the target image resolution
        in_channels=3,  # the number of input channels, 3 for RGB images
        out_channels=3,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        attention_head_dim=64,
        # attn_norm_num_groups=32,
        time_embedding_type="positional",
        upsample_type="resnet",
        downsample_type="resnet",
        # num_class_embeds=1000,
        block_out_channels=(
            128,
            128,
            256,
            256,
            512,
            512,
        ),

        down_block_types=(
            "ResnetDownsampleBlock2D",
            "ResnetDownsampleBlock2D",
            "ResnetDownsampleBlock2D",
            # "ResnetDownsampleBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
            # "ResnetDownsampleBlock2D"
            "AttnDownBlock2D"
        ),
        up_block_types=(
            # "ResnetUpsampleBlock2D",
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            # "ResnetUpsampleBlock2D",
            "ResnetUpsampleBlock2D",
            "ResnetUpsampleBlock2D",
            "ResnetUpsampleBlock2D"
        ),

    )
    return model


def create_unet_resnet1024(config):
    model = UNet2DModel(
        sample_size=config.image_size,  # the target image resolution
        in_channels=3,  # the number of input channels, 3 for RGB images
        out_channels=3,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        attention_head_dim=64,
        # attn_norm_num_groups=32,
        time_embedding_type="positional",
        upsample_type="resnet",
        downsample_type="resnet",
        # num_class_embeds=1000,
        block_out_channels=(
            256,
            256,
            512,
            512,
            1024,
            1024
        ),

        down_block_types=(
            "ResnetDownsampleBlock2D",
            "ResnetDownsampleBlock2D",
            "ResnetDownsampleBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D"
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "ResnetUpsampleBlock2D",
            "ResnetUpsampleBlock2D",
            "ResnetUpsampleBlock2D"
        ),

    )
    return model


def create_unet_resnet768(config):
    model = UNet2DModel(
        sample_size=config.image_size,  # the target image resolution
        in_channels=3,  # the number of input channels, 3 for RGB images
        out_channels=3,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        attention_head_dim=64,
        attn_norm_num_groups=32,
        time_embedding_type="positional",
        upsample_type="resnet",
        downsample_type="resnet",
        # num_class_embeds=1000,
        block_out_channels=(
            192,
            384,
            576,
            768
        ),

        down_block_types=(
            "ResnetDownsampleBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D"
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "ResnetUpsampleBlock2D",
        ),

    )
    return model
