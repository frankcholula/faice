# -*- coding: UTF-8 -*-
"""
@Time : 05/05/2025 18:09
@Author : xiaoguangliang
@File : unet_with_pretrain.py
@Project : code
"""
from diffusers import UNet2DConditionModel

from utils.model_tools import freeze_layers


class BaseUNetCondition(object):
    def __init__(
        self,
        config,
        compress_rate=8,
        attention_head_dim=8,
        layers_per_block=2,
        block_num=4,
    ):
        self.sample_size = int(config.image_size / compress_rate)
        self.attention_head_dim = attention_head_dim
        self.layers_per_block = layers_per_block
        self.block_num = block_num

    def unet_cond_l(self):
        model = UNet2DConditionModel(
            sample_size=self.sample_size,  # the target image resolution
            in_channels=4,  # the number of input channels, 3 for RGB images
            out_channels=4,  # the number of output channels
            cross_attention_dim=768,
            attention_head_dim=self.attention_head_dim,
            layers_per_block=self.layers_per_block,  # how many ResNet layers to use per UNet block
            **self.multi_attention_block(),
        )

        # Initialize pretrained model
        model = model.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",  # Base model
            subfolder="unet",
        )

        # Freeze some layers
        frozen_layers = 3
        freeze_layers(model, freeze_until_layer=frozen_layers)

        return model

    def multi_attention_block(self):
        block_out_channels = [320, 640, 1280, 1280]
        down_block_types = [
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ]
        up_block_types = [
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
        ]
        if self.block_num == 4:
            block_out_channels = block_out_channels
            down_block_types = down_block_types
            up_block_types = up_block_types
        elif self.block_num == 5:
            block_out_channels = block_out_channels + [1280]
            down_block_types = ["CrossAttnDownBlock2D"] + down_block_types
            up_block_types = up_block_types + ["CrossAttnUpBlock2D"]
        elif self.block_num == 6:
            block_out_channels = block_out_channels + [1120, 1344]
            down_block_types = ["CrossAttnDownBlock2D"] * 2 + down_block_types
            up_block_types = up_block_types + ["CrossAttnUpBlock2D"] * 2
        blocks = {
            "block_out_channels": tuple(block_out_channels),
            "down_block_types": tuple(down_block_types),
            "up_block_types": tuple(up_block_types),
        }
        return blocks


_compress_rate = 8


def unet_cond_l_block_4(config):
    return BaseUNetCondition(config, compress_rate=_compress_rate).unet_cond_l()



