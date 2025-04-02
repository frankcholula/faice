# -*- coding: UTF-8 -*-
"""
@Time : 27/03/2025 09:54
@Author : xiaoguangliang
@File : VQModels.py
@Project : faice
"""
from diffusers import VQModel
from src.conf.model_config import model_config

# TypeError: unsupported operand type(s) for +: 'NoneType' and 'int'
vqvae = VQModel(
    sample_size=model_config.image_size,  # the target image resolution
    in_channels=3,  # RGB images
    out_channels=3,
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(128, 128, 256, 256, 512, 512),
    act_fn="silu",
    latent_channels=3,
    num_vq_embeddings=512,  # Codebook size
    vq_embed_dim=64,  # Latent dimension
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

