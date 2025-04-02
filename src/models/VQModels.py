# -*- coding: UTF-8 -*-
"""
@Time : 27/03/2025 09:54
@Author : xiaoguangliang
@File : VQModels.py
@Project : faice
"""
from diffusers import VQModel
from conf.model_config import model_config

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
    vq_embed_dim=3,  # Latent dimension
    # the number of output channels for each UNet block
    down_block_types=(
        "DownEncoderBlock2D",  # a regular ResNet downsampling block
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "AttnDownEncoderBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownEncoderBlock2D",
    ),
    up_block_types=(
        "UpDecoderBlock2D",  # a regular ResNet upsampling block
        "AttnUpDecoderBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
    ),
)

# vqvae = VQModel()

