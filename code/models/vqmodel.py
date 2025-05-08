# -*- coding: UTF-8 -*-
"""
@Time : 17/04/2025 20:23
@Author : xiaoguangliang
@File : vqmodel.py
@Project : code
"""
from diffusers import VQModel


def base_vqvae(config, latent_channels=3):
    vqvae = VQModel(
        sample_size=config.image_size,  # the target image resolution
        in_channels=3,  # RGB images
        out_channels=3,
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(128, 256, 512),
        latent_channels=latent_channels,
        num_vq_embeddings=8192,  # Codebook size
        # scaling_factor=1,
        # the number of output channels for each UNet block
        down_block_types=(
            "DownEncoderBlock2D",  # a regular ResNet downsampling block
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
        ),
        up_block_types=(
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
        ),
    )

    return vqvae


def vqvae_b_3(config):
    return base_vqvae(config, latent_channels=3)


def vqvae_b_16(config):
    return base_vqvae(config, latent_channels=16)


def vqvae_b_32(config):
    return base_vqvae(config, latent_channels=32)


def vqvae_b_64(config):
    return base_vqvae(config, latent_channels=64)


if __name__ == "__main__":
    pass
