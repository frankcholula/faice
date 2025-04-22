# -*- coding: UTF-8 -*-
"""
@Time : 22/04/2025 19:08
@Author : xiaoguangliang
@File : dit.py
@Project : code
"""
from diffusers import Transformer2DModel


def create_transformers_2d(config):
    transformer_2d = Transformer2DModel(
        sample_size=config.image_size,
        in_channels=3,
        out_channels=3,
        activation_fn="gelu-approximate",
        attention_bias=True,
        attention_head_dim=72,
        norm_elementwise_affine=False,
        norm_type="ada_norm_zero",
        num_attention_heads=16,
        num_embeds_ada_norm=1000,
        num_layers=28,
        patch_size=2,

    )
    return transformer_2d
