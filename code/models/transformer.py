# -*- coding: UTF-8 -*-
"""
@Time : 22/04/2025 19:08
@Author : xiaoguangliang
@File : transformer.py
@Project : code
"""
from diffusers import DiTTransformer2DModel, Transformer2DModel


def create_dit_transformer(config):
    dit_transformer_2d = DiTTransformer2DModel(
        sample_size=int(config.image_size / 4),
        # sample_size=config.image_size,
        in_channels=16,
        out_channels=16,
        activation_fn="gelu-approximate",
        attention_bias=True,
        attention_head_dim=64,
        norm_type="ada_norm_zero",
        num_attention_heads=4,
        num_embeds_ada_norm=1000,
        num_layers=24,
        patch_size=1,

    )
    return dit_transformer_2d


def create_transformer_2d(config):
    transformer_2d = Transformer2DModel(
        sample_size=config.image_size,
        in_channels=3,
        out_channels=3,
        activation_fn="gelu-approximate",
        attention_bias=True,
        attention_head_dim=64,
        norm_type="ada_norm_zero",
        num_attention_heads=4,
        num_embeds_ada_norm=1000,
        num_layers=24,
        patch_size=2,
        # attention_type="flash",
    )

    return transformer_2d


def create_transformer_2d_vae(config):
    transformer_2d = Transformer2DModel(
        sample_size=int(config.image_size / 4),
        in_channels=16,
        out_channels=16,
        activation_fn="gelu-approximate",
        attention_bias=True,
        attention_head_dim=72,
        norm_type="ada_norm_zero",
        num_attention_heads=16,
        num_embeds_ada_norm=1000,
        num_layers=28,
        patch_size=2,
        attention_type="flash",
    )

    return transformer_2d


def create_transformer_2d_xformers(config):
    transformer_2d = Transformer2DModel(
        sample_size=config.image_size,
        in_channels=3,
        out_channels=3,
        activation_fn="gelu-approximate",
        attention_bias=True,
        attention_head_dim=64,
        norm_type="ada_norm_zero",
        num_attention_heads=4,
        num_embeds_ada_norm=1000,
        num_layers=12,
        patch_size=2,
        attention_type="xformers",
    )

    transformer_2d.enable_xformers_memory_efficient_attention()

    return transformer_2d


def create_transformer_2d_xformers_vae(config):
    transformer_2d = Transformer2DModel(
        sample_size=int(config.image_size / 4),
        in_channels=16,
        out_channels=16,
        activation_fn="gelu-approximate",
        attention_bias=True,
        attention_head_dim=32,
        norm_type="ada_norm_zero",
        num_attention_heads=4,
        num_embeds_ada_norm=1000,
        num_layers=12,
        patch_size=2,
        attention_type="xformers",
    )

    transformer_2d.enable_xformers_memory_efficient_attention()

    return transformer_2d


def create_transformer_2d_xformers_fast(config):
    transformer_2d = Transformer2DModel(
        sample_size=config.image_size,
        in_channels=3,
        out_channels=3,
        activation_fn="gelu-approximate",
        attention_bias=True,
        attention_head_dim=64,
        norm_type="ada_norm_zero",
        num_attention_heads=12,
        num_embeds_ada_norm=1000,
        num_layers=12,
        patch_size=4,
        attention_type="xformers",
    )

    transformer_2d.enable_xformers_memory_efficient_attention()

    return transformer_2d


def create_transformer_2d_xl(config):
    transformer_2d = Transformer2DModel(
        sample_size=config.image_size,
        in_channels=3,
        out_channels=3,
        activation_fn="gelu-approximate",
        attention_bias=True,
        attention_head_dim=72,
        norm_type="ada_norm_zero",
        num_attention_heads=16,
        num_embeds_ada_norm=1000,
        num_layers=28,
        patch_size=2,

    )
    return transformer_2d
