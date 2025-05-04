# -*- coding: UTF-8 -*-
"""
@Time : 22/04/2025 19:08
@Author : xiaoguangliang
@File : transformer.py
@Project : code
"""
from diffusers import DiTTransformer2DModel, Transformer2DModel


class DiT(object):
    def __init__(self, config, depth=28, hidden_size=1152, patch_size=2, num_heads=16, compress_rate=4,
                 channels=4, attention_type='default'):
        self.sample_size = int(config.image_size / compress_rate)
        self.num_layers = depth
        self.num_attention_heads = num_heads
        self.attention_head_dim = hidden_size // num_heads
        self.patch_size = patch_size
        self.channels = channels
        self.attention_type = attention_type

    def create_dit(self):
        dit = DiTTransformer2DModel(
            sample_size=self.sample_size,
            in_channels=self.channels,
            out_channels=self.channels,
            activation_fn="gelu-approximate",
            attention_bias=True,
            attention_head_dim=self.attention_head_dim,
            norm_type="ada_norm_zero",
            num_attention_heads=self.num_attention_heads,
            num_embeds_ada_norm=1000,
            num_layers=self.num_layers,
            patch_size=self.patch_size,
        )

        return dit

    def create_transformer_2d(self):
        dit = Transformer2DModel(
            sample_size=self.sample_size,
            in_channels=self.channels,
            out_channels=self.channels,
            activation_fn="gelu-approximate",
            attention_bias=True,
            attention_head_dim=self.attention_head_dim,
            norm_type="ada_norm_zero",
            num_attention_heads=self.num_attention_heads,
            num_embeds_ada_norm=1000,
            num_layers=self.num_layers,
            patch_size=self.patch_size,
            attention_type=self.attention_type
        )

        if self.attention_type == 'flash':
            dit.enable_flash_attention()
        elif self.attention_type == 'xformers':
            dit.enable_xformers_memory_efficient_attention()
        else:
            pass

        return dit


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(config, **kwargs):
    return DiT(config, depth=28, hidden_size=1152, patch_size=2, num_heads=16, compress_rate=1,
               channels=3, **kwargs).create_dit()


def DiT_XL_4(config, **kwargs):
    return DiT(config, depth=28, hidden_size=1152, patch_size=4, num_heads=16, compress_rate=1,
               channels=3, **kwargs).create_dit()


def DiT_XL_8(config, **kwargs):
    return DiT(config, depth=28, hidden_size=1152, patch_size=8, num_heads=16, compress_rate=1,
               channels=3, **kwargs).create_dit()


def DiT_L_2(config, **kwargs):
    return DiT(config, depth=24, hidden_size=1024, patch_size=2, num_heads=16, compress_rate=1,
               channels=3, **kwargs).create_dit()


def DiT_L_4(config, **kwargs):
    return DiT(config, depth=24, hidden_size=1024, patch_size=4, num_heads=16, compress_rate=1,
               channels=3, **kwargs).create_dit()


def DiT_L_8(config, **kwargs):
    return DiT(config, depth=24, hidden_size=1024, patch_size=8, num_heads=16, compress_rate=1,
               channels=3, **kwargs).create_dit()


def DiT_B_2(config, **kwargs):
    return DiT(config, depth=12, hidden_size=768, patch_size=2, num_heads=12, compress_rate=1,
               channels=3, **kwargs).create_dit()


def DiT_B_4(config, **kwargs):
    return DiT(config, depth=12, hidden_size=768, patch_size=4, num_heads=12, compress_rate=1,
               channels=3, **kwargs).create_dit()


def DiT_B_8(config, **kwargs):
    return DiT(config, depth=12, hidden_size=768, patch_size=8, num_heads=12, compress_rate=1,
               channels=3, **kwargs).create_dit()


def DiT_XL_2_vae_layers4(config, **kwargs):
    return DiT(config, depth=28, hidden_size=1152, patch_size=2, num_heads=16, compress_rate=8,
               channels=4, **kwargs).create_dit()


def DiT_XL_4_vae_layers4(config, **kwargs):
    return DiT(config, depth=28, hidden_size=1152, patch_size=4, num_heads=16, compress_rate=8,
               channels=4, **kwargs).create_dit()


def DiT_XL_8_vae_layers4(config, **kwargs):
    return DiT(config, depth=28, hidden_size=1152, patch_size=8, num_heads=16, compress_rate=8,
               channels=4, **kwargs).create_dit()


def DiT_L_2_vae_layers4(config, **kwargs):
    return DiT(config, depth=24, hidden_size=1024, patch_size=2, num_heads=16, compress_rate=8,
               channels=4, **kwargs).create_dit()


def DiT_L_4_vae_layers4(config, **kwargs):
    return DiT(config, depth=24, hidden_size=1024, patch_size=4, num_heads=16, compress_rate=8,
               channels=4, **kwargs).create_dit()


def DiT_L_8_vae_layers4(config, **kwargs):
    return DiT(config, depth=24, hidden_size=1024, patch_size=8, num_heads=16, compress_rate=8,
               channels=4, **kwargs).create_dit()


def DiT_B_2_vae_layers4(config, **kwargs):
    return DiT(config, depth=12, hidden_size=768, patch_size=2, num_heads=12, compress_rate=8,
               channels=4, **kwargs).create_dit()


def DiT_B_4_vae_layers4(config, **kwargs):
    return DiT(config, depth=12, hidden_size=768, patch_size=4, num_heads=12, compress_rate=8,
               channels=4, **kwargs).create_dit()


def DiT_B_8_vae_layers4(config, **kwargs):
    return DiT(config, depth=12, hidden_size=768, patch_size=8, num_heads=12, compress_rate=8,
               channels=4, **kwargs).create_dit()
