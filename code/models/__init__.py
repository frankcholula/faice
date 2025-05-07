from .unet import unet_b_block_6, unet_b_block_8, unet_b_block_6_head_dim_64, unet_b_block_8_head_dim_64, \
    unet_b_block_8_head_dim_64_layer_4, unet_l_block_4, unet_l_block_4_head_dim_64, \
    unet_l_block_4_head_dim_64_layer_4, unet_l_block_5, unet_l_block_5_head_dim_64, \
    unet_l_block_5_head_dim_64_layer_3, \
    unet_l_block_5_head_dim_64_layer_4, unet_l_block_6, unet_l_block_6_head_dim_64, \
    unet_l_block_6_head_dim_64_layer_4, unet_l_block_8_head_dim_64, unet_l_block_8_head_dim_64_layer_4
from .unet_with_pretrain import unet_cond_l_block_4
from .unet_resnet import create_unet_resnet512, create_unet_resnet1024, create_unet_resnet768
from .transformer import DiT_XL_2, DiT_XL_4, DiT_XL_8, DiT_L_2, DiT_L_4, DiT_L_8, DiT_B_2, DiT_B_4, DiT_B_8, \
    DiT_XL_2_vae_channels_4, DiT_XL_4_vae_channels_4, DiT_XL_8_vae_channels_4, DiT_L_2_vae_channels_4, \
    DiT_L_4_vae_channels_4, DiT_L_8_vae_channels_4, DiT_B_2_vae_channels_4, DiT_B_2_vae_channels_16, DiT_B_4_vae_channels_4, DiT_B_8_vae_channels_4, \
    DiT_XL_2_transformer_2d, DiT_XL_4_transformer_2d, DiT_XL_8_transformer_2d, DiT_L_2_transformer_2d, \
    DiT_L_4_transformer_2d, DiT_L_8_transformer_2d, DiT_B_2_transformer_2d, DiT_B_4_transformer_2d, \
    DiT_B_8_transformer_2d, DiT_XL_2_vae_channels_4_transformer_2d, DiT_XL_4_vae_channels_4_transformer_2d, \
    DiT_XL_8_vae_channels_4_transformer_2d, DiT_L_2_vae_channels_4_transformer_2d, \
    DiT_L_4_vae_channels_4_transformer_2d, DiT_L_8_vae_channels_4_transformer_2d, \
    DiT_B_2_vae_channels_4_transformer_2d, DiT_B_4_vae_channels_4_transformer_2d, \
    DiT_B_8_vae_channels_4_transformer_2d
from .vae import vae_b_4, vae_b_16, vae_l_4, vae_l_16
from .vqmodel import vqvae_b_3, vqvae_b_16, vqvae_b_32, vqvae_b_64

__model_factory = {
    "unet": unet_b_block_6,
    "unet_b_block_8": unet_b_block_8,
    "unet_b_block_6_head_dim_64": unet_b_block_6_head_dim_64,
    "unet_b_block_8_head_dim_64": unet_b_block_8_head_dim_64,
    "unet_b_block_8_head_dim_64_layer_4": unet_b_block_8_head_dim_64_layer_4,
    "unet_l_block_4": unet_l_block_4,
    "unet_l_block_4_head_dim_64": unet_l_block_4_head_dim_64,
    "unet_l_block_4_head_dim_64_layer_4": unet_l_block_4_head_dim_64_layer_4,
    "unet_l_block_5": unet_l_block_5,
    "unet_l_block_5_head_dim_64": unet_l_block_5_head_dim_64,
    "unet_l_block_5_head_dim_64_layer_3": unet_l_block_5_head_dim_64_layer_3,
    "unet_l_block_5_head_dim_64_layer_4": unet_l_block_5_head_dim_64_layer_4,
    "unet_l_block_6": unet_l_block_6,
    "unet_l_block_6_head_dim_64": unet_l_block_6_head_dim_64,
    "unet_l_block_6_head_dim_64_layer_4": unet_l_block_6_head_dim_64_layer_4,
    "unet_l_block_8_head_dim_64": unet_l_block_8_head_dim_64,
    "unet_l_block_8_head_dim_64_layer_4": unet_l_block_8_head_dim_64_layer_4,
    "unet_cond_l": unet_cond_l_block_4,
    "unet_resnet512": create_unet_resnet512,
    "unet_resnet1024": create_unet_resnet1024,
    "unet_resnet768": create_unet_resnet768,
    "DiT_XL_2": DiT_XL_2,
    "DiT_XL_4": DiT_XL_4,
    "DiT_XL_8": DiT_XL_8,
    "DiT_L_2": DiT_L_2,
    "DiT_L_4": DiT_L_4,
    "DiT_L_8": DiT_L_8,
    "DiT_B_2": DiT_B_2,
    "DiT_B_4": DiT_B_4,
    "DiT_B_8": DiT_B_8,
    "DiT_XL_2_vae_channels_4": DiT_XL_2_vae_channels_4,
    "DiT_XL_4_vae_channels_4": DiT_XL_4_vae_channels_4,
    "DiT_XL_8_vae_channels_4": DiT_XL_8_vae_channels_4,
    "DiT_L_2_vae_channels_4": DiT_L_2_vae_channels_4,
    "DiT_L_4_vae_channels_4": DiT_L_4_vae_channels_4,
    "DiT_L_8_vae_channels_4": DiT_L_8_vae_channels_4,
    "DiT_B_2_vae_channels_4": DiT_B_2_vae_channels_4,
    "DiT_B_2_vae_channels_16": DiT_B_2_vae_channels_16,
    "DiT_B_4_vae_channels_4": DiT_B_4_vae_channels_4,
    "DiT_B_8_vae_channels_4": DiT_B_8_vae_channels_4,
    "DiT_XL_2_transformer_2d": DiT_XL_2_transformer_2d,
    "DiT_XL_4_transformer_2d": DiT_XL_4_transformer_2d,
    "DiT_XL_8_transformer_2d": DiT_XL_8_transformer_2d,
    "DiT_L_2_transformer_2d": DiT_L_2_transformer_2d,
    "DiT_L_4_transformer_2d": DiT_L_4_transformer_2d,
    "DiT_L_8_transformer_2d": DiT_L_8_transformer_2d,
    "DiT_B_2_transformer_2d": DiT_B_2_transformer_2d,
    "DiT_B_4_transformer_2d": DiT_B_4_transformer_2d,
    "DiT_B_8_transformer_2d": DiT_B_8_transformer_2d,
    "DiT_XL_2_vae_channels_4_transformer_2d": DiT_XL_2_vae_channels_4_transformer_2d,
    "DiT_XL_4_vae_channels_4_transformer_2d": DiT_XL_4_vae_channels_4_transformer_2d,
    "DiT_XL_8_vae_channels_4_transformer_2d": DiT_XL_8_vae_channels_4_transformer_2d,
    "DiT_L_2_vae_channels_4_transformer_2d": DiT_L_2_vae_channels_4_transformer_2d,
    "DiT_L_4_vae_channels_4_transformer_2d": DiT_L_4_vae_channels_4_transformer_2d,
    "DiT_L_8_vae_channels_4_transformer_2d": DiT_L_8_vae_channels_4_transformer_2d,
    "DiT_B_2_vae_channels_4_transformer_2d": DiT_B_2_vae_channels_4_transformer_2d,
    "DiT_B_4_vae_channels_4_transformer_2d": DiT_B_4_vae_channels_4_transformer_2d,
    "DiT_B_8_vae_channels_4_transformer_2d": DiT_B_8_vae_channels_4_transformer_2d,
    "vae_b_4": vae_b_4,
    "vae_b_16": vae_b_16,
    "vae_l_4": vae_l_4,
    "vae_l_16": vae_l_16,
    "vqvae_channel_3": vqvae_b_3,
    "vqvae_channel_16": vqvae_b_16,
    "vqvae_channel_32": vqvae_b_32,
    "vqvae_channel_64": vqvae_b_64,
}


def get_names():
    return list(__model_factory.keys())


def init_model(name, *args, **kwargs):
    if name not in list(__model_factory.keys()):
        raise KeyError(f"Model type '{name}' is not supported")
    return __model_factory[name](*args, **kwargs)
