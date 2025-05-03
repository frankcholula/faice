from .unet import create_unet, create_latent_unet, create_unet_for_ldm, create_latent_unet_xl
from .unet_resnet import create_unet_resnet512, create_unet_resnet1024, create_unet_resnet768
from .transformer import create_dit_transformer, create_transformer_2d, create_transformer_2d_vae, \
    create_transformer_2d_xformers, create_transformer_2d_xformers_vae, create_transformer_2d_xformers_fast
from .vae import create_vae, create_vae_xl
from .vqmodel import create_vqmodel

__model_factory = {
    "unet": create_unet,
    "unet_resnet512": create_unet_resnet512,
    "unet_resnet1024": create_unet_resnet1024,
    "unet_resnet768": create_unet_resnet768,
    "unet_for_ldm": create_unet_for_ldm,
    "latent_unet": create_latent_unet,
    "latent_unet_xl": create_latent_unet_xl,
    "dit_transformer": create_dit_transformer,
    "transformer_2d": create_transformer_2d,
    "transformer_2d_vae": create_transformer_2d_vae,
    "transformer_2d_xformers": create_transformer_2d_xformers,
    "transformer_2d_xformers_vae": create_transformer_2d_xformers_vae,
    "transformer_2d_xformers_fast": create_transformer_2d_xformers_fast,
    "vae": create_vae,
    "vae_xl": create_vae_xl,
    "vqvae": create_vqmodel,
}


def get_names():
    return list(__model_factory.keys())


def init_model(name, *args, **kwargs):
    if name not in list(__model_factory.keys()):
        raise KeyError(f"Model type '{name}' is not supported")
    return __model_factory[name](*args, **kwargs)
