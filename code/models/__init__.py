from .unet import create_unet, create_lant_unet
from .unet_resnet import create_unet_resnet512, create_unet_resnet1024, create_unet_resnet768
from .transformer import create_dit_transformer, create_transformer_2d
from .vae import create_vae
from .vqmodel import create_vqmodel

__model_factory = {
    "unet": create_unet,
    "unet_resnet512": create_unet_resnet512,
    "unet_resnet1024": create_unet_resnet1024,
    "unet_resnet768": create_unet_resnet768,
    "dit_transformer": create_dit_transformer,
    "transformer_2d": create_transformer_2d,
    "vae": create_vae,
    "vqvae": create_vqmodel,
    "lant_unet": create_lant_unet,
}


def get_names():
    return list(__model_factory.keys())


def init_model(name, *args, **kwargs):
    if name not in list(__model_factory.keys()):
        raise KeyError(f"Model type '{name}' is not supported")
    return __model_factory[name](*args, **kwargs)
