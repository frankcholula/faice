from .unet import create_unet
from .unet_resnet import create_unet_resnet512, create_unet_resnet1024, create_unet_resnet768
from .transformer_2d import create_transformers_2d

__model_factory = {
    "unet": create_unet,
    "unet_resnet512": create_unet_resnet512,
    "unet_resnet1024": create_unet_resnet1024,
    "unet_resnet768": create_unet_resnet768,
    "transformer_2d": create_transformers_2d,
}


def get_names():
    return list(__model_factory.keys())


def init_model(name, *args, **kwargs):
    if name not in list(__model_factory.keys()):
        raise KeyError(f"Model type '{name}' is not supported")
    return __model_factory[name](*args, **kwargs)
