from diffusers import UNet2DModel


def create_unet(config):
    model = UNet2DModel(
        sample_size=config.image_size,  # the target image resolution
        in_channels=3,  # the number of input channels, 3 for RGB images
        out_channels=3,  # the number of output channels
        attention_head_dim=8,
        layers_per_block=2,  # how many ResNet layers to use per UNet block

        block_out_channels=(
            128,
            128,
            256,
            256,
            512,
            512,
        ),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )
    return model


def create_lant_unet(config):
    model = UNet2DModel(
        sample_size=int(config.image_size / 4),  # the target image resolution
        # in_channels=256,  # the number of input channels, 3 for RGB images
        # out_channels=256,  # the number of output channels
        in_channels=3,  # the number of input channels, 3 for RGB images
        out_channels=3,  # the number of output channels
        attention_head_dim=32,
        layers_per_block=2,  # how many ResNet layers to use per UNet block

        block_out_channels=(
            224,
            448,
            672,
            896
        ),
        down_block_types=(
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D"
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D"
        ),
    )
    return model
