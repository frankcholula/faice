from diffusers import UNet2DModel


class BaseUNet(UNet2DModel):
    """Baseline model given. Don't tweak this.
    This is technically wrong because it's built for 256 x 256 images.
    """

    def __init__(self, config):
        super().__init__(
            sample_size=config.image_size,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",  # 256 -> 128
                "DownBlock2D",  # 128 -> 64
                "DownBlock2D",  # 64 -> 32
                "DownBlock2D",  # 32 -> 16
                "AttnDownBlock2D",  # 16 -> 8
                "DownBlock2D",  # 8 -> 4
            ),
            up_block_types=(
                "UpBlock2D",  # 4 -> 8
                "AttnUpBlock2D",  # 8 -> 16
                "UpBlock2D",  # 16 -> 32
                "UpBlock2D",  # 32 -> 64
                "UpBlock2D",  # 64 -> 128
                "UpBlock2D",  # 128 -> 256
            ),
        )


class DDPMUNet(UNet2DModel):
    """This class mirrors the DDPM paper."""

    def __init__(self, config):
        super().__init__(
            sample_size=config.image_size,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            attention_head_dim=512,  # 512 for single head attention at the 16 x 16 resolution.
            time_embedding_type="positional",
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",  # 128 -> 64
                "DownBlock2D",  # 64 -> 32
                "DownBlock2D",  # 32 -> 16
                "AttnDownBlock2D",  # 16 -> 8
                "DownBlock2D",  # 8 -> 4
                "DownBlock2D",  # 4 -> 2
            ),
            up_block_types=(
                "UpBlock2D",  # 2 -> 4
                "UpBlock2D",  # 4 -> 8
                "AttnUpBlock2D",  # 8 -> 16
                "UpBlock2D",  # 16 -> 32
                "UpBlock2D",  # 32 -> 64
                "UpBlock2D",  # 64 -> 128
            ),
        )


class ADMUNet(UNet2DModel):
    """This is the model used in the AMDM paper. We should run some ablations using this class."""

    def __init__(self, config):
        super().__init__(
            sample_size=config.image_size,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            attention_head_dim=64,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )


ARCHITECTURES = {
    "base": BaseUNet,
    "ddpm": DDPMUNet,
    "adm": ADMUNet,
}


def create_unet(config):
    try:
        cls = ARCHITECTURES[config.unet_variant]
    except KeyError:
        raise ValueError(
            f"Unknown UNet variant {config.unet_variant!r}. "
            f"Choose from {list(ARCHITECTURES)}"
        )
    return cls(config)
