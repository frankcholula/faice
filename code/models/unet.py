from diffusers import UNet2DModel


class BaseUNet(UNet2DModel):
    """Baseline model given to us. DO NOT TWEAK."""

    def __init__(self, config):
        super().__init__(
            sample_size=config.image_size,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            attention_head_dim=8,
            block_out_channels=(128, 128, 256, 256, 512, 512),
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


class UNetAttentionHeads(UNet2DModel):
    """Change the attention resolution to 16 x 16 since we are starting with 128 x 128 images."""

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
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )


class UNetMultiResAttentionHeads(UNet2DModel):
    """Adding attention at 32 x 32, 16 x 16, and 8 x 8 resolutions."""

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
    "attention_heads": UNetAttentionHeads,
    "multi_res": UNetMultiResAttentionHeads,
}


def create_unet(config):
    """
    config.unet_variant should be one of: "base", "attention_heads", or "multi_res"
    """
    try:
        cls = ARCHITECTURES[config.unet_variant]
    except KeyError:
        raise ValueError(
            f"Unknown UNet variant {config.unet_variant!r}. "
            f"Choose from {list(ARCHITECTURES)}"
        )
    return cls(config)
