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
    """This class mirrors the DDPM paper. I've tweaked it to work with 128 x 128 images."""

    def __init__(self, config):
        if config.multi_res:
            # this is basically the same structure as the ADMUNet
            down_block_types = (
                "DownBlock2D",  # 128 -> 64
                "DownBlock2D",  # 64 -> 32
                "AttnDownBlock2D",  # 32 -> 16
                "AttnDownBlock2D",  # 16 -> 8
                "AttnDownBlock2D",  # 8 -> 4
                "DownBlock2D",  # 4 -> 2
            )
            up_block_types = (
                "UpBlock2D",  # 2 -> 4
                "AttnUpBlock2D",  # 4 -> 8
                "AttnUpBlock2D",  # 8 -> 16
                "AttnUpBlock2D",  # 16 -> 32
                "UpBlock2D",  # 32 -> 64
                "UpBlock2D",  # 64 -> 128
            )
        else:
            down_block_types = (
                "DownBlock2D",  # 128 -> 64
                "DownBlock2D",  # 64 -> 32
                "DownBlock2D",  # 32 -> 16
                "AttnDownBlock2D",  # 16 -> 8
                "DownBlock2D",  # 8 -> 4
                "DownBlock2D",  # 4 -> 2
            )
            up_block_types = (
                "UpBlock2D",  # 2 -> 4
                "UpBlock2D",  # 4 -> 8
                "AttnUpBlock2D",  # 8 -> 16
                "UpBlock2D",  # 16 -> 32
                "UpBlock2D",  # 32 -> 64
                "UpBlock2D",  # 64 -> 128
            )
        super().__init__(
            sample_size=config.image_size,
            in_channels=3,
            out_channels=3,
            layers_per_block=config.layers_per_block,
            attention_head_dim=config.attention_head_dim,  # 256 for single head attention at the 16 x 16 resolution.
            time_embedding_type="positional",
            block_out_channels=tuple(
                config.base_channels * m for m in (1, 1, 2, 2, 4, 4)
            ),
            down_block_types=down_block_types,
            up_block_types=up_block_types,
        )


class ADMUNet(UNet2DModel):
    """This is the model used in the ADM paper. We should run some ablations using this class.
    Stuff we should try ablating:
    - layers_per_block: this is the "depth" mentioned in the paper. We can try increasing it to 4.
    - channel width: the paper uses 160, so we can change block_out_channels to (160, 160, 320, 320, 640, 640)
    - fix channels-per-head, vary # heads: this is table 2 in the paper (this class fixes it to 64). We can try 32 and 128.
    - fix # heads, vary channels-per-head: this is also table 2 in the paper. (this requires us to do something like channel_dim // num_heads), with num_heads being [1, 2, 4, 8].
    - remove the attention resolution at 32 and 64: this is the "multi-res attention" ablation in the paper.
    - change the "upsample" and "downsample" attention from "resnet" to "default".
    - using a "wide" unet by changing the channels to [160, 160, 320, 320, 640, 640].
    """

    def __init__(self, config):
        super().__init__(
            sample_size=config.image_size,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            attention_head_dim=64,  # this gives varying attention heads for each layer.
            downsample_type="resnet",  # This gives BigGAN-style residual samplers.
            upsample_type="resnet",  # same as the above.
            resnet_time_scale_shift="scale_shift",  # This is the AdaGN portion.
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",  # 128 -> 64
                "AttnDownBlock2D",  # 64 -> 32  (2 attention heads)
                "AttnDownBlock2D",  # 32 -> 16 (4 attention heads)
                "AttnDownBlock2D",  # 16 -> 8 (8 attention heads)
                "DownBlock2D",  # 8 -> 4
                "DownBlock2D",  # 4 -> 2
            ),
            up_block_types=(
                "UpBlock2D",  # 2 -> 4
                "AttnUpBlock2D",  # 4 -> 8 (8 attention heads)
                "AttnUpBlock2D",  # 8 -> 16 (4 attention heads)
                "AttnUpBlock2D",  # 16 -> 32 (2 attention heads)
                "UpBlock2D",  # 32 -> 64
                "UpBlock2D",  # 64 -> 128
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
    model = cls(config)
    n_heads = config.fixed_heads
    if n_heads > 0:
        for blk, ch in zip(model.down_blocks, model.config.block_out_channels):
            if isinstance(blk, UNet2DModel.AttnDownBlock2D):
                blk.attn1.num_attention_heads = n_heads
                blk.attn1.attention_head_dim = ch // n_heads
        for blk, ch in zip(model.up_blocks, reversed(model.config.block_out_channels)):
            if isinstance(blk, UNet2DModel.AttnUpBlock2D):
                blk.attn1.num_attention_heads = n_heads
                blk.attn1.attention_head_dim = ch // n_heads

    return model
