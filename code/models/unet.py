from diffusers import UNet2DModel, UNet2DConditionModel


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
    """This class mirrors the DDPM paper. I've tweaked it to work with 128 x 128 images.
    We should run some ablations using this class so DO ARGIFY THIS.
    Stuff we should try ablating:
    - layers_per_block: this is the "depth" mentioned in the paper. We can try increasing it to 4.
    - channel width: the paper uses 160, so we can change block_out_channels to (160, 160, 320, 320, 640, 640)
    - fix channels-per-head, vary # heads: this is table 2 in the paper (this class fixes it to 64). We can try 32 and 128.
    - fix # heads, vary channels-per-head: this is also table 2 in the paper. (this requires us to do something like channel_dim // num_heads), with num_heads being [1, 2, 4, 8].
    - remove the attention resolution at 32 and 64: this is the "multi-res attention" ablation in the paper.
    - change the "upsample" and "downsample" attention from "resnet" to "default".
    - using a "wide" unet by changing the channels to [160, 160, 320, 320, 640, 640]."""

    def __init__(self, config):
        if config.multi_res:
            # this is basically the same structure as the ADMUNet, making this for ablation purposes.
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
                "ResnetDownsampleBlock2D",  # 128 -> 64
                "ResnetDownsampleBlock2D",  # 64 -> 32
                "ResnetDownsampleBlock2D",  # 32 -> 16
                "AttnDownBlock2D",  # 16 -> 8
                "ResnetDownsampleBlock2D",  # 8 -> 4
                "ResnetDownsampleBlock2D",  # 4 -> 2
            )
            up_block_types = (
                "ResnetUpsampleBlock2D",  # 2 -> 4
                "ResnetUpsampleBlock2D",  # 4 -> 8
                "AttnUpBlock2D",  # 8 -> 16
                "ResnetUpsampleBlock2D",  # 16 -> 32
                "ResnetUpsampleBlock2D",  # 32 -> 64
                "ResnetUpsampleBlock2D",  # 64 -> 128
            )
        super().__init__(
            sample_size=config.image_size,
            in_channels=3,
            out_channels=3,
            layers_per_block=config.layers_per_block,
            attention_head_dim=config.attention_head_dim,
            # 256 for single head attention at the 16 x 16 resolution.
            time_embedding_type="positional",
            block_out_channels=tuple(
                config.base_channels * m for m in (1, 1, 2, 2, 4, 4)
            ),
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            upsample_type=config.downsample_type,
            downsample_type=config.upsample_type,
        )


class ADMUNet(UNet2DModel):
    """This is the model used in the ADM paper. DO NOT ARGIFY THIS."""

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


class ClassConditionedUNet(UNet2DConditionModel):
    """For simplicity's sake and a quick proof of concept, we can just use the standard DDPM model and add class embeddings to it."""

    def __init__(self, config):
        super().__init__(
            sample_size=config.image_size,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",  # 128 -> 64
                "AttnDownBlock2D",  # 64 -> 32
                "AttnDownBlock2D",  # 32 -> 16
                "AttnDownBlock2D",  # 16 -> 8
                "DownBlock2D",  # 8 -> 4
                "DownBlock2D",  # 4 -> 2
            ),
            up_block_types=(
                "UpBlock2D",  # 2 -> 4
                "AttnUpBlock2D",  # 4 -> 8
                "AttnUpBlock2D",  # 8 -> 16
                "AttnUpBlock2D",  # 16 -> 32
                "UpBlock2D",  # 32 -> 64
                "UpBlock2D",  # 64 -> 128
            ),
            attention_head_dim=64,
            num_class_embeds=2,  # 2 classes for male and female.
            class_embed_type=None,  # keeping this simple since we just have 0 and 1
            mid_block_type="UNetMidBlock2D",  # disable cross attention
        )


ARCHITECTURES = {
    "base": BaseUNet,
    "ddpm": DDPMUNet,
    "adm": ADMUNet,
    "cond": ClassConditionedUNet,
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
    return model


_COMPRESS_RATE = 4


# TODO: refactor to use Liang's custom implementation.
class BasicUNet(object):
    def __init__(
            self,
            config,
            compress_rate=1,
            attention_head_dim=8,
            layers_per_block=2,
            block_num=6,
    ):
        self.sample_size = int(config.image_size / compress_rate)
        self.attention_head_dim = attention_head_dim
        self.layers_per_block = layers_per_block
        self.block_num = block_num

    def unet_b(self):
        model = UNet2DModel(
            sample_size=self.sample_size,  # the target image resolution
            in_channels=3,  # the number of input channels, 3 for RGB images
            out_channels=3,  # the number of output channels
            attention_head_dim=self.attention_head_dim,
            layers_per_block=self.layers_per_block,  # how many ResNet layers to use per UNet block
            **self.single_attention_block(),
        )
        return model

    def unet_l(self):
        model = UNet2DModel(
            sample_size=self.sample_size,  # the target image resolution
            in_channels=3,  # the number of input channels, 3 for RGB images
            out_channels=3,  # the number of output channels
            attention_head_dim=self.attention_head_dim,
            layers_per_block=self.layers_per_block,  # how many ResNet layers to use per UNet block
            **self.multi_attention_block(),
        )
        return model

    def unet_xl(self):
        model = UNet2DModel(
            sample_size=self.sample_size,  # the target image resolution
            in_channels=3,  # the number of input channels, 3 for RGB images
            out_channels=3,  # the number of output channels
            attention_head_dim=self.attention_head_dim,
            layers_per_block=self.layers_per_block,  # how many ResNet layers to use per UNet block
            **self.multi_attention_block_xl(),
        )
        return model

    def single_attention_block(self):
        block_out_channels = [128, 128, 256, 256, 512, 512]
        down_block_types = [
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ]
        up_block_types = [
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ]
        if self.block_num == 6:
            block_out_channels = block_out_channels
            down_block_types = down_block_types
            up_block_types = up_block_types
        elif self.block_num == 8:
            block_out_channels = block_out_channels + [1024] * 2
            down_block_types = ["DownBlock2D"] * 2 + down_block_types
            up_block_types = up_block_types + ["UpBlock2D"] * 2
        blocks = {
            "block_out_channels": tuple(block_out_channels),
            "down_block_types": tuple(down_block_types),
            "up_block_types": tuple(up_block_types),
        }
        return blocks

    def multi_attention_block(self):
        block_out_channels = [224, 448, 672, 896]
        down_block_types = [
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ]
        up_block_types = [
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
        ]
        if self.block_num == 4:
            block_out_channels = block_out_channels
            down_block_types = down_block_types
            up_block_types = up_block_types
        elif self.block_num == 5:
            block_out_channels = block_out_channels + [1120]
            down_block_types = down_block_types + ["AttnDownBlock2D"]
            up_block_types = ["AttnUpBlock2D"] + up_block_types
        elif self.block_num == 6:
            block_out_channels = block_out_channels + [1120, 1344]
            down_block_types = down_block_types + ["AttnDownBlock2D"] * 2
            up_block_types = ["AttnUpBlock2D"] * 2 + up_block_types
        blocks = {
            "block_out_channels": tuple(block_out_channels),
            "down_block_types": tuple(down_block_types),
            "up_block_types": tuple(up_block_types),
        }
        return blocks

    def multi_attention_block_xl(self):
        block_out_channels = [256, 512, 768, 1024]
        down_block_types = [
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ]
        up_block_types = [
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
        ]
        if self.block_num == 6:
            block_out_channels = block_out_channels + [1280, 1536]
            down_block_types = down_block_types + ["AttnDownBlock2D"] * 2
            up_block_types = ["AttnUpBlock2D"] * 2 + up_block_types
        elif self.block_num == 8:
            block_out_channels = block_out_channels + [1280, 1536, 1792, 2048, 2048, 2048]
            down_block_types = down_block_types + ["AttnDownBlock2D"] * 6
            up_block_types = ["AttnUpBlock2D"] * 6 + up_block_types
        blocks = {
            "block_out_channels": tuple(block_out_channels),
            "down_block_types": tuple(down_block_types),
            "up_block_types": tuple(up_block_types),
        }
        return blocks


def unet_b_block_6(config):
    return BasicUNet(config, compress_rate=_COMPRESS_RATE).unet_b()


def unet_b_block_8(config):
    return BasicUNet(config, compress_rate=_COMPRESS_RATE, block_num=8).unet_b()


def unet_b_block_6_head_dim_64(config):
    return BasicUNet(
        config, compress_rate=_COMPRESS_RATE, block_num=6, attention_head_dim=64
    ).unet_b()


def unet_b_block_8_head_dim_64(config):
    return BasicUNet(
        config, compress_rate=_COMPRESS_RATE, block_num=8, attention_head_dim=64
    ).unet_b()


def unet_b_block_8_head_dim_64_layer_4(config):
    return BasicUNet(
        config,
        compress_rate=_COMPRESS_RATE,
        block_num=8,
        attention_head_dim=64,
        layers_per_block=4,
    ).unet_b()


def unet_l_block_4(config):
    return BasicUNet(config, compress_rate=_COMPRESS_RATE, block_num=4).unet_l()


def unet_l_block_4_head_dim_64(config):
    return BasicUNet(
        config, compress_rate=_COMPRESS_RATE, block_num=4, attention_head_dim=64
    ).unet_l()


def unet_l_block_4_head_dim_64_layer_4(config):
    return BasicUNet(
        config,
        compress_rate=_COMPRESS_RATE,
        block_num=4,
        attention_head_dim=64,
        layers_per_block=4,
    ).unet_l()


def unet_l_block_5(config):
    return BasicUNet(config, compress_rate=_COMPRESS_RATE, block_num=5).unet_l()


def unet_l_block_5_head_dim_64(config):
    return BasicUNet(
        config, compress_rate=_COMPRESS_RATE, block_num=5, attention_head_dim=64
    ).unet_l()


def unet_l_block_5_head_dim_64_layer_3(config):
    return BasicUNet(
        config,
        compress_rate=_COMPRESS_RATE,
        block_num=5,
        attention_head_dim=64,
        layers_per_block=3,
    ).unet_l()


def unet_l_block_5_head_dim_64_layer_4(config):
    return BasicUNet(
        config,
        compress_rate=_COMPRESS_RATE,
        block_num=5,
        attention_head_dim=64,
        layers_per_block=4,
    ).unet_l()


def unet_l_block_6(config):
    return BasicUNet(config, compress_rate=_COMPRESS_RATE, block_num=6).unet_l()


def unet_l_block_6_head_dim_64(config):
    return BasicUNet(
        config, compress_rate=_COMPRESS_RATE, block_num=6, attention_head_dim=64
    ).unet_l()


def unet_l_block_6_head_dim_64_layer_4(config):
    return BasicUNet(
        config,
        compress_rate=_COMPRESS_RATE,
        block_num=6,
        attention_head_dim=64,
        layers_per_block=4,
    ).unet_l()


def unet_xl_block_6(config):
    return BasicUNet(config, compress_rate=_COMPRESS_RATE, block_num=6).unet_xl()


def unet_xl_block_6_head_dim_64(config):
    return BasicUNet(
        config, compress_rate=_COMPRESS_RATE, block_num=6, attention_head_dim=64
    ).unet_xl()


def unet_xl_block_6_head_dim_64_layer_4(config):
    return BasicUNet(
        config,
        compress_rate=_COMPRESS_RATE,
        block_num=6,
        attention_head_dim=64,
        layers_per_block=4,
    ).unet_xl()


def unet_xl_block_8_head_dim_64(config):
    return BasicUNet(
        config, compress_rate=_COMPRESS_RATE, block_num=8, attention_head_dim=64
    ).unet_xl()


def unet_xl_block_8_head_dim_64_layer_4(config):
    return BasicUNet(
        config,
        compress_rate=_COMPRESS_RATE,
        block_num=8,
        attention_head_dim=64,
        layers_per_block=4,
    ).unet_xl()
