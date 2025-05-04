from diffusers import UNet2DModel


class BaseUNet(object):
    def __init__(self, config, compress_rate=1, attention_head_dim=8, layers_per_block=2, block_num=6):
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

            **self.single_attention_block()

        )
        return model

    def unet_l(self):
        model = UNet2DModel(
            sample_size=self.sample_size,  # the target image resolution
            in_channels=3,  # the number of input channels, 3 for RGB images
            out_channels=3,  # the number of output channels
            attention_head_dim=self.attention_head_dim,
            layers_per_block=self.layers_per_block,  # how many ResNet layers to use per UNet block

            **self.multi_attention_block()

        )
        return model

    def single_attention_block(self):
        block_out_channels = [128, 128, 256, 256, 512, 512]
        down_block_types = ["DownBlock2D",
                            "DownBlock2D",
                            "DownBlock2D",
                            "DownBlock2D",
                            "AttnDownBlock2D",
                            "DownBlock2D"]
        up_block_types = ["UpBlock2D",
                          "AttnUpBlock2D",
                          "UpBlock2D",
                          "UpBlock2D",
                          "UpBlock2D",
                          "UpBlock2D"]
        if self.block_num == 6:
            block_out_channels = block_out_channels
            down_block_types = down_block_types
            up_block_types = up_block_types
        elif self.block_num == 8:
            block_out_channels = block_out_channels + [1024] * 2
            down_block_types = ["DownBlock2D"] * 2 + down_block_types
            up_block_types = up_block_types + ["UpBlock2D"] * 2
        blocks = {"block_out_channels": tuple(block_out_channels),
                  "down_block_types": tuple(down_block_types),
                  "up_block_types": tuple(up_block_types)}
        return blocks

    def multi_attention_block(self):
        block_out_channels = [224, 448, 672, 896]
        down_block_types = ["DownBlock2D",
                            "AttnDownBlock2D",
                            "AttnDownBlock2D",
                            "AttnDownBlock2D"]
        up_block_types = ["AttnUpBlock2D",
                          "AttnUpBlock2D",
                          "AttnUpBlock2D",
                          "UpBlock2D"]
        if self.block_num == 4:
            block_out_channels = block_out_channels
            down_block_types = down_block_types
            up_block_types = up_block_types
        elif self.block_num == 5:
            block_out_channels = block_out_channels + [1120]
            down_block_types = down_block_types + ['AttnDownBlock2D']
            up_block_types = ['AttnUpBlock2D'] + up_block_types
        elif self.block_num == 6:
            block_out_channels = block_out_channels + [1120, 1344]
            down_block_types = down_block_types + ['AttnDownBlock2D'] * 2
            up_block_types = ['AttnUpBlock2D'] * 2 + up_block_types
        blocks = {"block_out_channels": tuple(block_out_channels),
                  "down_block_types": tuple(down_block_types),
                  "up_block_types": tuple(up_block_types)}
        return blocks


def unet_b_block_6(config):
    return BaseUNet(config).unet_b()


def unet_b_block_8(config):
    return BaseUNet(config, block_num=8).unet_b()


def unet_b_block_6_head_dim_64(config):
    return BaseUNet(config, block_num=6, attention_head_dim=64).unet_b()


def unet_b_block_8_head_dim_64(config):
    return BaseUNet(config, block_num=8, attention_head_dim=64).unet_b()


def unet_b_block_8_head_dim_64_layer_4(config):
    return BaseUNet(config, block_num=8, attention_head_dim=64, layers_per_block=4).unet_b()


def unet_l_block_4(config):
    return BaseUNet(config, block_num=4).unet_l()


def unet_l_block_4_head_dim_64(config):
    return BaseUNet(config, block_num=4, attention_head_dim=64).unet_l()


def unet_l_block_4_head_dim_64_layer_4(config):
    return BaseUNet(config, block_num=4, attention_head_dim=64, layers_per_block=4).unet_l()


def unet_l_block_5(config):
    return BaseUNet(config, block_num=5).unet_l()


def unet_l_block_5_head_dim_64(config):
    return BaseUNet(config, block_num=5, attention_head_dim=64).unet_l()


def unet_l_block_5_head_dim_64_layer_3(config):
    return BaseUNet(config, block_num=5, attention_head_dim=64, layers_per_block=3).unet_l()


def unet_l_block_5_head_dim_64_layer_4(config):
    return BaseUNet(config, block_num=5, attention_head_dim=64, layers_per_block=4).unet_l()


def unet_l_block_6(config):
    return BaseUNet(config, block_num=6).unet_l()


def unet_l_block_6_head_dim_64(config):
    return BaseUNet(config, block_num=6, attention_head_dim=64).unet_l()


def unet_l_block_6_head_dim_64_layer_4(config):
    return BaseUNet(config, block_num=6, attention_head_dim=64, layers_per_block=4).unet_l()


if __name__ == "__main__":
    class Config:
        image_size = 128


    conf = Config()
    model = BaseUNet(conf).unet_b()
    # print(model)

    print(**BaseUNet(conf).multi_attention_block())
