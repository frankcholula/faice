# -*- coding: UTF-8 -*-
"""
@Time : 01/05/2025 21:49
@Author : xiaoguangliang
@File : unet_pp.py
@Project : code
"""
import segmentation_models_pytorch as smp


def unet_pp(config):
    """U-Net++

    Args:
        encoder_name (_type_): _description_
        encoder_weights (_type_): _description_
        classes (_type_): _description_
    """
    return smp.UnetPlusPlus(
        classes=2,
        in_channels=3,
        encoder_depth=5
    )
