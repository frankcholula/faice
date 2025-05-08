from typing import Tuple

import torchvision.transforms as T
from PIL import Image
import numpy as np


def build_transforms(config):
    # build train transformations
    transform_train = [
        T.Resize((config.image_size, config.image_size)),
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),
    ]

    transform_test = [
        T.Resize((config.image_size, config.image_size)),
        T.ToTensor(),
    ]

    if config.RHFlip:
        transform_train += [T.RandomHorizontalFlip()]
    if config.gblur:
        transform_train += [T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))]
    if config.center_crop_arr:
        transform_train += [T.CenterCrop((config.image_size, config.image_size))]

    transform_train = T.Compose(transform_train)
    transform_test = T.Compose(transform_test)

    return transform_train, transform_test
