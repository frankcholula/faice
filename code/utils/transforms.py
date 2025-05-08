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
        transform_train += [T.Lambda(lambda pil_image: center_crop_arr(pil_image, config.image_size))]
        # transform_train += [T.CenterCrop((config.image_size, config.image_size))]

    transform_train = T.Compose(transform_train)
    transform_test = T.Compose(transform_test)

    return transform_train, transform_test


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    print('pil_image.size', pil_image.size)
    print('pil_image', pil_image)
    print('pil_image type', type(pil_image))
    if min(pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])
