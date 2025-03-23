# -*- coding: UTF-8 -*-
"""
@Time : 09/03/2025 11:06
@Author : xiaoguangliang
@File : preprocess_data.py
@Project : faice
"""
import os

from PIL import Image
import numpy as np
import torch.utils.data as data
from torchvision import transforms
import matplotlib.pyplot as plt
from datasets import load_dataset

from codes.conf.global_setting import BASE_DIR
from codes.conf.model_config import config


def transform(examples):
    image_transform = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            # transforms.CenterCrop((config.image_size, config.image_size)),
            transforms.RandomAdjustSharpness(sharpness_factor=0.5),
            transforms.RandomHorizontalFlip(),
            # transforms.GaussianBlur(kernel_size=3), # Not good
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    images = [image_transform(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}


def get_data(data_path, do_trans=True):
    # Load local image data
    dataset = load_dataset("imagefolder", data_dir=data_path, split="train")

    # image datasets
    if do_trans:
        dataset.set_transform(transform)

    return dataset


def inspect_data(data):
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    for i, image in enumerate(data[:4]["image"]):
        # print('image size:', image.size)
        axs[i].imshow(image)
        axs[i].set_axis_off()
        if i == 3:
            break
    fig.show()


if __name__ == '__main__':
    data_path = BASE_DIR + "/data/celeba_hq_256/"
    dataset = get_data(data_path, False)
    inspect_data(dataset)
