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

from codes.conf.global_setting import BASE_DIR, config


class FaceData(data.Dataset):
    def __init__(self, root, files, transforms=None):
        # location of the dataset
        self.root = root
        # list of files
        self.files = files
        # transforms
        self.transforms = transforms

    def __getitem__(self, item):
        # read the image
        image = Image.open(os.path.join(self.root, self.files[item])).convert(mode="RGB")
        # class for that image
        # apply transformation
        if self.transforms:
            image = self.transforms(image)
            image = image.reshape(1, 3, config.image_size, config.image_size)
        # return the image and class
        return image

    def __len__(self):
        # return the total number of images
        return len(self.files)


def get_data(root, do_trans=True):
    # list of files
    files = os.listdir(root)

    image_transform = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    # image datasets
    if do_trans:
        dataset = FaceData(root, files, transforms=image_transform)
    else:
        dataset = FaceData(root, files)

    return dataset


def inspect_data(data):
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    for i, image in enumerate(data):
        axs[i].imshow(image)
        axs[i].set_axis_off()
        if i == 3:
            break
    fig.show()


if __name__ == '__main__':
    data_path = BASE_DIR + "/data/celeba_hq_256/"
    dataset = get_data(data_path)
    image_sample = dataset[0]
    print(image_sample.shape)
