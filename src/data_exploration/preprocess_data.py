# -*- coding: UTF-8 -*-
"""
@Time : 09/03/2025 11:06
@Author : xiaoguangliang
@File : preprocess_data.py
@Project : faice
"""

import cv2
import os

import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from datasets import load_dataset
from PIL import Image
from torchvision.utils import save_image

from conf.global_setting import BASE_DIR
from conf.model_config import model_config as config


def transform(examples):
    image_transform = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            # transforms.CenterCrop((config.image_size, config.image_size)),
            # transforms.RandomAdjustSharpness(sharpness_factor=0.5),
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

    # image data
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


def change_image_size(img_path, save_path):
    dataset = load_dataset("imagefolder", data_dir=img_path, split="train")
    # Convert all the image size into image_size*image_size
    for i, image in enumerate(dataset):
        image = image["image"]
        transform = transforms.ToTensor()
        image = transform(image)
        transform = transforms.Resize((config.image_size, config.image_size))
        resized_image = transform(image)
        # resized_image = resized_image.permute(1, 2, 0)
        # resized_image = resized_image.numpy()
        # images_uint8 = (resized_image * 255).astype(np.uint8)
        # images_uint8 = Image.fromarray(images_uint8)
        # images_uint8.save(save_path + "/" + str(i) + ".jpg")
        save_image(
            resized_image,
            os.path.join(save_path + "/" + str(i) + ".jpg"),
        )


def resize_image_with_opencv(image_path, new_size=(128, 128)):
    # Read the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found or unable to read: {image_path}")

    # Resize the image
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    return resized_image


def change_image_size_opencv(img_path, save_path, new_size=(128, 128)):
    dataset = load_dataset("imagefolder", data_dir=img_path, split="train")
    # Ensure the save directory exists
    os.makedirs(save_path, exist_ok=True)

    # Convert all the image size into new_size*new_size
    for i, image_dict in enumerate(dataset):
        image_path = image_dict["image"].filename
        save_image_path = os.path.join(save_path, f"{i}.jpg")
        resized_image = resize_image_with_opencv(image_path, new_size)
        # Save the resized image
        cv2.imwrite(save_image_path, resized_image)


if __name__ == "__main__":
    # data_path = BASE_DIR + "/data/celeba_hq_256/"
    # dataset = get_data(data_path, False)
    # inspect_data(dataset)

    test_dir = BASE_DIR + "/data/celeba_hq_split/test30"
    save_dir = BASE_DIR + "/data/celeba_hq_split/test30_128"
    change_image_size(test_dir, save_dir)
    # change_image_size_opencv(test_dir, save_dir)
