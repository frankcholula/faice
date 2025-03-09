# -*- coding: UTF-8 -*-
"""
@Time : 09/03/2025 11:01
@Author : xiaoguangliang
@File : __init__.py.py
@Project : faice
"""
# from datasets import load_dataset
# from codes.conf.global_setting import BASE_DIR, config
# from torchvision import transforms
#
# preprocess = transforms.Compose(
#     [
#         transforms.Resize((config.image_size, config.image_size)),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5], [0.5]),
#     ]
# )
#
#
# def transform(examples):
#     images = [preprocess(image.convert("RGB")) for image in examples["image"]]
#     return {"images": images}
#
#
# config.dataset_name = "huggan/smithsonian_butterflies_subset"
# dataset = load_dataset(config.dataset_name, split="train")
# dataset.set_transform(transform)
#
# sample_image = dataset[0]["images"].unsqueeze(0)
# print("Input shape:", sample_image.shape)
