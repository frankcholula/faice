# -*- coding: UTF-8 -*-
"""
@Time : 14/03/2025 19:13
@Author : xiaoguangliang
@File : FID_score.py
@Project : faice
"""
import os

import torch
from diffusers import DDPMPipeline
from diffusers import DDPMScheduler
from PIL import Image
import numpy as np
from torchvision.transforms import functional as F
from torchmetrics.image.fid import FrechetInceptionDistance
from loguru import logger

from codes.conf.global_setting import BASE_DIR, config


def preprocess_image(image):
    image = torch.tensor(image).unsqueeze(0)
    image = image.permute(0, 3, 1, 2) / 255.0
    return F.center_crop(image, (256, 256))


def make_real_images(dataset_path):
    logger.info(f"Loading real images from {dataset_path}")
    image_paths = sorted([os.path.join(dataset_path, x) for x in os.listdir(dataset_path)])

    real_images = [np.array(Image.open(path).convert("RGB")) for path in image_paths]

    real_images = torch.cat([preprocess_image(image) for image in real_images])
    print(real_images.shape)
    return real_images


def make_fake_images(model_ckpt, scheduler_path):
    # Load model
    logger.info(f"Loading model from {model_ckpt}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    scheduler = DDPMScheduler.from_pretrained(scheduler_path, subfolder="scheduler")

    pipeline = DDPMPipeline.from_pretrained(
        model_ckpt,
        scheduler=scheduler,
        torch_dtype=torch.float16,
        # variant="fp16",
        use_safetensors=True
    ).to(device)

    logger.info("Generate fake images")
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.manual_seed(config.seed),
    ).images

    # fake_images = torch.tensor(images)
    fake_images = images.permute(0, 3, 1, 2)
    return fake_images


def calculate_fid(dataset_path, model_ckpt, scheduler_path):
    logger.info("Calculate FID score")
    real_images = make_real_images(dataset_path)
    fake_images = make_fake_images(model_ckpt, scheduler_path)
    fid = FrechetInceptionDistance(normalize=True)
    fid.update(real_images, real=True)
    fid.update(fake_images, real=False)

    print(f"FID: {float(fid.compute())}")


if __name__ == '__main__':
    dataset_dir = BASE_DIR + '/data/celeba_hq_256'
    model_ckpt_dir = BASE_DIR + '/output/celeba_hq_256_training/'
    scheduler_dir = BASE_DIR + '/output/celeba_hq_256_training/scheduler/'
    test_data = BASE_DIR + '/data/test'

    calculate_fid(dataset_dir, model_ckpt_dir, scheduler_dir)
