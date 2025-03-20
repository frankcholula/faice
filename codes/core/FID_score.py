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

from codes.conf.global_setting import BASE_DIR
from codes.conf.model_config import config


def preprocess_image(image):
    image = torch.tensor(image).unsqueeze(0)
    image = image.permute(0, 3, 1, 2) / 255.0
    return F.center_crop(image, (256, 256))


def make_fid_input_images(images_path):
    logger.info(f"Loading real images from {images_path}")
    image_paths = sorted([os.path.join(images_path, x) for x in os.listdir(images_path)])

    real_images = [np.array(Image.open(path).convert("RGB")) for path in image_paths]

    real_images = torch.cat([preprocess_image(image) for image in real_images])
    logger.info(f"real images shape: {real_images.shape}")
    return real_images


def generate_images_from_model(model_ckpt, scheduler_path, device):
    # Load model
    logger.info(f"Loading model from {model_ckpt}")

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
        output_type="np"
    ).images

    fake_images = torch.tensor(images)
    fake_images = fake_images.permute(0, 3, 1, 2)
    return fake_images


def calculate_fid(dataset_path, model_ckpt, scheduler_path):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # if torch.backends.mps.is_available():
    #     device = torch.device("mps")
    logger.info("Calculate FID score")
    real_images = make_fid_input_images(dataset_path)
    # real_images = real_images.to(device)
    fake_images = generate_images_from_model(model_ckpt, scheduler_path, device)
    # fid = FrechetInceptionDistance(normalize=True).to(device)
    fid = FrechetInceptionDistance(normalize=True)
    fid.update(real_images, real=True)
    fid.update(fake_images, real=False)

    fid_score = round(float(fid.compute()), 3)

    print(f"FID: {fid_score}")


if __name__ == '__main__':
    dataset_dir = BASE_DIR + '/data/celeba_hq_256'
    # model_ckpt_dir = BASE_DIR + '/output/celeba_hq_256_training/'
    # scheduler_dir = BASE_DIR + '/output/celeba_hq_256_training/scheduler/'
    # test_data = BASE_DIR + '/data/test'
    model_ckpt_dir = BASE_DIR + '/output/celeba_hq_256_training_7_3/'
    scheduler_dir = BASE_DIR + '/output/celeba_hq_256_training_7_3/scheduler/'

    calculate_fid(dataset_dir, model_ckpt_dir, scheduler_dir)
