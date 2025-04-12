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
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance

from conf.log_conf import logger
from conf.global_setting import BASE_DIR
from conf.model_config import model_config
from src.data_exploration.preprocess_data import resize_image_with_opencv


def resize_image(image, size=(128, 128)):
    transform = transforms.Resize(size)
    resized_image = transform(image)
    return resized_image


def preprocess_image(image):
    # image = resize_image_with_opencv(image, new_size=(model_config.image_size, model_config.image_size))
    image = torch.tensor(image).unsqueeze(0)
    image = image.permute(0, 3, 1, 2) / 255.0

    # Resize the image to (model_config.image_size, model_config.image_size)
    image = resize_image(image, (model_config.image_size, model_config.image_size))
    # return F.center_crop(image, (model_config.image_size, model_config.image_size))
    return image


def make_fid_input_images(images_path):
    logger.info(f"Loading real images from {images_path}")
    image_paths = sorted(
        [os.path.join(images_path, x) for x in os.listdir(images_path)]
    )

    real_images = [np.array(Image.open(path).convert("RGB")) for path in image_paths]
    real_images = torch.cat([preprocess_image(image) for image in real_images])

    # real_images = torch.cat([preprocess_image(image) for image in image_paths])

    logger.info(f"real images shape: {real_images.shape}")
    return real_images


def generate_images_from_model(
    model_ckpt, scheduler_path, device, num_images=model_config.num_images
):
    # Load model
    logger.info(f"Loading model from {model_ckpt}")

    scheduler = DDPMScheduler.from_pretrained(scheduler_path, subfolder="scheduler")

    pipeline = DDPMPipeline.from_pretrained(
        model_ckpt,
        scheduler=scheduler,
        torch_dtype=torch.float16,
        # variant="fp16",
        use_safetensors=True,
    ).to(device)

    logger.info("Generate fake images")

    batch_size = model_config.eval_batch_size
    num_batches = (num_images + batch_size - 1) // batch_size  # Ceiling division

    all_fake_images = []

    for i in range(num_batches):
        if i == num_batches - 1:
            batch_size = num_images - i * batch_size
        batch_seed = (
            model_config.seed + i
        )  # Use a different seed for each batch to ensure diversity
        images = pipeline(
            batch_size=batch_size,
            generator=torch.manual_seed(batch_seed),
            output_type="np",
            num_inference_steps=1000,
        ).images

        fake_images = torch.tensor(images)
        fake_images = fake_images.permute(0, 3, 1, 2)
        all_fake_images.append(fake_images)

    # Concatenate all batches into a single tensor
    # fake_images = torch.cat(all_fake_images)[:num_images]  # Ensure exactly 300 images
    fake_images = torch.cat(all_fake_images)

    logger.info(f"Generated fake images shape: {fake_images.shape}")
    return fake_images


def calculate_fid(real_images, fake_images):
    fid = FrechetInceptionDistance(normalize=True)
    fid.update(real_images, real=True)
    fid.update(fake_images, real=False)

    fid_score = round(float(fid.compute()), 3)

    logger.info(f"FID score: {fid_score}")


def test_calculate_fid(dataset_path, model_ckpt, scheduler_path, fake_image_dir=None):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    logger.info("Calculate FID score")
    real_images = make_fid_input_images(dataset_path)
    # real_images = real_images.to(device)
    if fake_image_dir:
        fake_images = make_fid_input_images(fake_image_dir)
    else:
        fake_images = generate_images_from_model(model_ckpt, scheduler_path, device)
    calculate_fid(real_images, fake_images)


if __name__ == "__main__":
    dataset_dir = BASE_DIR + "/data/celeba_hq_256"
    # model_ckpt_dir = BASE_DIR + '/output/Training_log_splited_dataset/Consistency_DDPM/'
    # scheduler_dir = BASE_DIR + '/output/Training_log_splited_dataset/Consistency_DDPM/scheduler/'

    # model_ckpt_dir = BASE_DIR + '/output/celeba_hq_split_training/Consistency_DDPM/'
    # scheduler_dir = BASE_DIR + '/output/celeba_hq_split_training/Consistency_DDPM/scheduler/'

    model_ckpt_dir = (
        BASE_DIR + "/users/xl01339/aml/faice/output/ddpm-ddpm-face-500-test/"
    )
    scheduler_dir = (
        BASE_DIR + "/users/xl01339/aml/faice/output/ddpm-ddpm-face-500-test/scheduler/"
    )

    test_data = model_config.test_dir
    # fake_image_data = BASE_DIR + '/output/Training_log_splited_dataset/Consistency_DDPM/test_samples'
    fake_image_data = ""
    test_calculate_fid(
        test_data, model_ckpt_dir, scheduler_dir, fake_image_dir=fake_image_data
    )
