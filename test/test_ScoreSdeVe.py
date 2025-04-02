# -*- coding: UTF-8 -*-
"""
@Time : 26/03/2025 22:05
@Author : xiaoguangliang
@File : test_ScoreSdeVe.py
@Project : faice
"""
from diffusers import ScoreSdeVePipeline, ScoreSdeVeScheduler
import torch
import PIL.Image
import numpy as np

from conf import BASE_DIR


def test_ScoreSdeVe(model_path):
    pipe = ScoreSdeVePipeline.from_pretrained(model_path)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    pipe.scheduler = ScoreSdeVeScheduler.from_config(
        pipe.scheduler.config,
        num_train_timesteps=1000,
        snr=0.15,
        sigma_min=0.001,
        sigma_max=2348.0,
        sampling_eps=1e-3,
        correct_steps=3,
    )

    # Define parameters
    batch_size = 1  # Number of images to generate
    num_inference_steps = 5  # More steps = better quality (but slower)

    # Generate an image
    images = pipe(
        batch_size=batch_size,
        num_inference_steps=num_inference_steps,
        generator=torch.manual_seed(42),  # For reproducibility
        output_type="np"
    ).images

    # Save the image
    # image.save("generated_image.png")
    images_uint8 = (images * 255).astype(np.uint8)
    # images_uint8 = ((images + 1.0) * 127.5).astype(np.uint8)

    for j, image in enumerate(images_uint8):
        image = PIL.Image.fromarray(image)
        image.save(f"test{j}.png")


if __name__ == '__main__':
    m_path = BASE_DIR + '/output/Training_log_splited_dataset/old/ScoreSdeVe'
    test_ScoreSdeVe(m_path)
