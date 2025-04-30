# -*- coding: UTF-8 -*-
"""
@Time : 30/04/2025 22:35
@Author : xiaoguangliang
@File : test_vae.py
@Project : code
"""
import os
import torch
from models.vae import create_vae

from diffusers.utils.pil_utils import numpy_to_pil
from utils.metrics import make_grid

model_path = "runs/vae-vae-ddpm-face-500/checkpoints/model_vae.pth"


class config():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    image_size = 128


def vae_inference():
    checkpoint = torch.load(model_path)
    vae = create_vae(config)
    vae = vae.to(config.device)
    vae.load_state_dict(checkpoint['model_state_dict'])

    noise = torch.randn(16, 16, 16, 16).to(config.device)
    decoded = vae.decode(noise)[0]

    generated_images = (decoded / 2 + 0.5).clamp(0, 1)

    to_generate_images = generated_images.cpu().permute(0, 2, 3, 1).numpy()
    to_generate_images = numpy_to_pil(to_generate_images)
    # generated_images = generated_images.permute(0, 3, 2, 1)

    image_grid = make_grid(to_generate_images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join('runs', "vae_samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid_path = f"{test_dir}/000.png"
    image_grid.save(image_grid_path)


if __name__ == '__main__':
    vae_inference()