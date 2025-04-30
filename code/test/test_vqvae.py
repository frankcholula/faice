# -*- coding: UTF-8 -*-
"""
@Time : 30/04/2025 23:06
@Author : xiaoguangliang
@File : test_vqvae.py
@Project : code
"""
import os
import torch
from models.vqmodel import create_vqmodel

from diffusers.utils.pil_utils import numpy_to_pil
from utils.metrics import make_grid

model_path = "runs/vqvae-vqvae-ddpm-face-50/checkpoints/model_vqvae.pth"


class config():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    image_size = 128


def vae_inference():
    checkpoint = torch.load(model_path)
    vqvae = create_vqmodel(config)
    vqvae = vqvae.to(config.device)
    vqvae.load_state_dict(checkpoint['model_state_dict'])

    noise = torch.randn(16, 32, 32, 32).to(config.device)
    decoded = vqvae.decode(noise)[0]

    generated_images = (decoded / 2 + 0.5).clamp(0, 1)

    generated_images = generated_images.cpu().permute(0, 2, 3, 1).detach()
    generated_images = generated_images.numpy()
    generated_images = numpy_to_pil(generated_images)
    # generated_images = generated_images.permute(0, 3, 2, 1)

    image_grid = make_grid(generated_images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join('runs', "vae_samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid_path = f"{test_dir}/vqvae000.png"
    image_grid.save(image_grid_path)


if __name__ == '__main__':
    vae_inference()
