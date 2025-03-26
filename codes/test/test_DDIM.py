# -*- coding: UTF-8 -*-
"""
@Time : 26/03/2025 16:38
@Author : xiaoguangliang
@File : test_DDIM.py
@Project : faice
"""
import torch
from diffusers import DDIMPipeline
import PIL.Image
import numpy as np

from codes.conf.global_setting import BASE_DIR


def test_ddim(model_path):
    pipe = DDIMPipeline.from_pretrained(model_path)

    # image = pipe(eta=0.0, num_inference_steps=50)

    images = pipe(eta=0.5, num_inference_steps=10, output_type="np", batch_size=1).images

    # process image to PIL
    # fake_images = torch.tensor(images)
    # fake_images = fake_images.permute(0, 2, 3, 1)
    # # image_processed = image.cpu().permute(0, 2, 3, 1)
    # image_processed = (fake_images + 1.0) * 127.5
    # image_processed = image_processed.numpy()
    # image_processed = image_processed.astype(np.uint8)
    # image_pil = PIL.Image.fromarray(image_processed)
    #
    # # save image
    # image_pil.save("test.png")

    images_uint8 = (images * 255).astype(np.uint8)
    # images_uint8 = ((images + 1.0) * 127.5).astype(np.uint8)

    for j, image in enumerate(images_uint8):
        image = PIL.Image.fromarray(image)
        image.save(f"test{j}.png")


if __name__ == '__main__':
    m_path = BASE_DIR + '/output/Training_log_splited_dataset/old/DDIM'
    test_ddim(m_path)
