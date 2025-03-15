# -*- coding: UTF-8 -*-
"""
@Time : 09/03/2025 11:01
@Author : xiaoguangliang
@File : __init__.py.py
@Project : faice
"""
import os
import torch
from diffusers import DDPMPipeline
from diffusers import DDPMScheduler
from diffusers import AutoencoderKL



from codes.conf.global_setting import BASE_DIR, config
from codes.core.U_Net2D import make_grid

model_ckpt = BASE_DIR + '/output/celeba_hq_256_training/'
scheduler_path = BASE_DIR + '/output/celeba_hq_256_training/scheduler/'

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

scheduler = DDPMScheduler.from_pretrained(scheduler_path, subfolder="scheduler")

pipeline = DDPMPipeline.from_pretrained(
  model_ckpt,
  scheduler=scheduler,
  torch_dtype=torch.float16,
  # variant="fp16",
  use_safetensors=True
).to(device)

# pipeline = DDPMPipeline.from_pretrained(model_ckpt, torch_dtype=torch.float16, use_safetensors=True).to(device)

test_data = BASE_DIR + '/codes/data/test'

images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.manual_seed(config.seed),
    ).images

# Make a grid out of the images
image_grid = make_grid(images, rows=4, cols=4)

# Save the images
epoch = 0
test_dir = os.path.join(config.output_dir, "samples")
os.makedirs(test_dir, exist_ok=True)
image_grid.save(f"{test_dir}/{epoch:04d}.png")
