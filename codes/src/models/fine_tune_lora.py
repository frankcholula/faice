# -*- coding: UTF-8 -*-
"""
@Time : 15/03/2025 08:56
@Author : xiaoguangliang
@File : fine_tune_lora.py
@Project : faice
"""
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe.load_lora_weights("sassad/face-lora")


prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipe(prompt).images[0]