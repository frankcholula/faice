# -*- coding: UTF-8 -*-
"""
@Time : 11/05/2025 15:33
@Author : xiaoguangliang
@File : test_stable_diffusion.py
@Project : code
"""
import torch
from diffusers import StableDiffusionPipeline
from utils.metrics import evaluate
from pipelines.stable_diffusion import load_request_prompt

stable_diffusion_request_prompt_dir: str = (
    "datasets/celeba_hq_stable_diffusion/request_hq.txt"
)

test_prompts = load_request_prompt(stable_diffusion_request_prompt_dir)
model_path = 'runs/unet_cond_l_block_4-stable_diffusion-pndm-face_dialog-200'

pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe = pipe.to("cuda")


class Config():
    use_wandb = False
    num_inference_steps = 999
    seed = 0
    output_dir = "runs/stable_diffusion_inference"


config = Config()
evaluate(config, 1, pipe, prompt=test_prompts)

