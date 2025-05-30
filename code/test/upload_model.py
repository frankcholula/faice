# -*- coding: UTF-8 -*-
"""
@Time : 28/05/2025 10:20
@Author : xiaoguangliang
@File : upload_model.py
@Project : code
"""
import os
from huggingface_hub import HfApi, create_repo

api = HfApi()


def upload_model(repo_id, local_dir, token):
    """
    Upload a model to the Hugging Face Hub.

    Args:
        repo_id (str): The repository ID of the model.
        local_dir (str): The local directory containing the model files.
        token (str): The authentication token for the Hugging Face Hub.

    Returns:
        None
    """
    api.upload_folder(
        repo_id=repo_id,
        folder_path=local_dir,
        token=token,
    )


class Config():
    # output_dir = "runs/unet_cond_l_block_4-stable_diffusion-pndm-face_dialog-200"
    # output_dir = "runs/class_guidance_model"
    output_dir = "runs/unconditional_diffusion_unet_model"


config = Config()
token = os.environ.get("HUGGINGFACE_TOKEN")

# repo_id = "Ngene787/Faice_text2face"
# repo_id = "Ngene787/Faice_class_guidance"
repo_id = "Ngene787/Faice_unconditional_diffusion"

repo_id = create_repo(repo_id).repo_id
upload_model(repo_id=repo_id, local_dir=config.output_dir, token=token)
