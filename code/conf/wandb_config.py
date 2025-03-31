import wandb
import os
def setup_wandb():
    wandb_entity = os.getenv("WANDB_ENTITY")
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if not wandb_api_key:
        raise ValueError("WANDB_API_KEY not found in .env file")

    if not wandb_entity:
        raise ValueError("WANDB_ENTITY not found in .env file")
    wandb.login(key=wandb_api_key)
