# Standard library imports
import glob

# Image handling
from PIL import Image

# Deep learning framework
import torch
from torchvision import transforms

# Hugging Face
from datasets import load_dataset
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup

# Configuration
from conf.training_config import ButterflyConfig
from pipelines.ddpm import train_loop

from models.unet import create_unet

config = ButterflyConfig()
dataset = load_dataset(config.dataset_name, split="train")


preprocess = transforms.Compose(
    [
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}


dataset.set_transform(transform)
train_dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=config.train_batch_size, shuffle=True
)

model = create_unet(config)
sample_image = dataset[0]["images"].unsqueeze(0)
assert sample_image.shape == model(sample_image, timestep=0).sample.shape

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)

args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)

train_loop(*args)

sample_images = sorted(glob.glob(f"{config.output_dir}/samples/*.png"))
Image.open(sample_images[-1])