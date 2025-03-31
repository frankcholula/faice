# Standard library imports
import glob
import os

# Image handling
from PIL import Image

# Deep learning framework
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# Hugging Face
from diffusers import DDPMScheduler
from models.unet import create_unet
from diffusers.optimization import get_cosine_schedule_with_warmup

# Local configuration
from conf.training_config import FaceConfig
from diffusion_pipeline import train_loop


class CelebaAHQDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = glob.glob(os.path.join(root_dir, "*.jpg"))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {"images": image}


config = FaceConfig()
preprocess = transforms.Compose(
    [
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)
# training
train_dataset = CelebaAHQDataset(root_dir=config.train_dir, transform=preprocess)
train_dataloader = DataLoader(
    train_dataset, batch_size=config.train_batch_size, shuffle=True
)

# evaluation
test_dataset = CelebaAHQDataset(root_dir=config.test_dir, transform=preprocess)
test_dataloader = DataLoader(
    test_dataset, batch_size=config.eval_batch_size, shuffle=False
)


def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}


model = create_unet(config)
# some sanity checkss
sample_image = train_dataset[0]["images"].unsqueeze(0)
assert sample_image.shape == model(sample_image, timestep=0).sample.shape

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)

args = (
    config,
    model,
    noise_scheduler,
    optimizer,
    train_dataloader,
    lr_scheduler,
    test_dataloader,
)

train_loop(*args)

sample_images = sorted(glob.glob(f"{config.output_dir}/samples/*.png"))
Image.open(sample_images[-1])
