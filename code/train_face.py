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
from diffusers import UNet2DModel, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup

# Local configuration
from diffusion_pipeline import FaceConfig, train_loop


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
train_dataloader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)

# evaluation
test_dataset = CelebaAHQDataset(root_dir=config.test_dir, transform=preprocess)
test_dataloader = DataLoader(test_dataset, batch_size=config.eval_batch_size, shuffle=False)

def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}


model = UNet2DModel(
    sample_size=config.image_size,  # the target image resolution
    in_channels=3,  # the number of input channels, 3 for RGB images
    out_channels=3,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(
        128,
        128,
        256,
        256,
        512,
        512,
    ),
    down_block_types=(
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)
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

args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler, test_dataloader)

train_loop(*args)

sample_images = sorted(glob.glob(f"{config.output_dir}/samples/*.png"))
Image.open(sample_images[-1])
