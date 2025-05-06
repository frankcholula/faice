import os
import glob
import warnings

import torch
import csv

from PIL import Image
from args import get_config_and_components
from models.unet import ClassConditionedUNet
from diffusers.optimization import get_cosine_schedule_with_warmup
from conf.training_config import FaceConfig, ButterflyConfig
from utils.transforms import build_transforms
from utils.loggers import timer

# Suppress all FutureWarnings from the 'diffusers' module
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")


def setup_dataset(config):
    transform_train, transform_test = build_transforms(config)

    if isinstance(config, FaceConfig):
        from torch.utils.data import Dataset, DataLoader

        class CelebaAHQDataset(Dataset):
            def __init__(self, root_dir, transform=None):
                self.root_dir = root_dir
                self.transform = transform
                self.image_paths = glob.glob(os.path.join(root_dir, "*.jpg"))
                self.labels_path = os.path.join(root_dir, "labels.csv")
                self.labels_map = self._load_labels()

            def _load_labels(self):
                labels_map = {}
                with open(self.labels_path, newline="") as f:
                    reader = csv.reader(f)
                    header = next(reader, None)  # skip header
                    for img, label in reader:
                        labels_map[img] = int(label)
                return labels_map

            def __len__(self):
                return len(self.image_paths)

            def __getitem__(self, idx):
                img_path = self.image_paths[idx]
                image = Image.open(img_path).convert("RGB")
                image_name = os.path.splitext(os.path.basename(img_path))[0]
                label = self.labels_map.get(image_name, None)
                if self.transform:
                    image = self.transform(image)
                return {"images": image, "image_names": image_name, "labels": label}

        train_dataset = CelebaAHQDataset(
            root_dir=config.train_dir, transform=transform_train
        )
        train_dataloader = DataLoader(
            train_dataset, batch_size=config.train_batch_size, shuffle=True
        )
        test_dataset = CelebaAHQDataset(
            root_dir=config.test_dir, transform=transform_test
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=config.eval_batch_size, shuffle=False
        )

    elif isinstance(config, ButterflyConfig):
        from datasets import load_dataset

        dataset = load_dataset(config.dataset_name, split="train")

        def transform(examples):
            images = [
                transform_train(image.convert("RGB")) for image in examples["image"]
            ]
            return {"images": images}

        dataset.set_transform(transform)
        train_dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=config.train_batch_size, shuffle=True
        )
        test_dataloader = None

    return train_dataloader, test_dataloader


def main():
    config, model, noise_scheduler, train_loop = get_config_and_components()
    train_dataloader, test_dataloader = setup_dataset(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )

    with torch.no_grad():
        perform_model_sanity_check(model, train_dataloader)

    train_args = (
        config,
        model,
        noise_scheduler,
        optimizer,
        train_dataloader,
        lr_scheduler,
        test_dataloader,
    )

    train_loop(*train_args)
    # Print summary
    print(f"Training completed! Model saved to {config.output_dir}")
    try:
        sample_images = sorted(glob.glob(f"{config.output_dir}/samples/*.png"))
        print(f"Generated {len(sample_images)} sample images")
    except Exception as e:
        print(f"Error retrieving sample images: {e}")


def perform_model_sanity_check(model, train_dataloader):
    try:
        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Using model: {model.__class__.__name__}")
        print(f"Using device: {device}")

        # Move model to the device first
        model.to(device)

        sample_batch = next(iter(train_dataloader))
        sample_image = sample_batch["images"].to(device)
        sample_labels = sample_batch["labels"].to(device)
        print(f"Sample image shape: {sample_image.shape}")
        print(f"Sample label shape: {sample_labels.shape}")

        if len(sample_image.shape) == 3:  # Add batch dimension if missing
            sample_image = sample_image.unsqueeze(0)

            # Create timestep tensor on the same device
        timestep = torch.tensor([0], device=device)

        # Just check if the model runs
        if isinstance(model, ClassConditionedUNet):
            encoder_hidden_states = torch.zeros(
                sample_image.shape[0],
                1,
                model.config.cross_attention_dim,
                device=device,
            )
            _ = model(
                sample_image,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                class_labels=sample_labels,
            )

        else:
            _ = model(sample_image, timestep=timestep)
        print(f"Model sanity check passed on {device}!")
    except Exception as e:
        print(f"Model sanity check failed!")
        # raise e


if __name__ == "__main__":
    with timer("total training time"):
        main()
