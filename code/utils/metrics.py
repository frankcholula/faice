import os
import wandb
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchvision.transforms import functional as F
from torchvision.utils import save_image
from PIL import Image
from tqdm.auto import tqdm
from huggingface_hub import whoami, HfFolder


def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.manual_seed(config.seed),
    ).images

    # Make a grid out of the images
    image_grid = make_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid_path = f"{test_dir}/{epoch:04d}.png"
    image_grid.save(image_grid_path)

    if config.use_wandb:
        wandb.log(
            {
                "generated_images": wandb.Image(
                    image_grid_path, caption=f"Epoch {epoch}"
                ),
                "epoch": epoch,
            }
        )

def calculate_inception_score(config, pipeline, test_dataloader, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

def calculate_fid_score(config, pipeline, test_dataloader, device=None, save=True):
    """Calculate FID score between generated images and test dataset"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create FID instance with normalize=True since we'll provide images in [0,1] range
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    def preprocess_image(image, real):
        # Process fake images, need to convert to BCHW but no need to rescale.
        if real:
            # print("Processing real image...")
            # print(f"  Image shape: {image.shape} and  type: {type(image)}")
            # print(f"  Image min: {image.min().item()} and max: {image.max().item()}")
            # is_bchw = len(image.shape) == 4 and image.shape[1] == 3
            # print(f"  Format: {'BCHW' if is_bchw else 'NOT BCHW'}")
            image = (image + 1.0) / 2.0
        else:
            # Process fake images, need to convert to BCHW but no need to rescale.
            # print("Processing fake image...")
            # print(f"  Image shape: {image.shape} and type: {type(image)}")
            # print(f"  Image min: {np.min(image)} and max: {np.max(image)}")
            # is_bchw = len(image.shape) == 4 and image.shape[1] == 3
            # print(f"  Format: {'BCHW' if is_bchw else 'NOT BCHW'}")
            image = torch.tensor(image, device=device)
            image = image.permute(0, 3, 1, 2)  # Convert from BHWC to BCHW format

        image = F.center_crop(image, (config.image_size, config.image_size))
        return image

    real_count = 0
    fake_count = 0

    if save:
        real_dir = os.path.join(config.output_dir, "fid", "real")
        fake_dir = os.path.join(config.output_dir, "fid", "fake")
        os.makedirs(real_dir, exist_ok=True)
        os.makedirs(fake_dir, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Calculating FID (real images)"):
            real_images = batch["images"].to(device)
            processed_real = preprocess_image(real_images, real=True)
            if save:
                for image in processed_real:
                    save_image(image, os.path.join(real_dir, f"{real_count:03d}.jpg"))
                    real_count +=1
            fid.update(processed_real, real=True)

    with torch.no_grad():
        for batch in tqdm(
            range(0, len(test_dataloader.dataset), config.eval_batch_size),
            desc="Calculating FID (generated images)",
        ):
            # Generate images as numpy arrays
            output = pipeline(
                batch_size=min(
                    config.eval_batch_size, len(test_dataloader.dataset) - batch
                ),
                generator=torch.manual_seed(config.seed + batch),
                output_type="np.array",
            ).images
            processed_fake = preprocess_image(output, real=False)
            if save:
                for image in processed_fake:
                    save_image(
                        image,
                        os.path.join(fake_dir, f"{fake_count:03d}.jpg"),
                    )
                    fake_count +=1
            fid.update(processed_fake, real=False)

    # Compute final FID score
    fid_score = fid.compute().item()
    print(f"FID Score: {fid_score}")
    return fid_score


def get_full_repo_name(model_id: str, organization: str = None, token: str = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"
