import os
import torch
import wandb
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm
from huggingface_hub import whoami, HfFolder
from cleanfid import fid
from diffusers import DiTPipeline, DDPMPipeline, DDIMPipeline, StableDiffusionPipeline, LDMPipeline
from pipelines.ccddpm_pipeline import CCDDPMPipeline
from pipelines.custom_pipelines import CustomTransformer2DPipeline


def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


def pipeline_inference(
        config,
        pipeline,
        batch_size,
        output_type="pil",
        generator=torch.manual_seed(0),
        class_labels=[],
        prompt=[]
):
    if isinstance(pipeline, DiTPipeline) or isinstance(pipeline, CustomTransformer2DPipeline):
        images = pipeline(
            class_labels,
            guidance_scale=1.0,
            generator=generator,
            num_inference_steps=config.num_inference_steps,
            output_type=output_type,
        ).images
    elif isinstance(pipeline, CustomTransformer2DPipeline):
        images = pipeline(
            class_labels,
            generator=generator,
            num_inference_steps=config.num_inference_steps,
            output_type=output_type,
        ).images
    elif isinstance(pipeline, StableDiffusionPipeline):
        # Convert prompt to tensor
        images = pipeline(
            prompt=prompt,
            guidance_scale=7.5,
            eta=0.0,
            generator=generator,
            num_inference_steps=config.num_inference_steps,
            output_type=output_type,
        ).images
    elif isinstance(pipeline, DDIMPipeline) or isinstance(pipeline, LDMPipeline):
        images = pipeline(
            batch_size=batch_size,
            generator=generator,
            num_inference_steps=config.num_inference_steps,
            eta=config.eta,
            output_type=output_type,
        ).images
    elif isinstance(pipeline, CCDDPMPipeline):
        label_id = 1 if config.condition_on == "male" else 0
        class_labels = torch.full(
            (batch_size,), label_id, dtype=torch.long, device=pipeline.unet.device
        )
        encoder_hidden_states = torch.zeros(
            batch_size,
            1,
            pipeline.unet.config.cross_attention_dim,
            device=pipeline.unet.device,
        )
        images = pipeline(
            batch_size=batch_size,
            generator=generator,
            num_inference_steps=config.num_inference_steps,
            class_labels=class_labels,
            encoder_hidden_states=encoder_hidden_states,
        ).images
    else:
        images = pipeline(
            batch_size=batch_size,
            generator=generator,
            num_inference_steps=config.num_inference_steps,
            output_type=output_type,
        ).images
    return images


def evaluate(config, epoch, pipeline, class_labels=[], prompt=[]):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    batch_size = 16
    images = pipeline_inference(config, pipeline, batch_size, class_labels=class_labels, prompt=prompt)
    images_kwargs = {
        "batch_size": batch_size,
        "generator": torch.manual_seed(config.seed),
        "num_inference_steps": config.num_inference_steps,
    }

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


def preprocess_image(image, img_src, device):
    if img_src == "loaded":
        return image
    elif img_src == "generated":
        image = torch.tensor(image, device=device)
        image = image.permute(0, 3, 1, 2)
        return image


def calculate_inception_score(
        config, pipeline, test_dataloader, device=None, class_labels=[], prompts=[]
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inception_score = InceptionScore(
        feature="logits_unbiased", splits=10, normalize=True
    ).to(device)

    fake_dir = os.path.join(config.output_dir, "fid", "fake")
    fake_images = []
    with torch.no_grad():
        if os.path.exists(fake_dir) and len(os.listdir(fake_dir)) > 0:
            for filename in tqdm(os.listdir(fake_dir), desc="Loading Fake Images..."):
                image_path = os.path.join(fake_dir, filename)
                img = Image.open(image_path).convert("RGB")
                img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
                fake_images.append(img_tensor)
                if len(fake_images) == config.eval_batch_size:
                    processed_fake = preprocess_image(
                        torch.cat(fake_images, dim=0),
                        img_src="loaded",
                        device=device,
                    )
                    inception_score.update(processed_fake)
                    fake_images = []
            if fake_images:
                processed_fake = preprocess_image(
                    torch.cat(fake_images, dim=0),
                    img_src="loaded",
                    device=device,
                )
                inception_score.update(processed_fake)
        else:
            for batch in tqdm(test_dataloader, desc="Calculating Inception Score"):
                batch_size = min(
                    config.eval_batch_size, len(test_dataloader.dataset) - batch
                )
                generator = torch.manual_seed(config.seed + batch)

                if prompts:
                    # batch_data = test_dataloader.dataset
                    # prompts = [prompt_dict[int(x['image_names'])] for i, x in enumerate(batch_data) if
                    #            i in range(batch, batch + batch_size)]
                    prompts = prompts
                else:
                    prompts = []

                output = pipeline_inference(
                    config,
                    pipeline,
                    batch_size,
                    generator=generator,
                    output_type="np",
                    class_labels=class_labels,
                    prompt=prompts
                )
                processed_fake = preprocess_image(
                    output,
                    img_src="generated",
                    device=device,
                )
                inception_score.update(processed_fake)
    inception_mean, inception_std = inception_score.compute()
    print(f"Inception Score: {inception_mean:.2f} ± {inception_std:.2f}")
    return inception_mean, inception_std


def calculate_fid_score(
        config, pipeline, test_dataloader, device=None, save=True, class_labels=[],
        prompts=[]
):
    """Calculate FID score between generated images and test dataset"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create FID instance with normalize=True since we'll provide images in [0,1] range
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    fake_count = 0

    if save:
        real_dir = os.path.join(config.output_dir, "fid", "real")
        fake_dir = os.path.join(config.output_dir, "fid", "fake")
        os.makedirs(real_dir, exist_ok=True)
        os.makedirs(fake_dir, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(
                test_dataloader, desc="Loading Real Images for FID Calculation..."
        ):
            real_images = batch["images"].to(device)
            real_image_names = batch["image_names"]
            processed_real = preprocess_image(
                real_images,
                img_src="loaded",
                device=device,
            )
            if save:
                for i, image in enumerate(processed_real):
                    img_name = real_image_names[i]
                    save_image(image, os.path.join(real_dir, f"{img_name}.jpg"))
            fid.update(processed_real, real=True)

    with torch.no_grad():
        for batch in tqdm(
                range(0, len(test_dataloader.dataset), config.eval_batch_size),
                desc="Loading Fake Images for FID Calculation..",
        ):
            # Generate images as numpy arrays
            batch_size = min(
                config.eval_batch_size, len(test_dataloader.dataset) - batch
            )
            generator = torch.manual_seed(config.seed + batch)

            if prompts:
                prompts = prompts
                # batch_data = test_dataloader.dataset
                # prompts = [prompt_dict[int(x['image_names'])] for i, x in enumerate(batch_data) if
                #            i in range(batch, batch + batch_size)]
            else:
                prompts = []
            output = pipeline_inference(
                config,
                pipeline,
                batch_size,
                generator=generator,
                output_type="np",
                class_labels=class_labels,
                prompt=prompts
            )
            processed_fake = preprocess_image(
                output,
                img_src="generated",
                device=device,
            )
            if save:
                for image in processed_fake:
                    save_image(
                        image,
                        os.path.join(fake_dir, f"{fake_count:03d}.jpg"),
                    )
                    fake_count += 1
            fid.update(processed_fake, real=False)

    # Compute final FID score
    fid_score = fid.compute().item()
    print(f"FID Score: {fid_score}")
    clean_fid_score = calculate_clean_fid(real_dir, fake_dir)
    min_fid_score = min(float(clean_fid_score), float(fid_score))
    print(f"Minimum FID Score: {min_fid_score}")
    return min_fid_score


def get_full_repo_name(model_id: str, organization: str = None, token: str = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def calculate_clean_fid(real_images_dir, fake_images_dir, msg="Clean FID score"):
    score = fid.compute_fid(real_images_dir, fake_images_dir)
    fid_score = round(score, 5)

    print(f"{msg}: {fid_score}")
    return fid_score


def calculate_inception_score_vae(real_images, fake_images):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inception_score = InceptionScore(
        feature="logits_unbiased", splits=10, normalize=True
    ).to(device)

    inception_score.update(real_images)
    inception_score.update(fake_images)
    inception_mean, inception_std = inception_score.compute()
    print(f"Inception Score: {inception_mean:.2f} ± {inception_std:.2f}")
    return inception_mean, inception_std
