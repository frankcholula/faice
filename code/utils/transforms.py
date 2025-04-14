from typing import Tuple

import torchvision.transforms as T
from PIL import Image


def build_transforms(config):
    # build train transformations
    transform_train = [
        # T.Resize((config.image_size, config.image_size)),
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),
    ]

    transform_test = [
        # T.Resize((config.image_size, config.image_size)),
        T.ToTensor(),
    ]

    if config.RHFlip:
        transform_train += [T.RandomHorizontalFlip()]
    if config.gblur:
        transform_train += [T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))]

    transform_train = T.Compose(transform_train)
    transform_test = T.Compose(transform_test)

    return transform_train, transform_test


def resize_image(image_data, size: Tuple = (128, 128)):
    """
    Resize an image to a specific size.

    Args:
        image_data: The image data to be resized.
        size (tuple): Desired size of the image (width, height).

    Returns:
        None
    """

    # Resize to size with high-quality interpolation
    resized_img = image_data.resize(size, resample=Image.Resampling.LANCZOS)

    # Adjust DPI to preserve physical size (e.g., original DPI was 300)
    # Get the original DPI and calculate the new DPI
    original_dpi = image_data.info.get("dpi", (300, 300))
    resized_img.info["dpi"] = (original_dpi[0] * image_data.size[0] / size[0],
                               original_dpi[1] * image_data.size[1] / size[1])
    return resized_img


if __name__ == "__main__":
    image_file = "../datasets/test/1.jpg"
    # Open the image using Pillow
    img = Image.open(image_file).convert("RGB")
    new_img = resize_image(img)
    new_img.save("resized_image.jpg")
