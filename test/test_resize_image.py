from typing import Tuple

from PIL import Image


def resize_image(image_data, size: Tuple = (128, 128)):
    """
    Resize an image to a specific size.

    Args:
        image_data:
        output_path (str): Path to save the resized image.
        size (tuple): Desired size of the image (width, height).

    Returns:
        None
    """

    # Resize to size with high-quality interpolation
    resized_img = image_data.resize(size, resample=Image.Resampling.LANCZOS)

    # Adjust DPI to preserve physical size (e.g., original DPI was 300)
    # Get the original DPI and calculate the new DPI
    original_dpi = img.info.get("dpi", (300, 300))
    resized_img.info["dpi"] = (
        original_dpi[0] * img.size[0] / size[0],
        original_dpi[1] * img.size[1] / size[1],
    )
    return resized_img


if __name__ == "__main__":
    image_file = "../datasets/test/1.jpg"
    # Open the image using Pillow
    img = Image.open(image_file).convert("RGB")
    new_img = resize_image(img)
    new_img.save("resized_image.jpg")
