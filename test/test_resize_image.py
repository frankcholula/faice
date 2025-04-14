from PIL import Image
from conf.global_setting import BASE_DIR

image_path = BASE_DIR + "/data/celeba_hq_split/test/1.jpg"
# Load the 512x512 image
img = Image.open(image_path)

# Resize to 128x128 with high-quality interpolation
resized_img = img.resize((128, 128), resample=Image.Resampling.LANCZOS)

# Adjust DPI to preserve physical size (e.g., original DPI was 300)
resized_img.info["dpi"] = (1200, 1200)  # New DPI = (512/128)*300 = 1200
resized_img.save("resized_image.jpg")
