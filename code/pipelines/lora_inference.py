from diffusers import StableDiffusionPipeline
import torch

model_path = "test_lora-model/"
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16,
    safety_checker=None
)
pipe.unet.load_attn_procs(model_path)
pipe.to("cuda")

prompt = "Will smith eating spaghetti."
image = pipe(prompt, num_inference_steps=30, guidance_scale=6.0).images[0]
image.save("hello.png")
