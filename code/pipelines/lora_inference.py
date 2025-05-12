from diffusers import StableDiffusionPipeline
import torch

model_path = "finetuned_lora-model/"
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16,
    safety_checker=None
)
pipe.unet.load_attn_procs(model_path)
pipe.to("cuda")

prompt = "Close up face shot of Will smith eating spaghetti angrily."
image = pipe(prompt, num_inference_steps=30, guidance_scale=6.0).images[0]
image.save("assets/will_smith.png")
