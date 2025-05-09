from diffusers import StableDiffusionPipeline
import torch

model_path = "code/test_lora-model"
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16
)
pipe.unet.load_attn_procs(model_path)
pipe.to("cuda")

prompt = "Portrait of a young woman with long wavy hair, soft studio lighting, high contrast, 4k resolution, professional headshot"
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
image.save("hello.png")
