from diffusers import StableDiffusionPipeline
import torch

model_id = "./results"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

prompt = "A photo of sks dog in a bucket"
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

image.save("output_imgs/dog-bucket.png")




# LORA inference
from diffusers import DiffusionPipeline
pipe = DiffusionPipeline.from_pretrained("base-model-name").to("cuda")
pipe.load_lora_weights("path-to-the-lora-checkpoint")
image = pipe("A picture of a sks dog in a bucket", num_inference_steps=25).images[0]
