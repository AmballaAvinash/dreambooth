from diffusers import StableDiffusionPipeline
import torch

import argparse

parser = argparse.ArgumentParser(description="Dreambooth inference arguments")

parser.add_argument("--model_id", type=str, default="./results")
parser.add_argument("--prompt", type=str, default="A photo of sks dog in a bucket", help="prompt")
parser.add_argument("--save_output", type=str, default="output_imgs/dog-bucket.png", help="save output to")

input_args = parser.parse_args()


model_id = input_args.model_id
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

prompt = input_args.prompt
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

image.save(input_args.save_output)