# prompt: **GENERATE SIMILAR IMAGES SCRIPT:**
# Using Stable Diffusion and Python:
# **SCRIPT CODE:**
# ```python
# from PIL import Image
# import torch
# from diffusers import StableDiffusionPipeline
# Define Stable Diffusion pipeline
# model_id = "CompVis/stable-diffusion-v1-4"
# pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
# Original image prompt (modify this to match your image content)
# prompt = "Cyberpunk woman with green eyes and curly brown hair"
# Generate similar images with slight variations
# for i in range(10): # generate 10 similar images
#     variation_prompt = prompt + f" with slightly different lighting and expression {i}"
#     with torch.autocast("cuda"):
#         image = pipe(variation_prompt).images[0]
#     image.save(f"similar_image_{i}.jpg")
# print("Similar images generated and saved.")
# ```
# **VARIATION OPTIONS:**
# 1. `with slightly different lighting`
# 2. `and expression changed slightly`
# 3. `with hair styled differently`
# 4. `wearing alternative outfit`
# 5. `in a different pose`
# Shall I help with:
# 1. Running this script 
# 2. Modifying prompts for more variations 
# 3. Using generated images for Nexus Prime project?

import torch
from diffusers import StableDiffusionPipeline
from google.colab import files

# Assuming necessary installations and setup from previous code blocks

model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")  # Ensure the model is on the GPU

prompt = "Seductive Woman, 20 Years old, Selfie, Closeup Portrait, Full body, piercing green eyes, long curly brown hair, flawless skin"

variations = [
    "with slightly different lighting",
    "and expression changed slightly",
    "with hair styled differently",
    "wearing alternative outfit",
    "in a different pose"
]

for i, variation in enumerate(variations):
    variation_prompt = prompt + ", " + variation
    with torch.autocast("cuda"):
        image = pipe(variation_prompt).images[0]
    image.save(f"nexus_prime_queen_variation_{i}.png")
    files.download(f"nexus_prime_queen_variation_{i}.png")
    print(f"Image variation {i} generated, saved, and downloaded.")
