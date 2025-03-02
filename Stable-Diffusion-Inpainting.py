import torch
from diffusers import StableDiffusionInpaintPipeline #Loads the pretrained Stable Diffusion inpainting model.

from PIL import Image
import argparse

# Argument parser for CLI usage
parser = argparse.ArgumentParser(description="Stable Diffusion Inpainting Script")
parser.add_argument("--image", type=str, required=True, help="Path to the input image") #Input image to be inpainted.
parser.add_argument("--mask", type=str, required=True, help="Path to the mask image") #Black & white mask (white = area to inpaint, black = keep).
parser.add_argument("--output", type=str, default="output.png", help="Path to save the inpainted image") #File path to save the output.
parser.add_argument("--prompt", type=str, default="A beautiful landscape", help="Text prompt for inpainting") #Describes what should be filled in.

args = parser.parse_args()

# Load the inpainting model
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16
).to("cuda" if torch.cuda.is_available() else "cpu") # Automatically chooses GPU if available.

# Open images
image = Image.open(args.image).convert("RGB") #An image with missing parts.
mask = Image.open(args.mask).convert("L")  # Convert mask to grayscale , Corresponding mask (white = area to inpaint)

# Run inpainting
output = pipe(
    prompt=args.prompt,  # Text prompt for the model
    image=image,
    mask_image=mask
).images[0]

# Save the output image
output.save(args.output)

print(f"Inpainting completed! Saved as {args.output}")
