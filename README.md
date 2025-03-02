# Stable Diffusion Inpainting

This repository contains a simple **Stable Diffusion inpainting** for filling missing areas in an image.

---

## Features
- Uses **Stable Diffusion 2.0 Inpainting Model** (`stabilityai/stable-diffusion-2-inpainting`).
- Works with any **image & mask**.
- Fully automatic, requires **only an image, mask, and text prompt**.

---
### How It Works
Stable Diffusion inpainting works by:

- Taking an image + a mask (where the missing part is).
- Using a text prompt to describe what should be inpainted.
- Filling in the missing area using AI.

#### Model Details
- The model used is Stable Diffusion v2 Inpainting.
- Trained by Stability AI for realistic image inpainting.
---

## Dataset Structure

dataset/

│── image.jpg   # Image with missing part

│── mask.png    # Mask (white = to be inpainted)

---

## Install dependencies

### If needed install a virtual environment and then install the dependencies

pip install torch diffusers transformers pillow
 
---
