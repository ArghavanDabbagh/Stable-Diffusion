# Stable-Diffusion
# Image Inpainting with BLIP, CLIP, and Stable Diffusion

## Overview
This project performs **image inpainting** using **Stable Diffusion**, **BLIP (Bootstrapped Language-Image Pretraining)** for caption generation, and **CLIP (Contrastive Language-Image Pretraining)** for scoring. The pipeline evaluates inpainting results using **SSIM (Structural Similarity Index), PSNR (Peak Signal-to-Noise Ratio), and FID (Fréchet Inception Distance)**.

---

## Features
- **BLIP** generates captions for masked regions.
- **Stable Diffusion** fills in masked regions based on BLIP-generated captions.
- **CLIP** scores image-text similarity.
- **SSIM, PSNR, and FID** assess inpainting quality.
- Supports **fine-tuning BLIP** to improve caption relevance.

---

## Installation
Ensure you have Python **3.8+** and install dependencies:

```bash
pip install torch torchvision torchaudio
pip install transformers diffusers huggingface_hub
pip install scikit-image pytorch-fid tqdm pandas matplotlib
```

For CUDA acceleration:
```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
```

To access Hugging Face models, login:
```bash
huggingface-cli login
```

---

## Usage

### **1. Running the Inpainting Pipeline**
Run the main script to process images:
```bash
python main.py
```
This will:
- Load BLIP, CLIP, and Stable Diffusion models.
- Apply random masks to images.
- Generate captions for missing regions.
- Inpaint masked areas using Stable Diffusion.
- Save results and compute metrics.

---

### **2. Fine-Tuning BLIP**
To fine-tune BLIP for better caption generation:
```bash
python finetune_blip.py
```
This script:
- Uses an MLP head to predict **SSIM, PSNR, and CLIP scores**.
- Fine-tunes BLIP to generate better text prompts.
- Saves a fine-tuned BLIP model for later use.

---

### **3. Calculating FID**
To compute FID scores for inpainted images:
```bash
python calculate_fid.py
```
This script:
- Extracts features from real and inpainted images using InceptionV3.
- Computes FID scores for baseline and new methods.

---

## File Structure
```
project-folder/
│── main.py                # Runs the inpainting pipeline
│── finetune_blip.py       # Fine-tunes BLIP for better captioning
│── calculate_fid.py       # Computes FID score for inpainted images
│── dataset/               # Input dataset
│── Results/               # Output inpainted images & metrics
│── requirements.txt       # Dependencies
```

---

## Example Output
Images are saved in `Results/`. Metrics are saved in:
```
Results/inpainting_metrics_test2014.csv
Results/average_performance_test2014.png
```

---

## Acknowledgments

- **Hugging Face** for BLIP and CLIP.
- **Stable Diffusion** by Stability AI.
- **Torch and Diffusers** for inpainting models.



