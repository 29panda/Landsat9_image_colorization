# Thermal Infrared (TIR) Image Colorization using Pix2Pix GAN
### ISRO BAH 2026 Hackathon Project

## Overview

This project is part of the **ISRO BAH 2026 Hackathon** challenge on **Thermal Infrared (TIR) Image Enhancement and Colorization**.

The challenge aims to improve the interpretability of satellite Thermal Infrared imagery by developing a two-stage computational pipeline:

1. **Super-Resolution:** Convert low-resolution (200 m) Thermal Infrared images into higher-resolution (100 m) images.
2. **Colorization:** Generate realistic RGB representations from the high-resolution TIR images.

This repository implements the **second stage (Colorization)** using a Pix2Pix conditional Generative Adversarial Network (cGAN).

> **Current Status:**  
> Pix2Pix-based colorization implemented and trained.  
> Super-resolution module is not included in this repository.

---

# Problem Statement

Thermal Infrared imagery is widely used for applications such as:

- Wildfire monitoring
- Urban Heat Island analysis
- Volcanic activity detection
- Environmental monitoring
- Land surface temperature estimation

However, TIR images are typically single-band grayscale images, making them difficult for human interpretation.

The objective is to learn a mapping

```
TIR (1 channel)
        ↓
   Pix2Pix Generator
        ↓
Synthetic RGB Image
```

where the generated RGB image resembles the corresponding multispectral RGB satellite image.

---

# Dataset

The dataset was provided as part of the **ISRO BAH 2026 Hackathon**.
It is a sample of images of 4 bands from LandSat 9.

Each training sample contains:

- Low-resolution TIR image
- High-resolution TIR image
- Corresponding RGB image

During this implementation, the Pix2Pix model is trained using:

```
Input:
100 m TIR image (single channel)

Target:
100 m RGB image (3 channels)
```

The preprocessing pipeline includes:

- Patch extraction
- Normalization
- Tensor conversion
- Dataset generation for PyTorch

---

# Model Architecture

## Generator

A lightweight U-Net style encoder-decoder network.

```
Input (1×256×256)

↓

Encoder
Conv → Conv → Conv → Conv

↓

Decoder
Transpose Conv → Transpose Conv → Transpose Conv → Transpose Conv

↓

Output RGB Image
(3×256×256)
```

Output activation:

- Tanh

---

## Discriminator

PatchGAN discriminator conditioned on:

```
Input TIR image
+
Generated / Real RGB image
```

The discriminator classifies local image patches as:

- Real
- Fake

---

# Loss Function

The generator is trained using:

```
Generator Loss

L = L1
```

The project also contains implementations for:

- GAN Loss (Binary Cross Entropy)
- Structural Similarity (SSIM)

These were evaluated during experimentation.

---

# Training Pipeline

```
TIR Image
      │
      ▼
Generator (Pix2Pix)
      │
      ▼
Generated RGB
      │
      ├────────► L1 Loss
      │
      └────────► PatchGAN
                     │
                     ▼
            Adversarial Loss
```

Training was performed using:

- PyTorch
- Adam Optimizer
- Batch Size = 4
- 800 training epochs (checkpoint supported)

---

# Results

The repository saves:

- Input TIR images
- Ground Truth RGB images
- Generated RGB images

for every saved epoch to visualize training progress.


# Technologies Used

- Python
- PyTorch
- NumPy
- OpenCV
- Matplotlib
- Google Colab
- Pix2Pix Conditional GAN
- PatchGAN Discriminator

---

# Data Source

Dataset provided as part of:

**ISRO BAH 2026 Hackathon**

The dataset is intended solely for participation in the hackathon.

Therefore, the original dataset is **not redistributed** in this repository.

---

# Running the Project

Clone the repository

```bash
git clone https://github.com/<your_username>/<repository_name>.git
```

Install dependencies

```bash
pip install torch torchvision numpy opencv-python matplotlib
```

Train

```bash
python train.py
```

Inference

```bash
python inference.py
```

---

# Future Work

The complete challenge pipeline consists of two stages.

Future improvements include:

- SwinIR-based Super Resolution
- ESRGAN-based Super Resolution
- Semantic-aware colorization
- Perceptual Loss (VGG)
- Attention U-Net Generator
- Better PatchGAN architecture
- Deployment as an inference pipeline

---

# Disclaimer

This repository implements only the **colorization stage** of the original ISRO BAH 2026 challenge.

The super-resolution stage described in the challenge statement has **not** been implemented in this repository.

---

# Acknowledgements

- Indian Space Research Organisation (ISRO)
- BAH 2026 Hackathon Organizers
- Pix2Pix (Isola et al., 2017)
- PyTorch

---

# References

1. Isola, P., Zhu, J. Y., Zhou, T., & Efros, A. A. (2017). *Image-to-Image Translation with Conditional Adversarial Networks.*

2. Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.*

3. PyTorch Documentation

4. ISRO BAH 2026 Hackathon Documentation

---
