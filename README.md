# 🧠 AI vs Real Face Image Detection

🚧 Work in Progress

## Overview
A deep learning model to detect AI-generated face images.  
Focused on **generalization performance** across various AI generation models  
(GAN, DeepFake, StyleGAN, etc.)

## Goals
- Reproduce DenseNet-based baseline model (Accuracy ~0.98)
- Expand dataset beyond StyleGAN to diverse fake types
- Evaluate generalization performance (in/out-of-distribution)

## Dataset
| Type | Source |
|------|--------|
| Real | FFHQ |
| Fake | StyleGAN, ThisPersonDoesNotExist, Celeb-DF |

## Models
- DenseNet (Baseline)
- ResNet
- EfficientNet

## Tech Stack
- Python, PyTorch
- Grad-CAM, Confusion Matrix

## Progress
- [ ] Baseline reproduction
- [ ] Dataset expansion
- [ ] Multi-model experiments
- [ ] Generalization analysis
