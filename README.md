# 🩺 Development of a CNN-based Skin Disease Classification Model

## Overview
A CNN-based deep learning model to classify major skin diseases,  
enabling early detection and fast diagnosis assistance.

## Target Classes
| Class | Samples |
|-------|---------|
| Acne (여드름) | 593 |
| Eczema (습진) | 1,010 |
| Infestations/Bites (기생충감염/벌레물림) | 524 |
| Melanoma (흑색종) | 438 |

## Model
- Architecture: CNN (Transfer Learning, 3-stage fine-tuning)
- Stage 1: Frozen Base Training (50 epochs)
- Stage 2: Partial Fine-tuning (50 epochs)
- Stage 3: Full Fine-tuning (50 epochs)

## Results
| Metric | Score |
|--------|-------|
| Best Validation Accuracy | 94.04% |
| Final Validation Accuracy | 93.01% |
| Macro-average AUC | 0.98 |

### Per-class F1-score
| Class | F1-score |
|-------|----------|
| Acne | 0.9427 |
| Eczema | 0.9371 |
| Infestations/Bites | 0.8000 |
| Melanoma | 1.0000 |

## Tech Stack
- Python, TensorFlow/Keras
- Transfer Learning, Data Augmentation
- Grad-CAM, ROC Curve, Confusion Matrix

## Dataset
- Kaggle: Skin Diseases + Cancer Comprehensive Dataset

## Limitations & Future Work
- The limited data scope constrains generalizability to diverse clinical settings  
- Future work will focus on acquiring large-scale, high-quality datasets and enhancing model performance  
