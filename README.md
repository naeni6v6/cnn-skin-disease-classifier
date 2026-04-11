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

## Key Findings
- Acne & Melanoma: AUC 1.00 (완벽한 분류)
- Infestations/Bites: AUC 0.95 (가장 낮음)
- 제한된 데이터 범위로 인해 다양한 임상 환경 적용에 한계 존재
- 향후 고품질 대규모 데이터셋 확보 및 모델 고도화 예정
