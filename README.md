# CNN-Based Skin Disease Classification Model

## Overview
This project presents a CNN-based deep learning model for classifying major skin diseases.  
The objective is to support early detection and assist rapid diagnosis in clinical scenarios.

---

## Target Classes
| Class | Samples |
|-------|---------|
| Acne | 593 |
| Eczema | 1,010 |
| Infestations/Bites | 524 |
| Melanoma | 438 |

---

## Model Architecture
- CNN-based model with Transfer Learning
- 3-stage fine-tuning strategy:
  - Stage 1: Frozen base training (50 epochs)
  - Stage 2: Partial fine-tuning (50 epochs)
  - Stage 3: Full fine-tuning (50 epochs)

---

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

---

## Tech Stack
- Python
- TensorFlow / Keras
- Transfer Learning
- Data Augmentation
- Grad-CAM
- ROC Curve
- Confusion Matrix

---

## Dataset
- Kaggle: Skin Diseases + Cancer Comprehensive Dataset

---

## Key Insights
- Achieved high classification performance with Macro AUC of 0.98
- Acne and Melanoma showed near-perfect classification performance (AUC = 1.00)
- Infestations/Bites showed relatively lower performance (AUC = 0.95)
- Limited dataset size may affect generalization to diverse clinical environments

---

## Future Work
- Expand dataset size and diversity
- Improve model generalization in real-world scenarios
- Explore advanced architectures and ensemble methods
