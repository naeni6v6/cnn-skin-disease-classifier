<h1 align="center">🩺 CNN-based Skin Disease Classification</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Model-EfficientNetV2S-0066CC?style=for-the-badge" alt="EfficientNetV2S">
  <img src="https://img.shields.io/badge/Test_Accuracy-88.62%25-success?style=for-the-badge" alt="Accuracy 88.62%">
  <img src="https://img.shields.io/badge/Macro_F1-0.8906-success?style=for-the-badge" alt="Macro-F1 0.8906">
  <img src="https://img.shields.io/badge/Classes-7-success?style=for-the-badge" alt="7 classes">
</p>
<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/timm-000000?style=flat" alt="timm">
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Keras-D00000?logo=keras&logoColor=white" alt="Keras">
  <img src="https://img.shields.io/badge/images-21K-lightgrey" alt="21K images">
</p>
<p align="center">
  Classifying 7 major skin conditions — evolved from an in-school baseline (v1) into a research program (v2) with rigorous data curation, de-duplication, and leakage control.
</p>

## Overview
A CNN-based deep learning model to classify major skin diseases,
enabling early detection and fast diagnosis assistance.

This project was first built for an **in-school competition (v1)**. Finding its
scope and reliability lacking, it was extended into a **follow-up research
program (v2)** — expanding the class set, scaling up the data, and adding
rigorous data-quality control to make the results trustworthy.

---

## v1 — In-School Competition (Baseline)

### Target Classes
| Class | Samples |
|-------|---------|
| Acne (여드름) | 593 |
| Eczema (습진) | 1,010 |
| Infestations/Bites (기생충감염/벌레물림) | 524 |
| Melanoma (흑색종) | 438 |

### Model
- Architecture: CNN (Transfer Learning, 3-stage fine-tuning)
- Stage 1: Frozen Base Training / Stage 2: Partial / Stage 3: Full Fine-tuning

### Results
| Metric | Score |
|--------|-------|
| Best Validation Accuracy | 94.04% |
| Macro-average AUC | 0.98 |

**Limitations found:** no "normal skin" class (cannot tell disease from
non-disease), class imbalance, overfitting, and — most importantly — a suspicious
Melanoma F1 of 1.0000 that later turned out to be **data leakage** from
capture-method bias. These motivated v2.

---

## v2 — Research Program (Extended)

### Target Classes (4 → 7)
| Class | Samples |
|-------|---------|
| Normal (정상) | 1,331 |
| Acne (여드름) | 3,072 |
| Eczema (습진) | 4,000* |
| Insect Bites (벌레물림) | 1,971 |
| Melanoma (흑색종) | 4,434 |
| Psoriasis (건선) | 2,507 |
| Fungal Infection (진균감염) | 3,987 |

*\*Eczema under-sampled from 12,042 → 4,000.*

### 🔍 Data Curation & Quality Control
- Consolidated scattered sub-diagnoses into 7 classes by clinical/visual features
  (e.g. atopic → Eczema; perioral dermatitis → Acne). Rosacea and benign nevi excluded.
- **Removed 7,873 duplicate images (≈33%)** via pHash near-duplicate detection
- pHash group-split so identical images never leak across train/val/test
- Diagnosed capture-method bias as the cause of v1's inflated Melanoma score

### Model
- Framework: **PyTorch** (migrated from Keras/TensorFlow)
- Architecture: **EfficientNetV2-S** (timm pretrained)
- Backbone-freeze warmup → full fine-tuning, cosine LR, early stopping
- Overfitting control: Dropout, stochastic depth, weight decay, label smoothing, strong augmentation

### Results (held-out test, 3,059 images)
| Metric | Score |
|--------|-------|
| Accuracy | 88.62% |
| Macro-F1 | 0.8906 |

> The lower headline number vs. v1 is intentional: with duplicates removed and
> leakage blocked, this is a **more honest and generalizable** evaluation.

#### Per-class F1-score
| Class | F1-score |
|-------|----------|
| Normal | 0.9795 |
| Acne | 0.9737 |
| Eczema | 0.8081 |
| Insect Bites | 0.8750 |
| Melanoma | 0.9789 |
| Psoriasis | 0.7988 |
| Fungal Infection | 0.8203 |

---

## 🛠️ Tech Stack
- Python, PyTorch, timm (v2) · TensorFlow/Keras (v1)
- Transfer Learning, Data Augmentation, pHash de-duplication
- Grad-CAM, ROC Curve, Confusion Matrix

## 📂 Dataset Sources
Aggregated from ~15 public datasets (Kaggle & Roboflow), then merged, de-duplicated,
and re-labeled into 7 classes (~21K images after cleaning).

<details>
<summary>Full source list</summary>

**Core disease images (DermNet-derived)**
- [lysaapriani/skin-disease-and-normal-skin-dataset](https://www.kaggle.com/datasets/lysaapriani/skin-disease-and-normal-skin-dataset)
- [pacificrm/skindiseasedataset](https://www.kaggle.com/datasets/pacificrm/skindiseasedataset)
- [ismailpromus/skin-diseases-image-dataset](https://www.kaggle.com/datasets/ismailpromus/skin-diseases-image-dataset)
- *shubhamgoel27/dermnet — skipped (duplicate of the above)*

**Normal skin**
- [ahdasdwdasd/our-normal-skin](https://www.kaggle.com/datasets/ahdasdwdasd/our-normal-skin)
- [dipuiucse/monkeypox-skin-images](https://www.kaggle.com/datasets/dipuiucse/monkeypoxskinimagedataset) (normal subset)
- [shakyadissanayake/oily-dry-and-normal-skin-types](https://www.kaggle.com/datasets/shakyadissanayake/oily-dry-and-normal-skin-types-dataset)
- [sd20co001/image-dataset-for-skin-diseases](https://www.kaggle.com/datasets/sd20co001/image-dataset-for-skindiseases-dry-oily-normalskin) (normal / acne / eczema)

**Per-class augmentation**
- [anieetorudofia/skin-diseases-cancer-comprehensive](https://www.kaggle.com/datasets/anieetorudofia/skin-diseases-cancer-comprehensive-dataset) (all 7 classes)
- [youssefmohmmed/human-skin-diseases-image](https://www.kaggle.com/datasets/youssefmohmmed/human-skin-diseases-image) (6 classes, no melanoma)
- [tapakah68/skin-problems-34-on-the-iga-scale](https://www.kaggle.com/datasets/tapakah68/skin-problems-34-on-the-iga-scale) (acne)
- [roboflow/acne-data-cuenl](https://universe.roboflow.com/acne-data/acne-data-cuenl) (acne)
- [moonfallidk/bug-bite-images](https://www.kaggle.com/datasets/moonfallidk/bug-bite-images) (insect bites)
- [roboflow/bug-bites](https://universe.roboflow.com/object-detection-ttfpu/bug-bites) (insect bites)

</details>

## ⚠️ Limitations & Future Work
- Capture-method bias remains a dataset-level limitation (some classes separable by
  source rather than lesion) — documented, not fully solvable in code
- Melanoma class covers malignant lesions only; benign nevi/moles were excluded, so
  the model does not yet distinguish malignant melanoma from benign look-alikes
- Single-source origin still limits diversity across skin tones and imaging conditions
- **Next:** Grad-CAM++ / Score-CAM explainability, confidence-based
  "consult a specialist" thresholds, and FastAPI + Streamlit web deployment
