# Robust Deepfake Detection via Hybrid Convolutional and Transformer Architectures

This project implements a robust deepfake video detection model that integrates a **ResNet3D50** backbone with a **2D Swin Transformer** for spatial encoding and a **Temporal Transformer** for sequence modeling. The hybrid architecture is designed to capture both spatial and temporal inconsistencies introduced in manipulated videos.

## 🧠 Architecture Overview

- **ResNet3D50 (Truncated)**: Extracts high-resolution spatio-temporal features from video frames. Final layers removed to preserve `28×28` resolution.
- **Swin Transformer 2D**: Applies localized window-based self-attention to encode detailed spatial features on each video frame.
- **Temporal Transformer**: Models inter-frame dependencies to detect temporal artifacts.
- **Binary Classifier**: Predicts whether a video is `REAL` or `FAKE`.

---

## 📁 Dataset

- **Source**: [Kaggle Deepfake Dataset](https://www.kaggle.com/c/deepfake-detection-challenge/data)
- **Contents**: 1000 training and 400 testing videos, with `metadata.json` indicating labels.
- **Classes**: `REAL` (0), `FAKE` (1)

---

## 🏗️ Project Structure

```text
├── main.py # Training and testing pipeline
├── transformer.py # Legacy transformer components (not final)
├── swin_transformer.py # Swin Transformer architecture
├── model_weights.pth # (Optional) Trained weights
├── Balanced Sample/ # Training data directory
├── dfdc_train_part_0/ # Testing data directory
└── cached_test_samples/ # Optional test set cache
```text


## 📊 Performance

**Final Architecture**: Adjusted ResNet3D50 + Swin Transformer + Temporal Transformer  
**Epochs**: 20  
**Batch Size**: 16  
**Best Metrics:**

| Metric     | Value (%) |
|------------|-----------|
| Accuracy   | 69.1      |
| Precision  | 65.1      |
| Recall     | 82.7      |
| F1 Score   | 72.8      |
| ROC AUC    | 69.2      |

---

## ⚙️ Training Details

- **Optimizer**: AdamW  
- **Learning Rate**: 5e-5  
- **Scheduler**: StepLR (gamma=0.5 every 5 epochs)  
- **Loss Function**: BCEWithLogitsLoss

---

## 📌 Future Work

- Extend to 3D Swin Transformers for full volumetric attention  
- Integrate adversarial robustness techniques  
- Explore self-/semi-supervised pretraining  
- Multimodal detection with audio + video



