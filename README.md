# Robust Deepfake Detection via Hybrid Convolutional and Transformer Architectures

This project implements a state-of-the-art deepfake video detection model that combines a **truncated ResNet3D50** backbone with a **2D Swin Transformer** for spatial feature encoding and a **Temporal Transformer** for sequence modeling. The hybrid architecture is specifically designed to capture both spatial and temporal inconsistencies characteristic of manipulated videos.

## 🧠 Architecture Overview

The model consists of four main components working in sequence:

1. **Truncated ResNet3D50 Backbone**: Extracts high-resolution spatio-temporal features from video frames while preserving spatial resolution at 28×28 (layers 3 and 4 removed)
2. **2D Swin Transformer**: Applies hierarchical window-based self-attention to encode detailed spatial features for each video frame
3. **Temporal Transformer**: Models inter-frame dependencies across the video sequence to detect temporal artifacts
4. **Binary Classifier**: Final classification layer that predicts REAL (0) or FAKE (1)

### Key Architecture Details

- **Input**: Video frames resized to 224×224, normalized with computed dataset statistics
- **Frame Sampling**: 16 frames per video, uniformly sampled across the entire video duration
- **Feature Flow**: ResNet3D → Projection Layer → Swin Transformer → Temporal Attention → Classification
- **Embedding Dimension**: 96 (configurable)
- **Window Size**: 7×7 for Swin Transformer attention windows

## 📁 Dataset Structure

```
project_root/
├── Balanced Sample/          # Training dataset directory
│   ├── metadata.json        # Training labels and metadata
│   └── *.mp4                # Training video files
├── dfdc_train_part_0/       # Testing dataset directory
│   ├── metadata.json        # Testing labels and metadata
│   └── *.mp4                # Testing video files
└── cached_test_samples/     # Cached preprocessed test samples (auto-generated)
```

**Dataset Requirements:**
- **Format**: MP4 video files with corresponding metadata.json
- **Labels**: JSON format with video filenames as keys and label information
- **Classes**: REAL (label: 0), FAKE (label: 1)
- **Source**: Compatible with [Kaggle Deepfake Detection Challenge](https://www.kaggle.com/c/deepfake-detection-challenge/data) format

## 🏗️ Project Structure

```
├── main.py                  # Main training and evaluation pipeline
├── swin_transformer.py      # Swin Transformer implementation
├── transformer.py           # Legacy transformer (reference only)
├── model_weights.pth        # Saved model weights (generated after training)
├── README.md               # This file
└── requirements.txt        # Dependencies (if available)
```

## 🚀 Usage

### Training

```bash
python main.py
```

The training script will:
1. Load training data from `Balanced Sample/` directory
2. Apply data augmentations (horizontal flip, normalization)
3. Train the model for 20 epochs with 90/10 train/validation split
4. Save the best model weights to `model_weights.pth`

### Testing

Testing runs automatically after training, or you can modify the main script to run testing independently. The script will:
1. Load test data from `dfdc_train_part_0/` directory
2. Cache preprocessed samples for faster subsequent runs
3. Evaluate the model and display comprehensive metrics

### Key Training Parameters

- **Epochs**: 20
- **Batch Size**: 16
- **Learning Rate**: 5e-5 (AdamW optimizer)
- **Weight Decay**: 1e-4
- **Scheduler**: StepLR (γ=0.5 every 5 epochs)
- **Loss Function**: BCEWithLogitsLoss

## 📊 Performance Metrics

**Architecture**: Truncated ResNet3D50 + 2D Swin Transformer + Temporal Transformer

| Metric     | Value (%) |
|------------|-----------|
| Accuracy   | 69.1      |
| Precision  | 65.1      |
| Recall     | 82.7      |
| F1 Score   | 72.8      |
| ROC AUC    | 69.2      |

**Training Configuration:**
- **Dataset Split**: 90% training, 10% validation (per epoch)
- **Data Augmentation**: Random horizontal flip, normalization
- **Hardware**: CUDA-enabled GPU recommended

## 🔧 Technical Implementation Details

### Video Processing Pipeline
1. **Frame Extraction**: Uniform sampling of 16 frames per video
2. **Preprocessing**: Resize to 224×224, RGB conversion, normalization
3. **Batch Processing**: Frames organized as (Batch, Channels, Time, Height, Width)

### Model Architecture Flow
```
Input Video (B, C, T, H, W)
    ↓
ResNet3D Backbone (B, 512, T, 28, 28)
    ↓
Reshape & Project (B*T, 96, 28, 28)
    ↓
Swin Transformer Processing (B*T, num_patches, 96)
    ↓
Reshape to Temporal (B, T, 96)
    ↓
Temporal Transformer (B, T, 96)
    ↓
Global Average Pooling (B, 96)
    ↓
Classification Layer (B, 1)
```

### Caching System
- **Purpose**: Accelerates testing by preprocessing and caching test samples
- **Location**: `cached_test_samples/` directory
- **Format**: Pickle files containing preprocessed frame tensors and labels
- **Automatic**: Generated on first test run, reused subsequently

## 🛠️ Dependencies

Key dependencies include:
- PyTorch (with CUDA support recommended)
- torchvision
- OpenCV (cv2)
- scikit-learn
- NumPy
- Matplotlib

## 📈 Model Characteristics

**Strengths:**
- Hybrid architecture captures both spatial and temporal features
- Swin Transformer provides efficient hierarchical attention
- High recall (82.7%) - good at detecting fake videos
- Robust preprocessing pipeline with augmentation

**Considerations:**
- Moderate precision (65.1%) - some false positives
- Memory intensive due to 3D convolutions and attention mechanisms
- Requires GPU for practical training times

## 🔬 Future Enhancements

- **3D Swin Transformers**: Extend to full volumetric attention across space and time
- **Adversarial Training**: Improve robustness against sophisticated deepfakes
- **Multi-scale Analysis**: Process videos at multiple resolutions
- **Multimodal Detection**: Incorporate audio features for enhanced detection
- **Self-supervised Pretraining**: Leverage unlabeled video data
- **Model Compression**: Optimize for deployment scenarios

## 📝 Notes

- The model uses computed dataset statistics for normalization: 
  - Mean: [0.467, 0.449, 0.386]
  - Std: [0.254, 0.259, 0.242]
- Training uses dynamic train/validation splits with shuffling each epoch
- The architecture prioritizes recall over precision, making it conservative in classification
- OpenCV threading is disabled (`cv2.setNumThreads(0)`) to prevent conflicts

## 🤝 Contributing

This implementation serves as a research baseline for deepfake detection. Contributions for improvements in architecture, training strategies, or evaluation metrics are welcome.
