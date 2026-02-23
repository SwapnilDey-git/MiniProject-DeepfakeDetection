# 🎭 Deepfake Detection - Mini Project

A deep learning project for detecting deepfakes in images and videos using PyTorch, combining CNN and LSTM architectures.

## 📊 Project Overview

This project implements a binary classification system that distinguishes between **real** and **fake** (deepfake) media content. The model achieves **92.55% accuracy** on the validation dataset.

### Key Features
- ✅ **Image Detection**: CNN-based classification for static images
- ✅ **Video Detection**: LSTM + CNN temporal analysis for video deepfakes
- ✅ **Custom Dataset**: Built-in support for real/fake folder structures
- ✅ **Training & Inference**: Complete pipeline with model checkpoints
- ✅ **Visualization**: Training curves, confusion matrix, and classification metrics
- ✅ **GPU Support**: CUDA acceleration when available

---

## 📈 Model Performance

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | 92.55% |
| **Best Validation Accuracy** | 92.55% (Epoch 10) |
| **Best Validation Loss** | 0.1805 |
| **Real Precision** | 97% |
| **Fake Precision** | 88% |
| **Real Recall** | 89% |
| **Fake Recall** | 97% |

### Confusion Matrix Results
```
                Predicted Real    Predicted Fake
Actual Real          4,820              593
Actual Fake            149            4,392
```

---

## 🏗️ Architecture

### SimpleCNN (Backbone)
- 4 convolutional blocks with BatchNorm and ReLU
- Input: 3×128×128 images
- Output: 256-dimensional feature vector
- MaxPooling between layers for spatial reduction

### ImageClassifier
- SimpleCNN backbone
- Binary classification head
- Output: Real/Fake probability (0-1)

### VideoClassifier
- SimpleCNN backbone (frame-level features)
- Bidirectional LSTM (temporal modeling)
- 16 frames per video
- Output: Real/Fake probability

---

## 📁 Project Structure

```
deepfake_detection/
├── README.md                              # This file
├── detect.ipynb                           # Main training & inference notebook
├── infer.ipynb                            # Standalone inference notebook
├── image_model_best.pth                   # Trained image model weights
├── training_history.json                  # Training metrics history
├── train.csv, valid.csv, test.csv        # Data metadata
├── deepfake_dataset/
│   ├── train/images/
│   │   ├── real/                         # Real training images
│   │   └── fake/                         # Fake training images
│   ├── val/
│   │   ├── images/real/, fake/          # Validation images
│   │   └── videos/real/, fake/          # Validation videos
│   └── test_images/                      # Test images for inference
└── real_vs_fake/                         # Alternative dataset structure
```

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install torch torchvision opencv-python numpy scikit-learn matplotlib seaborn
```

### Training

1. **Open the notebook**:
   ```bash
   jupyter notebook detect.ipynb
   ```

2. **Run Cell 1** - Complete training pipeline:
   - Loads image/video datasets
   - Trains ImageClassifier for 10 epochs
   - Saves best model to `image_model_best.pth`
   - Optionally trains VideoClassifier (set `USE_VIDEO = True`)

### Inference

**Option 1: Cell 2 (Standalone Inference)**
```python
result = unified_predict(r"path/to/image_or_video.jpg")
print(f"Prediction: {result['file_type']}")
print(f"Fake Probability: {result['fake_probability']:.2%}")
```

**Option 2: Separate Inference Notebook**
```bash
jupyter notebook infer.ipynb
```

### Visualization & Metrics

**Run Cell 3** to generate:
- Training/validation loss curves
- Training/validation accuracy curves
- Confusion matrix on validation set
- Precision, recall, F1-scores

---

## 🛠️ Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Image Size | 128×128 | Input image resolution |
| Batch Size (Image) | 32 | Samples per batch |
| Batch Size (Video) | 4 | Frames batches for video |
| Num Frames | 16 | Frames extracted per video |
| Epochs | 10 | Training iterations |
| Learning Rate | 1e-3 | Adam optimizer LR |
| Model Size | ~2.5M | Total parameters (Image model) |

---

## 📊 Training History

The model shows good convergence:
- **Training Loss**: Steadily decreases from 0.47 → 0.19
- **Validation Loss**: Decreases to 0.18 (epoch 10)
- **Training Accuracy**: 78% → 93%
- **Validation Accuracy**: 83% → 92.55%

### Best Epoch: **10**
- Validation Accuracy: 92.55%
- Validation Loss: 0.1805

---

## 💡 Key Insights

1. **Model Strengths**:
   - Excellent real detection (97% precision, 89% recall)
   - Strong fake detection (88% precision, 97% recall)
   - Balanced performance across classes

2. **Potential Improvements**:
   - Data augmentation for robustness
   - Ensemble methods combining image + video models
   - Fine-tuning with pretrained backbones (ResNet, etc.)
   - Longer video sequences for better temporal modeling

---

## 📝 Dataset Format

Ensure your dataset follows this structure:

```
deepfake_dataset/
├── train/
│   └── images/
│       ├── real/
│       │   ├── image1.jpg
│       │   ├── image2.jpg
│       │   └── ...
│       └── fake/
│           ├── deepfake1.jpg
│           └── ...
├── val/
│   ├── images/
│   │   ├── real/
│   │   └── fake/
│   └── videos/
│       ├── real/
│       │   ├── real1.mp4
│       │   └── ...
│       └── fake/
│           ├── fake1.mp4
│           └── ...
└── test_images/
    ├── test1.jpg
    └── ...
```

---

## 🎯 Usage Examples

### Example 1: Predict on Single Image
```python
from detect import unified_predict, explain_result

result = unified_predict("path/to/image.jpg")
explain_result(result)
```

Output:
```
File Type : image
Prediction: REAL
Confidence: 95.32%
```

### Example 2: Batch Inference
```python
import os
test_dir = "deepfake_dataset/test_images"

for filename in os.listdir(test_dir):
    result = unified_predict(os.path.join(test_dir, filename))
    print(f"{filename}: {result['fake_probability']:.2%} fake")
```

---

## ⚙️ Advanced Configuration

Edit `detect.ipynb` Cell 1 to customize:

```python
USE_VIDEO = False                                    # Enable video training
DATA_ROOT = "B:\\deepfake_detection\\deepfake_dataset"  # Dataset path
IMG_SIZE = 128                                       # Image resolution
BATCH_SIZE_IMG = 32                                  # Image batch size
NUM_FRAMES = 16                                      # Frames per video
EPOCHS_IMG = 10                                      # Training epochs
LEARNING_RATE = 1e-3                                # Optimizer LR
```

---

## 🔍 Troubleshooting

| Issue | Solution |
|-------|----------|
| No training images found | Check dataset path and folder structure |
| CUDA out of memory | Reduce `BATCH_SIZE_IMG` or `IMG_SIZE` |
| Model weights not loading | Ensure `image_model_best.pth` exists in working directory |
| Video codec errors | Install FFmpeg: `pip install imageio imageio-ffmpeg` |

---

## 📚 Dependencies

- **PyTorch**: Deep learning framework
- **OpenCV** (cv2): Image/video processing
- **NumPy**: Numerical computing
- **scikit-learn**: Metrics and evaluation
- **Matplotlib & Seaborn**: Visualization
- **Torchvision**: Image transforms

---

## 📄 File Descriptions

| File | Purpose |
|------|---------|
| `detect.ipynb` | Main training, inference, and visualization notebook |
| `infer.ipynb` | Standalone inference notebook for deployment |
| `image_model_best.pth` | Trained image classifier weights |
| `training_history.json` | Epoch-wise training metrics |
| `test.csv` | Test dataset metadata |
| `train.csv` | Training dataset metadata |
| `valid.csv` | Validation dataset metadata |

---

## 🎓 Learning Outcomes

This project demonstrates:
- ✅ CNN architecture design from scratch
- ✅ LSTM for temporal sequence modeling
- ✅ Binary classification with PyTorch
- ✅ Model evaluation and visualization
- ✅ Data loading and preprocessing
- ✅ GPU acceleration with CUDA
- ✅ Best practices in deep learning

---

## 📞 Support & Contact

For issues or questions:
1. Check the troubleshooting section
2. Review notebook comments for detailed explanations
3. Verify dataset structure and paths
4. Ensure all dependencies are installed

---

## 📜 License

This project is open source and available for educational and research purposes.

---

## 🙏 Acknowledgments

- Dataset structure inspired by common deepfake detection benchmarks
- Model architecture optimized for real-time inference
- Community feedback and PyTorch documentation

---

**Last Updated**: February 23, 2026  
**Model Accuracy**: 92.55%  
**Best Epoch**: 10
