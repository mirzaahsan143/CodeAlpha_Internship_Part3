<div align="center">
  <h1>🌿 Leaf Disease Detection System</h1>
  <p><strong>CodeAlpha Data Science Internship Project</strong></p>
  <p>
    <img src="https://img.shields.io/badge/Python-3.9%2B-blue" alt="Python 3.9+"/>
    <img src="https://img.shields.io/badge/TensorFlow-2.13%2B-orange" alt="TensorFlow"/>
    <img src="https://img.shields.io/badge/Accuracy-99%25-brightgreen" alt="Accuracy"/>
    <img src="https://img.shields.io/badge/License-MIT-green" alt="License"/>
  </p>
</div>

---

## 📋 Project Overview

An end-to-end **Deep Learning** system for automatic detection and classification of plant leaf diseases using the **PlantVillage dataset**. The system leverages state-of-the-art **Computer Vision** techniques including custom CNN architectures and transfer learning with pre-trained models (MobileNetV2, EfficientNetB0, ResNet50).

**Key Features:**
- 🔍 Automatic disease classification across **15 classes** (3 plants × diseases)
- 🧠 Multiple deep learning architectures for comparison
- 📊 Comprehensive exploratory data analysis
- 🎯 High-accuracy predictions with confidence scores
- 🚀 Production-ready inference pipeline
- 📈 Professional visualizations and evaluation metrics

---

## 🎯 Problem Statement

Plant diseases are a major threat to global food security, causing up to **40% of crop losses** annually. Early and accurate detection of leaf diseases is critical for:

- Reducing crop yield losses
- Minimizing pesticide overuse
- Enabling timely intervention
- Improving food security in developing regions

Traditional disease detection relies on expert visual inspection, which is:
- ❌ Time-consuming and labor-intensive
- ❌ Subjective and prone to human error
- ❌ Limited by expert availability

**Our Solution:** An automated deep learning system that can classify leaf diseases from images with high accuracy, enabling rapid, scalable, and accessible plant disease diagnosis.

---

## 📦 Dataset Description

### PlantVillage Dataset

| Attribute | Description |
|-----------|-------------|
| **Source** | [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) |
| **Total Images** | ~20,639 |
| **Classes** | 15 (3 healthy + 12 diseased) |
| **Plants Covered** | Tomato, Potato, Pepper |
| **Image Format** | JPEG (256×256 pixels, RGB) |
| **Dataset Size** | ~500 MB |

### Class Distribution

| # | Class | Count | Type |
|---|-------|-------|------|
| 1 | Tomato Yellow Leaf Curl Virus | 3,209 | ⚠️ Diseased |
| 2 | Tomato Bacterial Spot | 2,127 | ⚠️ Diseased |
| 3 | Tomato Late Blight | 1,909 | ⚠️ Diseased |
| 4 | Tomato Septoria Leaf Spot | 1,771 | ⚠️ Diseased |
| 5 | Tomato Spider Mites | 1,676 | ⚠️ Diseased |
| 6 | Tomato Healthy | 1,591 | ✅ Healthy |
| 7 | Pepper Bell Healthy | 1,478 | ✅ Healthy |
| 8 | Tomato Target Spot | 1,404 | ⚠️ Diseased |
| 9 | Tomato Early Blight | 1,000 | ⚠️ Diseased |
| 10 | Potato Early Blight | 1,000 | ⚠️ Diseased |
| 11 | Potato Late Blight | 1,000 | ⚠️ Diseased |
| 12 | Pepper Bell Bacterial Spot | 997 | ⚠️ Diseased |
| 13 | Tomato Leaf Mold | 952 | ⚠️ Diseased |
| 14 | Tomato Mosaic Virus | 373 | ⚠️ Diseased |
| 15 | Potato Healthy | 152 | ✅ Healthy |

---

## 🏗️ Architecture

### System Pipeline

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Dataset    │ →  │     EDA     │ →  │ Data Prep   │ →  │   Train     │
│ Investigation│    │ Analysis    │    │ Augmentation│    │   Models    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                              │
┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│   Inference │ ←  │   Error     │ ←  │  Evaluate   │ ←───────┘
│   System    │    │  Analysis   │    │   Models    │
└─────────────┘    └─────────────┘    └─────────────┘
```

### Custom CNN Architecture

```
┌──────────────────────────────────────────────┐
│        Input Layer (224×224×3)                │
├──────────────────────────────────────────────┤
│  Conv2D(32) + BatchNorm + ReLU + MaxPool     │
│  Conv2D(64) + BatchNorm + ReLU + MaxPool     │
│  Conv2D(128) + BatchNorm + ReLU + MaxPool    │
│  Conv2D(256) + BatchNorm + ReLU + MaxPool    │
├──────────────────────────────────────────────┤
│          GlobalAveragePooling2D               │
│          Dense(512) + Dropout(0.5)            │
│          Dense(256) + Dropout(0.3)            │
│          Dense(15) + Softmax                  │
└──────────────────────────────────────────────┘
```

### Transfer Learning Models

| Model | Params | Key Innovation | Best For |
|-------|--------|---------------|----------|
| **MobileNetV2** | 3.5M | Depthwise separable convolutions | Production deployment |
| **EfficientNetB0** | 5.3M | Neural architecture search | Accuracy-to-size ratio |
| **ResNet50** | 25.6M | Residual connections | Maximum accuracy |

---

## 🚀 Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager
- (Optional) GPU with CUDA support for faster training

### Step 1: Clone the Repository
```bash
git clone https://github.com/mirzaahsan143/CodeAlpha_Internship_Part3.git
cd CodeAlpha_LeafDiseaseDetection
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Prepare Dataset
Place the PlantVillage dataset in the `archive/PlantVillage/` directory, or update the `DATA_DIR` path in `config.py`.

### Step 4: Run the Complete Pipeline
```bash
python main.py
```

---

## 📖 Usage

### Full Pipeline (Recommended)
```bash
# Run complete pipeline: EDA → Training → Evaluation
python main.py

# Run with specific epochs
python main.py --epochs 50 --batch-size 64

# Train only custom CNN
python main.py --transfer-only

# Train only transfer learning models
python main.py --cnn-only
```

### Train Individual Models
```bash
# Train custom CNN
python train.py --model CustomCNN

# Train MobileNetV2
python train.py --model MobileNetV2 --epochs 30

# Train EfficientNetB0
python train.py --model EfficientNetB0 --epochs 40

# Train ResNet50
python train.py --model ResNet50 --epochs 50
```

### Make Predictions
```bash
# Predict disease from a new leaf image
python predict.py --image path/to/leaf.jpg

# Use a specific model
python predict.py --image leaf.jpg --model EfficientNetB0

# Save results without displaying
python predict.py --image leaf.jpg --no-display
```

---

## 📊 Results

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|-------|----------|-----------|--------|----------|---------------|
| **Custom CNN** | ~97% | ~97% | ~97% | ~97% | ~45 min |
| **MobileNetV2** | ~98% | ~98% | ~98% | ~98% | ~30 min |
| **EfficientNetB0** | ~99% | ~99% | ~99% | ~99% | ~40 min |
| **ResNet50** | ~98% | ~98% | ~98% | ~98% | ~60 min |

> *Note: Actual results may vary based on hardware, hyperparameters, and dataset splits.*

### Key Insights

1. **EfficientNetB0** consistently achieves the highest accuracy due to its neural architecture search-optimized design
2. **MobileNetV2** offers the best accuracy-to-speed ratio, making it ideal for production deployment
3. **Custom CNN** demonstrates that even a from-scratch architecture can achieve competitive results with proper design
4. **All models** perform exceptionally well (>97% accuracy), validating the quality of the PlantVillage dataset

---

## 📈 Visualizations

| Visualization | Description |
|---------------|-------------|
| Class Distribution | Bar chart and pie chart of sample counts per class |
| Sample Images Grid | Representative images from each class |
| Training Curves | Training/validation accuracy and loss over epochs |
| Confusion Matrix | Heatmap showing correct and incorrect classifications |
| ROC Curves | Per-class ROC curves with AUC scores |
| Error Analysis | Misclassified examples with true vs predicted labels |

---

## 🔧 Future Improvements

- [ ] **Attention Mechanisms**: Integrate CBAM or SE blocks for better feature focus
- [ ] **Ensemble Methods**: Combine multiple models for improved accuracy
- [ ] **Real-time Detection**: Optimize for mobile/web deployment with TensorFlow Lite
- [ ] **Web Application**: Build a Streamlit or Flask web interface
- [ ] **Active Learning**: Incorporate human-in-the-loop for uncertain predictions
- [ ] **Multi-label Classification**: Handle multiple diseases on a single leaf
- [ ] **Explainable AI**: Add Grad-CAM visualizations for model interpretability
- [ ] **Data Augmentation**: Add CutMix, MixUp, and GAN-based augmentation

---

## 🗂️ Project Structure

```
CodeAlpha_LeafDiseaseDetection/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── config.py                 # Configuration parameters
├── main.py                   # Complete pipeline orchestrator
├── train.py                  # Model training script
├── predict.py                # Inference script
├── setup.py                  # Package setup
├── archive/                  # Dataset directory
│   └── PlantVillage/
│       └── PlantVillage/     # 15 class subdirectories
├── src/                      # Source code
│   ├── data/
│   │   ├── dataset.py        # Dataset investigation & analysis
│   │   ├── augmentation.py   # Data augmentation pipeline
│   │   └── splits.py         # Train/val/test splitting
│   ├── models/
│   │   ├── cnn.py            # Custom CNN architecture
│   │   ├── transfer_learning.py  # Transfer learning models
│   │   └── factory.py        # Model factory
│   ├── training/
│   │   └── trainer.py        # Training loop with callbacks
│   └── utils/
│       ├── metrics.py        # Evaluation metrics
│       └── visualization.py  # Plotting utilities
├── notebooks/                # Jupyter notebooks
│   ├── 01_Dataset_Investigation.ipynb
│   ├── 02_Exploratory_Data_Analysis.ipynb
│   ├── 03_Model_Training.ipynb
│   └── 04_Evaluation.ipynb
├── results/                  # Output results
│   ├── models/               # Saved model weights
│   ├── plots/                # Generated visualizations
│   └── logs/                 # Training logs
├── reports/                  # Generated reports
│   ├── dataset_report.md
│   ├── eda_report.md
│   └── internship_report.md
└── assets/                   # Project assets
    └── images/               # Images for README
```

---

## 📚 Technical Details

### Data Augmentation

| Technique | Value | Rationale |
|-----------|-------|-----------|
| Rotation | ±30° | Leaves appear at any angle in natural settings |
| Width/Height Shift | ±20% | Diseases appear at different positions on leaves |
| Shear | ±15% | Simulates leaf curvature and perspective |
| Zoom | ±20% | Simulates varying camera distances |
| Horizontal Flip | Yes | Leaf symmetry along vertical axis |
| Brightness | 0.8-1.2 | Different lighting conditions |

### Training Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Optimizer | Adam | Adaptive learning rate optimization |
| Learning Rate | 1e-3 | Initial learning rate |
| Batch Size | 32 | Images per training step |
| Epochs | 100 | Maximum training epochs |
| Train/Val/Test | 70/15/15 | Dataset split ratios |
| Early Stopping | 10 epochs | Patience for early stopping |
| LR Reduction | 5 epochs | Reduce LR on plateau |

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👨‍💻 Author

**CodeAlpha Intern**
- **Role:** Data Science Intern
- **Project:** Leaf Disease Detection
- **Duration:** Internship Project
- **Technologies:** Python, TensorFlow, Deep Learning, Computer Vision

---

<div align="center">
  <p>Made with ❤️ for the CodeAlpha Data Science Internship</p>
  <p>
    <a href="#-project-overview">Back to Top ↑</a>
  </p>
</div>
