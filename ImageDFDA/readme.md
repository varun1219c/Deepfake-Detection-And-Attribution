# Deepfake Image Detection Project

A comprehensive deep learning project for detecting deepfake images using XceptionNet architecture with an interactive Streamlit web interface.

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/33635489-ee7e-404e-b19e-63c21def2464" />

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/59123d8c-6b2e-4104-8add-48e7f6217dcc" />

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/69ab88b0-0e1e-4564-a315-45a193538f2c" />

<img width="1454" height="562" alt="image" src="https://github.com/user-attachments/assets/a10afa90-47c4-4396-82c4-4efd0985f992" />

<img width="844" height="606" alt="image" src="https://github.com/user-attachments/assets/af284b16-2567-4f42-acf1-f2ca41dcea87" />


## 🎯 Overview

This project implements a state-of-the-art deepfake detection system using:
- **XceptionNet** with attention mechanism
- Advanced data augmentation
- Comprehensive evaluation metrics
- Interactive Streamlit web interface for real-time predictions

## ✨ Features

- 🧠 **Advanced Model**: XceptionNet with attention mechanism
- 📊 **Comprehensive Metrics**: ROC curves, confusion matrices, precision-recall curves
- 🎨 **Data Augmentation**: Random flips, rotations, color jitter
- 📈 **Training Visualization**: Real-time training plots and performance dashboards
- 🌐 **Web Interface**: User-friendly Streamlit app for predictions
- ⚡ **Mixed Precision Training**: Faster training with AMP
- 🔄 **Early Stopping**: Automatic training termination
- 📉 **Learning Rate Scheduling**: Adaptive learning rate adjustment

## 📦 Dataset

This project uses the **Labeled Deepfake Image Collection** dataset from Kaggle.

### Dataset Download Instructions

**Method 1: Using KaggleHub (Recommended)**

1. **Install KaggleHub**:
   ```bash
   pip install kagglehub
   ```

2. **Download the dataset using Python**:
   ```python
   import kagglehub
   
   # Download latest version
   path = kagglehub.dataset_download("jayanthbottu/labeled-deepfake-image-collection")
   print("Path to dataset files:", path)
   ```

3. **Copy/Move dataset to project directory**:
   - The dataset will be downloaded to your kagglehub cache
   - Copy the files to your project's `ImagesDF/` directory
   - Or update `DATA_DIR` in `train.py` to point to the downloaded path

**Method 2: Using Kaggle API**

1. **Install Kaggle API**:
   ```bash
   pip install kaggle
   ```

2. **Set up Kaggle credentials**:
   - Visit [Kaggle Account Settings](https://www.kaggle.com/account)
   - Click "Create New API Token" in the API section
   - Place `kaggle.json` in:
     - **Linux/Mac**: `~/.kaggle/kaggle.json`
     - **Windows**: `C:\Users\<YourUsername>\.kaggle\kaggle.json`
   - Set permissions (Linux/Mac): `chmod 600 ~/.kaggle/kaggle.json`

3. **Download the dataset**:
   ```bash
   kaggle datasets download -d jayanthbottu/labeled-deepfake-image-collection
   ```

4. **Extract**:
   ```bash
   unzip labeled-deepfake-image-collection.zip -d ImagesDF/
   ```

### Expected Dataset Structure

```
ImagesDF/
├── DALL-E/
├── DeepFaceLab/
├── Face2Face/
├── FaceShifter/
├── FaceSwap/
├── Midjourney/
├── NeuralTextures/
├── Real/
├── Stable Diffusion/
└── StyleGAN/
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM recommended

### Setup Steps

1. **Clone or create project directory**:
   ```bash
   mkdir deepfake-detection
   cd deepfake-detection
   ```

2. **Create virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # Activate virtual environment
   # Windows:
   venv\Scripts\activate
   # Linux/Mac:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 📖 Usage

### Training the Model

Train the deepfake detection model:

```bash
python train.py
```

**Training outputs**:
- Model checkpoint: `saved_models/best_model_TIMESTAMP.pt`
- Training reports: `training_reports/TIMESTAMP/`
- Visualizations: confusion matrices, ROC curves, training history plots
- Metrics: JSON and CSV files with detailed performance metrics

**Training parameters** (edit in `train.py`):
- `NUM_EPOCHS`: 20 (default)
- `BATCH_SIZE`: 16 (default)
- `LEARNING_RATE`: 0.0001 (default)
- `IMAGE_SIZE`: 299 (XceptionNet requirement)

### Running the Streamlit App

Launch the interactive web interface:

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

**Features**:
- Upload images (PNG, JPG, JPEG)
- Real-time deepfake detection
- Confidence scores for predictions
- Visual feedback with prediction results

### Using the Web Interface

1. **Upload Image**: Click "Browse files" or drag & drop
2. **Analyze**: Click the "Analyze Image" button
3. **View Results**: See prediction (Real/Fake) with confidence score
4. **Try Another**: Upload a new image to test

## 📁 Project Structure

```
deepfake-detection/
│
├── ImagesDF/                      # Dataset directory
│   ├── DALL-E/
│   ├── DeepFaceLab/
│   ├── Face2Face/
│   ├── FaceShifter/
│   ├── FaceSwap/
│   ├── Midjourney/
│   ├── NeuralTextures/
│   ├── Real/
│   ├── Stable Diffusion/
│   └── StyleGAN/                    # Real images
│
├── saved_models/                  # Trained model checkpoints
│   └── best_model_TIMESTAMP.pt
│
├── training_reports/              # Training outputs
│   └── TIMESTAMP/
│       ├── plots/                 # Visualizations
│       │   ├── confusion_matrices.png
│       │   ├── roc_curve.png
│       │   ├── training_history_comprehensive.png
│       │   └── performance_dashboard.png
│       └── metrics/               # Performance metrics
│           ├── classification_report.csv
│           ├── detailed_metrics.json
│           └── training_summary.json
│
├── train.py                       # Model training script
├── app.py                         # Streamlit web interface
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🏗️ Model Architecture

**EnhancedXceptionNet** with:
- Pre-trained XceptionNet base (ImageNet weights)
- Custom attention mechanism
- Advanced classifier head:
  - Dropout layers (50% default)
  - Batch normalization
  - Two fully connected layers (2048 → 512 → num_classes)

**Training Features**:
- AdamW optimizer with weight decay
- Label smoothing (0.1)
- Mixed precision training (AMP)
- Gradient clipping
- ReduceLROnPlateau scheduler
- Early stopping

## 📊 Results

After training, you'll find:

### Visualizations
- **Confusion Matrices**: Raw counts and normalized percentages
- **ROC Curves**: Area Under Curve (AUC) scores
- **Precision-Recall Curves**: Average Precision scores
- **Training History**: Loss, accuracy, F1-score, precision, recall
- **Performance Dashboard**: Comprehensive training overview

### Metrics
- Classification report (precision, recall, F1-score per class)
- Overall accuracy
- Weighted and macro-averaged metrics
- Per-class performance analysis

## ⚙️ Configuration

Key hyperparameters in `train.py`:

```python
NUM_EPOCHS = 20
LEARNING_RATE = 0.0001
BATCH_SIZE = 16
IMAGE_SIZE = 299
DROPOUT_RATE = 0.5
EARLY_STOPPING_PATIENCE = 5
REDUCE_LR_PATIENCE = 3
```

Data splits:
- Training: 70%
- Validation: 15%
- Test: 15%

## 🔧 Troubleshooting

**CUDA Out of Memory**:
- Reduce `BATCH_SIZE` in `train.py`
- Reduce `NUM_WORKERS`

**Dataset Not Found**:
- Ensure `DATA_DIR` path in `train.py` matches your dataset location
- Verify dataset structure has `Fake/` and `Real/` subdirectories

**Model Not Found in Streamlit**:
- Train the model first using `train.py`
- Update model path in `app.py` to point to your trained model

**Slow Training**:
- Enable CUDA if available
- Increase `NUM_WORKERS` (but not more than CPU cores)
- Use mixed precision training (enabled by default)

## 📝 Requirements

See `requirements.txt` for complete list of dependencies.

Main packages:
- PyTorch
- torchvision
- timm (PyTorch Image Models)
- Streamlit
- scikit-learn
- matplotlib, seaborn
- pandas, numpy
- Pillow
- kagglehub

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional model architectures
- Enhanced data augmentation techniques
- Video deepfake detection
- Model ensemble methods
- Deployment optimization

## 📄 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

- Dataset: [Labeled Deepfake Image Collection](https://www.kaggle.com/datasets/jayanthbottu/labeled-deepfake-image-collection) by Jayanth Bottu
- XceptionNet: François Chollet
- PyTorch Team
- Streamlit Team

## 📧 Contact

For questions or issues, please open an issue in the repository.

---

**Note**: This is a research/educational project. For production use, additional validation and testing are required.
