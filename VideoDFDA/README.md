# 🎥 Video DeepFake Detection and Attribution (Video-DFDA)

> **A deep learning system for detecting and classifying AI-generated fake videos (Deepfakes) using XceptionNet architecture on the FaceForensics++ C23 dataset.**

---

## 🧠 Algorithm Overview

The **Video DFDA** pipeline leverages a **frame-based CNN classification** approach to detect fake videos.  
By extracting frames from videos and analysing them individually through a deep neural network (XceptionNet), the system identifies synthetic manipulations such as **FaceSwap**, **Face2Face**, and **NeuralTextures**.

### Key Steps:
1. **Frame Extraction** – Sample frames from video files at equal intervals.  
2. **Feature Learning** – Feed extracted frames into a CNN (Xception) pre-trained on ImageNet.  
3. **Prediction Aggregation** – Average frame-wise predictions to determine the overall video authenticity.  
4. **Attribution** – Classify the specific deepfake generation technique.

---

## 🔁 Flow Chart

```
    A[Input Video File] --> B[Extract Frames (Every Nth Frame)]
    B --> C[Frame Preprocessing (Resize, Normalize)]
    C --> D[XceptionNet Feature Extraction]
    D --> E[Softmax Classification]
    E --> F[Aggregate Frame Probabilities]
    F --> G[Final Prediction (Real / Fake Type)]
    G --> H[Streamlit App Visualization]
````

---
<img width="1994" height="3289" alt="localhost_8501_ (3)" src="https://github.com/user-attachments/assets/2e55ec13-a8ba-45d2-8c32-a4e7f89b8451" />

## 📂 Folder Structure

```
Video-DFDA/
│
├── app.py                         # Streamlit-based video inference app
├── train.py                       # Model training and evaluation script
├── dataset_index.csv              # Pre-generated frame index file
├── VideoOut.txt                   # Training logs
├── VIDEO_DFDA/
│   └── Results/
│       ├── best_model.pt
│       ├── metrics_log.csv
│       ├── confusion_matrix.png
│       ├── accuracy_curve.png
│       ├── loss_curve.png
│       ├── classification_report.txt
│       ├── prec_recall_f1.png
│       └── roc_curves.png
│
├── Dataset/
│   └── FF++C32-Frames/            # Dataset (from Kaggle)
│       ├── Deepfakes/
│       ├── Face2Face/
│       ├── FaceShifter/
│       ├── FaceSwap/
│       ├── NeuralTextures/
│       └── Original/
│
└── README.md
```

---

## ✨ Features

* 🎞 **Frame-based DeepFake Detection**
* 🧠 **XceptionNet architecture (from TIMM)**
* 📊 **Per-class precision, recall, and F1 visualization**
* 📈 **ROC curves, confusion matrices, and accuracy plots**
* 🌐 **Streamlit interface for interactive analysis**
* ⚙️ **Automatic model checkpointing and training reports**

---

## 📘 Dataset

**Dataset Used:** [FaceForensics++ Extracted Dataset (C23 Quality)](https://www.kaggle.com/datasets/fatimahirshad/faceforensics-extracted-dataset-c23)

### Dataset Description

| Attribute    | Details                                                                               |
| :----------- | :------------------------------------------------------------------------------------ |
| Source       | FaceForensics++ C23 subset                                                            |
| Type         | Frame-level deepfake images                                                           |
| Classes      | 6 (`Deepfakes`, `Face2Face`, `FaceShifter`, `FaceSwap`, `NeuralTextures`, `Original`) |
| Format       | JPG/PNG                                                                               |
| Total Frames | ~20,000+                                                                              |
| Resolution   | 256×256 – 512×512                                                                     |

---

## 🧩 Adding the Dataset

1. **Install Kaggle CLI**

   ```bash
   pip install kaggle
   ```

2. **Authenticate**

   * Download `kaggle.json` from Kaggle → Account → API → Create New Token
   * Place in:

     ```
     ~/.kaggle/kaggle.json
     ```
   * Then run:

     ```bash
     chmod 600 ~/.kaggle/kaggle.json
     ```

3. **Download the Dataset**

   ```bash
   kaggle datasets download -d fatimahirshad/faceforensics-extracted-dataset-c23 -p ./Dataset --unzip
   ```

4. **Expected Folder Layout**

   ```
   Dataset/
   └── FF++C32-Frames/
       ├── Deepfakes/
       ├── Face2Face/
       ├── FaceShifter/
       ├── FaceSwap/
       ├── NeuralTextures/
       └── Original/
   ```

---

## ⚙️ Installing Requirements

```bash
pip install torch torchvision timm opencv-python numpy pandas matplotlib seaborn scikit-learn streamlit
```

Or install from `requirements.txt` (if provided).

---

## 🏋️ Training the Model

Run the training script:

```bash
python train.py
```

### Training Pipeline

1. **Create Frame Index CSV** (`dataset_index.csv`)
   Automatically scans `Dataset/FF++C32-Frames` and assigns labels.
   *(Auto-created on first run.)*

2. **Train/Test Split**

   * 80% training, 20% validation.

3. **Model Architecture**

   * **XceptionNet** (from `timm`)
   * Input Size: 300×300
   * Batch Size: 16
   * Optimizer: Adam
   * Scheduler: ReduceLROnPlateau

4. **Metrics Tracked**

   * Training/Validation Loss
   * Accuracy
   * Confusion Matrix
   * ROC Curves
   * Precision/Recall/F1 per class

---

## 📊 Training Metrics (From `VideoOut.txt`)

| Metric        |          Training | Validation |
| :------------ | ----------------: | ---------: |
| Best Accuracy |             99.6% | **90.00%** |
| Loss (final)  |             0.011 |     0.4228 |
| Epochs        |                15 |            |
| Model         | `legacy_xception` |            |
| Time          |           219 min |            |

---

## 🧠 Model Architecture

```
Input: 300x300 RGB Frame
│
├── XceptionNet (Pretrained on ImageNet)
│   ├── Depthwise Separable Convolutions
│   ├── Residual Connections
│   └── Global Average Pooling
│
└── Fully Connected Layer (6 Classes)
    → Softmax Output
```

**Classes:**

* Deepfakes
* Face2Face
* FaceShifter
* FaceSwap
* NeuralTextures
* Original

---

## 🧾 Results

| Class          | Precision | Recall |   F1 |
| :------------- | --------: | -----: | ---: |
| Deepfakes      |      0.89 |   0.88 | 0.88 |
| Face2Face      |      0.90 |   0.87 | 0.88 |
| FaceShifter    |      0.92 |   0.91 | 0.91 |
| FaceSwap       |      0.88 |   0.89 | 0.88 |
| NeuralTextures |      0.87 |   0.86 | 0.86 |
| Original       |      0.99 |   1.00 | 0.99 |

**Validation Accuracy:** 90.0%
**Weighted F1:** 0.89

---

## 🚀 Using Streamlit for Video DFDA

Launch the video analysis dashboard:

```bash
streamlit run app.py
```

### Features:

* 🎞 Upload any `.mp4`, `.avi`, or `.mkv` file.
* ⚙️ Automatically extracts frames every 10th frame.
* 📊 Displays class probabilities as a bar chart and pie chart.
* 🎯 Shows final predicted label with confidence score.
* 💾 Downloadable analysis report.

Example output:

```
🎯 Predicted Class: Deepfakes (91.8% confidence)
```

Local Access: [http://localhost:8501](http://localhost:8501)

---

## 🧾 Project Structure (Detailed)

```
Video-DFDA/
│
├── train.py                   # Training and evaluation logic
├── app.py                     # Streamlit interface
├── dataset_index.csv          # Frame metadata
├── VideoOut.txt               # Training output
├── VIDEO_DFDA/
│   └── Results/
│       ├── best_model.pt
│       ├── loss_curve.png
│       ├── accuracy_curve.png
│       ├── confusion_matrix.png
│       ├── classification_report.txt
│       ├── prec_recall_f1.png
│       ├── roc_curves.png
│       └── run_metadata.csv
│
├── Dataset/
│   └── FF++C32-Frames/
│       ├── Deepfakes/
│       ├── Face2Face/
│       ├── FaceShifter/
│       ├── FaceSwap/
│       ├── NeuralTextures/
│       └── Original/
```

---

## ⚙️ Configuration

| Parameter     |                            Default | Description             |
| :------------ | ---------------------------------: | :---------------------- |
| `IMAGE_SIZE`  |                                300 | Frame resize dimension  |
| `BATCH_SIZE`  |                                 16 | Batch size for training |
| `NUM_EPOCHS`  |                                 15 | Total epochs            |
| `LR`          |                               1e-4 | Learning rate           |
| `NUM_WORKERS` |                                  4 | Dataloader threads      |
| `MODEL_PATH`  | `VIDEO_DFDA/Results/best_model.pt` | Streamlit model path    |

---

## 🧰 Troubleshooting

| Issue                                  | Cause                | Fix                                  |
| :------------------------------------- | :------------------- | :----------------------------------- |
| `FileNotFoundError: dataset_index.csv` | Missing CSV          | Run `train.py` once to auto-generate |
| CUDA memory error                      | Low GPU memory       | Reduce `BATCH_SIZE`                  |
| Streamlit model mismatch               | Incorrect model path | Update `MODEL_PATH` in `app.py`      |
| Slow frame extraction                  | Large video          | Increase `frame_skip` parameter      |

---

## 🤝 Contributing

1. Fork the repository
2. Create a branch: `git checkout -b feature/video-updates`
3. Commit changes: `git commit -m "Improve video frame handling"`
4. Push: `git push origin feature/video-updates`
5. Open a Pull Request

---

## 📜 License

Released under the **MIT License**.
Free for research and educational use with proper attribution.

---

## 🙏 Acknowledgements

* **Dataset:** [FaceForensics++ Extracted Dataset C23](https://www.kaggle.com/datasets/fatimahirshad/faceforensics-extracted-dataset-c23)
* **Frameworks:** PyTorch, Streamlit, OpenCV, TIMM, Scikit-Learn
* **Hardware:** NVIDIA RTX 3050 6GB GPU
* **Author:** [Jayanth Bottu](https://www.linkedin.com/in/jayanthbottu/)

---

## 📞 Contact

**Author:** Jayanth Bottu
🔗 LinkedIn: [linkedin.com/in/jayanthbottu](https://www.linkedin.com/in/jayanthbottu/)

---

## ⚠️ Note

> **This is a research and educational project.**
> For real-world or production deployment, additional robustness testing, bias evaluation, and multi-environment validation are required.
