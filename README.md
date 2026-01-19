# 🌐 DeepFake Detection and Attribution (DFDA)

> **A Unified AI System for Multi-Modal DeepFake Detection and Source Attribution**
>
> 🧠 *Detects, classifies, and visualises synthetic content across Audio, Image, and Video modalities using advanced deep learning models.*

---
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/63328f54-e17d-4b59-812b-6adcf763eae5" />

## 🚀 Overview

**DeepFake Detection and Attribution (DFDA)** is an **AI-powered media forensics platform** that performs **multi-modal deepfake analysis** across:
- 🎧 **Audio** – Voice cloning, text-to-speech, and conversion detection  
- 🖼️ **Image** – AI-generated or manipulated visual content  
- 🎥 **Video** – Frame-based deepfake and facial reenactment analysis  

Developed using **PyTorch**, **Streamlit**, and **TIMM**, DFDA integrates custom CNNs, XceptionNet, and attention-based networks into one unified analytical system.

---

## 🧩 Core Capabilities

| Modality | Model | Techniques Detected | Benchmark Accuracy |
|:--|:--|:--|:--|
| **Audio** | Custom 4-layer CNN on Mel-spectrograms | Voice cloning, TTS, voice conversion | **94.7%** |
| **Image** | Enhanced XceptionNet + Attention | DALL-E, Midjourney, FaceSwap, Real | **96.2%** |
| **Video** | Legacy Xception + Temporal aggregation | Deepfakes, Face2Face, NeuralTextures | **93.8%** |

---

## ⚙️ System Architecture

```
    A[User Uploads Media] --> B{Select Modality}
    B -->|Audio| C1[Mel-Spectrogram Extraction]
    B -->|Image| C2[Image Preprocessing]
    B -->|Video| C3[Frame Extraction]
    C1 --> D1[Audio CNN Model]
    C2 --> D2[XceptionNet + Attention]
    C3 --> D3[Video Xception Model]
    D1 --> E[Softmax Prediction + Attribution]
    D2 --> E
    D3 --> E
    E --> F[Confidence Visualisation (Plotly + Streamlit)]
    F --> G[Detailed Report Export]
````

---
https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe
## 🧠 Algorithmic Highlights

### 🎧 Audio DFDA

* **Feature Extraction:** Converts audio to **Mel-Spectrograms (128 bins)**
* **Model:** 4-layer CNN with BatchNorm + Dropout
* **Loss:** CrossEntropy with Label Smoothing
* **Inference:** Confidence-based classification and spectrogram visualisation
<img width="4520" height="5942" alt="localhost_8501_ (13)" src="https://github.com/user-attachments/assets/0ffe8e9b-0bdb-4a03-94cb-3d2414544992" />


### 🖼️ Image DFDA

* **Model:** Enhanced **XceptionNet** with integrated **Attention Block**
* **Features:** Pretrained on ImageNet, refined with Dropout and BatchNorm
* **Inference:** Frame-level classification with attribution
<img width="4520" height="6074" alt="localhost_8501_ (12)" src="https://github.com/user-attachments/assets/5c0c94d8-d27c-4539-8ab3-b34806699f20" />

### 🎥 Video DFDA

* **Frame Analysis:** Extracts frames every *N* frames using OpenCV
* **Model:** Legacy Xception
* **Prediction Fusion:** Temporal average of frame-level softmax outputs
* **Outputs:** Confidence per class, frame count, ROC visualisation
<img width="4520" height="7166" alt="localhost_8501_ (11)" src="https://github.com/user-attachments/assets/c8477f63-9052-45b3-8676-5acc03a52481" />

---

## 📦 Folder Structure

```
DFDA/
│
├── app.py                         # Integrated Streamlit Application (All Modalities)
├── requirements.txt                # Lightweight dependency list
├── Environment.txt                 # Full environment spec (GPU + CUDA)
│
├── Audio.pth                       # Trained Audio DeepFake Model
├── Image.pt                        # Trained Image DeepFake Model
├── Video.pt                        # Trained Video DeepFake Model
│
├── Datasets/
│   ├── audio_dfda/                # Audio DeepFake dataset
│   ├── image_dfda/                # Labeled DeepFake Image Collection
│   └── video_dfda/                # FF++ C23 Frames Dataset
│
├── Results/
│   ├── metrics/                   # Metrics, classification reports, ROC curves
│   └── plots/                     # Accuracy/Loss plots and confusion matrices
│
└── README.md
```

---

## 📚 Datasets Used

| Modality  | Source                                | Classes | Dataset Link                                                                                            |
| :-------- | :------------------------------------ | :------ | :------------------------------------------------------------------------------------------------------ |
| **Audio** | Custom Audio DeepFake Dataset         | 5       | [Kaggle: Audio DeepFake](https://www.kaggle.com/datasets/jayanthbottu/audio-deepfake)                   |
| **Image** | Labeled DeepFake Image Collection     | 10      | [Kaggle: Image DFDA](https://www.kaggle.com/datasets/jayanthbottu/labeled-deepfake-image-collection)    |
| **Video** | FaceForensics++ (C23, FF++C32-Frames) | 6       | [Kaggle: Video DFDA](https://www.kaggle.com/datasets/fatimahirshad/faceforensics-extracted-dataset-c23) |

---

## 🧰 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/jayanthbottu/DFDA.git
cd DFDA
```

### 2️⃣ Create a Virtual Environment

```bash
python -m venv dfdavenv
source dfdavenv/bin/activate    # (Linux/Mac)
dfdavenv\Scripts\activate       # (Windows)
```

### 3️⃣ Install Dependencies

GPU-optimised install:

```bash
pip install -r requirements.txt
```

Full environment (for NVIDIA RTX GPUs):

```bash
pip install -r Environment.txt
```

### 4️⃣ Launch the Application

```bash
streamlit run app.py
```

Then open your browser at → [http://localhost:8501](http://localhost:8501)

---

## 🔍 Usage Guide

### 🏠 **Home**

Displays overview, architecture, team, and technical specifications.

### 🎧 **Audio Detection**

* Upload `.wav`, `.mp3`, `.ogg`, `.flac`, or `.m4a`
* Detects: *Voice Conversion*, *TTS*, *ElevenLabs*, *Tacotron*, *Original*
* Displays Mel-spectrogram, confidence scores, and probabilities.

### 🖼️ **Image Detection**

* Upload `.jpg`, `.png`, `.bmp`, or `.webp`
* Detects: *DALL-E*, *Midjourney*, *StyleGAN*, *FaceSwap*, *Real*
* Visualises probability bars and attention confidence matrix.

### 🎥 **Video Detection**

* Upload `.mp4`, `.avi`, `.mov`, `.mkv`, or `.webm`
* Extracts every Nth frame for prediction.
* Aggregates frame-level probabilities → final video classification.

---

## 📊 Example Results

| Modality | Sample                    | Prediction                 | Confidence |
| :------- | :------------------------ | :------------------------- | :--------- |
| Audio    | Voice clone (ElevenLabs)  | **Synthetic - ElevenLabs** | 96.4%      |
| Image    | DALL-E Generated Portrait | **Synthetic - DALL-E**     | 98.2%      |
| Video    | Face2Face Reenactment     | **Fake - Face2Face**       | 90.1%      |

---

## 🧱 Model Architecture Summary

| Model                   | Parameters | Input                   | Highlights                           |
| :---------------------- | :--------- | :---------------------- | :----------------------------------- |
| AudioDeepfakeCNN        | ~2.1M      | Mel-Spectrogram (128xN) | Multi-conv, dropout regularised      |
| EnhancedXceptionNet     | ~22.8M     | RGB (299x299)           | Attention-enhanced Xception          |
| Legacy Xception (Video) | ~22.9M     | Frames (300x300)        | Temporal consistency, softmax fusion |

---

## 📈 Performance Metrics

| Dataset    | Accuracy | F1-Score | Loss |
| :--------- | -------: | -------: | ---: |
| Audio DFDA |    94.7% |    0.948 | 0.26 |
| Image DFDA |    96.2% |    0.962 | 0.23 |
| Video DFDA |    90.0% |     0.89 | 0.42 |

📊 Visual Reports:

* Confusion Matrices
* ROC Curves
* Precision/Recall Heatmaps
* Epoch-wise Accuracy/Loss Plots

---

## 🎨 Streamlit UI Highlights

* Modern gradient interface with **custom CSS styling**
* Interactive **Plotly visualisations** for class probabilities
* Real-time GPU utilization display
* Configurable **frame skip** for video analysis
* Downloadable result summaries

---

## ⚙️ Configuration

| Parameter                 | Default    | Description                |
| :------------------------ | :--------- | :------------------------- |
| `AudioConfig.SAMPLE_RATE` | 16000 Hz   | Target sample rate         |
| `ImageConfig.IMG_SIZE`    | 299 px     | Input size for XceptionNet |
| `VideoConfig.FRAME_SKIP`  | 10         | Frame extraction interval  |
| `MAX_AUDIO_LENGTH`        | 5 sec      | Audio clip length          |
| `DEVICE`                  | CUDA / CPU | Auto-detected              |

---

## 🧰 Troubleshooting

| Issue                     | Cause                         | Solution                             |
| :------------------------ | :---------------------------- | :----------------------------------- |
| `Model file not found`    | Missing `.pt` or `.pth` files | Place trained models in project root |
| `CUDA out of memory`      | GPU RAM insufficient          | Reduce `batch_size` or frame count   |
| `Streamlit not launching` | Port conflict                 | Run with `--server.port 8502`        |
| Slow analysis             | Large video files             | Increase `FRAME_SKIP`                |

---

## 👩‍💻 Team & Contributions

| Name                   | Roll No    | Role                                |
| :--------------------- | :--------- | :---------------------------------- |
| **Jayanth Bottu**      | 2303A51LA7 | System Architect & Integration Lead |
| **Sri Varsha Janagam** | 2303A51L99 | Model Development (Audio)           |
| **Sindhu Kodati**      | 2303A51LA0 | UI Design & Testing                 |
| **Mahendra Gaddam**    | 2303A51LA9 | Model Training (Image)              |
| **Nikhil Kuchana**     | 2303A51LB0 | Data Processing & Video DFDA        |

---

## 📜 License

This project is released under the **MIT License**.
You may freely use, modify, and distribute this work with proper attribution.

---

## 🙏 Acknowledgements

* **Datasets:**

  * [Audio DeepFake Dataset](https://www.kaggle.com/datasets/jayanthbottu/audio-deepfake)
  * [Labeled DeepFake Image Collection](https://www.kaggle.com/datasets/jayanthbottu/labeled-deepfake-image-collection)
  * [FaceForensics++ C23 Dataset](https://www.kaggle.com/datasets/fatimahirshad/faceforensics-extracted-dataset-c23)

* **Frameworks:** PyTorch • Streamlit • TIMM • Librosa • OpenCV • Plotly

* **Hardware:** NVIDIA RTX 3050 6 GB GPU (CUDA 11.8)

---

## 📞 Contact

**Author:** [Jayanth Bottu](https://www.linkedin.com/in/jayanthbottu/)

---

## ⚠️ Note

> ⚙️ **This is a research and educational project.**
> For production deployment, perform extended validation, adversarial robustness testing, and real-world dataset calibration.

---

## 🌟 Project Snapshot

| Modality  | Example UI                                      | Description                             |
| :-------- | :---------------------------------------------- | :-------------------------------------- |
| 🎧 Audio  | 🎵 Waveform & Mel-spectrogram visualisation     | Detects TTS & cloned voices             |
| 🖼️ Image | 🖼️ Confidence bar and matrix view              | Identifies AI-generated art & deepfakes |
| 🎥 Video  | 🎞️ Frame-wise classification with progress bar | Analyses temporal manipulation          |

---

<p align="center">
  <b>🔹 DFDA — Advancing Trust in Digital Media 🔹</b><br>
  <i>“Truth is the new frontier of Artificial Intelligence.”</i>
</p>
