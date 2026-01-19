import streamlit as st
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import timm
import cv2
import numpy as np
import tempfile
from pathlib import Path
import matplotlib.pyplot as plt

CLASS_NAMES = ["Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures", "Original"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "VIDEO_DFDA/Results/best_model.pt"  

st.set_page_config(page_title="🎥 Deepfake Video Detection", layout="wide")
st.title("🎥 Deepfake Video Detection System")
st.markdown("Upload a video to detect if it’s **synthetic or original** using your trained Xception model.")

@st.cache_resource
def load_model(num_classes=len(CLASS_NAMES)):
    model = timm.create_model("legacy_xception", pretrained=False, num_classes=num_classes)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

    model_dict = model.state_dict()
    filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict, strict=False)

    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

def extract_frames(video_path, frame_skip=10):
    """Extract frames every N frames from video."""
    frames = []
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_skip == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        frame_count += 1
    cap.release()
    return frames

def preprocess_frame(frame):
    """Resize and normalise frame for model input."""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(frame).unsqueeze(0)

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    st.video(uploaded_file)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
        tmpfile.write(uploaded_file.read())
        video_path = tmpfile.name

    st.info("Extracting frames... ⏳")
    frames = extract_frames(video_path, frame_skip=10)
    st.success(f"✅ Extracted {len(frames)} frames.")

    st.info("Classifying video... please wait ⚙️")
    progress = st.progress(0)
    all_probs = []

    for idx, frame in enumerate(frames):
        tensor = preprocess_frame(frame).to(DEVICE)
        with torch.no_grad():
            output = model(tensor)
            probs = F.softmax(output, dim=1).cpu().numpy()[0]
            all_probs.append(probs)
        progress.progress((idx + 1) / len(frames))

    avg_probs = np.mean(all_probs, axis=0)
    pred_class = CLASS_NAMES[np.argmax(avg_probs)]
    conf_score = np.max(avg_probs) * 100

    st.markdown("---")
    st.success(f"### 🎯 Predicted Class: **{pred_class}** ({conf_score:.2f}% confidence)")

    st.markdown("#### 📊 Confidence Scores per Class")
    conf_data = {CLASS_NAMES[i]: float(avg_probs[i]) for i in range(len(CLASS_NAMES))}
    st.bar_chart(conf_data)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(avg_probs, labels=CLASS_NAMES, autopct='%1.1f%%', startangle=140)
    ax.set_title("Class Probability Distribution")
    st.pyplot(fig)

    st.markdown("---")
    st.markdown("**Developed by Jayanth** | Deepfake Video Detection | PyTorch + Streamlit")

else:
    st.info("👆 Please upload a video file to start detection.")