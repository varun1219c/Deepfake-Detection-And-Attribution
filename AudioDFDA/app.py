import streamlit as st
import torch
import torchaudio
import torchaudio.transforms as T
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from inference import Config, AudioDeepfakeCNN  

st.set_page_config(page_title="Audio Deepfake Detector", layout="wide")
st.title("🎧 Audio Deepfake Detection System")
st.markdown("Upload an audio file to detect if it’s **synthetic or original** using your trained CNN model.")

@st.cache_resource
def load_model():
    model_path = "training_results\\models\\best_model.pth"
    model = AudioDeepfakeCNN(num_classes=len(Config.CLASS_NAMES))

    checkpoint = torch.load(model_path, map_location=Config.DEVICE)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(Config.DEVICE)
    model.eval()
    return model

model = load_model()

def preprocess_audio(file):
    try:
        waveform, sr = torchaudio.load(file)
        if waveform.shape[0] > 1:  
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sr != Config.SAMPLE_RATE:
            resampler = T.Resample(sr, Config.SAMPLE_RATE)
            waveform = resampler(waveform)
    except Exception:
        y, sr = librosa.load(file, sr=Config.SAMPLE_RATE, mono=True)
        waveform = torch.FloatTensor(y).unsqueeze(0)

    max_length = Config.SAMPLE_RATE * Config.MAX_AUDIO_LENGTH
    if waveform.shape[1] < max_length:
        pad = max_length - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad))
    else:
        waveform = waveform[:, :max_length]

    mel_spec = T.MelSpectrogram(
        sample_rate=Config.SAMPLE_RATE,
        n_fft=Config.N_FFT,
        hop_length=Config.HOP_LENGTH,
        n_mels=Config.N_MELS,
        power=2.0
    )(waveform)

    mel_spec_db = T.AmplitudeToDB(top_db=80)(mel_spec)
    mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
    return waveform, mel_spec_db.unsqueeze(0)  

uploaded_file = st.file_uploader(
    "Upload an audio file", 
    type=["wav", "mp3", "flac", "ogg", "m4a"]
)

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    with st.spinner("🔍 Processing and classifying..."):
        waveform, mel = preprocess_audio(uploaded_file)
        mel = mel.to(Config.DEVICE)

        with torch.no_grad():
            outputs = model(mel)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            pred_class = Config.CLASS_NAMES[np.argmax(probs)]

    st.success(f"### 🎤 Predicted Class: **{pred_class}**")

    st.markdown("#### 🔍 Confidence Scores")
    conf_df = {Config.CLASS_NAMES[i]: float(probs[i]) for i in range(len(Config.CLASS_NAMES))}
    st.bar_chart(conf_df)

    st.markdown("#### 🎵 Waveform")
    fig1, ax1 = plt.subplots(figsize=(8, 2))
    ax1.plot(waveform.squeeze().numpy(), color='teal')
    ax1.set_title("Audio Waveform", fontsize=12)
    ax1.set_xlabel("Samples")
    ax1.set_ylabel("Amplitude")
    ax1.grid(alpha=0.3)
    st.pyplot(fig1)

    st.markdown("#### 🎼 Mel Spectrogram (dB)")
    mel_np = mel.detach().cpu().numpy()

    if mel_np.ndim == 4:
        mel_np = mel_np[0, 0, :, :]
    elif mel_np.ndim == 3:
        mel_np = mel_np[0, :, :]
    elif mel_np.ndim == 2:
        mel_np = mel_np
    else:
        raise ValueError(f"Unexpected mel shape: {mel_np.shape}")

    fig2, ax2 = plt.subplots(figsize=(8, 3))
    img = librosa.display.specshow(
        mel_np,
        sr=Config.SAMPLE_RATE,
        hop_length=Config.HOP_LENGTH,
        x_axis="time",
        y_axis="mel",
        cmap="magma",
        ax=ax2
    )
    fig2.colorbar(img, ax=ax2, format="%+2.0f dB")
    ax2.set_title("Mel Spectrogram (Log Scale)", fontsize=12)
    st.pyplot(fig2)

st.markdown("---")
st.markdown("Deepfake Audio Detection | PyTorch + Streamlit | CNN Model")