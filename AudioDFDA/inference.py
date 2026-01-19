import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import librosa
import numpy as np
from pathlib import Path
import json
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class Config:
    MODEL_PATH = Path("training_results/models/best_model.pth")
    CONFIG_PATH = Path("training_results/metrics/config.json")

    SAMPLE_RATE = 16000
    N_MELS = 128
    N_FFT = 2048
    HOP_LENGTH = 512
    MAX_AUDIO_LENGTH = 5

    CLASS_NAMES = ["ElevenLabs", "Original", "Tacotron", "Text To Speech", "Voice Conversion"]

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AudioDeepfakeCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(AudioDeepfakeCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.3)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.3)
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def load_audio(file_path):
    """Load audio file with support for multiple formats"""
    try:
        waveform, sr = torchaudio.load(file_path)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sr != Config.SAMPLE_RATE:
            resampler = T.Resample(sr, Config.SAMPLE_RATE)
            waveform = resampler(waveform)

        return waveform
    except:
        waveform, sr = librosa.load(file_path, sr=Config.SAMPLE_RATE, mono=True)
        waveform = torch.FloatTensor(waveform).unsqueeze(0)
        return waveform

def extract_features(waveform):
    """Extract mel spectrogram features"""
    max_length = Config.SAMPLE_RATE * Config.MAX_AUDIO_LENGTH
    if waveform.shape[1] < max_length:
        padding = max_length - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, padding))
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

    return mel_spec_db

def load_model(model_path, device):
    """Load trained model"""
    model = AudioDeepfakeCNN(num_classes=len(Config.CLASS_NAMES))
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model

def predict_single_file(model, audio_path, device):
    """Predict class for a single audio file"""
    try:
        waveform = load_audio(audio_path)
        mel_spec = extract_features(waveform)
        mel_spec = mel_spec.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(mel_spec)
            probabilities = torch.softmax(output, dim=1)[0]
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()

        return {
            'predicted_class': Config.CLASS_NAMES[predicted_class],
            'predicted_index': predicted_class,
            'confidence': confidence,
            'probabilities': {
                class_name: prob.item() 
                for class_name, prob in zip(Config.CLASS_NAMES, probabilities)
            }
        }
    except Exception as e:
        return {'error': str(e)}

def predict_batch(model, audio_dir, device, output_file=None):
    """Predict classes for all audio files in a directory"""
    audio_dir = Path(audio_dir)
    audio_files = []

    for ext in ['*.wav', '*.mp3', '*.flac', '*.ogg', '*.m4a']:
        audio_files.extend(list(audio_dir.glob(ext)))

    if not audio_files:
        print(f"No audio files found in {audio_dir}")
        return []

    results = []
    print(f"\nProcessing {len(audio_files)} audio files...")

    for audio_file in tqdm(audio_files):
        result = predict_single_file(model, audio_file, device)
        result['filename'] = audio_file.name
        result['filepath'] = str(audio_file)
        results.append(result)

    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\nResults saved to: {output_file}")

    return results

def print_prediction(result):
    """Pretty print prediction result"""
    if 'error' in result:
        print(f"Error: {result['error']}")
        return

    print("\n" + "="*60)
    print("PREDICTION RESULT")
    print("="*60)
    if 'filename' in result:
        print(f"File: {result['filename']}")
    print(f"Predicted Class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']*100:.2f}%")
    print("\nClass Probabilities:")
    print("-"*60)

    sorted_probs = sorted(result['probabilities'].items(), 
                         key=lambda x: x[1], reverse=True)

    for class_name, prob in sorted_probs:
        bar_length = int(prob * 40)
        bar = '#' * bar_length + '-' * (40 - bar_length)
        print(f"{class_name:<20} [{bar}] {prob*100:6.2f}%")

    print("="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(
        description='Audio Deepfake Detection - Inference'
    )
    parser.add_argument('--file', type=str, help='Path to single audio file')
    parser.add_argument('--dir', type=str, help='Path to directory with audio files')
    parser.add_argument('--output', type=str, help='Output JSON file for batch predictions')
    parser.add_argument('--model', type=str, default=str(Config.MODEL_PATH),
                       help='Path to trained model')

    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using the training script.")
        return

    print("\n" + "="*60)
    print("AUDIO DEEPFAKE DETECTION - INFERENCE")
    print("="*60)
    print(f"Device: {Config.DEVICE}")
    print(f"Model: {model_path}")

    print("\nLoading model...")
    model = load_model(model_path, Config.DEVICE)
    print("Model loaded successfully!")

    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: File not found - {file_path}")
            return

        print(f"\nProcessing: {file_path}")
        result = predict_single_file(model, file_path, Config.DEVICE)
        print_prediction(result)

    elif args.dir:
        dir_path = Path(args.dir)
        if not dir_path.exists():
            print(f"Error: Directory not found - {dir_path}")
            return

        output_file = args.output if args.output else dir_path / 'predictions.json'
        results = predict_batch(model, dir_path, Config.DEVICE, output_file)

        if results:
            print("\n" + "="*60)
            print("BATCH PREDICTION SUMMARY")
            print("="*60)

            class_counts = {}
            for result in results:
                if 'predicted_class' in result:
                    class_name = result['predicted_class']
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1

            for class_name, count in sorted(class_counts.items()):
                percentage = (count / len(results)) * 100
                print(f"{class_name:<20}: {count:3d} files ({percentage:5.1f}%)")

            print("="*60 + "\n")

    else:
        print("\nError: Please specify either --file or --dir")
        print("Usage examples:")
        print("  Single file:  python inference.py --file audio.wav")
        print("  Batch:        python inference.py --dir audio_folder/")
        print("  With output:  python inference.py --dir audio_folder/ --output results.json")

if __name__ == "__main__":
    main()