import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchaudio
import torchaudio.transforms as T
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, 
                            accuracy_score, precision_recall_fscore_support,
                            roc_curve, auc, roc_auc_score)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
import warnings
import gc
import psutil
from datetime import datetime
from collections import Counter
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)

class Config:

    DATASET_PATH = Path("Dataset")
    OUTPUT_DIR = Path("training_results")
    MODEL_SAVE_PATH = OUTPUT_DIR / "models"
    METRICS_DIR = OUTPUT_DIR / "metrics"
    GRAPHS_DIR = OUTPUT_DIR / "graphs"

    SAMPLE_RATE = 16000
    N_MELS = 128
    N_FFT = 2048
    HOP_LENGTH = 512
    MAX_AUDIO_LENGTH = 5  

    BATCH_SIZE = 16
    NUM_EPOCHS = 30
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 1e-5
    PATIENCE = 4  

    NUM_CLASSES = 5
    CLASS_NAMES = ["ElevenLabs", "Original", "Tacotron", "Text To Speech", "Voice Conversion"]

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 2
    PIN_MEMORY = True

for dir_path in [Config.OUTPUT_DIR, Config.MODEL_SAVE_PATH, 
                 Config.METRICS_DIR, Config.GRAPHS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

class AudioDeepfakeDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None, augment=False):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.augment = augment

        self.time_stretch = T.TimeStretch(n_freq=Config.N_MELS)
        self.freq_mask = T.FrequencyMasking(freq_mask_param=15)
        self.time_mask = T.TimeMasking(time_mask_param=35)

    def __len__(self):
        return len(self.file_paths)

    def load_audio(self, file_path):
        """Load audio with support for multiple formats"""
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

    def extract_features(self, waveform):
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

        return mel_spec_db

    def augment_spectrogram(self, spec):
        """Apply data augmentation to spectrogram"""
        if np.random.random() > 0.5:
            spec = self.freq_mask(spec)
        if np.random.random() > 0.5:
            spec = self.time_mask(spec)
        return spec

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        waveform = self.load_audio(file_path)
        mel_spec = self.extract_features(waveform)

        if self.augment:
            mel_spec = self.augment_spectrogram(mel_spec)

        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)

        return mel_spec, label

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

def load_dataset():
    """Load dataset from folder structure"""
    print("\n" + "="*60)
    print("Loading Dataset...")
    print("="*60)

    file_paths = []
    labels = []
    class_counts = {class_name: 0 for class_name in Config.CLASS_NAMES}

    for class_idx, class_name in enumerate(Config.CLASS_NAMES):
        class_dir = Config.DATASET_PATH / class_name

        if not class_dir.exists():
            print(f"Warning: Directory not found - {class_dir}")
            continue

        audio_files = []
        for ext in ['*.wav', '*.mp3', '*.flac', '*.ogg', '*.m4a']:
            audio_files.extend(list(class_dir.glob(ext)))

        class_counts[class_name] = len(audio_files)

        for audio_file in audio_files:
            file_paths.append(str(audio_file))
            labels.append(class_idx)

        print(f"  {class_name}: {len(audio_files)} files")

    print(f"\nTotal files: {len(file_paths)}")
    print("="*60)

    return file_paths, labels, class_counts

def get_weighted_sampler(labels):
    """Create weighted sampler for imbalanced dataset"""
    class_counts = Counter(labels)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    return sampler

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                         'acc': f'{100.*correct/total:.2f}%'})

        if total % 100 == 0:
            torch.cuda.empty_cache()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation", leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                             'acc': f'{100.*correct/total:.2f}%'})

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc, all_preds, all_labels, all_probs

def plot_training_history(history, save_path):
    """Plot training and validation metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['train_acc'], label='Train Acc', linewidth=2)
    axes[1].plot(history['val_acc'], label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved training history plot: {save_path}")

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix: {save_path}")

def plot_normalized_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot normalized confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='YlOrRd',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Percentage'})
    plt.title('Normalized Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved normalized confusion matrix: {save_path}")

def plot_class_performance(y_true, y_pred, class_names, save_path):
    """Plot per-class performance metrics"""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0)

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim([0, 1.1])

    for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
        ax.text(i - width, p + 0.02, f'{p:.2f}', ha='center', va='bottom', fontsize=8)
        ax.text(i, r + 0.02, f'{r:.2f}', ha='center', va='bottom', fontsize=8)
        ax.text(i + width, f + 0.02, f'{f:.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved class performance plot: {save_path}")

def plot_roc_curves(y_true, y_probs, class_names, save_path):
    """Plot ROC curves for all classes"""
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))

    plt.figure(figsize=(12, 10))
    colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))

    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], np.array(y_probs)[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2, 
                label=f'{class_name} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Multi-Class', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC curves: {save_path}")

def plot_class_distribution(class_counts, save_path):
    """Plot class distribution"""
    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, counts, color=plt.cm.Set3(np.linspace(0, 1, len(classes))))
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.title('Dataset Class Distribution', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved class distribution plot: {save_path}")

def save_metrics_report(y_true, y_pred, y_probs, class_names, save_path):
    """Save detailed metrics report"""

    report = classification_report(y_true, y_pred, target_names=class_names, 
                                   zero_division=0, digits=4)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0)

    overall_acc = accuracy_score(y_true, y_pred)

    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    roc_auc_scores = []
    for i in range(len(class_names)):
        try:
            score = roc_auc_score(y_true_bin[:, i], np.array(y_probs)[:, i])
            roc_auc_scores.append(score)
        except:
            roc_auc_scores.append(0.0)

    report_text = f"""
{'='*70}
AUDIO DEEPFAKE DETECTION - PERFORMANCE REPORT
{'='*70}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERALL METRICS:
{'-'*70}
Overall Accuracy: {overall_acc*100:.2f}%

CLASSIFICATION REPORT:
{'-'*70}
{report}

PER-CLASS DETAILED METRICS:
{'-'*70}
{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'ROC-AUC':<12} {'Support':<10}
{'-'*70}
"""

    for i, class_name in enumerate(class_names):
        report_text += f"{class_name:<20} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {roc_auc_scores[i]:<12.4f} {support[i]:<10}\n"

    report_text += f"\n{'='*70}\n"

    with open(save_path, 'w') as f:
        f.write(report_text)

    print(f"Saved metrics report: {save_path}")

    json_path = save_path.with_suffix('.json')
    metrics_dict = {
        'overall_accuracy': float(overall_acc),
        'per_class_metrics': {
            class_name: {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i]),
                'roc_auc': float(roc_auc_scores[i]),
                'support': int(support[i])
            }
            for i, class_name in enumerate(class_names)
        },
        'timestamp': datetime.now().isoformat()
    }

    with open(json_path, 'w') as f:
        json.dump(metrics_dict, f, indent=4)

    print(f"Saved metrics JSON: {json_path}")

def main():
    print("\n" + "="*70)
    print(" AUDIO DEEPFAKE DETECTION - TRAINING PIPELINE")
    print("="*70)
    print(f"Device: {Config.DEVICE}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("="*70)

    file_paths, labels, class_counts = load_dataset()

    if len(file_paths) == 0:
        print("Error: No audio files found!")
        return

    plot_class_distribution(class_counts, Config.GRAPHS_DIR / 'class_distribution.png')

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        file_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )

    print(f"\nDataset Split:")
    print(f"  Training samples: {len(train_paths)}")
    print(f"  Validation samples: {len(val_paths)}")

    train_dataset = AudioDeepfakeDataset(train_paths, train_labels, augment=True)
    val_dataset = AudioDeepfakeDataset(val_paths, val_labels, augment=False)

    train_sampler = get_weighted_sampler(train_labels)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE,
        sampler=train_sampler,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )

    model = AudioDeepfakeCNN(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, 
                          weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    best_val_acc = 0.0
    patience_counter = 0

    print("\n" + "="*70)
    print(" TRAINING START")
    print("="*70 + "\n")

    for epoch in range(Config.NUM_EPOCHS):
        print(f"Epoch [{epoch+1}/{Config.NUM_EPOCHS}]")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, 
                                           optimizer, Config.DEVICE)

        val_loss, val_acc, val_preds, val_labels, val_probs = validate(
            model, val_loader, criterion, Config.DEVICE
        )

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, Config.MODEL_SAVE_PATH / 'best_model.pth')

            print(f"  >>> New best model saved! Val Acc: {val_acc:.2f}%")
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"  Best Val Acc: {best_val_acc:.2f}%")
        print(f"  Patience: {patience_counter}/{Config.PATIENCE}\n")

        gc.collect()
        torch.cuda.empty_cache()

    print("\n" + "="*70)
    print(" TRAINING COMPLETED")
    print("="*70)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%\n")

    checkpoint = torch.load(Config.MODEL_SAVE_PATH / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    print("Performing final evaluation...")
    val_loss, val_acc, val_preds, val_labels, val_probs = validate(
        model, val_loader, criterion, Config.DEVICE
    )

    print("\n" + "="*70)
    print(" GENERATING METRICS AND VISUALIZATIONS")
    print("="*70 + "\n")

    plot_training_history(history, Config.GRAPHS_DIR / 'training_history.png')
    plot_confusion_matrix(val_labels, val_preds, Config.CLASS_NAMES, 
                         Config.GRAPHS_DIR / 'confusion_matrix.png')
    plot_normalized_confusion_matrix(val_labels, val_preds, Config.CLASS_NAMES,
                                    Config.GRAPHS_DIR / 'confusion_matrix_normalized.png')
    plot_class_performance(val_labels, val_preds, Config.CLASS_NAMES,
                          Config.GRAPHS_DIR / 'class_performance.png')
    plot_roc_curves(val_labels, val_probs, Config.CLASS_NAMES,
                   Config.GRAPHS_DIR / 'roc_curves.png')

    save_metrics_report(val_labels, val_preds, val_probs, Config.CLASS_NAMES,
                       Config.METRICS_DIR / 'metrics_report.txt')

    history_df = pd.DataFrame(history)
    history_df.to_csv(Config.METRICS_DIR / 'training_history.csv', index=False)
    print(f"Saved training history CSV: {Config.METRICS_DIR / 'training_history.csv'}")

    with open(Config.METRICS_DIR / 'model_summary.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("MODEL ARCHITECTURE SUMMARY\n")
        f.write("="*70 + "\n\n")
        f.write(str(model) + "\n\n")
        f.write(f"Total Parameters: {total_params:,}\n")
        f.write(f"Trainable Parameters: {trainable_params:,}\n")
        f.write("="*70 + "\n")
    print(f"Saved model summary: {Config.METRICS_DIR / 'model_summary.txt'}")

    config_dict = {
        'dataset_path': str(Config.DATASET_PATH),
        'sample_rate': Config.SAMPLE_RATE,
        'n_mels': Config.N_MELS,
        'n_fft': Config.N_FFT,
        'hop_length': Config.HOP_LENGTH,
        'max_audio_length': Config.MAX_AUDIO_LENGTH,
        'batch_size': Config.BATCH_SIZE,
        'num_epochs': Config.NUM_EPOCHS,
        'learning_rate': Config.LEARNING_RATE,
        'weight_decay': Config.WEIGHT_DECAY,
        'num_classes': Config.NUM_CLASSES,
        'class_names': Config.CLASS_NAMES,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'best_validation_accuracy': float(best_val_acc),
        'training_samples': len(train_paths),
        'validation_samples': len(val_paths),
        'class_distribution': class_counts
    }

    with open(Config.METRICS_DIR / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=4)
    print(f"Saved configuration: {Config.METRICS_DIR / 'config.json'}")

    summary_text = f"""
{'='*70}
AUDIO DEEPFAKE DETECTION - TRAINING SUMMARY
{'='*70}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DATASET INFORMATION:
{'-'*70}
Total Samples: {len(file_paths)}
Training Samples: {len(train_paths)}
Validation Samples: {len(val_paths)}

Class Distribution:
"""

    for class_name, count in class_counts.items():
        summary_text += f"  - {class_name}: {count} samples\n"

    summary_text += f"""
MODEL INFORMATION:
{'-'*70}
Architecture: Custom CNN
Total Parameters: {total_params:,}
Trainable Parameters: {trainable_params:,}

TRAINING CONFIGURATION:
{'-'*70}
Batch Size: {Config.BATCH_SIZE}
Learning Rate: {Config.LEARNING_RATE}
Weight Decay: {Config.WEIGHT_DECAY}
Number of Epochs: {len(history['train_loss'])}
Early Stopping Patience: {Config.PATIENCE}

RESULTS:
{'-'*70}
Best Validation Accuracy: {best_val_acc:.2f}%
Final Training Loss: {history['train_loss'][-1]:.4f}
Final Validation Loss: {history['val_loss'][-1]:.4f}

OUTPUT FILES:
{'-'*70}
Models:
  - best_model.pth

Graphs:
  - training_history.png
  - confusion_matrix.png
  - confusion_matrix_normalized.png
  - class_performance.png
  - roc_curves.png
  - class_distribution.png

Metrics:
  - metrics_report.txt
  - metrics_report.json
  - training_history.csv
  - model_summary.txt
  - config.json
  - summary.txt

{'='*70}
"""

    with open(Config.METRICS_DIR / 'summary.txt', 'w') as f:
        f.write(summary_text)
    print(f"Saved summary report: {Config.METRICS_DIR / 'summary.txt'}")

    print("\n" + "="*70)
    print(" ALL OUTPUTS SAVED SUCCESSFULLY")
    print("="*70)
    print(f"\nResults saved to: {Config.OUTPUT_DIR}")
    print(f"  - Models: {Config.MODEL_SAVE_PATH}")
    print(f"  - Metrics: {Config.METRICS_DIR}")
    print(f"  - Graphs: {Config.GRAPHS_DIR}")

    print("\n" + "="*70)
    print(" TRAINING PIPELINE COMPLETED")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()