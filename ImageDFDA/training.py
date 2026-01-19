import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import timm
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score, f1_score,
    accuracy_score, precision_score, recall_score
)
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import json
import warnings
from tqdm import tqdm
import pandas as pd

warnings.filterwarnings('ignore')

DATA_DIR = r"ImagesDF"
OUTPUT_DIR = "training_reports"
MODELS_DIR = "saved_models"
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_OUTPUT_DIR = os.path.join(OUTPUT_DIR, RUN_TIMESTAMP)

NUM_EPOCHS = 20
LEARNING_RATE = 0.0001
BATCH_SIZE = 16
IMAGE_SIZE = 299
NUM_WORKERS = 4

EARLY_STOPPING_PATIENCE = 5
REDUCE_LR_PATIENCE = 3
MIN_LR = 1e-7
LR_REDUCTION_FACTOR = 0.5
GRADIENT_CLIP_VALUE = 1.0

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

USE_PRETRAINED = True
DROPOUT_RATE = 0.5
USE_ATTENTION = True

class EnhancedDeepFakeDataset(Dataset):
    """Enhanced Dataset with advanced preprocessing."""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        if not os.path.exists(root_dir) or not os.path.isdir(root_dir):
            raise FileNotFoundError(f"Dataset directory not found at {root_dir}")

        class_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

        if not class_names:
            raise ValueError(f"No class subfolders found in {root_dir}")

        print(f"Found {len(class_names)} classes: {class_names}")

        for i, class_name in enumerate(class_names):
            class_dir = os.path.join(root_dir, class_name)
            self.class_to_idx[class_name] = i
            self.idx_to_class[i] = class_name

            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(i)

        print(f"Total images loaded: {len(self.image_paths)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label, img_path

class AttentionBlock(nn.Module):
    """Attention mechanism for XceptionNet."""
    def __init__(self, in_features):
        super(AttentionBlock, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, in_features // 8),
            nn.ReLU(),
            nn.Linear(in_features // 8, in_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        att_weights = self.attention(x)
        return x * att_weights

class EnhancedXceptionNet(nn.Module):
    """XceptionNet with attention mechanism and improved head."""
    def __init__(self, num_classes, dropout_rate=0.3, use_attention=True):
        super(EnhancedXceptionNet, self).__init__()
        self.base_model = timm.create_model('xception', pretrained=USE_PRETRAINED)
        in_features = self.base_model.get_classifier().in_features
        self.base_model.reset_classifier(0)

        self.use_attention = use_attention
        if use_attention:
            self.attention = AttentionBlock(in_features)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.base_model(x)
        if self.use_attention:
            features = self.attention(features)
        output = self.classifier(features)
        return output

class AdvancedTrainer:
    """Enhanced trainer with comprehensive tracking."""
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=0.01
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            patience=config['reduce_lr_patience'],
            factor=config['lr_reduction_factor'],
            min_lr=config['min_lr'],
            verbose=True
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

        self.history = {
            'train_loss': [], 'train_acc': [], 'train_f1': [], 'train_precision': [], 'train_recall': [],
            'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_precision': [], 'val_recall': [],
            'learning_rates': [], 'epoch_times': []
        }

        self.early_stopping_counter = 0
        self.best_val_f1 = 0.0
        self.best_val_acc = 0.0

    def train_one_epoch(self, epoch_num):
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []

        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch_num} - Training')

        for images, labels, _ in progress_bar:
            images, labels = images.to(self.device), labels.to(self.device)

            with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda')):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clip_value'])
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct/total:.4f}'
            })

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
        epoch_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        epoch_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)

        return epoch_loss, epoch_acc, epoch_f1, epoch_precision, epoch_recall

    def validate_one_epoch(self, epoch_num):
        self.model.eval()
        running_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels, _ in tqdm(self.val_loader, desc=f'Epoch {epoch_num} - Validation'):
                images, labels = images.to(self.device), labels.to(self.device)

                with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda')):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
        epoch_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        epoch_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)

        return epoch_loss, epoch_acc, epoch_f1, epoch_precision, epoch_recall

    def fit(self):
        print("\n" + "="*70)
        print("STARTING ENHANCED XCEPTIONNET TRAINING")
        print("="*70 + "\n")

        for epoch in range(self.config['num_epochs']):
            import time
            start_time = time.time()

            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']}")
            print("-" * 70)

            train_loss, train_acc, train_f1, train_prec, train_rec = self.train_one_epoch(epoch+1)
            val_loss, val_acc, val_f1, val_prec, val_rec = self.validate_one_epoch(epoch+1)

            epoch_time = time.time() - start_time
            current_lr = self.optimizer.param_groups[0]['lr']

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['train_f1'].append(train_f1)
            self.history['train_precision'].append(train_prec)
            self.history['train_recall'].append(train_rec)

            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            self.history['val_precision'].append(val_prec)
            self.history['val_recall'].append(val_rec)

            self.history['learning_rates'].append(current_lr)
            self.history['epoch_times'].append(epoch_time)

            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Train -> Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f} | Prec: {train_prec:.4f} | Rec: {train_rec:.4f}")
            print(f"  Valid -> Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f} | Prec: {val_prec:.4f} | Rec: {val_rec:.4f}")
            print(f"  Learning Rate: {current_lr:.2e} | Time: {epoch_time:.2f}s")

            self.scheduler.step(val_f1)

            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_val_acc = val_acc
                self.save_checkpoint(epoch, val_f1, val_acc)
                print(f"   New best model saved! (F1: {val_f1:.4f}, Acc: {val_acc:.4f})")
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
                if self.early_stopping_counter >= self.config['early_stopping_patience']:
                    print(f"\n Early stopping triggered after {epoch+1} epochs")
                    break

        print("\n" + "="*70)
        print("TRAINING COMPLETED")
        print(f"Best Validation F1-Score: {self.best_val_f1:.4f}")
        print(f"Best Validation Accuracy: {self.best_val_acc:.4f}")
        print("="*70 + "\n")

        return self.history

    def save_checkpoint(self, epoch, val_f1, val_acc):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_f1': val_f1,
            'val_acc': val_acc,
            'config': self.config
        }
        torch.save(checkpoint, self.config['model_save_path'])

class ComprehensiveEvaluator:
    """Generate comprehensive evaluation metrics and visualizations."""
    def __init__(self, model, test_loader, device, class_names, output_dir):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.class_names = class_names
        self.output_dir = output_dir
        self.num_classes = len(class_names)

        self.metrics_dir = os.path.join(output_dir, 'metrics')
        self.plots_dir = os.path.join(output_dir, 'plots')
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

    def evaluate(self):
        print("\n" + "="*70)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("="*70 + "\n")

        y_true, y_pred, y_proba = self._get_predictions()

        self._save_classification_metrics(y_true, y_pred)
        self._plot_confusion_matrix(y_true, y_pred)
        self._plot_roc_curves(y_true, y_proba)
        self._plot_precision_recall_curves(y_true, y_proba)
        self._plot_per_class_metrics(y_true, y_pred)
        self._save_detailed_metrics(y_true, y_pred, y_proba)

        print(f"\n Evaluation complete! Results saved to: {self.output_dir}\n")

    def _get_predictions(self):
        self.model.eval()
        y_true, y_pred, y_proba = [], [], []

        with torch.no_grad():
            for images, labels, _ in tqdm(self.test_loader, desc='Evaluating'):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                y_proba.extend(probabilities.cpu().numpy())

        return np.array(y_true), np.array(y_pred), np.array(y_proba)

    def _save_classification_metrics(self, y_true, y_pred):
        report = classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(os.path.join(self.metrics_dir, 'classification_report.csv'))

        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=self.class_names))

    def _plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names,
                    ax=axes[0], cbar_kws={'label': 'Count'})
        axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Predicted Label', fontsize=12)
        axes[0].set_ylabel('True Label', fontsize=12)

        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens',
                    xticklabels=self.class_names, yticklabels=self.class_names,
                    ax=axes[1], cbar_kws={'label': 'Proportion'})
        axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Predicted Label', fontsize=12)
        axes[1].set_ylabel('True Label', fontsize=12)

        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_roc_curves(self, y_true, y_proba):
        plt.figure(figsize=(12, 9))

        if self.num_classes == 2:
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2.5, label=f'ROC curve (AUC = {roc_auc:.3f})')
        else:
            for i in range(self.num_classes):
                y_true_binary = (y_true == i).astype(int)
                fpr, tpr, _ = roc_curve(y_true_binary, y_proba[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2.5, label=f'{self.class_names[i]} (AUC = {roc_auc:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Guess')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_precision_recall_curves(self, y_true, y_proba):
        plt.figure(figsize=(12, 9))

        if self.num_classes == 2:
            precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
            avg_precision = average_precision_score(y_true, y_proba[:, 1])
            plt.plot(recall, precision, lw=2.5, label=f'PR curve (AP = {avg_precision:.3f})')
        else:
            for i in range(self.num_classes):
                y_true_binary = (y_true == i).astype(int)
                precision, recall, _ = precision_recall_curve(y_true_binary, y_proba[:, i])
                avg_precision = average_precision_score(y_true_binary, y_proba[:, i])
                plt.plot(recall, precision, lw=2.5, label=f'{self.class_names[i]} (AP = {avg_precision:.3f})')

        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="best", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'precision_recall_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_per_class_metrics(self, y_true, y_pred):
        report = classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True)

        classes = self.class_names
        precision = [report[c]['precision'] for c in classes]
        recall = [report[c]['recall'] for c in classes]
        f1 = [report[c]['f1-score'] for c in classes]

        x = np.arange(len(classes))
        width = 0.25

        fig, ax = plt.subplots(figsize=(14, 8))

        bars1 = ax.bar(x - width, precision, width, label='Precision', color='#2E86AB')
        bars2 = ax.bar(x, recall, width, label='Recall', color='#A23B72')
        bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#F18F01')

        ax.set_xlabel('Classes', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.05])

        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'per_class_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _save_detailed_metrics(self, y_true, y_pred, y_proba):
        metrics = {
            'overall_accuracy': accuracy_score(y_true, y_pred),
            'weighted_precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'weighted_recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'weighted_f1': f1_score(y_true, y_pred, average='weighted'),
            'macro_precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'macro_recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'macro_f1': f1_score(y_true, y_pred, average='macro')
        }

        with open(os.path.join(self.metrics_dir, 'detailed_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)

        print("\nDetailed Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")

def plot_training_history(history, save_dir):
    """Comprehensive training history visualization."""
    plots_dir = os.path.join(save_dir, 'plots')

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2, color='#FF6B6B')
    axes[0, 0].plot(history['val_loss'], label='Validation Loss', linewidth=2, color='#4ECDC4')
    axes[0, 0].set_title('Loss vs. Epochs', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epochs', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(history['train_acc'], label='Train Accuracy', linewidth=2, color='#FF6B6B')
    axes[0, 1].plot(history['val_acc'], label='Validation Accuracy', linewidth=2, color='#4ECDC4')
    axes[0, 1].set_title('Accuracy vs. Epochs', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epochs', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy', fontsize=12)
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(history['train_f1'], label='Train F1-Score', linewidth=2, color='#FF6B6B')
    axes[1, 0].plot(history['val_f1'], label='Validation F1-Score', linewidth=2, color='#4ECDC4')
    axes[1, 0].set_title('F1-Score vs. Epochs', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epochs', fontsize=12)
    axes[1, 0].set_ylabel('F1-Score', fontsize=12)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(history['learning_rates'], linewidth=2, color='#95E1D3', marker='o')
    axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epochs', fontsize=12)
    axes[1, 1].set_ylabel('Learning Rate', fontsize=12)
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'training_history_comprehensive.png'), dpi=300, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].plot(history['train_precision'], label='Train Precision', linewidth=2, color='#FF6B6B')
    axes[0].plot(history['val_precision'], label='Validation Precision', linewidth=2, color='#4ECDC4')
    axes[0].set_title('Precision vs. Epochs', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epochs', fontsize=12)
    axes[0].set_ylabel('Precision', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['train_recall'], label='Train Recall', linewidth=2, color='#FF6B6B')
    axes[1].plot(history['val_recall'], label='Validation Recall', linewidth=2, color='#4ECDC4')
    axes[1].set_title('Recall vs. Epochs', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epochs', fontsize=12)
    axes[1].set_ylabel('Recall', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'precision_recall_history.png'), dpi=300, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(16, 8))

    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_acc'], label='Train Accuracy', linewidth=2, marker='o', markersize=4)
    ax.plot(epochs, history['val_acc'], label='Val Accuracy', linewidth=2, marker='s', markersize=4)
    ax.plot(epochs, history['train_f1'], label='Train F1', linewidth=2, marker='^', markersize=4)
    ax.plot(epochs, history['val_f1'], label='Val F1', linewidth=2, marker='v', markersize=4)
    ax.plot(epochs, history['train_precision'], label='Train Precision', linewidth=2, marker='d', markersize=4)
    ax.plot(epochs, history['val_precision'], label='Val Precision', linewidth=2, marker='*', markersize=6)

    ax.set_title('All Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epochs', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.legend(loc='best', fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'all_metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f" Training history plots saved to {plots_dir}")

def plot_model_performance_summary(history, save_dir):
    """Create a comprehensive performance summary dashboard."""
    plots_dir = os.path.join(save_dir, 'plots')

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(history['train_loss'], label='Train', linewidth=2.5, color='#E63946')
    ax1.plot(history['val_loss'], label='Validation', linewidth=2.5, color='#457B9D')
    ax1.fill_between(range(len(history['train_loss'])), history['train_loss'], alpha=0.3, color='#E63946')
    ax1.fill_between(range(len(history['val_loss'])), history['val_loss'], alpha=0.3, color='#457B9D')
    ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    best_val_acc = max(history['val_acc'])
    best_val_f1 = max(history['val_f1'])
    final_train_acc = history['train_acc'][-1]
    final_val_acc = history['val_acc'][-1]

    summary_text = f"""
    BEST PERFORMANCE
    {'='*25}
    Best Val Accuracy: {best_val_acc:.4f}
    Best Val F1-Score: {best_val_f1:.4f}

    FINAL METRICS
    {'='*25}
    Final Train Acc: {final_train_acc:.4f}
    Final Val Acc: {final_val_acc:.4f}

    Total Epochs: {len(history['train_loss'])}
    """
    ax2.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(history['train_acc'], label='Train', linewidth=2, color='#2A9D8F')
    ax3.plot(history['val_acc'], label='Validation', linewidth=2, color='#E76F51')
    ax3.set_title('Accuracy', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Accuracy')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(history['train_f1'], label='Train', linewidth=2, color='#2A9D8F')
    ax4.plot(history['val_f1'], label='Validation', linewidth=2, color='#E76F51')
    ax4.set_title('F1-Score', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('F1-Score')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(gs[1, 2])
    ax5.plot(history['learning_rates'], linewidth=2, color='#F4A261', marker='o')
    ax5.set_title('Learning Rate', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Epochs')
    ax5.set_ylabel('LR')
    ax5.set_yscale('log')
    ax5.grid(True, alpha=0.3)

    ax6 = fig.add_subplot(gs[2, 0])
    ax6.plot(history['train_precision'], label='Train', linewidth=2, color='#264653')
    ax6.plot(history['val_precision'], label='Validation', linewidth=2, color='#E9C46A')
    ax6.set_title('Precision', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Epochs')
    ax6.set_ylabel('Precision')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    ax7 = fig.add_subplot(gs[2, 1])
    ax7.plot(history['train_recall'], label='Train', linewidth=2, color='#264653')
    ax7.plot(history['val_recall'], label='Validation', linewidth=2, color='#E9C46A')
    ax7.set_title('Recall', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Epochs')
    ax7.set_ylabel('Recall')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    ax8 = fig.add_subplot(gs[2, 2])
    ax8.bar(range(len(history['epoch_times'])), history['epoch_times'], color='#8338EC', alpha=0.7)
    ax8.axhline(y=np.mean(history['epoch_times']), color='r', linestyle='--',
                label=f"Avg: {np.mean(history['epoch_times']):.2f}s")
    ax8.set_title('Epoch Duration', fontsize=12, fontweight='bold')
    ax8.set_xlabel('Epochs')
    ax8.set_ylabel('Time (seconds)')
    ax8.legend()
    ax8.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Model Training Performance Dashboard', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(os.path.join(plots_dir, 'performance_dashboard.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f" Performance dashboard saved to {plots_dir}")

def save_training_summary(history, config, save_dir):
    """Save comprehensive training summary as JSON and text."""
    metrics_dir = os.path.join(save_dir, 'metrics')

    summary = {
        'configuration': config,
        'training_results': {
            'total_epochs': len(history['train_loss']),
            'best_validation_accuracy': float(max(history['val_acc'])),
            'best_validation_f1': float(max(history['val_f1'])),
            'final_train_accuracy': float(history['train_acc'][-1]),
            'final_val_accuracy': float(history['val_acc'][-1]),
            'final_train_f1': float(history['train_f1'][-1]),
            'final_val_f1': float(history['val_f1'][-1]),
            'final_learning_rate': float(history['learning_rates'][-1]),
            'average_epoch_time': float(np.mean(history['epoch_times'])),
            'total_training_time': float(sum(history['epoch_times']))
        },
        'history': {
            'train_loss': [float(x) for x in history['train_loss']],
            'val_loss': [float(x) for x in history['val_loss']],
            'train_acc': [float(x) for x in history['train_acc']],
            'val_acc': [float(x) for x in history['val_acc']],
            'train_f1': [float(x) for x in history['train_f1']],
            'val_f1': [float(x) for x in history['val_f1']],
            'learning_rates': [float(x) for x in history['learning_rates']]
        }
    }

    with open(os.path.join(metrics_dir, 'training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)

    with open(os.path.join(metrics_dir, 'training_summary.txt'), 'w') as f:
        f.write("="*70 + "\n")
        f.write("TRAINING SUMMARY REPORT\n")
        f.write("="*70 + "\n\n")

        f.write("CONFIGURATION\n")
        f.write("-"*70 + "\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")

        f.write("\n" + "="*70 + "\n")
        f.write("TRAINING RESULTS\n")
        f.write("="*70 + "\n")
        for key, value in summary['training_results'].items():
            f.write(f"{key}: {value}\n")

    print(f" Training summary saved to {metrics_dir}")

if __name__ == '__main__':
    print("\n" + "="*70)
    print("DEEPFAKE DETECTION - ENHANCED TRAINING PIPELINE")
    print("="*70 + "\n")

    os.makedirs(RUN_OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(os.path.join(RUN_OUTPUT_DIR, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(RUN_OUTPUT_DIR, 'metrics'), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device.type.upper()}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print()

    model_save_path = os.path.join(MODELS_DIR, f"best_model_{RUN_TIMESTAMP}.pt")

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    print("Loading dataset...")
    full_dataset = EnhancedDeepFakeDataset(root_dir=DATA_DIR)

    total_size = len(full_dataset)
    train_size = int(TRAIN_SPLIT * total_size)
    val_size = int(VAL_SPLIT * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_test_transform
    test_dataset.dataset.transform = val_test_transform

    print(f"\nDataset Distribution:")
    print(f"  Training:   {len(train_dataset):,} images ({len(train_dataset)/total_size*100:.1f}%)")
    print(f"  Validation: {len(val_dataset):,} images ({len(val_dataset)/total_size*100:.1f}%)")
    print(f"  Test:       {len(test_dataset):,} images ({len(test_dataset)/total_size*100:.1f}%)")
    print(f"  Total:      {total_size:,} images\n")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    num_classes = len(full_dataset.class_to_idx)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {list(full_dataset.class_to_idx.keys())}\n")

    model = EnhancedXceptionNet(
        num_classes=num_classes,
        dropout_rate=DROPOUT_RATE,
        use_attention=USE_ATTENTION
    )

    config = {
        'num_epochs': NUM_EPOCHS,
        'learning_rate': LEARNING_RATE,
        'batch_size': BATCH_SIZE,
        'image_size': IMAGE_SIZE,
        'early_stopping_patience': EARLY_STOPPING_PATIENCE,
        'reduce_lr_patience': REDUCE_LR_PATIENCE,
        'min_lr': MIN_LR,
        'lr_reduction_factor': LR_REDUCTION_FACTOR,
        'gradient_clip_value': GRADIENT_CLIP_VALUE,
        'model_save_path': model_save_path,
        'use_pretrained': USE_PRETRAINED,
        'dropout_rate': DROPOUT_RATE,
        'use_attention': USE_ATTENTION
    }

    trainer = AdvancedTrainer(model, train_loader, val_loader, device, config)

    history = trainer.fit()

    print("\nGenerating training visualizations...")
    plot_training_history(history, RUN_OUTPUT_DIR)
    plot_model_performance_summary(history, RUN_OUTPUT_DIR)
    save_training_summary(history, config, RUN_OUTPUT_DIR)

    print("\nLoading best model for evaluation...")
    best_model = EnhancedXceptionNet(
        num_classes=num_classes,
        dropout_rate=DROPOUT_RATE,
        use_attention=USE_ATTENTION
    )
    checkpoint = torch.load(model_save_path)
    best_model.load_state_dict(checkpoint['model_state_dict'])

    class_names = list(full_dataset.idx_to_class.values())
    evaluator = ComprehensiveEvaluator(best_model, test_loader, device, class_names, RUN_OUTPUT_DIR)
    evaluator.evaluate()

    print("\n" + "="*70)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"\nAll results saved to: {RUN_OUTPUT_DIR}")
    print(f"Best model saved to: {model_save_path}")
    print(f"\nGenerated files:")
    print(f"   Training plots: {os.path.join(RUN_OUTPUT_DIR, 'plots')}")
    print(f"   Metrics & reports: {os.path.join(RUN_OUTPUT_DIR, 'metrics')}")
    print(f"   Model checkpoint: {model_save_path}")
    print("\n" + "="*70 + "\n")