import os
import time
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_fscore_support
)

try:
    import timm
    TIMM_AVAILABLE = True
except Exception:
    TIMM_AVAILABLE = False

DATASET_DIR = "Dataset/FF++C32-Frames"    
INDEX_CSV = "dataset_index.csv"           
RESULTS_DIR = os.path.join("VIDEO_DFDA", "Results")
IMAGE_SIZE = 300
BATCH_SIZE = 16
NUM_EPOCHS = 15
LR = 1e-4
NUM_WORKERS = 4   
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_EPOCH_MODELS = True  
SEED = 42
os.makedirs(RESULTS_DIR, exist_ok=True)

def build_index_csv(root_dir, out_csv):
    classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    label_map = {c: i for i, c in enumerate(classes)}
    rows = []
    for c in classes:
        folder = os.path.join(root_dir, c)
        for fname in os.listdir(folder):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                full = os.path.join(folder, fname)
                rows.append({"filename": fname, "label": c, "filepath": full, "label_id": label_map[c]})
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df, label_map

class FrameDatasetFromCSV(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform

        if "filepath" not in self.df.columns:
            raise KeyError("CSV must contain 'filepath' column.")
        if "label_id" not in self.df.columns and "label" not in self.df.columns:
            raise KeyError("CSV must contain either 'label_id' or 'label' column.")

        if "label_id" not in self.df.columns:
            labels = sorted(self.df["label"].unique())
            self.label_to_id = {lab: i for i, lab in enumerate(labels)}
            self.df["label_id"] = self.df["label"].map(self.label_to_id)
        else:
            self.label_to_id = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = str(row["filepath"])

        path = path.replace("\\", os.sep).replace("/", os.sep)
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = int(row["label_id"])
        return img, label

def create_model(num_classes):
    if TIMM_AVAILABLE:
        try:
            model = timm.create_model("legacy_xception", pretrained=True, num_classes=num_classes)
            return model
        except Exception:
            pass

    try:
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model
    except Exception:

        class SmallNet(nn.Module):
            def __init__(self, n_classes):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(3, 16, 3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(16, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(32, n_classes)
                )
            def forward(self, x):
                return self.net(x)
        return SmallNet(num_classes)

def save_loss_acc_plots(train_losses, val_losses, train_accs, val_accs, out_dir):
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="train_loss", marker='o')
    plt.plot(val_losses, label="val_loss", marker='o')
    plt.title("Loss per epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "loss_curve.png"))
    plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(train_accs, label="train_acc", marker='o')
    plt.plot(val_accs, label="val_acc", marker='o')
    plt.title("Accuracy per epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "accuracy_curve.png"))
    plt.close()

def save_confusion_matrix(y_true, y_pred, class_names, out_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def save_classification_report(y_true, y_pred, class_names, out_path):
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    with open(out_path, "w") as fh:
        fh.write(report)

def save_roc_curves(y_true, y_score, n_classes, class_names, out_path):

    from sklearn.preprocessing import label_binarize
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {roc_auc:.3f})")
    plt.plot([0,1], [0,1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def save_precision_recall_heatmap(y_true, y_pred, class_names, out_path):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=list(range(len(class_names))))
    data = np.vstack([p, r, f1])
    df = pd.DataFrame(data, index=["Precision", "Recall", "F1"], columns=class_names)
    plt.figure(figsize=(max(6, len(class_names)*0.8), 4))
    import seaborn as sns
    sns.heatmap(df, annot=True, fmt=".3f", cmap="YlGnBu")
    plt.title("Precision / Recall / F1")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def train_and_evaluate(csv_index, results_dir):

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])

    dataset = FrameDatasetFromCSV(csv_index, transform=transform)
    n_samples = len(dataset)

    torch.manual_seed(SEED)
    train_size = int(0.8 * n_samples)
    val_size = n_samples - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    df_index = pd.read_csv(csv_index)
    if "label" in df_index.columns:
        class_names = sorted(df_index["label"].unique(), key=lambda x: x)

    else:

        label_ids = df_index["label_id"].unique()
        class_names = [str(i) for i in sorted(label_ids.tolist())]

    if "label" in df_index.columns and "label_id" in df_index.columns:

        tmp = df_index[["label", "label_id"]].drop_duplicates().sort_values("label_id")
        class_names = tmp["label"].tolist()

    num_classes = len(class_names)
    model = create_model(num_classes)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)

    best_val_acc = 0.0
    history = []
    all_val_preds = []
    all_val_labels = []
    all_val_probs = []

    start_time = time.time()
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        epoch_preds = []
        epoch_labels = []
        epoch_probs = []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * imgs.size(0)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(probs, 1)

                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                epoch_preds.extend(preds.cpu().numpy())
                epoch_labels.extend(labels.cpu().numpy())
                epoch_probs.extend(probs.cpu().numpy())

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

        all_val_preds.extend(epoch_preds)
        all_val_labels.extend(epoch_labels)
        all_val_probs.extend(epoch_probs)

        scheduler.step(val_loss)

        if SAVE_EPOCH_MODELS:
            epoch_model_path = os.path.join(results_dir, f"model_epoch_{epoch}.pt")
            torch.save(model.state_dict(), epoch_model_path)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(results_dir, "best_model.pt"))

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch}/{NUM_EPOCHS} - train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, "
              f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}, epoch_time: {epoch_time:.1f}s")

    total_time = time.time() - start_time
    print(f"Training finished in {total_time/60:.2f} minutes. Best val acc: {best_val_acc:.4f}")

    torch.save(model.state_dict(), os.path.join(results_dir, f"last_model_epoch_{NUM_EPOCHS}.pt"))

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(os.path.join(results_dir, "metrics_log.csv"), index=False)

    y_true = np.array(all_val_labels)
    y_pred = np.array(all_val_preds)
    y_probs = np.array(all_val_probs)  

    save_confusion_matrix(y_true, y_pred, class_names, os.path.join(results_dir, "confusion_matrix.png"))
    save_classification_report(y_true, y_pred, class_names, os.path.join(results_dir, "classification_report.txt"))
    save_loss_acc_plots(
        hist_df["train_loss"].tolist(),
        hist_df["val_loss"].tolist(),
        (hist_df["train_acc"]*100).tolist(),
        (hist_df["val_acc"]*100).tolist(),
        results_dir
    )
    save_precision_recall_heatmap(y_true, y_pred, class_names, os.path.join(results_dir, "prec_recall_f1.png"))

    if num_classes >= 2:
        try:
            save_roc_curves(y_true, y_probs, num_classes, class_names, os.path.join(results_dir, "roc_curves.png"))
        except Exception as ex:
            print("Could not compute/save ROC curves:", ex)

    meta = {
        "dataset_index_csv": os.path.abspath(csv_index),
        "dataset_dir": os.path.abspath(DATASET_DIR),
        "results_dir": os.path.abspath(results_dir),
        "num_classes": num_classes,
        "class_names": class_names,
        "batch_size": BATCH_SIZE,
        "image_size": IMAGE_SIZE,
        "num_epochs": NUM_EPOCHS,
        "lr": LR,
        "device": str(DEVICE),
        "best_val_acc": float(best_val_acc)
    }
    pd.Series(meta).to_csv(os.path.join(results_dir, "run_metadata.csv"), header=False)

    print("All outputs saved to:", results_dir)
    return results_dir

if __name__ == "__main__":

    if not os.path.exists(INDEX_CSV):
        print("Building dataset index CSV...")
        df_index, label_map = build_index_csv(DATASET_DIR, INDEX_CSV)
        print(f"Index built with {len(df_index)} samples and {len(label_map)} classes.")
    else:
        print(f"Using existing index CSV: {INDEX_CSV}")
        df_index = pd.read_csv(INDEX_CSV)

        if "filepath" not in df_index.columns:
            raise KeyError("Index CSV must contain 'filepath' column.")

    train_and_evaluate(INDEX_CSV, RESULTS_DIR)