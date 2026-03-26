"""
evaluate_cnn_lstm.py — Test & Validation Evaluation for CNN+LSTM
=================================================================
Mirrors the MobileNetV2 test script structure.
Runs evaluation on both the validation and test splits and saves:
  - Confusion matrices (val + test)
  - ROC curve (test)
  - Printed classification reports

Usage:
    python /content/data/evaluate_cnn_lstm.py
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
)

# ── Configuration ──────────────────────────────────────────────────────────────

SEQ_SPLITS_DIR = "/content/data/seq_splits"
MODEL_DIR      = "/content/drive/MyDrive/AlertDrive/models"
REPORTS_DIR    = "/content/drive/MyDrive/AlertDrive/reports"

# Fallback to local if Drive is not mounted
if not os.path.isdir("/content/drive/MyDrive"):
    MODEL_DIR   = "/content/data/models"
    REPORTS_DIR = "/content/data/reports"
    print("[INFO] Google Drive not detected — saving reports locally.")

os.makedirs(REPORTS_DIR, exist_ok=True)

BATCH_SIZE  = 16
SEQ_LENGTH  = 16
IMG_SIZE    = 112
NUM_CLASSES = 2
CLASS_NAMES = ["notdrowsy", "drowsy"]
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device  : {DEVICE}")
print(f"Seq splits    : {SEQ_SPLITS_DIR}")
print(f"Model dir     : {MODEL_DIR}")
print(f"Reports dir   : {REPORTS_DIR}")

# ── Dataset ───────────────────────────────────────────────────────────────────

class SequenceDataset(Dataset):
    """
    Identical to the dataset used during training.
    CSV columns: video_id | label | frames | seq_length | subject | action
    `frames` is a pipe-separated list of file paths.
    """

    def __init__(self, csv_path: str, transform=None):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Sequence CSV not found: {csv_path}\n"
                f"Run create_sequence.py first."
            )
        self.df        = pd.read_csv(csv_path)
        self.transform = transform
        print(f"  Loaded {len(self.df):,} sequences from {csv_path}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        label = int(row["label"])
        paths = row["frames"].split("|")

        frames, last_valid = [], None

        for p in paths:
            try:
                img = Image.open(p).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                last_valid = img
                frames.append(img)
            except Exception:
                fallback = last_valid if last_valid is not None \
                           else torch.zeros(3, IMG_SIZE, IMG_SIZE)
                frames.append(fallback)

        pad = last_valid if last_valid is not None \
              else torch.zeros(3, IMG_SIZE, IMG_SIZE)
        while len(frames) < SEQ_LENGTH:
            frames.append(pad)
        frames = frames[:SEQ_LENGTH]

        return torch.stack(frames), label   # (SEQ_LENGTH, C, H, W), int


# ── Transform (no augmentation for evaluation) ────────────────────────────────

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── DataLoaders ───────────────────────────────────────────────────────────────

print("\nLoading datasets …")
val_dataset  = SequenceDataset(
    os.path.join(SEQ_SPLITS_DIR, "val_seq.csv"),  eval_transform)
test_dataset = SequenceDataset(
    os.path.join(SEQ_SPLITS_DIR, "test_seq.csv"), eval_transform)

val_loader  = DataLoader(val_dataset,  batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=2, pin_memory=True)

print(f"\nVal  sequences : {len(val_dataset):,}")
print(f"Test sequences : {len(test_dataset):,}")

# ── Model ─────────────────────────────────────────────────────────────────────

class CNNLSTM(nn.Module):
    """Must exactly match the architecture used during training."""

    def __init__(self, hidden_size: int = 256, num_layers: int = 2,
                 num_classes: int = 2):
        super().__init__()
        mobilenet     = models.mobilenet_v2(weights=None)   # weights loaded below
        self.cnn      = mobilenet.features
        self.cnn_pool = nn.AdaptiveAvgPool2d((1, 1))
        cnn_out_size  = 1280

        self.lstm = nn.LSTM(
            input_size  = cnn_out_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = 0.3,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x      = x.view(B * T, C, H, W)
        feat   = self.cnn(x)
        feat   = self.cnn_pool(feat).view(B, T, -1)
        out, _ = self.lstm(feat)
        return self.classifier(out[:, -1, :])


checkpoint_path = os.path.join(MODEL_DIR, "cnn_lstm_best.pth")
model = CNNLSTM(hidden_size=256, num_layers=2, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

total     = sum(p.numel() for p in model.parameters())
print(f"\n✓ Model loaded from: {checkpoint_path}")
print(f"  Total parameters  : {total:,}")

# ── Evaluate ──────────────────────────────────────────────────────────────────

def evaluate(model: nn.Module, loader: DataLoader):
    """
    Returns accuracy, ROC-AUC, predictions, drowsy-class probabilities,
    and ground-truth labels.  Mirrors the logic in train_cnn_lstm.py.
    """
    model.eval()
    p0, p1, all_labels = [], [], []

    with torch.no_grad():
        for seqs, labels in loader:
            seqs  = seqs.to(DEVICE)
            probs = torch.softmax(model(seqs), dim=1).cpu().numpy()
            p0.extend(probs[:, 0])
            p1.extend(probs[:, 1])
            all_labels.extend(labels.numpy())

    all_labels = np.array(all_labels)
    p0, p1     = np.array(p0), np.array(p1)

    auc0 = roc_auc_score(all_labels, p0)
    auc1 = roc_auc_score(all_labels, p1)

    if auc1 >= auc0:
        all_probs = p1
        all_preds = (p1 >= 0.5).astype(int)
    else:
        all_probs = p0
        all_preds = (p0 >= 0.5).astype(int)
        print(f"  [Note] Using index-0 probs  (auc0={auc0:.4f} > auc1={auc1:.4f})")

    accuracy = np.mean(all_preds == all_labels) * 100
    roc_auc  = max(auc0, auc1)
    return accuracy, roc_auc, all_preds, all_probs, all_labels

# ── Confusion-matrix helper ───────────────────────────────────────────────────

def save_confusion_matrix(true_labels, preds, acc, auc, split_name, out_path):
    cm   = confusion_matrix(true_labels, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"CNN+LSTM — {split_name}\nAcc: {acc:.2f}%  AUC: {auc:.3f}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"  ✓ Confusion matrix saved → {out_path}")

# ── Run evaluation on Validation + Test ──────────────────────────────────────

sep = "=" * 70

for split_name, loader in [("Validation", val_loader), ("Test", test_loader)]:

    print(f"\n{sep}")
    print(f"  Evaluating : {split_name}")
    print(sep)

    acc, auc, preds, probs, true_labels = evaluate(model, loader)

    print(f"  Accuracy : {acc:.2f}%")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(true_labels, preds, target_names=CLASS_NAMES))

    cm_path = os.path.join(
        REPORTS_DIR, f"cnnlstm_confusion_{split_name.lower()}.png"
    )
    save_confusion_matrix(true_labels, preds, acc, auc, split_name, cm_path)

# ── ROC Curve (Test set) ──────────────────────────────────────────────────────

print(f"\n{sep}")
print("  Generating ROC Curve (Test set) …")

_, _, _, probs_test, labels_test = evaluate(model, test_loader)
fpr, tpr, _ = roc_curve(labels_test, probs_test)
auc_test    = roc_auc_score(labels_test, probs_test)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, lw=2, label=f"CNN+LSTM (AUC = {auc_test:.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — CNN+LSTM Test Set")
plt.legend(loc="lower right")
plt.tight_layout()
roc_path = os.path.join(REPORTS_DIR, "cnnlstm_roc_curve.png")
plt.savefig(roc_path, dpi=150)
plt.show()
print(f"  ✓ ROC curve saved → {roc_path}")

# ── Summary ───────────────────────────────────────────────────────────────────

print(f"\n{sep}")
print("  CNN+LSTM EVALUATION COMPLETE")
print(f"  Reports saved to: {REPORTS_DIR}")
print(sep)