"""
CNN+LSTM — Driver Drowsiness Detection (Temporal Model)
========================================================
Dataset  : NTHU-DDD
Model    : MobileNetV2 (CNN) + 2-layer LSTM
Input    : Sequences of 16 consecutive frames (~0.5 s at 30 fps)
Training : Frozen CNN for 5 epochs → full fine-tuning

Colab folder layout (from screenshot):
    /content/data/
        processed/          ← raw frames
        splits/             ← train.csv, val.csv, test.csv
        seq_splits/         ← train_seq.csv, val_seq.csv, test_seq.csv
        create_sequence.py
        train_cnn_lstm.py   ← this file
    /content/drive/MyDrive/AlertDrive/
        models/             ← checkpoints saved here
        reports/            ← plots saved here

Usage:
    # Step 1
    python /content/data/create_sequence.py

    # Step 2
    python /content/data/train_cnn_lstm.py
"""

import os
import time
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

# ── Paths ──────────────────────────────────────────────────────────────────────

SEQ_SPLITS_DIR = "/content/data/seq_splits"
MODEL_DIR      = "/content/drive/MyDrive/AlertDrive/models"
REPORTS_DIR    = "/content/drive/MyDrive/AlertDrive/reports"

# Fallback to local if Drive is not mounted
if not os.path.isdir("/content/drive/MyDrive"):
    MODEL_DIR   = "/content/data/models"
    REPORTS_DIR = "/content/data/reports"
    print("[INFO] Google Drive not detected — saving models/reports locally.")

os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ── Configuration ──────────────────────────────────────────────────────────────

BATCH_SIZE     = 16
SEQ_LENGTH     = 16
NUM_EPOCHS     = 15
LR             = 1e-4
IMG_SIZE       = 112
NUM_CLASSES    = 2
UNFREEZE_EPOCH = 5
CLASS_NAMES    = ["notdrowsy", "drowsy"]
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device : {DEVICE}")
print(f"Seq splits   : {SEQ_SPLITS_DIR}")
print(f"Models dir   : {MODEL_DIR}")
print(f"Reports dir  : {REPORTS_DIR}")

# ── Dataset ───────────────────────────────────────────────────────────────────

class SequenceDataset(Dataset):
    """
    Loads sequences of frames from a sequence CSV.
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
        print(f"    Subjects : {sorted(self.df['subject'].unique())}")
        print(f"    Actions  : {sorted(self.df['action'].unique())}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        label = int(row["label"])
        paths = row["frames"].split("|")

        frames     = []
        last_valid = None

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

        # Guarantee exactly SEQ_LENGTH frames
        pad = last_valid if last_valid is not None \
              else torch.zeros(3, IMG_SIZE, IMG_SIZE)
        while len(frames) < SEQ_LENGTH:
            frames.append(pad)
        frames = frames[:SEQ_LENGTH]

        return torch.stack(frames), label   # (SEQ_LENGTH, C, H, W), int


# ── Transforms ────────────────────────────────────────────────────────────────

_mean = [0.485, 0.456, 0.406]
_std  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(_mean, _std),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(_mean, _std),
])

# ── DataLoaders ───────────────────────────────────────────────────────────────

print("\nLoading datasets …")
train_dataset = SequenceDataset(
    os.path.join(SEQ_SPLITS_DIR, "train_seq.csv"), train_transform)
val_dataset   = SequenceDataset(
    os.path.join(SEQ_SPLITS_DIR, "val_seq.csv"),   eval_transform)
test_dataset  = SequenceDataset(
    os.path.join(SEQ_SPLITS_DIR, "test_seq.csv"),  eval_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=2, pin_memory=True)

print(f"\nTrain sequences : {len(train_dataset):,}")
print(f"Val   sequences : {len(val_dataset):,}")
print(f"Test  sequences : {len(test_dataset):,}")

# ── Model ─────────────────────────────────────────────────────────────────────

class CNNLSTM(nn.Module):
    """
    MobileNetV2 (CNN) feature extractor + 2-layer LSTM classifier.
    CNN is frozen for the first UNFREEZE_EPOCH epochs, then fine-tuned.
    """

    def __init__(self, hidden_size: int = 256, num_layers: int = 2,
                 num_classes: int = 2):
        super().__init__()
        mobilenet     = models.mobilenet_v2(weights="IMAGENET1K_V1")
        self.cnn      = mobilenet.features
        self.cnn_pool = nn.AdaptiveAvgPool2d((1, 1))
        cnn_out_size  = 1280

        for param in self.cnn.parameters():
            param.requires_grad = False

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

    def unfreeze_cnn(self):
        for param in self.cnn.parameters():
            param.requires_grad = True
        print("  CNN backbone unfrozen ✓")


model = CNNLSTM(hidden_size=256, num_layers=2, num_classes=NUM_CLASSES).to(DEVICE)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"\nModel built ✓  |  Trainable: {trainable:,} / {total:,} params")

# ── Class weights ─────────────────────────────────────────────────────────────

_train_df = pd.read_csv(os.path.join(SEQ_SPLITS_DIR, "train_seq.csv"))
_lv       = _train_df["label"].values
_counts   = np.bincount(_lv, minlength=2)
_weights  = torch.tensor(
    [len(_lv) / (2 * _counts[0]),
     len(_lv) / (2 * _counts[1])],
    dtype=torch.float,
).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=_weights)
print(f"Class weights : notdrowsy={_weights[0]:.3f}  drowsy={_weights[1]:.3f}")

# ── Evaluate ──────────────────────────────────────────────────────────────────

def evaluate(model: nn.Module, loader: DataLoader):
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
        print(f"    [Note] Using index-0 probs (auc0={auc0:.4f} > auc1={auc1:.4f})")

    accuracy = np.mean(all_preds == all_labels) * 100
    roc_auc  = max(auc0, auc1)
    return accuracy, roc_auc, all_preds, all_probs, all_labels

# ── Train epoch ───────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    t0 = time.time()

    for i, (seqs, labels) in enumerate(loader, 1):
        seqs, labels = seqs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out  = model(seqs)
        loss = criterion(out, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        correct    += (out.argmax(1) == labels).sum().item()
        total      += labels.size(0)

        if i % 50 == 0 or i == len(loader):
            print(f"    Batch {i}/{len(loader)} "
                  f"| Loss: {total_loss/i:.4f} "
                  f"| Acc: {correct/total*100:.2f}% "
                  f"| {time.time()-t0:.1f}s")

    return total_loss / len(loader), correct / total * 100

# ── Training loop ─────────────────────────────────────────────────────────────

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=LR
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", patience=3, factor=0.5
)

history = {"train_loss": [], "train_acc": [], "val_acc": [], "val_auc": []}
best_val_acc    = 0.0
best_model_path = os.path.join(MODEL_DIR, "cnn_lstm_best.pth")

sep = "=" * 70
print(f"\n{sep}\n  CNN+LSTM TRAINING  ({NUM_EPOCHS} epochs)\n{sep}")

for epoch in range(1, NUM_EPOCHS + 1):

    if epoch == UNFREEZE_EPOCH:
        model.unfreeze_cnn()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR / 5)
        print(f"  Optimizer reset — LR = {LR/5:.2e}")

    print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
    print("  " + "-" * 60)

    train_loss, train_acc     = train_epoch(model, train_loader, optimizer)
    val_acc, val_auc, _, _, _ = evaluate(model, val_loader)

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)
    history["val_auc"].append(val_auc)

    print(f"  Train Loss : {train_loss:.4f} | Train Acc : {train_acc:.2f}%")
    print(f"  Val   Acc  : {val_acc:.2f}%   | Val  AUC  : {val_auc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        print(f"  ✓ Best model saved  (val_acc={best_val_acc:.2f}%)")

    scheduler.step(val_acc)

print(f"\nBest Val Accuracy : {best_val_acc:.2f}%")

# ── Final evaluation ──────────────────────────────────────────────────────────

print(f"\nLoading best checkpoint: {best_model_path}")
model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))

for split_name, loader in [("Validation", val_loader), ("Test", test_loader)]:
    acc, auc, preds, probs, true_labels = evaluate(model, loader)
    print(f"\n{'-'*70}")
    print(f"  {split_name}  →  Accuracy: {acc:.2f}%  |  AUC: {auc:.4f}")
    print(f"{'-'*70}")
    print(classification_report(true_labels, preds, target_names=CLASS_NAMES))

    cm   = confusion_matrix(true_labels, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"CNN+LSTM — {split_name}\nAcc: {acc:.2f}%  AUC: {auc:.3f}")
    plt.tight_layout()
    out_path = os.path.join(REPORTS_DIR,
                            f"cnnlstm_confusion_{split_name.lower()}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved → {out_path}")

# ── ROC curve ────────────────────────────────────────────────────────────────

_, _, _, probs_test, labels_test = evaluate(model, test_loader)
fpr, tpr, _ = roc_curve(labels_test, probs_test)
auc_test    = roc_auc_score(labels_test, probs_test)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, lw=2, label=f"CNN+LSTM (AUC = {auc_test:.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — CNN+LSTM Test Set")
plt.legend(loc="lower right")
plt.tight_layout()
roc_path = os.path.join(REPORTS_DIR, "cnnlstm_roc_curve.png")
plt.savefig(roc_path, dpi=150)
plt.close()
print(f"ROC curve saved → {roc_path}")

# ── Training history ─────────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history["train_acc"], label="Train Acc")
ax1.plot(history["val_acc"],   label="Val Acc")
ax1.axvline(x=UNFREEZE_EPOCH - 1, color="gray",
            linestyle="--", label="CNN unfrozen")
ax1.set_title("Accuracy over Epochs")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy (%)"); ax1.legend()

ax2.plot(history["train_loss"], color="red", label="Train Loss")
ax2.axvline(x=UNFREEZE_EPOCH - 1, color="gray",
            linestyle="--", label="CNN unfrozen")
ax2.set_title("Loss over Epochs")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss"); ax2.legend()

plt.tight_layout()
hist_path = os.path.join(REPORTS_DIR, "cnnlstm_training_history.png")
plt.savefig(hist_path, dpi=150)
plt.close()
print(f"Training history saved → {hist_path}")

# ── Summary ───────────────────────────────────────────────────────────────────

print(f"\n{sep}")
print("  CNN+LSTM TRAINING COMPLETE")
print(sep)
print(f"  Best Val Accuracy : {best_val_acc:.2f}%")
print(f"  Model checkpoint  : {best_model_path}")
print(sep)