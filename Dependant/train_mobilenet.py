"""
MobileNetV2 — Driver Drowsiness Detection (Pretrained CNN)
===========================================================
Dataset  : NTHU-DDD (with balanced subject-dependent splits)
Model    : MobileNetV2 pretrained on ImageNet, fine-tuned for binary classification
Training : Two-phase — frozen base (5 epochs) then full fine-tuning (10 epochs)

All paths configured for Colab with flat dataset structure:
  /content/processed/drowsy/     - all drowsy images
  /content/processed/notdrowsy/  - all notdrowsy images
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

# ── Mount Google Drive (for saving models) ───────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

# ── Configuration ─────────────────────────────────────────────────────────────
SPLITS_DIR          = "/content/splits"
MODEL_DIR           = "/content/drive/MyDrive/AlertDrive/models"
REPORTS_DIR         = "/content/drive/MyDrive/AlertDrive/reports"

BATCH_SIZE          = 32
NUM_EPOCHS_FROZEN   = 5    # Phase 1: classifier head only
NUM_EPOCHS_FINETUNE = 10   # Phase 2: full model fine-tuning
LR_FROZEN           = 1e-3
LR_FINETUNE         = 1e-4
IMG_SIZE            = 224
NUM_CLASSES         = 2
CLASS_NAMES         = ["notdrowsy", "drowsy"]  # 0=notdrowsy, 1=drowsy
DEVICE              = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
print(f"Using device: {DEVICE}")

# ── Dataset ───────────────────────────────────────────────────────────────────
class DrowsinessDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, int(row["label"])

# Data augmentation for training
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# No augmentation for validation/test
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Load datasets
print("\n" + "="*70)
print("LOADING DATASETS")
print("="*70)

train_dataset = DrowsinessDataset(f"{SPLITS_DIR}/train.csv", train_transform)
val_dataset   = DrowsinessDataset(f"{SPLITS_DIR}/val.csv",   val_transform)
test_dataset  = DrowsinessDataset(f"{SPLITS_DIR}/test.csv",  val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=2, pin_memory=True)

print(f"Train: {len(train_dataset):,} images")
print(f"Val:   {len(val_dataset):,} images")
print(f"Test:  {len(test_dataset):,} images")

# Print class distribution
train_labels = train_dataset.df['label'].values
print(f"\nTrain class distribution:")
print(f"  Not drowsy: {(train_labels==0).sum():,} ({(train_labels==0).mean()*100:.1f}%)")
print(f"  Drowsy:     {(train_labels==1).sum():,} ({train_labels.mean()*100:.1f}%)")

# ── Model ─────────────────────────────────────────────────────────────────────
def build_model(freeze_base=True):
    model = models.mobilenet_v2(weights="IMAGENET1K_V1")
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.last_channel, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, NUM_CLASSES),
    )
    if freeze_base:
        for param in model.features.parameters():
            param.requires_grad = False
        print("\nBase frozen — training classifier head only")
    else:
        for param in model.parameters():
            param.requires_grad = True
        print("\nFull model unfrozen — fine-tuning all layers")
    return model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

# ── Evaluation Function ──────────────────────────────────────────────────────
def evaluate(model, loader):
    """Evaluate model and return metrics"""
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            drowsy_probs = probs[:, 1].cpu().numpy()
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(drowsy_probs)
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    accuracy = np.mean(all_preds == all_labels) * 100
    roc_auc = roc_auc_score(all_labels, all_probs)
    
    return accuracy, roc_auc, all_preds, all_probs, all_labels

# ── Training Epoch ───────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    t0 = time.time()
    
    for batch_idx, (imgs, labels) in enumerate(loader, 1):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        if batch_idx % 100 == 0:
            pct = batch_idx / len(loader) * 100
            print(f"    Batch {batch_idx}/{len(loader)} ({pct:.0f}%) | "
                  f"Loss: {total_loss/batch_idx:.4f} | "
                  f"Acc: {correct/total*100:.2f}% | "
                  f"{time.time()-t0:.1f}s")
    
    return total_loss / len(loader), correct / total * 100

# ── Training Loop ────────────────────────────────────────────────────────────
history = {"train_loss": [], "train_acc": [], "val_acc": [], "val_auc": []}
best_val_acc = 0

def run_training(model, num_epochs, lr, phase_name):
    global best_val_acc
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    
    print(f"\n{'='*70}")
    print(f"  {phase_name}")
    print(f"{'='*70}")

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs} — LR: {scheduler.get_last_lr()[0]:.6f}")
        print("  " + "-" * 60)
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer)
        val_acc, val_auc, _, _, _ = evaluate(model, val_loader)
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)
        
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val   Acc:  {val_acc:.2f}%   | Val AUC:   {val_auc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),
                       os.path.join(MODEL_DIR, "mobilenet_best.pth"))
            print(f"  ✓ Best model saved (val_acc={best_val_acc:.2f}%)")
        
        scheduler.step()
    
    return model

# ── Phase 1: Frozen Base ─────────────────────────────────────────────────────
model = build_model(freeze_base=True)
model = run_training(model, NUM_EPOCHS_FROZEN, LR_FROZEN,
                     "PHASE 1 — CLASSIFIER ONLY")

# ── Phase 2: Full Fine-tuning ─────────────────────────────────────────────────
for param in model.features.parameters():
    param.requires_grad = True
model = run_training(model, NUM_EPOCHS_FINETUNE, LR_FINETUNE,
                     "PHASE 2 — FULL FINE-TUNING")

# ── Final Evaluation ─────────────────────────────────────────────────────────
print("\n" + "="*70)
print("FINAL EVALUATION")
print("="*70)

# Load best model
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "mobilenet_best.pth")))

for split_name, loader in [("Validation", val_loader), ("Test", test_loader)]:
    print(f"\n{split_name.upper()} SET RESULTS:")
    print("-" * 50)
    
    acc, auc, preds, probs, labels_eval = evaluate(model, loader)
    print(f"  Accuracy : {acc:.2f}%")
    print(f"  ROC-AUC  : {auc:.4f}")
    print()
    print(classification_report(labels_eval, preds, target_names=CLASS_NAMES))
    
    # Confusion matrix
    cm = confusion_matrix(labels_eval, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"MobileNetV2 — {split_name}\nAcc: {acc:.2f}%  AUC: {auc:.3f}")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, f"mobilenet_confusion_{split_name.lower()}.png"), dpi=150)
    plt.close()
    print(f"  ✓ Confusion matrix saved")

# ── ROC Curve ─────────────────────────────────────────────────────────────────
_, _, _, probs_test, labels_test = evaluate(model, test_loader)
fpr, tpr, _ = roc_curve(labels_test, probs_test)
auc = roc_auc_score(labels_test, probs_test)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, lw=2, label=f"MobileNetV2 (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.500)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — MobileNetV2 Test Set")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "mobilenet_roc_curve.png"), dpi=150)
plt.close()
print("✓ ROC curve saved")

# ── Training History Plot ────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history["train_acc"], label="Train Acc", linewidth=2)
ax1.plot(history["val_acc"], label="Val Acc", linewidth=2)
ax1.axvline(x=NUM_EPOCHS_FROZEN - 0.5, color="gray", linestyle="--", 
            label="Fine-tune start", alpha=0.7)
ax1.set_title("Accuracy over Epochs")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy (%)")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(history["train_loss"], label="Train Loss", color="red", linewidth=2)
ax2.axvline(x=NUM_EPOCHS_FROZEN - 0.5, color="gray", linestyle="--", 
            label="Fine-tune start", alpha=0.7)
ax2.set_title("Loss over Epochs")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "mobilenet_training_history.png"), dpi=150)
plt.close()
print("✓ Training history plot saved")

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  MOBILENETV2 TRAINING COMPLETE")
print("="*70)
print(f"  Best Validation Accuracy : {best_val_acc:.2f}%")
print(f"  Test Accuracy            : {acc:.2f}%")
print(f"  Test ROC-AUC             : {auc:.4f}")
print(f"\n  Models saved to: {MODEL_DIR}/mobilenet_best.pth")
print(f"  Reports saved to: {REPORTS_DIR}/")
print("="*70)
