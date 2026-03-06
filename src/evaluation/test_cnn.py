import sys
import os
import json

# -------------------------------------------------
# Add project root to path
# -------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)

from src.models.baseline.hf_cnn_resnet import HFCNNResNet
from src.preprocessing.hf_dataloader import get_dataloaders

# -------------------------------------------------
# Create Results Folder
# -------------------------------------------------
RESULT_DIR = "results/cnn_ddd"
os.makedirs(RESULT_DIR, exist_ok=True)

# -------------------------------------------------
# Device
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------
# Load Model
# -------------------------------------------------
model = HFCNNResNet(num_classes=2).to(device)

model.load_state_dict(
    torch.load(
        "trained_model/best_model_ddd.pth",
        map_location=device,
    )
)

model.eval()
processor = model.processor

# -------------------------------------------------
# Load Test Data
# -------------------------------------------------
_, test_loader = get_dataloaders(
    "splits/ddd_test.csv",
    "splits/ddd_test.csv",
    processor,
)

# -------------------------------------------------
# Inference
# -------------------------------------------------
all_preds = []
all_probs = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        probs = torch.softmax(outputs, dim=1)[:, 1]
        _, preds = torch.max(outputs, 1)

        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)

# -------------------------------------------------
# Classification Report
# -------------------------------------------------
print("\n================ Classification Report ================\n")

report_dict = classification_report(
    all_labels,
    all_preds,
    output_dict=True,
)

report_text = classification_report(all_labels, all_preds)
print(report_text)

with open(os.path.join(RESULT_DIR, "classification_report.txt"), "w") as f:
    f.write(report_text)

# -------------------------------------------------
# Confusion Matrix
# -------------------------------------------------
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

cm_path = os.path.join(RESULT_DIR, "confusion_matrix.png")
plt.savefig(cm_path)
plt.close()

# -------------------------------------------------
# ROC Curve + AUC
# -------------------------------------------------
fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()

roc_path = os.path.join(RESULT_DIR, "roc_curve.png")
plt.savefig(roc_path)
plt.close()

with open(os.path.join(RESULT_DIR, "roc_auc.txt"), "w") as f:
    f.write(f"ROC AUC Score: {roc_auc:.4f}")

print(f"\nROC AUC Score: {roc_auc:.4f}")

# -------------------------------------------------
# Save Metrics (Accuracy, Precision, Recall, F1, AUC, CM)
# -------------------------------------------------

metrics_dict = {
    "accuracy": report_dict["accuracy"],
    "precision": report_dict["1"]["precision"]
    if "1" in report_dict else None,
    "recall": report_dict["1"]["recall"]
    if "1" in report_dict else None,
    "f1_score": report_dict["1"]["f1-score"]
    if "1" in report_dict else None,
    "auc": float(roc_auc),
    "confusion_matrix": cm.tolist(),
}

metrics_path = os.path.join(RESULT_DIR, "metrics.json")

with open(metrics_path, "w") as f:
    json.dump(metrics_dict, f, indent=4)

print(f"\nMetrics saved → {metrics_path}")

# -------------------------------------------------
# Show & Save Sample Predictions
# -------------------------------------------------

def show_samples(images, labels, preds, num=5):
    images = images[:num]
    labels = labels[:num]
    preds = preds[:num]

    fig, axes = plt.subplots(1, num, figsize=(15, 3))

    for i in range(num):
        img = images[i].permute(1, 2, 0).cpu().numpy()
        axes[i].imshow(img)
        axes[i].set_title(f"P:{preds[i]} | T:{labels[i]}")
        axes[i].axis("off")

    sample_path = os.path.join(RESULT_DIR, "sample_predictions.png")
    plt.savefig(sample_path)
    plt.close()


# Visualize one batch
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)

        _, preds = torch.max(outputs, 1)

        show_samples(images, labels.numpy(), preds.cpu().numpy())
        break

print(f"\nAll results saved inside → {RESULT_DIR}/")