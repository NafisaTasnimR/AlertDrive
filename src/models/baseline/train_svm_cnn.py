import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib
import os
import time

print("="*80)
print("SVM TRAINING WITH CNN FEATURES ")
print("="*80)

# Load CNN features
print("\n[1/6] Loading CNN features...")
X_train = np.load("data/features/X_train_cnn.npy")
y_train = np.load("data/features/y_train_cnn.npy")
X_val = np.load("data/features/X_val_cnn.npy")
y_val = np.load("data/features/y_val_cnn.npy")

print(f"  Full training samples: {X_train.shape[0]}")
print(f"  Feature dimensions: {X_train.shape[1]}")

# Use subset for faster training
SUBSET_SIZE = 15000  
print(f"\n[2/6] Using subset of {SUBSET_SIZE} samples...")

X_train_subset, _, y_train_subset, _ = train_test_split(
    X_train, y_train, 
    train_size=SUBSET_SIZE, 
    stratify=y_train, 
    random_state=42
)

print(f"  Subset samples: {X_train_subset.shape[0]}")
print(f"  Class distribution: Drowsy={sum(y_train_subset)}, Not Drowsy={len(y_train_subset)-sum(y_train_subset)}")

# Better normalization
print("\n[3/6] Normalizing features with StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_subset)
X_val_scaled = scaler.transform(X_val)

# Check normalization
print(f"  Mean: {X_train_scaled.mean():.6f} (should be ~0)")
print(f"  Std:  {X_train_scaled.std():.6f} (should be ~1)")
print("   Features normalized")

# Try RBF kernel (often better for CNN features)
print("\n[4/6] Training RBF SVM (better for complex patterns)...")
start_time = time.time()

# Increased max_iter and using RBF
model = SVC(
    kernel="rbf",      # RBF often works better than linear for CNN features
    C=10,              # Regularization
    gamma='scale',     # Auto-compute gamma
    max_iter=5000,     # Increased from 1000
    verbose=True,
    cache_size=500     # More cache for speed
)

model.fit(X_train_scaled, y_train_subset)

train_time = time.time() - start_time
print(f"\n   Training completed in {train_time:.2f} seconds")

# Evaluate
print("\n[5/6] Evaluating model...")
y_pred = model.predict(X_val_scaled)
accuracy = accuracy_score(y_val, y_pred)

print(f"\n{'='*80}")
print("VALIDATION RESULTS (RBF SVM):")
print(f"{'='*80}")
print(f"Accuracy: {accuracy*100:.2f}%")
print(f"Training Time: {train_time:.2f}s")
print(f"Trained on: {SUBSET_SIZE} samples")
print(f"Kernel: RBF (better for non-linear patterns)")

print("\nClassification Report:")
print(classification_report(y_val, y_pred, target_names=['Not Drowsy', 'Drowsy']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_val, y_pred)
print(cm)
print(f"\n  True Negatives:  {cm[0][0]:>6} (correctly identified alert)")
print(f"  False Positives: {cm[0][1]:>6} (alert classified as drowsy)")
print(f"  False Negatives: {cm[1][0]:>6} (drowsy classified as alert) CRITICAL")
print(f"  True Positives:  {cm[1][1]:>6} (correctly identified drowsy)")

# Calculate important metrics for drowsiness detection
drowsy_recall = cm[1][1] / (cm[1][1] + cm[1][0])
drowsy_precision = cm[1][1] / (cm[1][1] + cm[0][1])

print(f"\n DROWSINESS DETECTION METRICS:")
print(f"  Recall (catch drowsy drivers):    {drowsy_recall*100:.1f}%")
print(f"  Precision (avoid false alarms):   {drowsy_precision*100:.1f}%")

# Save model
print(f"\n[6/6] Saving model...")
os.makedirs("models/baseline", exist_ok=True)
joblib.dump(model, "models/baseline/svm_cnn_rbf.pkl")
joblib.dump(scaler, "models/baseline/scaler_cnn.pkl")

print(f"\n{'='*80}")
print("TRAINING COMPLETE")
print(f"{'='*80}")
print(f"\nModel saved: models/baseline/svm_cnn_rbf.pkl")
print(f"Scaler saved: models/baseline/scaler_cnn.pkl")
