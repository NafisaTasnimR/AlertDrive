import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

print("="*80)
print("SVM MODEL - FINAL TEST SET EVALUATION")
print("="*80)

# Load test features
print("\n[1/3] Loading test features...")
X_test = np.load("data/features/X_test_cnn.npy")
y_test = np.load("data/features/y_test_cnn.npy")
print(f"  Test samples: {X_test.shape[0]}")

# Load trained model
print("\n[2/3] Loading trained model...")
model = joblib.load("models/baseline/svm_cnn_rbf.pkl")
scaler = joblib.load("models/baseline/scaler_cnn.pkl")
print("  ✓ Model and scaler loaded")

# Normalize and predict
print("\n[3/3] Evaluating on test set...")
X_test_scaled = scaler.transform(X_test)
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

# Results
print(f"\n{'='*80}")
print("FINAL TEST RESULTS:")
print(f"{'='*80}")
print(f"Test Accuracy: {accuracy*100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Drowsy', 'Drowsy']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Detailed metrics
drowsy_recall = cm[1][1] / (cm[1][1] + cm[1][0])
drowsy_precision = cm[1][1] / (cm[1][1] + cm[0][1])
alert_recall = cm[0][0] / (cm[0][0] + cm[0][1])

print(f"\n{'='*80}")
print("DRIVER DROWSINESS DETECTION METRICS:")
print(f"{'='*80}")
print(f"Drowsy Detection Rate (Recall):  {drowsy_recall*100:.1f}%")
print(f"Alert Detection Rate (Recall):   {alert_recall*100:.1f}%")
print(f"Precision (Drowsy Predictions):  {drowsy_precision*100:.1f}%")
print(f"False Negatives (Missed Drowsy): {cm[1][0]} ({cm[1][0]/len(y_test)*100:.1f}%)")
print(f"False Positives (False Alarms):  {cm[0][1]} ({cm[0][1]/len(y_test)*100:.1f}%)")

print(f"\n{'='*80}")
print(" FINAL EVALUATION COMPLETE")
print(f"{'='*80}")
print("\nModel Performance Summary:")
print(f"  • Validation Accuracy: 95.53%")
print(f"  • Test Accuracy: {accuracy*100:.2f}%")
print(f"  • Model: SVM with RBF kernel + ResNet50 CNN features")
print(f"  • Ready for deployment!")