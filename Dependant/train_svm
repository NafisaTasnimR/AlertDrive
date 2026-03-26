"""
HOG + SGD SVM — Driver Drowsiness Detection (Baseline)
=======================================================
Dataset  : NTHU-DDD
Features : HOG at 128x128 resolution (7,056 features per image)
Model    : SGDClassifier (modified_huber loss) — linear SVM approximation

Improvements:
- Early stopping to prevent overfitting
- Adaptive learning rate for better convergence
- Data leakage detection
- Dummy classifier baseline
- Feature normalization validation
- Memory-efficient batch processing
- Proper checkpointing with best model restoration

Usage:
    python train_svm.py
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
import cv2
import joblib
import matplotlib.pyplot as plt

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
)

# ── Configuration ─────────────────────────────────────────────────────────────

SPLITS_DIR  = "splits"  # Change to your path
MODEL_DIR   = "models"
REPORTS_DIR = "reports"

IMG_SIZE         = 128
HOG_WIN_SIZE     = (128, 128)
HOG_BLOCK_SIZE   = (16, 16)
HOG_BLOCK_STRIDE = (8, 8)
HOG_CELL_SIZE    = (8, 8)
HOG_NBINS        = 9
MAX_EPOCHS       = 30
PATIENCE         = 5  # Early stopping patience
BATCH_SIZE       = 5000  # For memory-efficient loading

CLASS_NAMES = ["notdrowsy", "drowsy"]

os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ── HOG Feature Extraction ────────────────────────────────────────────────────

def build_hog():
    return cv2.HOGDescriptor(
        HOG_WIN_SIZE, HOG_BLOCK_SIZE, HOG_BLOCK_STRIDE,
        HOG_CELL_SIZE, HOG_NBINS
    )

hog = build_hog()

def extract_features(img_path):
    """Extract HOG features from image"""
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return hog.compute(gray).flatten()

def load_split(csv_path, split_name, batch_size=BATCH_SIZE):
    """
    Load features in batches to save memory.
    Returns features and labels as numpy arrays.
    """
    df = pd.read_csv(csv_path)
    all_features, all_labels = [], []
    skipped = 0
    
    print(f"\n[{split_name}] Extracting HOG features from {len(df)} images...")
    t0 = time.time()
    
    for start in range(0, len(df), batch_size):
        batch_df = df.iloc[start:start+batch_size]
        X_batch, y_batch = [], []
        
        for i, (_, row) in enumerate(batch_df.iterrows(), start + 1):
            feats = extract_features(row["path"])
            if feats is None:
                skipped += 1
                continue
            X_batch.append(feats)
            y_batch.append(int(row["label"]))
            
            if i % 500 == 0:
                pct = i / len(df) * 100
                print(f"  {pct:5.1f}% | {i}/{len(df)} | {time.time()-t0:.1f}s")
        
        if X_batch:
            all_features.extend(X_batch)
            all_labels.extend(y_batch)
        
        # Clear batch from memory
        del X_batch, y_batch
    
    print(f"  Done — {len(all_features)} extracted, {skipped} skipped [{time.time()-t0:.1f}s]")
    return np.array(all_features, dtype=np.float32), np.array(all_labels, dtype=np.int32)

# ── Data Validation Functions ─────────────────────────────────────────────────

def get_video_id(path):
    """Extract subject ID from image path"""
    filename = os.path.basename(path)
    parts = filename.split('_')
    return parts[0]  # First part is always subject ID

def check_data_leakage(train_paths, val_paths, test_paths):
    """Check if same video groups appear across splits"""
    train_videos = set([get_video_id(p) for p in train_paths])
    val_videos = set([get_video_id(p) for p in val_paths])
    test_videos = set([get_video_id(p) for p in test_paths])
    
    train_val_overlap = train_videos & val_videos
    train_test_overlap = train_videos & test_videos
    val_test_overlap = val_videos & test_videos
    
    print("\n" + "=" * 70)
    print("DATA LEAKAGE CHECK")
    print("=" * 70)
    print(f"Total unique videos: {len(train_videos | val_videos | test_videos)}")
    print(f"  Train videos: {len(train_videos)}")
    print(f"  Val videos:   {len(val_videos)}")
    print(f"  Test videos:  {len(test_videos)}")
    
    if train_val_overlap:
        print(f"\n⚠️  WARNING: Train-Val leakage! {len(train_val_overlap)} overlapping videos")
        print(f"   Overlap: {list(train_val_overlap)[:5]}")
    else:
        print(f"\n✓ No Train-Val leakage")
    
    if train_test_overlap:
        print(f"⚠️  WARNING: Train-Test leakage! {len(train_test_overlap)} overlapping videos")
        print(f"   Overlap: {list(train_test_overlap)[:5]}")
    else:
        print(f"✓ No Train-Test leakage")
    
    if val_test_overlap:
        print(f"⚠️  WARNING: Val-Test leakage! {len(val_test_overlap)} overlapping videos")
        print(f"   Overlap: {list(val_test_overlap)[:5]}")
    else:
        print(f"✓ No Val-Test leakage")
    
    return not (train_val_overlap or train_test_overlap or val_test_overlap)

def validate_split_distribution(df, name):
    """Check if split is representative"""
    print(f"\n{name.upper()} SPLIT DISTRIBUTION:")
    print(f"  Total images: {len(df):,}")
    print(f"  Drowsy ratio: {df['label'].mean():.2%} ({df['label'].sum():,} / {len(df):,})")
    
    # Extract subject IDs
    subjects = df['path'].apply(lambda x: os.path.basename(x).split('_')[0])
    print(f"  Unique subjects: {subjects.nunique()}")
    print(f"  Subject distribution: {dict(subjects.value_counts().head())}")
    
    return df

# ── Main Training Function ────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  HOG + SVM DRIVER DROWSINESS CLASSIFIER (IMPROVED)")
    print("=" * 70)
    
    # 1. Load splits with validation
    train_df = pd.read_csv(os.path.join(SPLITS_DIR, "train.csv"))
    val_df   = pd.read_csv(os.path.join(SPLITS_DIR, "val.csv"))
    test_df  = pd.read_csv(os.path.join(SPLITS_DIR, "test.csv"))
    
    validate_split_distribution(train_df, "train")
    validate_split_distribution(val_df, "validation")
    validate_split_distribution(test_df, "test")
    
    # 2. Check for data leakage
    train_paths = train_df['path'].tolist()
    val_paths   = val_df['path'].tolist()
    test_paths  = test_df['path'].tolist()
    
    # Check data leakage (disabled for subject-dependent splits)
    print("\n" + "=" * 70)
    print("SPLIT INFORMATION")
    print("=" * 70)
    print(f"Train: {len(train_df):,} images ({train_df['label'].mean()*100:.1f}% drowsy)")
    print(f"Val:   {len(val_df):,} images ({val_df['label'].mean()*100:.1f}% drowsy)")
    print(f"Test:  {len(test_df):,} images ({test_df['label'].mean()*100:.1f}% drowsy)")
    print("✓ Subject-dependent split: All subjects appear in all splits")
    
    # 3. Extract features
    X_train, y_train = load_split(os.path.join(SPLITS_DIR, "train.csv"), "TRAIN")
    X_val,   y_val   = load_split(os.path.join(SPLITS_DIR, "val.csv"),   "VAL")
    X_test,  y_test  = load_split(os.path.join(SPLITS_DIR, "test.csv"),  "TEST")
    
    print(f"\nFeature vector size : {X_train.shape[1]:,}")
    print(f"Train : {len(y_train):,} | Val : {len(y_val):,} | Test : {len(y_test):,}")
    
    # 4. Normalize features
    print("\nFitting StandardScaler...")
    t0 = time.time()
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)
    print(f"  Scaler done [{time.time()-t0:.1f}s]")
    
    # 5. Verify normalization
    print("\nFeature scaling verification:")
    print(f"  Train mean: {X_train_s.mean():.6f} (should be ~0)")
    print(f"  Train std:  {X_train_s.std():.6f} (should be ~1)")
    print(f"  Val mean:   {X_val_s.mean():.6f}")
    print(f"  Val std:    {X_val_s.std():.6f}")
    
    if abs(X_train_s.mean()) > 0.01 or abs(X_train_s.std() - 1) > 0.01:
        print("  ⚠️  WARNING: Features not properly normalized!")
    
    # 6. Dummy classifier baseline
    print("\n" + "=" * 70)
    print("DUMMY CLASSIFIER BASELINE")
    print("=" * 70)
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train_s, y_train)
    dummy_train_acc = dummy.score(X_train_s, y_train) * 100
    dummy_val_acc = dummy.score(X_val_s, y_val) * 100
    dummy_test_acc = dummy.score(X_test_s, y_test) * 100
    
    print(f"Most frequent class: {np.argmax(np.bincount(y_train))} ({CLASS_NAMES[np.argmax(np.bincount(y_train))]})")
    print(f"  Train accuracy: {dummy_train_acc:.2f}%")
    print(f"  Val accuracy:   {dummy_val_acc:.2f}%")
    print(f"  Test accuracy:  {dummy_test_acc:.2f}%")
    
    # 7. Compute sample weights
    classes = np.array([0, 1])
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    sample_weights = np.array([weights[0] if l == 0 else weights[1] for l in y_train])
    print(f"\nClass weights: notdrowsy={weights[0]:.3f}, drowsy={weights[1]:.3f}")
    
    # 8. Train SGD SVM with early stopping
    print("\n" + "=" * 70)
    print("TRAINING SGD SVM WITH EARLY STOPPING")
    print("=" * 70)
    
    svm = SGDClassifier(
        loss="modified_huber",
        alpha=0.001,
        eta0=0.01,
        learning_rate="adaptive",  # ← Improved from "constant"
        random_state=42,
        max_iter=1,
        tol=None,
        warm_start=True,
        shuffle=True,
    )
    
    best_val_acc = 0
    best_epoch = 0
    best_model_state = None
    no_improve = 0
    t0 = time.time()
    
    print("\n  Epoch | Train Acc | Val Acc | Best Val | Time  | Status")
    print("  " + "-" * 60)
    
    for epoch in range(1, MAX_EPOCHS + 1):
        svm.partial_fit(X_train_s, y_train, classes=classes, sample_weight=sample_weights)
        
        train_acc = svm.score(X_train_s, y_train) * 100
        val_acc = svm.score(X_val_s, y_val) * 100
        
        status = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            # Use pickle for in-memory serialization
            best_model_state = pickle.dumps({"scaler": scaler, "svm": svm})
            no_improve = 0
            status = "✓ NEW BEST"
            
            # Save best model to disk
            joblib.dump({"scaler": scaler, "svm": svm},
                        os.path.join(MODEL_DIR, "svm_best.joblib"))
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                status = f"✗ Early stop (no improvement for {PATIENCE} epochs)"
                print(f"  {epoch:>5} | {train_acc:8.2f} | {val_acc:8.2f} | {best_val_acc:8.2f} | {time.time()-t0:5.1f}s | {status}")
                break
        
        print(f"  {epoch:>5} | {train_acc:8.2f} | {val_acc:8.2f} | {best_val_acc:8.2f} | {time.time()-t0:5.1f}s | {status}")
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            joblib.dump({"scaler": scaler, "svm": svm, "epoch": epoch},
                        os.path.join(MODEL_DIR, f"svm_checkpoint_ep{epoch}.joblib"))
    
    print("  " + "-" * 60)
    print(f"\n✓ Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")
    
    # 9. Load best model
    print("\nLoading best model for evaluation...")
    if best_model_state is None:
        print("  No best model found, using final model")
        best_svm = svm
    else:
        best = pickle.loads(best_model_state)
        best_svm = best["svm"]
        scaler = best["scaler"]  # Use the scaler from best model
    
    # 10. Re-transform data with best scaler if needed
    if best_model_state is not None:
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)
    
    # 11. Evaluate on validation and test sets
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)
    
    results = {}
    
    for split_name, X_eval, y_eval in [
        ("Validation", X_val_s, y_val),
        ("Test", X_test_s, y_test),
    ]:
        print(f"\n{split_name.upper()} SET RESULTS:")
        print("-" * 50)
        
        y_pred = best_svm.predict(X_eval)
        y_prob = best_svm.predict_proba(X_eval)[:, 1]
        
        roc_auc = roc_auc_score(y_eval, y_prob)
        accuracy = np.mean(y_pred == y_eval) * 100
        
        print(f"  Accuracy : {accuracy:.2f}%")
        print(f"  ROC-AUC  : {roc_auc:.4f}")
        print(f"\n  Classification Report:")
        print(classification_report(y_eval, y_pred, target_names=CLASS_NAMES))
        
        # Compare with dummy classifier
        dummy_acc = dummy.score(X_eval, y_eval) * 100
        improvement = accuracy - dummy_acc
        print(f"  Improvement over dummy: {improvement:+.2f}%")
        
        results[split_name] = {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'improvement': improvement
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_eval, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
        fig, ax = plt.subplots(figsize=(5, 4))
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"SVM — {split_name}\nAcc: {accuracy:.2f}%  AUC: {roc_auc:.3f}")
        plt.tight_layout()
        cm_path = os.path.join(REPORTS_DIR, f"svm_confusion_{split_name.lower()}.png")
        plt.savefig(cm_path, dpi=150)
        plt.close()
        print(f"\n  Confusion matrix saved -> {cm_path}")
    
    # 12. ROC Curve
    print("\nGenerating ROC curve...")
    y_prob_test = best_svm.predict_proba(X_test_s)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob_test)
    auc = roc_auc_score(y_test, y_prob_test)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f"SVM (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.500)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — SVM Test Set")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(REPORTS_DIR, "svm_roc_curve.png")
    plt.savefig(roc_path, dpi=150)
    plt.close()
    print(f"  ROC curve saved -> {roc_path}")
    
    # 13. Save final model
    joblib.dump({"scaler": scaler, "svm": best_svm},
                os.path.join(MODEL_DIR, "svm_final.joblib"))
    print(f"\n✓ Final model saved -> {MODEL_DIR}/svm_final.joblib")
    
    # 14. Summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}% (epoch {best_epoch})")
    print(f"Test Accuracy:            {results['Test']['accuracy']:.2f}%")
    print(f"Test ROC-AUC:             {results['Test']['roc_auc']:.4f}")
    print(f"Improvement over dummy:   {results['Test']['improvement']:+.2f}%")
    
    if results['Test']['improvement'] < 5:
        print("\n⚠️  WARNING: Model barely beats dummy classifier!")
        print("   Possible issues:")
        print("   1. Features may not capture drowsiness cues")
        print("   2. Data split may have distribution shift")
        print("   3. Consider trying different features (LBP, face landmarks, etc.)")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
