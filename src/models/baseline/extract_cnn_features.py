import cv2
import numpy as np
import pandas as pd
import os
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
import tensorflow as tf

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("="*80)
print("CNN FEATURE EXTRACTION FOR SVM")
print("Using ResNet50 pre-trained on ImageNet")
print("="*80)

IMG_SIZE = 224
OUT_DIR = "data/features"
os.makedirs(OUT_DIR, exist_ok=True)

# Load pre-trained ResNet50 
print("\n[1/4] Loading pre-trained ResNet50...")
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(IMG_SIZE, IMG_SIZE, 3))
print("  ✓ Model loaded (2048 features per image)")

def extract_features(csv_path, save_name):
    """Extract CNN features from images"""
    print(f"\n[Processing {save_name}]")
    df = pd.read_csv(csv_path)
    
    features_list = []
    labels_list = []
    
    batch_size = 32
    total_images = len(df)
    
    for i in range(0, total_images, batch_size):
        batch_df = df.iloc[i:i+batch_size]
        batch_images = []
        
        for _, row in batch_df.iterrows():
            img = cv2.imread(row["path"])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = preprocess_input(img)
            batch_images.append(img)
        
        # Extract features for batch
        batch_images = np.array(batch_images)
        batch_features = base_model.predict(batch_images, verbose=0)
        
        features_list.extend(batch_features)
        labels_list.extend(batch_df["label"].values)
        
        # Progress
        processed = min(i + batch_size, total_images)
        print(f"  Progress: {processed}/{total_images} ({processed/total_images*100:.1f}%)", end='\r')
    
    print(f"  ✓ Completed: {total_images} images processed")
    
    # Save
    X = np.array(features_list)
    y = np.array(labels_list)
    
    np.save(os.path.join(OUT_DIR, f"X_{save_name}_cnn.npy"), X)
    np.save(os.path.join(OUT_DIR, f"y_{save_name}_cnn.npy"), y)
    print(f"  ✓ Saved: {X.shape}")

# Extract features for all splits
print("\n" + "="*80)
print("[2/4] Extracting Training Features...")
extract_features("splits/train.csv", "train")

print("\n" + "="*80)
print("[3/4] Extracting Validation Features...")
extract_features("splits/val.csv", "val")

print("\n" + "="*80)
print("[4/4] Extracting Test Features...")
extract_features("splits/test.csv", "test")

print("\n" + "="*80)
print("✅ FEATURE EXTRACTION COMPLETE")
print("="*80)
print(f"\nFeatures saved in: {OUT_DIR}/")
print("Feature dimension: 2048 (much smaller than HOG!)")
print("\nNext step: Run train_svm_cnn.py")