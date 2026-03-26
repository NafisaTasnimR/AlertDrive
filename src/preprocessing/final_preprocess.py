import cv2
import os
import numpy as np
from typing import Tuple

IN_DIR = "data/interim/faces"
OUT_DIR = "data/processed"
IMG_SIZE = 224

# ImageNet normalization stats for later use
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

os.makedirs(OUT_DIR, exist_ok=True)

def augment_image(img: np.ndarray) -> np.ndarray:
    """
    Apply light data augmentation to reduce overfitting.
    Augmentation is applied only during training (controlled in sequence_dataset.py).
    
    Args:
        img: Image as numpy array (224, 224, 3) in BGR format, normalized [0,1]
    
    Returns:
        Augmented image in same format
    """
    h, w = img.shape[:2]
    
    # Random horizontal flip (70% probability for eyedness symmetry)
    if np.random.random() < 0.7:
        img = cv2.flip(img, 1)
    
    # Random brightness adjustment (±10%)
    brightness = np.random.uniform(0.9, 1.1)
    img = np.clip(img * brightness, 0, 1)
    
    # Random contrast adjustment (±10%)
    contrast = np.random.uniform(0.9, 1.1)
    img = np.clip((img - 0.5) * contrast + 0.5, 0, 1)
    
    # Random Gaussian blur (small kernel)
    if np.random.random() < 0.3:
        img = cv2.GaussianBlur(img, (3, 3), sigma=np.random.uniform(0.1, 0.5))
        img = np.clip(img, 0, 1)
    
    # Random small rotation (±5 degrees)
    if np.random.random() < 0.4:
        angle = np.random.uniform(-5, 5)
        center = (w // 2, h // 2)
        mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, mat, (w, h), borderValue=(0.5, 0.5, 0.5))
        img = np.clip(img, 0, 1)
    
    # Random small translation (±5% of image size)
    if np.random.random() < 0.3:
        tx = np.random.uniform(-int(0.05*w), int(0.05*w))
        ty = np.random.uniform(-int(0.05*h), int(0.05*h))
        mat = np.float32([[1, 0, tx], [0, 1, ty]])
        img = cv2.warpAffine(img, mat, (w, h), borderValue=(0.5, 0.5, 0.5))
        img = np.clip(img, 0, 1)
    
    return img


for label in ["drowsy", "notdrowsy"]:
    in_path = os.path.join(IN_DIR, label)
    out_path = os.path.join(OUT_DIR, label)
    os.makedirs(out_path, exist_ok=True)

    for img_name in os.listdir(in_path):
        img_path = os.path.join(in_path, img_name)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Warning: Could not read {img_name}")
            continue

        # Resize
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
        # Normalize to [0, 1]
        img = img / 255.0
        
        # Save base image without augmentation
        # (augmentation will be applied during training in sequence_dataset.py)
        save_path = os.path.join(out_path, img_name)
        cv2.imwrite(save_path, (img * 255).astype(np.uint8))

print("✓ Images resized to (224, 224)")
print("✓ Images saved in [0, 1] range")