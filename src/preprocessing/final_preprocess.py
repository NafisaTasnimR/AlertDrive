import cv2
import os
import numpy as np

IN_DIR = "data/interim/faces"
OUT_DIR = "data/processed"
IMG_SIZE = 224

os.makedirs(OUT_DIR, exist_ok=True)

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

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        # Keep in BGR format for cv2.imwrite
        img = img / 255.0

        save_path = os.path.join(out_path, img_name)
        cv2.imwrite(save_path, (img * 255).astype(np.uint8))