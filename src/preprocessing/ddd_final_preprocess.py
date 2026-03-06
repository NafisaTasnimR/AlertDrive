import cv2
import os
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
IN_DIR  = "data/interim/faces_ddd"   # output of ddd_collect.py
OUT_DIR = "data/processed_ddd"       # ← separate from data/processed
IMG_SIZE = 224                        # same target size as final_preprocess.py

os.makedirs(OUT_DIR, exist_ok=True)

total_saved   = 0
total_skipped = 0

for label in ["drowsy", "notdrowsy"]:
    in_path  = os.path.join(IN_DIR,  label)
    out_path = os.path.join(OUT_DIR, label)
    os.makedirs(out_path, exist_ok=True)

    if not os.path.exists(in_path):
        print(f"[WARNING] Input folder not found: {in_path}")
        continue

    images = [f for f in os.listdir(in_path)
              if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]

    label_saved = 0
    for img_name in images:
        img_path = os.path.join(in_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Warning: Could not read {img_name}")
            total_skipped += 1
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        # Keep in BGR format for cv2.imwrite  (same as final_preprocess.py)
        img = img / 255.0

        save_path = os.path.join(out_path, img_name)
        cv2.imwrite(save_path, (img * 255).astype(np.uint8))
        label_saved += 1

    total_saved += label_saved
    print(f"[{label}] Saved {label_saved} images → {out_path}")

print(f"\nDDD final preprocessing complete.")
print(f"  Saved:   {total_saved}")
print(f"  Skipped: {total_skipped}")
print(f"  Output:  {OUT_DIR}/")