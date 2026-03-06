import os
import shutil

# ── Paths ──────────────────────────────────────────────────────────────────
RAW_BASE = "data/raw/Driver Drowsiness Dataset (DDD)"  # root of the DDD dataset
OUT_BASE = "data/interim/faces_ddd"  # separate interim folder — does NOT touch
                                      # data/interim/faces used by the other pipeline

# Map dataset folder names → internal label names (matching existing convention)
LABEL_MAP = {
    "Drowsy":     "drowsy",
    "Non Drowsy": "notdrowsy",
}

# ── Copy ───────────────────────────────────────────────────────────────────
os.makedirs(OUT_BASE, exist_ok=True)

total_copied = 0
for raw_label, out_label in LABEL_MAP.items():
    src_dir = os.path.join(RAW_BASE, raw_label)
    dst_dir = os.path.join(OUT_BASE, out_label)
    os.makedirs(dst_dir, exist_ok=True)

    if not os.path.exists(src_dir):
        print(f"[WARNING] Source folder not found: {src_dir}")
        continue

    images = [
        f for f in os.listdir(src_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]

    for img_name in images:
        src = os.path.join(src_dir, img_name)
        dst = os.path.join(dst_dir, img_name)
        if not os.path.exists(dst):          # skip already-copied files
            shutil.copy(src, dst)

    total_copied += len(images)
    print(f"[{raw_label}] Copied {len(images)} images → {dst_dir}")

print(f"\nDDD collect complete. Total images staged: {total_copied}")