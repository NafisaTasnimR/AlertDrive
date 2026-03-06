import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Script lives at: ALERTDRIVE/src/utils/ddd_create_splits.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed_ddd"   # DDD processed images
OUT_DIR  = PROJECT_ROOT / "splits"                    # same splits folder, ddd_ prefixed CSVs
os.makedirs(OUT_DIR, exist_ok=True)

LABELS = {
    "notdrowsy": 0,
    "drowsy":    1
}

data = []
for label, class_id in LABELS.items():
    folder = DATA_DIR / label
    if not folder.exists():
        print(f"[WARNING] Folder not found: {folder}")
        continue
    for img in os.listdir(folder):
        data.append([str(folder / img), class_id])

df = pd.DataFrame(data, columns=["path", "label"])

print(f"Total images: {len(df)}")
print(f"  Drowsy:     {(df['label'] == 1).sum()}")
print(f"  Non Drowsy: {(df['label'] == 0).sum()}")

# 70% train / 15% val / 15% test  — same ratio as create_splits.py
train, temp = train_test_split(df,   test_size=0.3,  stratify=df["label"],   random_state=42)
val,   test = train_test_split(temp, test_size=0.5,  stratify=temp["label"], random_state=42)

# Save with ddd_ prefix — does NOT overwrite train.csv / val.csv / test.csv
train.to_csv(OUT_DIR / "ddd_train.csv", index=False)
val.to_csv(  OUT_DIR / "ddd_val.csv",   index=False)
test.to_csv( OUT_DIR / "ddd_test.csv",  index=False)

print(f"\nSplits saved to {OUT_DIR}/")
print(f"  ddd_train.csv : {len(train)} images")
print(f"  ddd_val.csv   : {len(val)}   images")
print(f"  ddd_test.csv  : {len(test)}  images")