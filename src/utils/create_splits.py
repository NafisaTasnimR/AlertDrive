import os
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = "data/processed"
OUT_DIR = "splits"
os.makedirs(OUT_DIR, exist_ok=True)

data = []

LABELS = {
    "notdrowsy": 0,
    "drowsy": 1
}

for label, class_id in LABELS.items():
    folder = os.path.join(DATA_DIR, label)
    for img in os.listdir(folder):
        data.append([os.path.join(folder, img), class_id])

df = pd.DataFrame(data, columns=["path", "label"])

train, temp = train_test_split(df, test_size=0.3, stratify=df["label"])
val, test = train_test_split(temp, test_size=0.5, stratify=temp["label"])

train.to_csv("splits/train.csv", index=False)
val.to_csv("splits/val.csv", index=False)
test.to_csv("splits/test.csv", index=False)
