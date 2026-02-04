import os
import shutil

RAW_BASE = "data/raw/Multi class/train"
OUT_BASE = "data/interim/images"

os.makedirs(OUT_BASE, exist_ok=True)

# ---- DROWSY (multiple subfolders) ----
drowsy_out = os.path.join(OUT_BASE, "drowsy")
os.makedirs(drowsy_out, exist_ok=True)

drowsy_path = os.path.join(RAW_BASE, "drowsy")
for sub in os.listdir(drowsy_path):
    sub_path = os.path.join(drowsy_path, sub)
    for img in os.listdir(sub_path):
        src = os.path.join(sub_path, img)
        dst = os.path.join(drowsy_out, f"{sub}_{img}")
        shutil.copy(src, dst)

# ---- NOT DROWSY ----
notdrowsy_out = os.path.join(OUT_BASE, "notdrowsy")
os.makedirs(notdrowsy_out, exist_ok=True)

notdrowsy_path = os.path.join(RAW_BASE, "notdrowsy")
for img in os.listdir(notdrowsy_path):
    src = os.path.join(notdrowsy_path, img)
    dst = os.path.join(notdrowsy_out, img)
    shutil.copy(src, dst)

print("Image collection complete.")
