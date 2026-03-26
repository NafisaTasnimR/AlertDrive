import os
import shutil

RAW_BASE = "data/raw/Multi class/train"
OUT_BASE = "data/interim/faces"

os.makedirs(OUT_BASE, exist_ok=True)

# ---- DROWSY (preserve original filenames) ----
drowsy_out = os.path.join(OUT_BASE, "drowsy")
os.makedirs(drowsy_out, exist_ok=True)

drowsy_path = os.path.join(RAW_BASE, "drowsy")
for sub in os.listdir(drowsy_path):
    sub_path = os.path.join(drowsy_path, sub)
    for img in os.listdir(sub_path):
        src = os.path.join(sub_path, img)
        # Keep original filename - don't add action prefix
        dst = os.path.join(drowsy_out, img)
        shutil.copy(src, dst)

# ---- NOT DROWSY (unchanged) ----
notdrowsy_out = os.path.join(OUT_BASE, "notdrowsy")
os.makedirs(notdrowsy_out, exist_ok=True)

notdrowsy_path = os.path.join(RAW_BASE, "notdrowsy")
for img in os.listdir(notdrowsy_path):
    src = os.path.join(notdrowsy_path, img)
    dst = os.path.join(notdrowsy_out, img)
    shutil.copy(src, dst)

print("Image collection complete.")