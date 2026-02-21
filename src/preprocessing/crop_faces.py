import cv2
import os


FACE_CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
EYE_CASCADE = cv2.data.haarcascades + "haarcascade_eye.xml"

IN_DIR = "data/interim/images"
OUT_DIR = "data/interim/faces"
LOG_FILE = "logs/face_failures.txt"

PADDING_RATIO = 0.30
MIN_FACE_SIZE = (50, 50)

os.makedirs("logs", exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

face_cascade = cv2.CascadeClassifier(FACE_CASCADE)
eye_cascade = cv2.CascadeClassifier(EYE_CASCADE)

log = open(LOG_FILE, "a")

for label in ["drowsy", "notdrowsy"]:
    in_path = os.path.join(IN_DIR, label)
    out_path = os.path.join(OUT_DIR, label)
    os.makedirs(out_path, exist_ok=True)

    images = os.listdir(in_path)

    for i, img_name in enumerate(images):
        out_file = os.path.join(out_path, img_name)

        if os.path.exists(out_file):
            continue

        img_path = os.path.join(in_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            log.write(f"{img_name} - unreadable\n")
            continue

        img = cv2.resize(img, (640, 480))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization for better detection
        gray_eq = cv2.equalizeHist(gray)

        # Multi-pass face detection
        faces = face_cascade.detectMultiScale(gray_eq, 1.3, 5, minSize=MIN_FACE_SIZE)
        
        if len(faces) == 0:
            faces = face_cascade.detectMultiScale(gray_eq, 1.1, 3, minSize=MIN_FACE_SIZE)
        
        if len(faces) == 0:
            faces = face_cascade.detectMultiScale(gray, 1.2, 4, minSize=MIN_FACE_SIZE)

        if len(faces) == 0:
            log.write(f"{img_name} - no face\n")
            continue

        # Use largest face
        if len(faces) > 1:
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        
        x, y, w, h = faces[0]
        
        # Basic quality checks
        if w < 50 or h < 50:
            log.write(f"{img_name} - face too small\n")
            continue
        
        aspect_ratio = w / float(h)
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            log.write(f"{img_name} - unusual proportions\n")
            continue
        
        # KEY CHANGE: Relaxed eye validation
        # We check IF eyes can be detected, but we're very lenient
        # This prevents "mouth-only" crops while allowing faces with glasses/shadows
        gray_face = gray_eq[y:y+h, x:x+w]
        
        # Try to detect eyes with VERY lenient parameters
        eyes = eye_cascade.detectMultiScale(
            gray_face,
            scaleFactor=1.1,
            minNeighbors=1,  # Very low threshold
            minSize=(8, 8)   # Very small minimum
        )
        
        # If still no eyes, try on non-equalized gray
        if len(eyes) == 0:
            gray_face_orig = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(
                gray_face_orig,
                scaleFactor=1.05,
                minNeighbors=1,
                minSize=(8, 8)
            )
        
        # RELAXED VALIDATION: Just check if there's SOMETHING eye-like in upper 75%
        # This catches actual eyes while rejecting pure mouth/chin crops
        has_eye_region = False
        for (ex, ey, ew, eh) in eyes:
            # Very lenient - just needs to be in upper 75% of face
            if ey < (h * 0.75):
                has_eye_region = True
                break
        
        # If no eye-like features detected in upper region, it's likely a bad crop
        if not has_eye_region and len(eyes) == 0:
            log.write(f"{img_name} - no eye region detected (likely bad crop)\n")
            continue

        # Expand bounding box
        pad = int(PADDING_RATIO * h)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img.shape[1], x + w + pad)
        y2 = min(img.shape[0], y + h + pad)

        face = img[y1:y2, x1:x2]
        
        if face.shape[0] < 40 or face.shape[1] < 40:
            log.write(f"{img_name} - final crop too small\n")
            continue
        
        cv2.imwrite(out_file, face)

        if i % 100 == 0:
            print(f"[{label}] Processed {i}/{len(images)} images")

log.close()
print("Face cropping completed.")
print(f"Check {LOG_FILE} for details on rejected images.")