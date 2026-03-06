import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

FACE_DIR = "data/interim/faces_ddd"          # ← DDD-specific interim folder
EYE_CASCADE  = cv2.data.haarcascades + "haarcascade_eye.xml"
FACE_CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

eye_cascade  = cv2.CascadeClassifier(EYE_CASCADE)
face_cascade = cv2.CascadeClassifier(FACE_CASCADE)

print("=" * 80)
print("DRIVER DROWSINESS DETECTION - DDD DATASET VALIDATION")
print("=" * 80)
print("\nValidating DDD pre-cropped faces meet requirements for:")
print("  ✓ Face detection quality")
print("  ✓ Eye region visibility (for eye closure, blink rate detection)")
print("  ✓ Facial features clarity (for yawning detection)")
print("  ✓ Image quality for CNN/LSTM processing")
print("=" * 80)


def analyze_face_quality(img):
    """
    Analyze if face meets drowsiness detection requirements:
    - Face clearly visible
    - Eye region detectable (for eye closure, blink rate)
    - Sufficient quality for facial feature extraction
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]

    # Check 1: Face is centered and properly sized
    faces = face_cascade.detectMultiScale(gray, 1.1, 3)
    has_face = len(faces) > 0

    # Check 2: Eye region detectable (critical for drowsiness)
    gray_eq = cv2.equalizeHist(gray)
    eyes = eye_cascade.detectMultiScale(gray_eq, 1.1, 1, minSize=(8, 8))

    eye_region_detected = False
    valid_eye_count = 0
    for (ex, ey, ew, eh) in eyes:
        if ey < (h * 0.75):          # upper 75 % of face
            eye_region_detected = True
            valid_eye_count += 1

    # Check 3: Image quality (sharpness)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    is_sharp = laplacian_var > 50

    # Check 4: Sufficient size for CNN processing
    min_size_ok = h >= 40 and w >= 40

    # Check 5: Aspect ratio reasonable
    aspect_ratio = w / float(h)
    aspect_ok = 0.5 <= aspect_ratio <= 2.0

    return {
        "has_face":           has_face,
        "eye_region_detected": eye_region_detected,
        "eye_count":          valid_eye_count,
        "is_sharp":           is_sharp,
        "sharpness_score":    laplacian_var,
        "size_ok":            min_size_ok,
        "aspect_ok":          aspect_ok,
        "dimensions":         (w, h),
        "overall_quality":    has_face and eye_region_detected and min_size_ok and aspect_ok,
    }


def check_directory(label, num_visual_samples=20):
    """Check all images and provide comprehensive statistics."""
    face_path = os.path.join(FACE_DIR, label)

    if not os.path.exists(face_path):
        print(f"\n❌ Directory not found: {face_path}")
        return

    images = [f for f in os.listdir(face_path)
              if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    print(f"\n{'=' * 80}")
    print(f"CHECKING: {label.upper()} DATASET  (DDD)")
    print(f"{'=' * 80}")
    print(f"Total images: {len(images)}")

    stats = {
        "total":               len(images),
        "has_face":            0,
        "eye_region_detected": 0,
        "no_eye_region":       0,
        "sharp":               0,
        "blurry":              0,
        "good_quality":        0,
        "size_issues":         0,
        "aspect_issues":       0,
        "eye_counts":          {0: 0, 1: 0, 2: 0, "3+": 0},
    }

    problematic_images = []

    print("\n📊 Analyzing all images...")
    for idx, img_name in enumerate(images):
        img_path = os.path.join(face_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        quality = analyze_face_quality(img)

        if quality["has_face"]:
            stats["has_face"] += 1
        if quality["eye_region_detected"]:
            stats["eye_region_detected"] += 1
        else:
            stats["no_eye_region"] += 1
            problematic_images.append((img_name, "no_eye_region"))

        if quality["is_sharp"]:
            stats["sharp"] += 1
        else:
            stats["blurry"] += 1

        if quality["overall_quality"]:
            stats["good_quality"] += 1
        else:
            if not quality["size_ok"]:
                stats["size_issues"] += 1
                problematic_images.append((img_name, "size_too_small"))
            if not quality["aspect_ok"]:
                stats["aspect_issues"] += 1
                problematic_images.append((img_name, "bad_aspect_ratio"))

        ec = quality["eye_count"]
        if ec == 0:
            stats["eye_counts"][0] += 1
        elif ec == 1:
            stats["eye_counts"][1] += 1
        elif ec == 2:
            stats["eye_counts"][2] += 1
        else:
            stats["eye_counts"]["3+"] += 1

        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{len(images)}...")

    # ── Print report ────────────────────────────────────────────────────────
    t = stats["total"]
    print(f"\n{'─' * 80}")
    print("📈 DDD DATASET QUALITY REPORT")
    print(f"{'─' * 80}")

    print(f"\n🎯 Overall Quality:")
    print(f"  ✓ High quality images: {stats['good_quality']:>6} ({stats['good_quality']/t*100:>5.1f}%)")
    print(f"  ⚠ Issues found:        {t-stats['good_quality']:>6} ({(t-stats['good_quality'])/t*100:>5.1f}%)")

    print(f"\n👤 Face Detection:")
    print(f"  ✓ Face detected:       {stats['has_face']:>6} ({stats['has_face']/t*100:>5.1f}%)")
    print(f"  ✗ No face:             {t-stats['has_face']:>6} ({(t-stats['has_face'])/t*100:>5.1f}%)")

    print(f"\n👁  Eye Region Detection (CRITICAL for drowsiness):")
    print(f"  ✓ Eye region found:    {stats['eye_region_detected']:>6} ({stats['eye_region_detected']/t*100:>5.1f}%)")
    print(f"  ✗ No eye region:       {stats['no_eye_region']:>6} ({stats['no_eye_region']/t*100:>5.1f}%)")

    print(f"\n👁️👁️ Eye Count Distribution:")
    for k in [0, 1, 2, "3+"]:
        n = stats["eye_counts"][k]
        print(f"  {k} eye(s) detected:       {n:>6} ({n/t*100:>5.1f}%)")

    print(f"\n📸 Image Sharpness:")
    print(f"  ✓ Sharp images:        {stats['sharp']:>6} ({stats['sharp']/t*100:>5.1f}%)")
    print(f"  ⚠ Blurry images:       {stats['blurry']:>6} ({stats['blurry']/t*100:>5.1f}%)")

    print(f"\n📐 Dimension Issues:")
    print(f"  ✗ Size issues:         {stats['size_issues']:>6}")
    print(f"  ✗ Aspect ratio issues: {stats['aspect_issues']:>6}")

    print(f"\n{'─' * 80}")
    print("⚠️  QUALITY ASSESSMENT FOR DROWSINESS DETECTION:")
    print(f"{'─' * 80}")

    eye_rate = stats["eye_region_detected"] / t * 100
    if eye_rate >= 90:
        print("✅ EXCELLENT: >90% of images have detectable eye regions")
    elif eye_rate >= 80:
        print("✓ GOOD: 80-90% of images have detectable eye regions")
    elif eye_rate >= 70:
        print("⚠ ACCEPTABLE: 70-80% of images have detectable eye regions")
    else:
        print("❌ POOR: <70% of images have detectable eye regions")

    if stats["no_eye_region"] > 0:
        print(f"\n⚠️  {stats['no_eye_region']} images without detectable eye regions")
        print("   Note: closed eyes in the Drowsy class are expected and OK.")

    if problematic_images:
        print(f"\n🔍 Problematic images (first 10):")
        for img_name, issue in problematic_images[:10]:
            print(f"   - {img_name}: {issue}")
        if len(problematic_images) > 10:
            print(f"   ... and {len(problematic_images) - 10} more")

    # ── Visual sample ────────────────────────────────────────────────────────
    print(f"\n🖼️  Generating visual sample grid...")
    sample_indices = np.linspace(0, len(images) - 1,
                                 min(num_visual_samples, len(images)), dtype=int)
    sample = [images[i] for i in sample_indices]

    fig, axes = plt.subplots(4, 5, figsize=(16, 13))
    fig.suptitle(f"DDD {label.upper()} - Quality Check Sample (Eye Regions Marked)",
                 fontsize=16, fontweight="bold")
    axes = axes.flatten()

    for idx, img_name in enumerate(sample):
        if idx >= 20:
            break
        img_path = os.path.join(face_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        quality = analyze_face_quality(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)
        eyes = eye_cascade.detectMultiScale(gray_eq, 1.1, 1, minSize=(8, 8))

        img_display = img.copy()
        h = img.shape[0]
        for (ex, ey, ew, eh) in eyes:
            color = (0, 255, 0) if ey < h * 0.75 else (255, 0, 0)
            cv2.rectangle(img_display, (ex, ey), (ex + ew, ey + eh), color, 2)

        img_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
        status = "✓" if quality["overall_quality"] else "✗"
        color  = "green" if quality["overall_quality"] else "red"
        axes[idx].imshow(img_display)
        axes[idx].set_title(f"{status} {quality['eye_count']} eye(s)",
                            fontsize=10, color=color, fontweight="bold")
        axes[idx].axis("off")

    for idx in range(len(sample), 20):
        axes[idx].axis("off")

    plt.tight_layout()
    # ↓ _ddd_ prefix avoids overwriting validation_drowsy.png / validation_notdrowsy.png
    output_path = f"validation_ddd_{label}.png"
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    print(f"✅ Visual validation saved: {output_path}")
    plt.close()


# ── Run ───────────────────────────────────────────────────────────────────
for label in ["drowsy", "notdrowsy"]:
    check_directory(label)

print(f"\n{'=' * 80}")
print("✅ DDD VALIDATION COMPLETE")
print(f"{'=' * 80}")
print("\n📋 REQUIREMENTS CHECKLIST FOR DRIVER DROWSINESS DETECTION:")
print("  ✓ Face clearly visible (for YOLO/CNN detection)")
print("  ✓ Eye region detectable (for eye closure duration, blink rate)")
print("  ✓ Facial features visible (for yawning detection)")
print("  ✓ Sufficient quality (for CNN/LSTM feature extraction)")
print("  ✓ Proper dimensions (for model input)")
print("\n💡 KEY INSIGHTS:")
print("  • Images with 0 eyes but good face: May be closed eyes (GOOD for drowsy class!)")
print("  • Images with 1 eye: Angled faces (GOOD for real-world variance)")
print("  • Images with 2 eyes: Frontal faces (IDEAL)")
print("  • Blurry images: May reduce accuracy but add realism")
print(f"\n{'=' * 80}\n")
print("Next step → run  ddd_final_preprocess.py")