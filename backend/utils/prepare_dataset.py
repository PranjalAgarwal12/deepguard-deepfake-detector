"""
Dataset Preparation Utility
Organizes, preprocesses, and splits dataset for deepfake detection training.

Supported datasets:
  - FaceForensics++ (downloaded via their script)
  - DFDC (Deepfake Detection Challenge)
  - Kaggle 140k Real/Fake Faces
  - Any custom folder with real/ and fake/ subdirectories

Usage:
  python prepare_dataset.py --source_dir /path/to/raw --output_dir ../dataset
"""

import os
import shutil
import argparse
import random
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
IMG_SIZE = 224
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ─────────────────────────────────────────────
# FACE DETECTION (optional cropping)
# ─────────────────────────────────────────────
def load_face_detector():
    """Load OpenCV Haar cascade for face detection."""
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(cascade_path)


def crop_face(image: np.ndarray, face_detector, padding: float = 0.2) -> np.ndarray | None:
    """
    Detect and crop the largest face in image.
    Returns cropped face or None if no face detected.
    padding: fractional padding around face bounding box.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    if len(faces) == 0:
        return None

    # Take the largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    # Add padding
    pad_x = int(w * padding)
    pad_y = int(h * padding)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(image.shape[1], x + w + pad_x)
    y2 = min(image.shape[0], y + h + pad_y)

    return image[y1:y2, x1:x2]


# ─────────────────────────────────────────────
# IMAGE PREPROCESSING
# ─────────────────────────────────────────────
def preprocess_and_save(
    src_path: str,
    dst_path: str,
    face_detector=None,
    crop_faces: bool = False,
) -> bool:
    """
    Preprocess a single image:
      1. Load image
      2. Optionally crop face
      3. Resize to IMG_SIZE x IMG_SIZE
      4. Save as high-quality JPEG
    Returns True if successful, False otherwise.
    """
    try:
        img = cv2.imread(src_path)
        if img is None:
            return False

        if crop_faces and face_detector is not None:
            face = crop_face(img, face_detector)
            if face is not None:
                img = face
            # If no face detected, use full image

        # Resize
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LANCZOS4)

        # Save
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        cv2.imwrite(dst_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return True

    except Exception as e:
        print(f"  ⚠ Failed to process {src_path}: {e}")
        return False


# ─────────────────────────────────────────────
# DATASET SPLITTING
# ─────────────────────────────────────────────
def split_and_copy(
    source_dir: str,
    output_dir: str,
    class_name: str,
    crop_faces: bool = False,
):
    """
    Split images from source_dir into train/val/test splits
    and save to output_dir/{split}/{class_name}/.
    """
    # Collect all image paths
    all_images = []
    for root, _, files in os.walk(source_dir):
        for fname in files:
            if Path(fname).suffix.lower() in SUPPORTED_EXTS:
                all_images.append(os.path.join(root, fname))

    if not all_images:
        print(f"  ⚠ No images found in {source_dir}")
        return

    # Shuffle
    random.seed(RANDOM_SEED)
    random.shuffle(all_images)

    n = len(all_images)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    splits = {
        "train": all_images[:n_train],
        "val": all_images[n_train : n_train + n_val],
        "test": all_images[n_train + n_val :],
    }

    print(f"\n  Class '{class_name}': {n} total images")
    print(f"    Train: {len(splits['train'])}")
    print(f"    Val:   {len(splits['val'])}")
    print(f"    Test:  {len(splits['test'])}")

    face_detector = load_face_detector() if crop_faces else None
    counters = {"success": 0, "failed": 0}

    for split, paths in splits.items():
        print(f"\n  Processing {split} set...")
        for i, src in enumerate(tqdm(paths, desc=f"  {split}/{class_name}")):
            ext = ".jpg"
            dst = os.path.join(output_dir, split, class_name, f"{i:06d}{ext}")
            ok = preprocess_and_save(src, dst, face_detector, crop_faces)
            if ok:
                counters["success"] += 1
            else:
                counters["failed"] += 1

    print(f"\n  ✅ {counters['success']} images processed, {counters['failed']} failed")


# ─────────────────────────────────────────────
# DATASET STATS
# ─────────────────────────────────────────────
def print_dataset_stats(output_dir: str):
    """Print a summary table of the processed dataset."""
    print("\n" + "=" * 50)
    print("  DATASET STATISTICS")
    print("=" * 50)
    print(f"  {'Split':<10} {'real':<10} {'fake':<10} {'Total':<10}")
    print("  " + "-" * 40)

    for split in ["train", "val", "test"]:
        counts = {}
        for cls in ["real", "fake"]:
            path = os.path.join(output_dir, split, cls)
            if os.path.exists(path):
                counts[cls] = len(os.listdir(path))
            else:
                counts[cls] = 0
        total = sum(counts.values())
        print(f"  {split:<10} {counts.get('real', 0):<10} {counts.get('fake', 0):<10} {total:<10}")

    print("=" * 50)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Prepare deepfake detection dataset")
    parser.add_argument("--source_dir", required=True,
                        help="Root dir with 'real/' and 'fake/' subdirectories")
    parser.add_argument("--output_dir", default="../dataset",
                        help="Output directory for processed dataset")
    parser.add_argument("--crop_faces", action="store_true",
                        help="Crop faces using OpenCV (recommended for better accuracy)")
    args = parser.parse_args()

    print("=" * 50)
    print("  DATASET PREPARATION")
    print("=" * 50)
    print(f"  Source:      {args.source_dir}")
    print(f"  Output:      {args.output_dir}")
    print(f"  Crop faces:  {args.crop_faces}")

    # Process real images
    real_dir = os.path.join(args.source_dir, "real")
    if os.path.exists(real_dir):
        split_and_copy(real_dir, args.output_dir, "real", args.crop_faces)
    else:
        print(f"⚠ Real images directory not found: {real_dir}")

    # Process fake images
    fake_dir = os.path.join(args.source_dir, "fake")
    if os.path.exists(fake_dir):
        split_and_copy(fake_dir, args.output_dir, "fake", args.crop_faces)
    else:
        print(f"⚠ Fake images directory not found: {fake_dir}")

    # Print stats
    print_dataset_stats(args.output_dir)
    print("\n✅ Dataset preparation complete!")


if __name__ == "__main__":
    main()
