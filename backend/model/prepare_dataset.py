"""
========================================================
  Deepfake Detection — Dataset Preparation Script
  Run this BEFORE training to organise + preprocess
  your raw dataset into train / val / test splits.
========================================================

Supported datasets:
  • FaceForensics++ (FF++)
  • DFDC (DeepFake Detection Challenge)
  • Any folder with real/ and fake/ subfolders

Usage:
  python prepare_dataset.py --src /path/to/raw --dst ../../dataset
"""

import os
import argparse
import shutil
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
IMG_SIZE   = (224, 224)
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST_RATIO  = 0.15  (remainder)
SEED = 42
random.seed(SEED)


# ─────────────────────────────────────────────
#  FACE DETECTOR (optional but recommended)
# ─────────────────────────────────────────────
def get_face_detector():
    """Return OpenCV DNN face detector."""
    prototxt = Path(__file__).parent / "deploy.prototxt"
    caffemodel = Path(__file__).parent / "res10_300x300_ssd_iter_140000.caffemodel"
    if prototxt.exists() and caffemodel.exists():
        net = cv2.dnn.readNetFromCaffe(str(prototxt), str(caffemodel))
        return net
    return None


def crop_face(img: np.ndarray, net, conf_threshold=0.6):
    """
    Detect largest face and return cropped region.
    Falls back to full image if no face found.
    """
    if net is None:
        return img

    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(img, (300, 300)), 1.0, (300, 300), (104, 117, 123)
    )
    net.setInput(blob)
    detections = net.forward()

    best_conf = 0
    best_box  = None
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold and confidence > best_conf:
            best_conf = confidence
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            best_box = box.astype(int)

    if best_box is None:
        return img

    x1, y1, x2, y2 = best_box
    # Add small padding
    pad = 20
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    return img[y1:y2, x1:x2]


# ─────────────────────────────────────────────
#  PREPROCESSING
# ─────────────────────────────────────────────
def preprocess_image(img_path: Path, out_path: Path, net=None):
    """
    Read → face-crop → resize → save.
    Normalisation (rescale /255) is done inside the Keras generator
    so we save the raw uint8 image here.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return False

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = crop_face(img, net)
    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return True


# ─────────────────────────────────────────────
#  SPLIT HELPER
# ─────────────────────────────────────────────
def split_paths(paths):
    random.shuffle(paths)
    n = len(paths)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    return (
        paths[:n_train],
        paths[n_train:n_train + n_val],
        paths[n_train + n_val:],
    )


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def prepare(src: Path, dst: Path, use_face_crop: bool = True):
    """
    src must contain:
        src/real/  → real face images
        src/fake/  → deepfake images
    """
    net = get_face_detector() if use_face_crop else None
    if net:
        print("✔  Face detector loaded — will crop faces")
    else:
        print("⚠  No face detector found — using full images")

    for label in ("real", "fake"):
        src_label = src / label
        if not src_label.exists():
            print(f"  ⚠  Folder not found: {src_label} — skipping")
            continue

        exts = {".jpg", ".jpeg", ".png", ".webp"}
        paths = sorted([p for p in src_label.rglob("*") if p.suffix.lower() in exts])
        print(f"\n  {label.upper()}: {len(paths)} images found")

        train_p, val_p, test_p = split_paths(paths)
        print(f"    Train {len(train_p)} | Val {len(val_p)} | Test {len(test_p)}")

        for split_name, split_paths_list in [
            ("train", train_p), ("val", val_p), ("test", test_p)
        ]:
            print(f"    Processing {split_name}…")
            ok = fail = 0
            for p in tqdm(split_paths_list, desc=f"    {split_name}/{label}"):
                out = dst / split_name / label / p.name
                if preprocess_image(p, out, net):
                    ok += 1
                else:
                    fail += 1
            print(f"      ✔ {ok} saved   ✗ {fail} failed")

    print("\n✅  Dataset prepared at:", dst)
    print_dataset_stats(dst)


def print_dataset_stats(dst: Path):
    print("\nDataset Statistics:")
    print("-" * 40)
    for split in ("train", "val", "test"):
        for label in ("real", "fake"):
            folder = dst / split / label
            count = len(list(folder.glob("*"))) if folder.exists() else 0
            print(f"  {split:5s} / {label:4s} : {count:6d} images")
    print("-" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare deepfake detection dataset")
    parser.add_argument("--src", required=True, help="Source raw dataset folder")
    parser.add_argument("--dst", default="../../dataset", help="Output folder")
    parser.add_argument("--no-face-crop", action="store_true")
    args = parser.parse_args()

    prepare(
        src=Path(args.src),
        dst=Path(args.dst),
        use_face_crop=not args.no_face_crop,
    )
