#!/usr/bin/env python3
"""
Build train/val image folders from ISIC 2019 (patient-level split by lesion_id).

Download ISIC 2019 from: https://challenge.isic-archive.com/data/#2019
Unzip so you have under data/isic2019/raw/:
  ISIC_2019_Training_Input/*.jpg
  ISIC_2019_Training_GroundTruth.csv
  ISIC_2019_Training_Metadata.csv

Run from project root:
  python scripts/prepare_isic2019.py
  python scripts/prepare_isic2019.py --raw-dir /path/to/isic/folder
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.skin_classifier import CLASSES

# ISIC 2019 column name -> our classifier code
ISIC_LABEL_TO_CLASS = {
    "MEL": "MEL",
    "NV": "NV",
    "BCC": "BCC",
    "AKIEC": "AK",
    "BKL": "BKL",
    "DF": "DF",
    "VASC": "VASC",
    "SCC": "SCC",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare ISIC 2019 train/val folders")
    p.add_argument(
        "--raw-dir",
        type=Path,
        default=ROOT / "data" / "isic2019" / "raw",
        help="Folder containing ISIC_2019_Training_* files",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "isic2019" / "splits",
        help="Output: out-dir/train/MEL/, out-dir/val/MEL/, ...",
    )
    p.add_argument("--val-ratio", type=float, default=0.15, help="Fraction of lesions for val")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of symlink (uses more disk)",
    )
    return p.parse_args()


def _is_positive_label(value: str | None) -> bool:
    if value is None or value == "":
        return False
    try:
        return float(value) >= 0.5
    except (TypeError, ValueError):
        return str(value).strip().lower() in ("1", "1.0", "true")


def read_ground_truth(gt_path: Path) -> dict[str, str]:
    """image_id -> class code (one-hot row in CSV)."""
    labels: dict[str, str] = {}
    with open(gt_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row.get("image") or row.get("image_id")
            if not image_id:
                continue
            for isic_col, cls in ISIC_LABEL_TO_CLASS.items():
                if _is_positive_label(row.get(isic_col)):
                    labels[image_id] = cls
                    break
    return labels


def read_lesion_ids(meta_path: Path) -> dict[str, str]:
    """image_id -> lesion_id for grouped split."""
    mapping: dict[str, str] = {}
    with open(meta_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row.get("image") or row.get("image_id")
            lesion_id = row.get("lesion_id")
            if image_id and lesion_id:
                mapping[image_id] = str(lesion_id)
    return mapping


def find_image(raw_dir: Path, image_id: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png", ".JPG"):
        p = raw_dir / "ISIC_2019_Training_Input" / f"{image_id}{ext}"
        if p.is_file():
            return p
        p = raw_dir / f"{image_id}{ext}"
        if p.is_file():
            return p
    return None


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir
    out_dir = args.out_dir

    gt_path = raw_dir / "ISIC_2019_Training_GroundTruth.csv"
    meta_path = raw_dir / "ISIC_2019_Training_Metadata.csv"
    if not gt_path.is_file():
        print(f"Missing: {gt_path}")
        print("Download ISIC 2019 and point --raw-dir at the folder with Training CSVs.")
        sys.exit(1)
    if not meta_path.is_file():
        print(f"Missing: {meta_path}")
        sys.exit(1)

    labels = read_ground_truth(gt_path)
    lesion_map = read_lesion_ids(meta_path)

    # lesion_id -> list of (image_id, class)
    by_lesion: dict[str, list[tuple[str, str]]] = defaultdict(list)
    missing_img = 0
    for image_id, cls in labels.items():
        if cls not in CLASSES:
            continue
        lesion_id = lesion_map.get(image_id, image_id)
        src = find_image(raw_dir, image_id)
        if src is None:
            missing_img += 1
            continue
        by_lesion[lesion_id].append((image_id, cls, src))

    if not by_lesion:
        print("No images found. Check --raw-dir and ISIC_2019_Training_Input/")
        sys.exit(1)

    lesion_ids = list(by_lesion.keys())
    random.seed(args.seed)
    random.shuffle(lesion_ids)
    n_val = max(1, int(len(lesion_ids) * args.val_ratio))
    val_lesions = set(lesion_ids[:n_val])
    train_lesions = set(lesion_ids[n_val:])

    for split in ("train", "val"):
        for cls in CLASSES:
            (out_dir / split / cls).mkdir(parents=True, exist_ok=True)

    counts = {"train": defaultdict(int), "val": defaultdict(int)}
    for lesion_id, items in by_lesion.items():
        split = "val" if lesion_id in val_lesions else "train"
        for image_id, cls, src in items:
            dst = out_dir / split / cls / f"{image_id}{src.suffix.lower()}"
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            if args.copy:
                import shutil

                shutil.copy2(src, dst)
            else:
                dst.symlink_to(src.resolve())
            counts[split][cls] += 1

    print(f"Lesions: {len(train_lesions)} train, {len(val_lesions)} val")
    print(f"Images missing on disk: {missing_img}")
    print(f"Output: {out_dir}")
    for split in ("train", "val"):
        total = sum(counts[split].values())
        print(f"  {split}: {total} images")
        for cls in CLASSES:
            if counts[split][cls]:
                print(f"    {cls}: {counts[split][cls]}")


if __name__ == "__main__":
    main()
