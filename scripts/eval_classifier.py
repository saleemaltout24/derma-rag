#!/usr/bin/env python3
"""
Test the skin classifier on labeled images.

Folder layout (class name = folder name):
  data/eval/MEL/photo1.jpg
  data/eval/NV/photo2.jpg
  ...

Run from project root:
  python scripts/eval_classifier.py
  python scripts/eval_classifier.py path/to/your/eval/folder
"""

import sys
from pathlib import Path

# Project root on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.skin_classifier import CLASS_NAMES, CLASSES, classify_skin_image

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main() -> None:
    eval_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "eval"
    if not eval_dir.is_dir():
        print(f"Missing folder: {eval_dir}")
        print("Create it and add subfolders named like: MEL, NV, BCC, AK, BKL, DF, VASC, SCC")
        sys.exit(1)

    rows = []
    for class_code in CLASSES:
        folder = eval_dir / class_code
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in IMAGE_EXT:
                continue
            result = classify_skin_image(str(path))
            predicted = result.get("predicted_class", "?")
            conf = result.get("confidence", 0)
            ok = predicted == class_code
            rows.append((path.name, class_code, predicted, conf, ok))

    if not rows:
        print(f"No images found under {eval_dir}")
        print("Add photos inside folders named: " + ", ".join(CLASSES))
        sys.exit(1)

    correct = sum(1 for r in rows if r[4])
    total = len(rows)

    print()
    print("=" * 60)
    print(f"  Results: {correct}/{total} correct ({100 * correct / total:.0f}%)")
    print("=" * 60)
    print()
    print(f"{'File':<24} {'True':<6} {'Predicted':<10} {'Conf%':<8} OK?")
    print("-" * 60)
    for name, true, pred, conf, ok in rows:
        mark = "yes" if ok else "NO"
        true_name = CLASS_NAMES.get(true, true)
        pred_name = CLASS_NAMES.get(pred, pred)
        print(f"{name:<24} {true:<6} {pred:<10} {conf:<8.1f} {mark}")
        if not ok:
            print(f"    expected: {true_name}  got: {pred_name}")

    print()
    print("Wrong guesses by true class:")
    by_true: dict[str, list[str]] = {}
    for _, true, pred, _, ok in rows:
        if not ok:
            by_true.setdefault(true, []).append(pred)
    if not by_true:
        print("  (none — all correct!)")
    else:
        for true, preds in by_true.items():
            print(f"  {true} → {', '.join(preds)}")


if __name__ == "__main__":
    main()
