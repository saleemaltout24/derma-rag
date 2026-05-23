#!/usr/bin/env python3
"""
Fine-tune EfficientNet-B0 skin classifier (8 classes, letterbox 224).

Expects folder layout (use scripts/prepare_isic2019.py first):
  data/isic2019/splits/train/MEL/*.jpg
  data/isic2019/splits/val/AK/*.jpg
  ...

Run from project root:
  python scripts/train_skin_classifier.py
  python scripts/train_skin_classifier.py --epochs 20 --batch-size 16

Saves: models/skin_classifier_v3.pth
Then run: python scripts/eval_classifier.py
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.skin_classifier import CLASS_NAMES, CLASSES, letterbox_rgb, load_image_rgb_exif

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train skin classifier v3")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "isic2019" / "splits",
        help="Root with train/ and val/ class subfolders",
    )
    p.add_argument(
        "--pretrained",
        type=Path,
        default=ROOT / "models" / "skin_classifier_v2.pth",
        help="Checkpoint to fine-tune from",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "models" / "skin_classifier_v3.pth",
    )
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=4, help="Early stop if val acc stalls")
    return p.parse_args()


class LetterboxFolderDataset(Dataset):
    def __init__(self, root: Path, class_to_idx: dict[str, int], train: bool = False):
        self.samples: list[tuple[Path, int]] = []
        self.train = train
        self.augment = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            ]
        )
        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

        for cls, idx in class_to_idx.items():
            folder = root / cls
            if not folder.is_dir():
                continue
            for path in sorted(folder.iterdir()):
                if path.suffix.lower() in IMAGE_EXT:
                    self.samples.append((path, idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        pil = letterbox_rgb(load_image_rgb_exif(str(path)))
        if self.train:
            pil = self.augment(pil)
        tensor = self.to_tensor(pil)
        return tensor, label


def build_model(num_classes: int) -> nn.Module:
    try:
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
    except Exception:
        model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def load_checkpoint_weights(model: nn.Module, path: Path) -> None:
    if not path.is_file():
        print(f"No checkpoint at {path} — training from ImageNet init.")
        return
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    prev_acc = ckpt.get("val_accuracy") if isinstance(ckpt, dict) else None
    print(f"Loaded weights from {path}" + (f" (prev val acc {prev_acc:.2f}%)" if prev_acc else ""))


def class_weights_from_dataset(dataset: LetterboxFolderDataset) -> torch.Tensor:
    counts = Counter(label for _, label in dataset.samples)
    n = len(dataset)
    weights = []
    for i in range(len(CLASSES)):
        c = counts.get(i, 0)
        weights.append(n / (len(CLASSES) * c) if c > 0 else 1.0)
    w = torch.tensor(weights, dtype=torch.float32)
    return w / w.mean()


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, dict[str, float]]:
    model.eval()
    correct = 0
    total = 0
    per_class_correct: Counter[str] = Counter()
    per_class_total: Counter[str] = Counter()

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        for pred, label in zip(preds.cpu().tolist(), labels.cpu().tolist()):
            code = CLASSES[label]
            per_class_total[code] += 1
            if pred == label:
                per_class_correct[code] += 1

    acc = 100.0 * correct / total if total else 0.0
    per_class_acc = {
        c: 100.0 * per_class_correct[c] / per_class_total[c] if per_class_total[c] else 0.0
        for c in CLASSES
    }
    return acc, per_class_acc


def main() -> None:
    args = parse_args()
    train_root = args.data_dir / "train"
    val_root = args.data_dir / "val"

    if not train_root.is_dir() or not val_root.is_dir():
        print(f"Missing train/val under {args.data_dir}")
        print("1. Download ISIC 2019")
        print("2. python scripts/prepare_isic2019.py")
        print("3. python scripts/train_skin_classifier.py")
        sys.exit(1)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    class_to_idx = {c: i for i, c in enumerate(CLASSES)}

    train_ds = LetterboxFolderDataset(train_root, class_to_idx, train=True)
    val_ds = LetterboxFolderDataset(val_root, class_to_idx, train=False)
    if len(train_ds) == 0:
        print(f"No training images in {train_root}")
        sys.exit(1)

    print(f"Train images: {len(train_ds)}  Val images: {len(val_ds)}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")
    if device.type == "cpu":
        print("Tip: training on CPU is slow (hours). Use a GPU machine or --epochs 8 --batch-size 16.")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(len(CLASSES))
    load_checkpoint_weights(model, args.pretrained)
    model.to(device)

    weights = class_weights_from_dataset(train_ds).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    stale = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1

        scheduler.step()
        val_acc, per_class = evaluate(model, val_loader, device)
        train_loss = running_loss / max(n_batches, 1)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch}/{args.epochs}  loss={train_loss:.4f}  val_acc={val_acc:.2f}%  "
            f"({elapsed:.0f}s)  best={best_acc:.2f}%"
        )
        ak_acc = per_class.get("AK", 0.0)
        scc_acc = per_class.get("SCC", 0.0)
        print(f"  AK val acc: {ak_acc:.1f}%   SCC val acc: {scc_acc:.1f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            stale = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": CLASSES,
                    "class_names": CLASS_NAMES,
                    "val_accuracy": val_acc,
                    "epoch": epoch,
                    "per_class_val_accuracy": per_class,
                },
                args.output,
            )
            print(f"  -> saved {args.output}")
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stop (no improvement for {args.patience} epochs).")
                break

    print()
    print(f"Done. Best val accuracy: {best_acc:.2f}%")
    print(f"Weights: {args.output}")
    print("Restart backend, or set SKIN_CLASSIFIER_PATH to use v3.")


if __name__ == "__main__":
    main()
