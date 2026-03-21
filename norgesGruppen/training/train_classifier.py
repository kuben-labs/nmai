"""
Train a product classifier (EfficientNet-V2-S) on extracted crops + reference images.

Requires: run prepare_crops.py first.

Usage:
    python train_classifier.py                         # defaults
    python train_classifier.py --epochs 50 --batch 64  # custom
    python train_classifier.py --device 0              # GPU id
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from PIL import Image

CROPS_DIR = Path(__file__).parent / "crops"
RUNS_DIR = Path(__file__).parent / "runs"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class CropDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples  # list of (path_str, class_id)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def build_model(num_classes):
    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def get_transforms(train=True, img_size=224):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.2),
        ])
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(
        f"cuda:{args.device}"
        if args.device != "cpu" and torch.cuda.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    with open(CROPS_DIR / "class_names.json") as f:
        class_names = json.load(f)
    num_classes = len(class_names)

    # Collect samples
    all_samples = []
    for cls_id in range(num_classes):
        cls_dir = CROPS_DIR / str(cls_id)
        if not cls_dir.exists():
            continue
        for img_path in cls_dir.glob("*.jpg"):
            all_samples.append((str(img_path), cls_id))

    print(f"Total samples: {len(all_samples)}, classes: {num_classes}")

    random.seed(42)
    random.shuffle(all_samples)

    # 90/10 split
    n_val = max(1, int(len(all_samples) * 0.1))
    val_samples = all_samples[:n_val]
    train_samples = all_samples[n_val:]
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    # Class-balanced sampling
    train_labels = np.array([s[1] for s in train_samples])
    class_counts = np.bincount(train_labels, minlength=num_classes).astype(np.float64)
    class_counts = np.maximum(class_counts, 1.0)
    sample_weights = 1.0 / class_counts[train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=len(train_samples),
        replacement=True,
    )

    # Class-weighted loss
    total = class_counts.sum()
    loss_weights = total / (num_classes * class_counts)
    loss_weights = np.clip(loss_weights, 0.1, 10.0)
    loss_weights = torch.FloatTensor(loss_weights).to(device)

    train_ds = CropDataset(train_samples, get_transforms(True, args.img_size))
    val_ds = CropDataset(val_samples, get_transforms(False, args.img_size))

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, sampler=sampler,
        num_workers=args.workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    # Model
    model = build_model(num_classes).to(device)

    # Differential LR: backbone gets 100x lower LR
    backbone_params, head_params = [], []
    for name, param in model.named_parameters():
        (head_params if "classifier" in name else backbone_params).append(param)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.01},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(weight=loss_weights, label_smoothing=0.1)

    # Mixed precision
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    save_dir = RUNS_DIR / "classifier"
    save_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    for epoch in range(args.epochs):
        # --- Train ---
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total

        # --- Validate ---
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    outputs = model(images)
                _, preds = outputs.max(1)
                val_total += labels.size(0)
                val_correct += preds.eq(labels).sum().item()

        val_acc = val_correct / val_total
        scheduler.step()

        lr = optimizer.param_groups[1]["lr"]
        print(
            f"Epoch {epoch + 1:3d}/{args.epochs} | "
            f"Loss: {train_loss:.4f} | Train: {train_acc:.4f} | "
            f"Val: {val_acc:.4f} | LR: {lr:.2e}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "num_classes": num_classes,
                "class_names": class_names,
                "img_size": args.img_size,
                "best_acc": best_acc,
                "epoch": epoch + 1,
            }, save_dir / "best_classifier.pt")
            print(f"  -> Saved (acc={best_acc:.4f})")

    # Export FP16 for submission
    ckpt = torch.load(save_dir / "best_classifier.pt", map_location="cpu",
                       weights_only=False)
    state = ckpt["model_state_dict"]
    ckpt["model_state_dict"] = {k: v.half() for k, v in state.items()}
    fp16_path = save_dir / "best_classifier_fp16.pt"
    torch.save(ckpt, fp16_path)
    size_mb = fp16_path.stat().st_size / 1e6
    print(f"\nBest val accuracy: {best_acc:.4f}")
    print(f"FP16 saved: {fp16_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
