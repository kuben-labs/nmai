"""
Train a product classifier on prepared data.

Uses EfficientNet-B0 from timm (pre-installed in sandbox as timm==0.9.12).
Exports to ONNX for submission.

Run on VM:
    python train_classifier.py --data classifier_data --epochs 30 --output weights/classifier.onnx
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import timm


class ProductDataset(Dataset):
    def __init__(self, data_dir, manifest, transform=None, split="train", val_ratio=0.1):
        self.data_dir = Path(data_dir)
        self.transform = transform

        samples = manifest["samples"]
        # Deterministic split
        np.random.seed(42)
        indices = np.random.permutation(len(samples))
        val_size = int(len(samples) * val_ratio)

        if split == "train":
            self.samples = [samples[i] for i in indices[val_size:]]
        else:
            self.samples = [samples[i] for i in indices[:val_size]]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rec = self.samples[idx]
        if rec["source"] == "reference":
            img_path = self.data_dir / "references" / rec["filename"]
        else:
            img_path = self.data_dir / "crops" / rec["filename"]

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, rec["category_id"]


def build_transforms(img_size=224):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform


def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=scaler is not None):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def export_onnx(model, output_path, img_size=224, num_classes=357, opset=17):
    """Export classifier to ONNX (FP16-compatible)."""
    model.eval()
    model.cpu()

    dummy = torch.randn(1, 3, img_size, img_size)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        opset_version=opset,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
    )

    # Convert to FP16
    try:
        import onnx
        from onnxruntime.transformers.float16 import convert_float_to_float16
        model_onnx = onnx.load(str(output_path))
        model_fp16 = convert_float_to_float16(model_onnx, keep_io_types=True)
        onnx.save(model_fp16, str(output_path))
        print("Converted to FP16")
    except ImportError:
        print("onnx/onnxruntime not available, saved as FP32")

    size_mb = output_path.stat().st_size / 1e6
    print(f"Exported classifier: {output_path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="classifier_data", help="Data directory")
    parser.add_argument("--output", default="weights/classifier.onnx", help="Output ONNX path")
    parser.add_argument("--model", default="efficientnet_b0", help="timm model name")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-classes", type=int, default=357,
                        help="Number of product categories (0-355 + unknown=356)")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--save-pt", action="store_true",
                        help="Also save PyTorch weights")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = Path(args.data)
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True)

    # Load manifest
    with open(data_dir / "manifest.json") as f:
        manifest = json.load(f)
    print(f"Dataset: {manifest['total_samples']} samples")

    # Datasets
    train_tf, val_tf = build_transforms(args.img_size)
    train_ds = ProductDataset(data_dir, manifest, transform=train_tf, split="train")
    val_ds = ProductDataset(data_dir, manifest, transform=val_tf, split="val")
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # Model
    model = timm.create_model(
        args.model, pretrained=args.pretrained, num_classes=args.num_classes
    )
    model = model.to(device)
    print(f"Model: {args.model}, params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # Training setup
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler() if device == "cuda" else None

    best_acc = 0
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), output_path.with_suffix(".pt"))
            print(f"  Saved best model (acc={val_acc:.3f})")

    # Load best and export
    model.load_state_dict(torch.load(output_path.with_suffix(".pt"), weights_only=True))
    export_onnx(model, output_path, img_size=args.img_size, num_classes=args.num_classes)

    if not args.save_pt:
        output_path.with_suffix(".pt").unlink(missing_ok=True)

    print(f"\nBest validation accuracy: {best_acc:.3f}")


if __name__ == "__main__":
    main()
