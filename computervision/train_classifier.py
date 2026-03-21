"""Train EfficientNet-V2-S classifier for product identification (multi-GPU DDP)."""
import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
import timm
from tqdm import tqdm


def get_transforms(split, imgsz=224):
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(imgsz, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize(int(imgsz * 1.14)),
        transforms.CenterCrop(imgsz),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def is_main_process():
    return not is_distributed() or dist.get_rank() == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=32, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--model", default="tf_efficientnetv2_s.in21k_ft_in1k")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", default="runs/classify")
    parser.add_argument("--backend", default="nccl", choices=["nccl", "gloo"])
    args = parser.parse_args()

    # DDP setup
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend=args.backend)
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}")
    dataset_dir = Path(__file__).parent / "cls_dataset"
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Datasets
    train_ds = datasets.ImageFolder(dataset_dir / "train", transform=get_transforms("train", args.imgsz))
    val_ds = datasets.ImageFolder(dataset_dir / "val", transform=get_transforms("val", args.imgsz))

    num_classes = len(train_ds.classes)
    if is_main_process():
        print(f"Classes: {num_classes}, Train: {len(train_ds)}, Val: {len(val_ds)}")
        with open(output_dir / "class_to_idx.json", "w") as f:
            json.dump(train_ds.class_to_idx, f, indent=2)

    train_sampler = DistributedSampler(train_ds) if is_distributed() else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if is_distributed() else None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        sampler=val_sampler,
        num_workers=args.workers, pin_memory=True,
    )

    # Model
    model = timm.create_model(args.model, pretrained=True, num_classes=num_classes)
    model = model.to(device)
    if is_distributed():
        model = DDP(model, device_ids=[local_rank])
        if is_main_process():
            print(f"Using {dist.get_world_size()} GPUs with DDP")

    # Optimizer with lower LR for backbone, scaled by world size
    raw_model = model.module if is_distributed() else model
    backbone_params = []
    head_params = []
    for name, param in raw_model.named_parameters():
        if "classifier" in name or "head" in name or "fc" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    world_size = dist.get_world_size() if is_distributed() else 1
    scaled_lr = args.lr * (world_size ** 0.5)  # sqrt scaling rule

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": scaled_lr * 0.1},
        {"params": head_params, "lr": scaled_lr},
    ], weight_decay=0.01)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = 0.0
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        loader = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [train]") if is_main_process() else train_loader
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += images.size(0)
            if is_main_process() and hasattr(loader, 'set_postfix'):
                loader.set_postfix(loss=f"{train_loss/train_total:.4f}", acc=f"{train_correct/train_total:.4f}")

        scheduler.step()

        # Validate (all ranks compute, main rank reports)
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += images.size(0)

        # Aggregate across ranks
        if is_distributed():
            stats = torch.tensor([val_correct, val_total], device=device, dtype=torch.float64)
            dist.all_reduce(stats)
            val_correct, val_total = int(stats[0].item()), int(stats[1].item())

        val_acc = val_correct / val_total if val_total > 0 else 0

        if is_main_process():
            print(f"Epoch {epoch+1}: train_acc={train_correct/train_total:.4f}, val_acc={val_acc:.4f}")

            if val_acc > best_acc:
                best_acc = val_acc
                state_dict = raw_model.state_dict()
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": state_dict,
                    "val_acc": val_acc,
                    "num_classes": num_classes,
                    "model_name": args.model,
                    "imgsz": args.imgsz,
                }, output_dir / "best.pt")
                print(f"  -> New best: {val_acc:.4f}")

    if is_main_process():
        print(f"\nBest val accuracy: {best_acc:.4f}")
        print(f"Model saved to {output_dir / 'best.pt'}")

    if is_distributed():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
