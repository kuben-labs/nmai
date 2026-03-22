"""Training script for YOLOv8 on the NorgesGruppen grocery detection dataset.

Usage:
    # Quick baseline (YOLOv8m, 640px, fast iteration)
    python scripts/train.py --model yolov8m.pt --imgsz 640 --epochs 50 --name baseline

    # Full training (YOLOv8m, 1280px, longer)
    python scripts/train.py --model yolov8m.pt --imgsz 1280 --epochs 150 --name full

    # Scale up (YOLOv8l, best accuracy)
    python scripts/train.py --model yolov8l.pt --imgsz 1280 --epochs 150 --name large

    # Resume interrupted training
    python scripts/train.py --resume runs/full/weights/last.pt
"""

import argparse
from pathlib import Path

import torch

# Fix PyTorch 2.6+ compatibility with ultralytics 8.1.0:
# PyTorch 2.6 changed torch.load default to weights_only=True,
# but ultralytics 8.1.0 predates this change and needs weights_only=False.
_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

from ultralytics import YOLO  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8 for grocery detection")
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8m.pt",
        help="Model: yolov8n/s/m/l/x.pt (default: yolov8m.pt)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="configs/dataset.yaml",
        help="Dataset config YAML",
    )
    parser.add_argument(
        "--imgsz", type=int, default=1280, help="Image size (default: 1280)"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Training epochs (default: 100)"
    )
    parser.add_argument("--batch", type=int, default=-1, help="Batch size (-1 = auto)")
    parser.add_argument("--device", type=str, default="0", help="GPU device(s)")
    parser.add_argument("--name", type=str, default="exp", help="Experiment name")
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Early stopping patience (0 = disabled)",
    )
    args = parser.parse_args()

    if args.resume:
        # Resume training
        print(f"Resuming training from {args.resume}")
        model = YOLO(args.resume)
        model.train(resume=True)
        return

    print(f"Training {args.model} at imgsz={args.imgsz} for {args.epochs} epochs")
    print(f"Dataset: {args.data}")
    print(f"Experiment: runs/{args.name}")

    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="runs",
        name=args.name,
        # Optimization
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=5,
        # Augmentation — strong augmentation for small dataset
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,  # Important: handles varying image sizes
        shear=2.0,
        flipud=0.0,  # No vertical flip — shelves have orientation
        fliplr=0.5,
        mosaic=1.0,  # Mosaic augmentation — great for dense scenes
        mixup=0.1,
        copy_paste=0.1,
        # Training settings
        patience=args.patience,
        cos_lr=True,
        close_mosaic=15,  # Disable mosaic for last 15 epochs
        amp=True,  # Mixed precision
        # Saving
        save=True,
        save_period=10,
        val=True,
        plots=True,
        verbose=True,
    )

    # Run validation after training
    best_weights = Path("runs") / args.name / "weights" / "best.pt"
    if best_weights.exists():
        print(f"\nRunning final validation with {best_weights}")
        val_model = YOLO(str(best_weights))
        metrics = val_model.val(data=args.data, imgsz=args.imgsz, device=args.device)
        print(f"\nFinal mAP@0.5: {metrics.box.map50:.4f}")
        print(f"Final mAP@0.5:0.95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
