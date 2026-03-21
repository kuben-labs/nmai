"""
Single-GPU training script. Accepts all config via CLI args.
Designed to be launched by launch_experiments.py, one per GPU.

Usage:
    CUDA_VISIBLE_DEVICES=3 python train_single.py --name exp03_lr_high --model yolo11x --lr0 0.002
"""

import argparse
from pathlib import Path


DATASET_YAML = Path(__file__).parent / "dataset_tiled" / "dataset.yaml"
RUNS_DIR = Path(__file__).parent / "runs"


def parse_args():
    p = argparse.ArgumentParser()
    # Experiment identity
    p.add_argument("--name", required=True, help="Experiment name (used as run dir)")

    # Model
    p.add_argument("--model", default="yolo11x",
                   choices=["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"])
    p.add_argument("--weights", default=None,
                   help="Path to .pt weights to fine-tune from (default: pretrained)")

    # Data
    p.add_argument("--data", default=str(DATASET_YAML))
    p.add_argument("--imgsz", type=int, default=1280)

    # Training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=80)

    # Optimizer
    p.add_argument("--optimizer", default="AdamW")
    p.add_argument("--lr0", type=float, default=0.001)
    p.add_argument("--lrf", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=0.0005)
    p.add_argument("--nbs", type=int, default=64)
    p.add_argument("--warmup_epochs", type=float, default=5)

    # Augmentation
    p.add_argument("--hsv_h", type=float, default=0.015)
    p.add_argument("--hsv_s", type=float, default=0.3)
    p.add_argument("--hsv_v", type=float, default=0.3)
    p.add_argument("--degrees", type=float, default=3.0)
    p.add_argument("--translate", type=float, default=0.1)
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--shear", type=float, default=2.0)
    p.add_argument("--perspective", type=float, default=0.0001)
    p.add_argument("--fliplr", type=float, default=0.5)
    p.add_argument("--mosaic", type=float, default=1.0)
    p.add_argument("--close_mosaic", type=int, default=50)
    p.add_argument("--mixup", type=float, default=0.0)
    p.add_argument("--copy_paste", type=float, default=0.0)

    # Loss
    p.add_argument("--cls", type=float, default=1.5)
    p.add_argument("--box", type=float, default=7.5)
    p.add_argument("--dfl", type=float, default=1.5)

    # Regularization
    p.add_argument("--dropout", type=float, default=0.1)

    return p.parse_args()


def main():
    args = parse_args()
    from ultralytics import YOLO

    weights = args.weights if args.weights else f"{args.model}.pt"
    print(f"[{args.name}] Training {weights} @ imgsz={args.imgsz}, batch={args.batch}")

    model = YOLO(weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=0,  # CUDA_VISIBLE_DEVICES already set by launcher
        project=str(RUNS_DIR),
        name=args.name,

        # Augmentation
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        perspective=args.perspective,
        fliplr=args.fliplr,
        flipud=0.0,
        mosaic=args.mosaic,
        close_mosaic=args.close_mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste,

        # Schedule
        cos_lr=True,
        patience=args.patience,
        warmup_epochs=args.warmup_epochs,
        warmup_momentum=0.5,

        # Loss
        cls=args.cls,
        box=args.box,
        dfl=args.dfl,

        # Regularization
        dropout=args.dropout,

        # Optimizer
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        weight_decay=args.weight_decay,
        nbs=args.nbs,

        # Other
        deterministic=False,
        save=True,
        plots=True,
    )

    # Export best to ONNX
    best_pts = sorted(RUNS_DIR.glob(f"**/{args.name}*/weights/best.pt"))
    if best_pts:
        model_exp = YOLO(str(best_pts[-1]))
        model_exp.export(
            format="onnx", imgsz=args.imgsz,
            half=True, opset=17, simplify=True,
        )
        print(f"[{args.name}] Exported ONNX: {best_pts[-1]}")


if __name__ == "__main__":
    main()
