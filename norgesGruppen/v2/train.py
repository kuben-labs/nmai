"""
Train YOLO11x on NorgesGruppen shelf dataset, export to ONNX for submission.

Requires latest ultralytics (pip install ultralytics --upgrade).
Sandbox has ultralytics 8.1.0, so we export to ONNX for inference.

Augmentation philosophy for shelf product detection:
  - Mosaic: YES, helps with small dense objects, but close early (last 50 epochs)
  - MixUp: NO, blending two shelf images creates ghost products
  - CopyPaste: NO, pasting onto dense shelves creates unrealistic overlaps
  - HSV: MODERATE, store lighting varies but don't distort product colors
  - FlipLR: YES, shelves are left-right symmetric
  - FlipUD: NO, products are never upside down on shelves
  - Geometric: SMALL rotation/translate for camera angle variation

Usage:
    python train.py                                          # defaults
    python train.py --model yolo11l --epochs 200             # lighter model
    python train.py --resume                                 # resume training
    python train.py --export-only runs/.../best.pt           # just export ONNX
"""

import argparse
from pathlib import Path


DATASET_YAML = Path(__file__).parent / "dataset_tiled" / "dataset.yaml"
RUNS_DIR = Path(__file__).parent / "runs"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolo11x",
                   choices=["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"])
    p.add_argument("--weights", default=None,
                   help="Path to existing .pt weights to fine-tune from")
    p.add_argument("--data", default=str(DATASET_YAML))
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=4,
                   help="Batch size PER GPU (4 fits L4 24GB at imgsz=1280)")
    p.add_argument("--workers", type=int, default=4,
                   help="Workers per GPU (keep low for multi-GPU, avoid /dev/shm exhaustion)")
    p.add_argument("--device", default="0",
                   help="GPU ids: '0' single, '0,1' 2-GPU, '0,1,2,3' 4-GPU DDP")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--export-only", default=None,
                   help="Skip training, just export this .pt to ONNX")
    return p.parse_args()


def export_onnx(pt_path, imgsz=1280):
    """Export ultralytics .pt to ONNX for sandbox inference."""
    from ultralytics import YOLO

    model = YOLO(str(pt_path))
    model.export(
        format="onnx",
        imgsz=imgsz,
        half=True,
        opset=17,
        simplify=True,
    )
    onnx_path = Path(str(pt_path).replace(".pt", ".onnx"))
    if onnx_path.exists():
        size_mb = onnx_path.stat().st_size / 1e6
        print(f"\nONNX exported: {onnx_path} ({size_mb:.1f} MB)")
        if size_mb > 420:
            print("WARNING: over 420 MB limit — try yolo11l")
    return onnx_path


def main():
    args = parse_args()
    from ultralytics import YOLO

    if args.export_only:
        export_onnx(args.export_only, args.imgsz)
        return

    # Parse device: "0" → single GPU, "0,1,2,3" → multi-GPU DDP
    device = args.device
    if "," in device:
        device = [int(d) for d in device.split(",")]
        n_gpus = len(device)
    else:
        n_gpus = 1

    if args.resume:
        last_pts = sorted(RUNS_DIR.glob("**/last.pt"))
        if not last_pts:
            raise FileNotFoundError("No runs found to resume.")
        model = YOLO(str(last_pts[-1]))
        model.train(resume=True)
    else:
        weights = args.weights if args.weights else f"{args.model}.pt"
        print(f"Training on {n_gpus} GPU(s): {device}")
        print(f"Effective batch size: {args.batch * n_gpus}")
        model = YOLO(weights)
        model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch * n_gpus,  # total batch = per-gpu * n_gpus
            workers=args.workers,
            device=device,
            project=str(RUNS_DIR),
            name=f"{args.model}_ngd",

            # ── Augmentation (careful for dense shelf images) ────────
            # Color: moderate — store lighting varies but product colors matter
            hsv_h=0.015,        # small hue shift
            hsv_s=0.3,          # moderate saturation (v1 used 0.7 — too much!)
            hsv_v=0.3,          # moderate brightness

            # Geometric: small — cameras are roughly fixed but have slight variation
            degrees=3.0,        # small rotation for slight camera tilt
            translate=0.1,      # small translation
            scale=0.5,          # default scale — important for multi-size products
            shear=2.0,          # tiny shear
            perspective=0.0001, # tiny perspective

            # Flips
            fliplr=0.5,         # shelves are symmetric left-right
            flipud=0.0,         # products are NEVER upside down

            # Mosaic: helps small-object detection but breaks spatial context
            mosaic=1.0,
            close_mosaic=50,    # close mosaic 50 epochs before end (v1 used 30)

            # DISABLED — these hurt dense shelf detection:
            mixup=0.0,          # blending shelves creates ghost products
            copy_paste=0.0,     # pasting on dense shelves = unrealistic overlaps

            # ── Training schedule ────────────────────────────────────
            cos_lr=True,
            patience=80,        # generous — small dataset has noisy training
            warmup_epochs=5,
            warmup_momentum=0.5,

            # ── Loss weights ─────────────────────────────────────────
            # Classification is 30% of competition score, boost it
            cls=1.5,            # default 0.5, v1 used 1.0
            box=7.5,            # default
            dfl=1.5,            # default

            # ── Regularization ───────────────────────────────────────
            dropout=0.1,           # YOLO11 supports head dropout

            # ── Other ────────────────────────────────────────────────
            deterministic=False,  # deterministic=True segfaults with DDP+NCCL
            save=True,
            plots=True,
            optimizer="AdamW",
            lr0=0.001,
            lrf=0.01,             # final LR = lr0 * lrf = 1e-5
            weight_decay=0.0005,
            nbs=64,               # nominal batch size for LR scaling
        )

    # Auto-export best weights to ONNX
    best_pts = sorted(RUNS_DIR.glob(f"**/{args.model}_ngd*/weights/best.pt"))
    if best_pts:
        export_onnx(best_pts[-1], args.imgsz)


if __name__ == "__main__":
    main()
