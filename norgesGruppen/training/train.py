"""
Train YOLOv8x on the NorgesGruppen shelf dataset.

Pin: ultralytics==8.1.0

Usage:
    python train.py                                          # defaults (tiled dataset)
    python train.py --data dataset/dataset.yaml              # non-tiled
    python train.py --model yolov8l --epochs 100 --imgsz 640 # quick test
    python train.py --resume                                 # resume last run
    python train.py --weights runs/.../best.pt --epochs 50   # fine-tune
"""

import argparse
import torch
from pathlib import Path

# Fix for PyTorch 2.6+ changing weights_only default, which breaks ultralytics 8.1.0
_orig_torch_load = torch.load
def _patched_torch_load(f, map_location=None, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(f, map_location=map_location, **kwargs)
torch.load = _patched_torch_load


DATASET_YAML = Path(__file__).parent / "dataset_tiled" / "dataset.yaml"
RUNS_DIR = Path(__file__).parent / "runs"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolov8x",
                   choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
                   help="YOLOv8 variant")
    p.add_argument("--weights", default=None,
                   help="Path to existing .pt weights to fine-tune from")
    p.add_argument("--data", default=str(DATASET_YAML),
                   help="Path to dataset.yaml (default: tiled dataset)")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="0", help="CUDA device id or 'cpu'")
    p.add_argument("--resume", action="store_true", help="Resume last run")
    return p.parse_args()


def main():
    args = parse_args()

    from ultralytics import YOLO

    if args.resume:
        last_pts = sorted(RUNS_DIR.glob("**/last.pt"))
        if not last_pts:
            raise FileNotFoundError("No runs found to resume.")
        model_path = last_pts[-1]
        print(f"Resuming from {model_path}")
        model = YOLO(str(model_path))
        model.train(resume=True)
    else:
        weights = args.weights if args.weights else f"{args.model}.pt"
        model = YOLO(weights)
        model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
            project=str(RUNS_DIR),
            name=f"{args.model}_ngd",

            # ── Augmentation (aggressive for dense shelf images) ──
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.15,
            copy_paste=0.3,    # very effective for product detection

            # ── Training schedule ──
            cos_lr=True,
            patience=50,
            close_mosaic=30,   # disable mosaic for last 30 epochs

            # ── Loss weights ──
            # Increase cls loss since classification is 30% of score
            cls=1.0,           # default 0.5
            box=7.5,           # default
            dfl=1.5,           # default

            # ── Regularization ──
            label_smoothing=0.1,

            # ── Other ──
            save=True,
            plots=True,
            half=False,
            optimizer="auto",
        )

    # Export best weights to FP16 for submission
    best = sorted(RUNS_DIR.glob("**/best.pt"))[-1]
    print(f"\nBest weights: {best}")
    print("Exporting FP16 copy for submission...")
    ckpt = torch.load(best, map_location="cpu")
    if "model" in ckpt:
        ckpt["model"] = ckpt["model"].half()
    if "ema" in ckpt and ckpt["ema"] is not None:
        ckpt["ema"] = ckpt["ema"].half()
    # Strip optimizer (not needed for inference, saves space)
    ckpt.pop("optimizer", None)
    ckpt.pop("updates", None)
    fp16_path = best.parent / "best_fp16.pt"
    torch.save(ckpt, fp16_path)
    size_mb = fp16_path.stat().st_size / 1e6
    print(f"FP16 weights saved: {fp16_path}  ({size_mb:.1f} MB)")
    if size_mb > 420:
        print("WARNING: still over 420MB — try yolov8l or yolov8m")


if __name__ == "__main__":
    main()
