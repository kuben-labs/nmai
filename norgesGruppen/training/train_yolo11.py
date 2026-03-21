"""
Train YOLO11x on the NorgesGruppen shelf dataset and export to ONNX.

IMPORTANT: Requires latest ultralytics (NOT 8.1.0).
    pip install ultralytics --upgrade

The sandbox only has ultralytics 8.1.0, so we export to ONNX for inference.

Usage:
    python train_yolo11.py                                    # defaults
    python train_yolo11.py --model yolo11l --epochs 200       # lighter model
    python train_yolo11.py --export-only runs/.../best.pt     # just export
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
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="0")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--export-only", default=None,
                   help="Skip training, just export this .pt to ONNX")
    return p.parse_args()


def export_onnx(pt_path, imgsz=1280):
    """Export ultralytics .pt to ONNX FP16."""
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
            print("WARNING: over 420 MB — try yolo11l or quantize further")
    return onnx_path


def main():
    args = parse_args()
    from ultralytics import YOLO

    if args.export_only:
        export_onnx(args.export_only, args.imgsz)
        return

    if args.resume:
        last_pts = sorted(RUNS_DIR.glob("**/last.pt"))
        if not last_pts:
            raise FileNotFoundError("No runs found to resume.")
        model = YOLO(str(last_pts[-1]))
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

            # ── Augmentation (minimal — heavy aug can hurt on dense shelves) ──
            hsv_h=0.015,
            hsv_s=0.4,
            hsv_v=0.3,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.0,
            copy_paste=0.0,

            # ── Schedule ──
            cos_lr=True,
            patience=50,
            close_mosaic=30,

            # ── Loss ──
            cls=1.0,
            label_smoothing=0.1,

            # ── Other ──
            save=True,
            plots=True,
        )

    # Auto-export best weights to ONNX
    best_pts = sorted(RUNS_DIR.glob(f"**/{args.model}_ngd*/weights/best.pt"))
    if best_pts:
        export_onnx(best_pts[-1], args.imgsz)


if __name__ == "__main__":
    main()
