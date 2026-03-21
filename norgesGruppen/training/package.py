"""
Package submission zip.

Places run.py + weights at the zip root (no subfolders).
Supports both .pt (ultralytics) and .onnx (YOLO11/any) weights.

Usage:
    python package.py --yolo runs/yolov8x_ngd/weights/best.pt
    python package.py --yolo runs/yolo11x_ngd/weights/best.onnx
    python package.py --yolo ... --classifier runs/classifier/best_classifier_fp16.pt
"""

import argparse
import zipfile
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).parent

_orig_torch_load = torch.load
def _patched(f, map_location=None, **kw):
    kw.setdefault("weights_only", False)
    return _orig_torch_load(f, map_location=map_location, **kw)
torch.load = _patched


def export_yolo_fp16(src, dst):
    """Convert ultralytics checkpoint to FP16."""
    ckpt = torch.load(src, map_location="cpu")

    if "model" in ckpt:
        model = ckpt["model"]
        if hasattr(model, "half"):
            ckpt["model"] = model.half()
        if "ema" in ckpt and ckpt["ema"] is not None:
            ckpt["ema"] = ckpt["ema"].half()

    ckpt.pop("optimizer", None)
    ckpt.pop("updates", None)

    torch.save(ckpt, dst)
    return dst.stat().st_size / 1e6


def export_classifier_fp16(src, dst):
    """Convert classifier state_dict to FP16."""
    ckpt = torch.load(src, map_location="cpu")
    if "model_state_dict" in ckpt:
        ckpt["model_state_dict"] = {
            k: v.half() for k, v in ckpt["model_state_dict"].items()
        }
    torch.save(ckpt, dst)
    return dst.stat().st_size / 1e6


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--yolo", required=True,
                   help="Path to YOLO weights (.pt or .onnx)")
    p.add_argument("--classifier", default=None)
    p.add_argument("--out", default=str(SCRIPT_DIR / "submission.zip"))
    p.add_argument("--no-fp16", action="store_true")
    args = p.parse_args()

    out_path = Path(args.out)
    tmp_dir = SCRIPT_DIR / "_tmp_pkg"
    tmp_dir.mkdir(exist_ok=True)

    yolo_src = Path(args.yolo)
    is_onnx = yolo_src.suffix == ".onnx"

    # YOLO weights
    if is_onnx:
        # ONNX is already exported — just use it directly
        yolo_pkg = yolo_src
        yolo_mb = yolo_src.stat().st_size / 1e6
        yolo_zip_name = "best.onnx"
    elif args.no_fp16:
        yolo_pkg = yolo_src
        yolo_mb = yolo_src.stat().st_size / 1e6
        yolo_zip_name = "best.pt"
    else:
        yolo_pkg = tmp_dir / "best.pt"
        yolo_mb = export_yolo_fp16(yolo_src, yolo_pkg)
        yolo_zip_name = "best.pt"

    print(f"YOLO:       {yolo_mb:.1f} MB  ({yolo_zip_name})"
          f"  (src: {yolo_src.stat().st_size / 1e6:.1f} MB)")

    total_mb = yolo_mb
    cls_pkg = None
    if args.classifier:
        cls_src = Path(args.classifier)
        if cls_src.exists():
            if args.no_fp16 or "fp16" in cls_src.name:
                cls_pkg = cls_src
            else:
                cls_pkg = tmp_dir / "best_classifier.pt"
                export_classifier_fp16(cls_src, cls_pkg)
            cls_mb = cls_pkg.stat().st_size / 1e6
            total_mb += cls_mb
            print(f"Classifier: {cls_mb:.1f} MB")

    print(f"Total wt:   {total_mb:.1f} MB")
    if total_mb > 420:
        print("WARNING: over 420 MB weight limit!")

    # Build zip — everything at root
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(SCRIPT_DIR / "run.py", "run.py")
        zf.write(yolo_pkg, yolo_zip_name)
        if cls_pkg:
            zf.write(cls_pkg, "best_classifier.pt")

    zip_mb = out_path.stat().st_size / 1e6
    print(f"\n-> {out_path} ({zip_mb:.1f} MB)")

    # Verify structure
    with zipfile.ZipFile(out_path, "r") as zf:
        print("\nZip contents:")
        for info in zf.infolist():
            print(f"  {info.filename:30s} {info.file_size / 1e6:.1f} MB")

    # Cleanup temp
    for f in tmp_dir.glob("*"):
        f.unlink()
    tmp_dir.rmdir()


if __name__ == "__main__":
    main()
