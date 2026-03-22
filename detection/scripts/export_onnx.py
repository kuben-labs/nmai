"""Export trained YOLOv8 model to ONNX format for sandbox inference.

Usage:
    python scripts/export_onnx.py --weights runs/full2/weights/best.pt --imgsz 1280
"""

import argparse
from pathlib import Path

import torch

# Fix PyTorch 2.6+ compatibility
_torch_load = torch.load


def _safe_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _torch_load(*args, **kwargs)


torch.load = _safe_load

from ultralytics import YOLO  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLO to ONNX")
    parser.add_argument("--weights", required=True, help="Path to .pt weights")
    parser.add_argument("--imgsz", type=int, default=1280, help="Export image size")
    parser.add_argument("--half", action="store_true", help="Export FP16 ONNX")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    print(f"Loading model from {weights_path}")
    model = YOLO(str(weights_path))

    print(f"Exporting to ONNX (imgsz={args.imgsz}, half={args.half})...")
    export_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        half=args.half,
        opset=17,
        simplify=True,
    )

    print(f"\nExported to: {export_path}")
    onnx_size = Path(export_path).stat().st_size / (1024 * 1024)
    print(f"ONNX size: {onnx_size:.1f} MB")

    if onnx_size > 420:
        print("WARNING: ONNX file exceeds 420 MB limit!")
        print("Try exporting with --half for FP16")


if __name__ == "__main__":
    main()
