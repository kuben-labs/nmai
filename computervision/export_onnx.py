"""Export YOLO11 multi-class model to ONNX for sandbox submission."""
import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="runs/detect/runs/exp_11x_1280/weights/best.pt")
    parser.add_argument("--imgsz", type=int, default=1280)
    args = parser.parse_args()

    weights = Path(args.weights)
    model = YOLO(str(weights))
    exported = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=17,
        simplify=True,
        half=True,
    )
    print(f"Exported: {exported}")

    # Check size
    onnx_path = weights.with_suffix(".onnx")
    if onnx_path.exists():
        size_mb = onnx_path.stat().st_size / (1024 * 1024)
        print(f"Size: {size_mb:.1f} MB (limit: 420 MB)")


if __name__ == "__main__":
    main()
