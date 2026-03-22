"""Build submission zip for multi-class YOLO approach."""
import argparse
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo-onnx", default="runs/detect/runs/exp_11x_1280/weights/best.onnx")
    parser.add_argument("--output", default="submission.zip")
    args = parser.parse_args()

    onnx_path = Path(args.yolo_onnx)
    run_py = Path(__file__).parent / "run.py"

    files = {
        "run.py": run_py,
        "yolo.onnx": onnx_path,
    }

    total_size = 0
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as zf:
        for arcname, filepath in files.items():
            if not filepath.exists():
                print(f"ERROR: {filepath} not found!")
                return
            size_mb = filepath.stat().st_size / (1024 * 1024)
            total_size += size_mb
            print(f"  Adding {arcname}: {size_mb:.1f} MB")
            zf.write(filepath, arcname)

    print(f"\nTotal: {total_size:.1f} MB (limit: 420 MB)")
    print(f"Files: {len(files)} (limit: 1000)")
    print(f"Weight files: 1 (limit: 3)")
    print(f"Zip created: {args.output}")


if __name__ == "__main__":
    main()
