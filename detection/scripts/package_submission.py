"""Package a trained model into a submission zip.

Usage:
    # ONNX submission (recommended)
    python scripts/package_submission.py --weights runs/full2/weights/best.onnx

    # .pt submission (may fail on sandbox due to torch.load issue)
    python scripts/package_submission.py --weights runs/full2/weights/best.pt

This script:
1. Copies run.py and the weights file into submission/
2. Creates submission.zip with run.py at the root
3. Validates the zip structure and limits
"""

import argparse
import zipfile
from pathlib import Path


def copy_file(src: Path, dst: Path) -> None:
    """Copy a file without using shutil (blocked in sandbox)."""
    dst.write_bytes(src.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser(description="Package submission zip")
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to model weights (.onnx or .pt)",
    )
    parser.add_argument(
        "--run-script",
        type=str,
        default="scripts/run.py",
        help="Path to run.py inference script",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="submission.zip",
        help="Output zip file path",
    )
    args = parser.parse_args()

    weights_path = Path(args.weights)
    run_script_path = Path(args.run_script)
    output_path = Path(args.output)

    if not weights_path.exists():
        print(f"ERROR: Weights not found: {weights_path}")
        raise SystemExit(1)
    if not run_script_path.exists():
        print(f"ERROR: run.py not found: {run_script_path}")
        raise SystemExit(1)

    weights_size_mb = weights_path.stat().st_size / (1024 * 1024)
    print(f"Weights: {weights_path} ({weights_size_mb:.1f} MB)")

    if weights_size_mb > 420:
        print(f"WARNING: Weights exceed 420 MB limit ({weights_size_mb:.1f} MB)")

    # Determine the destination filename based on what run.py expects
    weights_suffix = weights_path.suffix
    if weights_suffix == ".onnx":
        dst_weights_name = "best.onnx"
    else:
        dst_weights_name = "best.pt"

    # Create submission directory
    submission_dir = Path("submission")
    submission_dir.mkdir(exist_ok=True)

    dst_run = submission_dir / "run.py"
    dst_weights = submission_dir / dst_weights_name

    copy_file(run_script_path, dst_run)
    copy_file(weights_path, dst_weights)
    print(f"Copied {run_script_path} -> {dst_run}")
    print(f"Copied {weights_path} -> {dst_weights}")

    # Remove old files from submission dir that shouldn't be there
    for f in submission_dir.iterdir():
        if f.name not in ("run.py", dst_weights_name):
            f.unlink()
            print(f"Removed stale file: {f.name}")

    # Create zip
    print(f"\nCreating {output_path}...")
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(submission_dir.iterdir()):
            if file_path.name.startswith(".") or file_path.name == "__MACOSX":
                continue
            zf.write(file_path, file_path.name)
            print(f"  Added: {file_path.name}")

    # Validate zip
    zip_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nZip size: {zip_size_mb:.1f} MB (compressed)")

    with zipfile.ZipFile(output_path, "r") as zf:
        names = zf.namelist()
        total_uncompressed = sum(info.file_size for info in zf.infolist())
        total_uncompressed_mb = total_uncompressed / (1024 * 1024)

    print(f"Uncompressed size: {total_uncompressed_mb:.1f} MB (limit: 420 MB)")
    print(f"Files in zip: {names}")

    if "run.py" not in names:
        print("ERROR: run.py is NOT at zip root!")
        raise SystemExit(1)

    if total_uncompressed_mb > 420:
        print(f"ERROR: Uncompressed size exceeds 420 MB limit!")
        raise SystemExit(1)

    weight_extensions = {".pt", ".pth", ".onnx", ".safetensors", ".npy"}
    weight_files = [n for n in names if Path(n).suffix in weight_extensions]
    py_files = [n for n in names if n.endswith(".py")]

    if len(weight_files) > 3:
        print(f"ERROR: {len(weight_files)} weight files (limit: 3)")
        raise SystemExit(1)
    if len(py_files) > 10:
        print(f"ERROR: {len(py_files)} Python files (limit: 10)")
        raise SystemExit(1)

    print(f"\nSubmission ready: {output_path}")
    print(f"  Python files: {len(py_files)}")
    print(f"  Weight files: {len(weight_files)}")


if __name__ == "__main__":
    main()
