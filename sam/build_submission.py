"""
Build submission zip for the NorgesGruppen competition.

Run after exporting models:
    python build_submission.py

Creates: submission.zip with run.py + weight files at the root.
"""

import json
import zipfile
from pathlib import Path


ALLOWED_EXTENSIONS = {".py", ".json", ".yaml", ".yml", ".cfg",
                      ".pt", ".pth", ".onnx", ".safetensors", ".npy"}
WEIGHT_EXTENSIONS = {".pt", ".pth", ".onnx", ".safetensors", ".npy"}
MAX_ZIP_SIZE = 420 * 1024 * 1024  # 420 MB
MAX_WEIGHT_FILES = 3
MAX_PYTHON_FILES = 10
MAX_TOTAL_FILES = 1000


def build_zip(output_path="submission.zip", weights_dir="weights"):
    script_dir = Path(__file__).parent
    weights_path = script_dir / weights_dir

    files_to_include = []

    # 1. run.py (required)
    run_py = script_dir / "run.py"
    if not run_py.exists():
        print("ERROR: run.py not found!")
        return
    files_to_include.append(("run.py", run_py))

    # 2. Weight files from weights directory
    if weights_path.exists():
        for f in sorted(weights_path.iterdir()):
            if f.suffix.lower() in WEIGHT_EXTENSIONS:
                files_to_include.append((f.name, f))

    # Also check for weight files next to run.py
    for f in sorted(script_dir.iterdir()):
        if f.suffix.lower() in WEIGHT_EXTENSIONS and f.name not in [n for n, _ in files_to_include]:
            files_to_include.append((f.name, f))

    # 3. Optional config files
    for name in ["categories.json", "config.json"]:
        cfg = script_dir / name
        if cfg.exists():
            files_to_include.append((name, cfg))

    # Validation
    weight_files = [n for n, p in files_to_include if Path(n).suffix.lower() in WEIGHT_EXTENSIONS]
    py_files = [n for n, p in files_to_include if Path(n).suffix.lower() == ".py"]

    print("=== Submission Contents ===")
    total_size = 0
    for name, path in files_to_include:
        size = path.stat().st_size
        total_size += size
        marker = " [WEIGHT]" if Path(name).suffix.lower() in WEIGHT_EXTENSIONS else ""
        print(f"  {name}: {size / 1e6:.1f} MB{marker}")

    print(f"\nTotal: {total_size / 1e6:.1f} MB")
    print(f"Weight files: {len(weight_files)}/{MAX_WEIGHT_FILES}")
    print(f"Python files: {len(py_files)}/{MAX_PYTHON_FILES}")

    # Check limits
    errors = []
    if len(weight_files) > MAX_WEIGHT_FILES:
        errors.append(f"Too many weight files: {len(weight_files)} > {MAX_WEIGHT_FILES}")
    if len(py_files) > MAX_PYTHON_FILES:
        errors.append(f"Too many Python files: {len(py_files)} > {MAX_PYTHON_FILES}")
    if total_size > MAX_ZIP_SIZE:
        errors.append(f"Total size {total_size/1e6:.0f}MB exceeds {MAX_ZIP_SIZE/1e6:.0f}MB limit")
    if len(files_to_include) > MAX_TOTAL_FILES:
        errors.append(f"Too many files: {len(files_to_include)} > {MAX_TOTAL_FILES}")

    for name, path in files_to_include:
        if Path(name).suffix.lower() not in ALLOWED_EXTENSIONS:
            errors.append(f"Disallowed file type: {name}")

    if errors:
        print("\nERRORS:")
        for e in errors:
            print(f"  - {e}")
        return

    # Check for banned imports in Python files
    banned_imports = [
        "import os", "import sys", "import subprocess", "import socket",
        "import ctypes", "import pickle", "import marshal", "import shelve",
        "import shutil", "import yaml", "import requests", "import urllib",
        "import multiprocessing", "import threading", "import signal",
        "import gc", "import code", "import codeop", "import pty",
        "from os ", "from sys ", "from subprocess ",
    ]
    for name, path in files_to_include:
        if not name.endswith(".py"):
            continue
        content = path.read_text()
        for banned in banned_imports:
            if banned in content:
                print(f"\nWARNING: '{banned}' found in {name} — will be blocked by sandbox!")

    # Create zip
    out = Path(output_path)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, path in files_to_include:
            zf.write(path, name)

    zip_size = out.stat().st_size
    print(f"\nCreated: {out} ({zip_size / 1e6:.1f} MB compressed)")
    print("Ready to upload at https://app.ainm.no/submit/norgesgruppen-data")


if __name__ == "__main__":
    build_zip()
