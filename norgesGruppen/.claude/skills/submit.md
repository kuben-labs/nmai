---
name: submit
description: Package and verify a submission zip for upload
user_invocable: true
---

# Package Submission

## Quick Package (uses package.py)
```bash
cd training/
python package.py \
    --yolo runs/yolov8x_ngd/weights/best.pt \
    --classifier runs/classifier/best_classifier_fp16.pt
```
This creates `submission.zip` with run.py + FP16 weights at the zip root.

## Verify Before Upload

### 1. Check zip structure (run.py MUST be at root, not in subfolder)
```bash
unzip -l submission.zip | head -20
```
Expected: `run.py`, `best.pt`, optionally `best_classifier.pt` — all at root level.

### 2. Check file limits
- Max 1000 files, max 10 .py files, max 3 weight files
- Max 420 MB uncompressed, max 420 MB total weights
- Allowed types: .py, .json, .yaml, .yml, .cfg, .pt, .pth, .onnx, .safetensors, .npy

### 3. Check for blocked imports in run.py
Scan run.py for these blocked modules — the security scanner will reject the zip:
```
os, sys, subprocess, socket, ctypes, builtins, importlib, pickle, marshal,
shelve, shutil, yaml, requests, urllib, http.client, multiprocessing,
threading, signal, gc, code, codeop, pty
```
Also blocked: `eval()`, `exec()`, `compile()`, `__import__()`, `getattr()` with dangerous names.

### 4. Local smoke test
```bash
python run.py --input ../coco/train/images --output test_predictions.json
python evaluate.py --predictions test_predictions.json --val-only
```

## Manual Zip Creation (if not using package.py)
```bash
cd my_submission_dir/
zip -r ../submission.zip . -x ".*" "__MACOSX/*"
```
NEVER zip the folder itself — zip the contents. `run.py` must be at root.

## Common Submission Errors
| Error | Fix |
|-------|-----|
| run.py not found | Zip contents, not folder |
| __MACOSX files | Use terminal zip with `-x "__MACOSX/*"` |
| .bin file type | Rename .bin → .pt (same format) |
| Security violations | Remove blocked imports, use pathlib/json |
| No predictions.json | Ensure run.py writes to `--output` path |
| Timeout (300s) | Use GPU, reduce model size, check time budget |
| Exit 137 (OOM) | Reduce batch size, use FP16 |
| Exit 139 (segfault) | Version mismatch — re-export or use ONNX |
| ModuleNotFoundError | Package not in sandbox — export to ONNX |

## Submission Limits
- 3 per day per team (resets midnight UTC)
- 2 in-flight at once
- Upload at the competition submit page
