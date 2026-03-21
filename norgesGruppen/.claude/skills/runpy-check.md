---
name: runpy-check
description: Audit run.py for sandbox compatibility before submission
user_invocable: true
---

# Audit run.py for Sandbox Compatibility

Read `training/run.py` and check for all of the following issues:

## 1. Blocked Imports
Scan for any import of: `os`, `sys`, `subprocess`, `socket`, `ctypes`, `builtins`, `importlib`, `pickle`, `marshal`, `shelve`, `shutil`, `yaml`, `requests`, `urllib`, `http.client`, `multiprocessing`, `threading`, `signal`, `gc`, `code`, `codeop`, `pty`

**Fix**: Use `pathlib` instead of `os`, `json` instead of `yaml`.

## 2. Blocked Calls
Scan for: `eval()`, `exec()`, `compile()`, `__import__()`, `getattr()` with dangerous attribute names.

## 3. Resource Constraints
- **Timeout**: Total inference must complete in <300s (use 280s budget). Check for time management logic.
- **Memory**: 8 GB RAM limit. Check batch sizes, ensure no unnecessary tensor accumulation.
- **GPU**: NVIDIA L4, 24 GB VRAM. Verify GPU is used via `torch.cuda.is_available()`.

## 4. Weight File References
- All weight files must be referenced relative to `run.py` location (use `Path(__file__).parent`)
- Verify the weight filenames match what `package.py` puts in the zip (`best.pt`, `best_classifier.pt`)

## 5. Output Format
Verify predictions JSON matches:
```json
[{"image_id": int, "category_id": int, "bbox": [x, y, w, h], "score": float}]
```
- `image_id` extracted from filename: `img_00042.jpg` → `42`
- `bbox` in COCO format: `[x, y, width, height]` (NOT xyxy)
- `category_id` range: 0-355 (or 0 for detection-only)
- `score` range: 0.0-1.0

## 6. Package Versions
- `ultralytics` usage must be compatible with 8.1.0
- `torchvision` models must be compatible with 0.21.0
- `timm` backbones must be compatible with 0.9.12
- PyTorch 2.6 `weights_only` default change — patch `torch.load` if loading .pt files

## 7. File I/O
- Only use `pathlib.Path` for file operations
- Write output to exact `--output` path
- Create parent dirs with `mkdir(parents=True, exist_ok=True)`

Report all issues found with specific line numbers and fixes.
