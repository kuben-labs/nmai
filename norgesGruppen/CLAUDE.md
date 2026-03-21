# NorgesGruppen Shelf Product Detection — NM i AI Competition

## Competition Goal
Detect and classify grocery products on store shelf images. Score = **0.7 * detection_mAP@0.5 + 0.3 * classification_mAP@0.5**. Detection-only (all category_id=0) caps at 70%.

## Project Layout
```
norgesGruppen/
├── coco/train/                  # Ground truth data
│   ├── annotations.json         # COCO format: 248 images, ~22700 annotations, 357 categories (0-356)
│   └── images/                  # Shelf images from Norwegian grocery stores
├── product/                     # Reference product images by barcode
│   └── {barcode}/{main,front,back,left,right,top,bottom}.jpg
│   └── metadata.json            # Product names and annotation counts
├── training/                    # YOLO+classifier approach (primary)
│   ├── run.py                   # Submission entry point: 2-stage (YOLOv8x + EfficientNet-V2-S)
│   ├── train.py                 # YOLOv8x training (ultralytics==8.1.0)
│   ├── train_classifier.py      # EfficientNet-V2-S product classifier
│   ├── prepare_dataset.py       # COCO → YOLO format (90/10 split, seed=42)
│   ├── prepare_tiled_dataset.py # Tiled + full-image dataset for high-res training
│   ├── prepare_crops.py         # Extract crops + reference images for classifier
│   ├── evaluate.py              # Local scoring (mirrors competition formula)
│   ├── package.py               # Build submission.zip (FP16 export)
│   ├── Makefile                 # Workflow commands (GCP sync, train, evaluate, package)
│   └── runs/                    # Training outputs (weights, logs)
├── samapproach/                 # Alternative: SAM3 + DINOv2 pipeline (experimental)
│   ├── run_pipeline.py          # 4-step pipeline orchestrator
│   ├── step1_segment.py         # SAM3 text-prompted segmentation
│   ├── step2_squares.py         # Masks → bounding squares
│   ├── step3_embed_products.py  # DINOv2 product embeddings
│   ├── step4_match.py           # Cosine similarity matching
│   └── config.py                # Pipeline configuration
```

## Submission Sandbox Constraints (CRITICAL)
All run.py code executes in a locked-down Docker container. Violating these causes instant failure.

### Blocked imports (security scanner rejects the zip)
`os`, `sys`, `subprocess`, `socket`, `ctypes`, `builtins`, `importlib`, `pickle`, `marshal`, `shelve`, `shutil`, `yaml`, `requests`, `urllib`, `http.client`, `multiprocessing`, `threading`, `signal`, `gc`, `code`, `codeop`, `pty`

### Blocked calls
`eval()`, `exec()`, `compile()`, `__import__()`, `getattr()` with dangerous names

### Use instead
- `pathlib` instead of `os` for file operations
- `json` instead of `yaml` for config files

### Resource limits
| Resource | Limit |
|----------|-------|
| Python | 3.11 |
| GPU | NVIDIA L4, 24 GB VRAM, CUDA 12.4 |
| CPU | 4 vCPU |
| Memory | 8 GB |
| Timeout | 300 seconds |
| Network | None (fully offline) |

### Zip limits
| Limit | Value |
|-------|-------|
| Max uncompressed size | 420 MB |
| Max files | 1000 |
| Max Python files | 10 |
| Max weight files (.pt/.onnx/.safetensors/.npy) | 3 |
| Max weight size total | 420 MB |
| Allowed types | .py, .json, .yaml, .yml, .cfg, .pt, .pth, .onnx, .safetensors, .npy |

### Pre-installed packages (exact versions)
PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124, ultralytics 8.1.0, onnxruntime-gpu 1.20.0, opencv-python-headless 4.9.0.80, albumentations 1.3.1, Pillow 10.2.0, numpy 1.26.4, scipy 1.12.0, scikit-learn 1.4.0, pycocotools 2.0.7, ensemble-boxes 1.0.9, timm 0.9.12, supervision 0.18.0, safetensors 0.4.2

### Version pinning (MUST match for .pt weights)
- `ultralytics==8.1.0` (8.2+ changes model class, weights won't load)
- `torchvision==0.21.0` (for torchvision detectors)
- `timm==0.9.12` (1.0+ changes layer names)
- ONNX opset ≤ 20 (onnxruntime 1.20.0 limit)

## Submission Format
```
submission.zip
├── run.py              # Required entry point
├── best.pt             # YOLO weights (FP16)
└── best_classifier.pt  # Classifier weights (FP16, optional)
```

run.py contract: `python run.py --input /data/images --output /output/predictions.json`

Output JSON array:
```json
[{"image_id": 42, "category_id": 0, "bbox": [x, y, w, h], "score": 0.923}]
```
- `image_id`: numeric from filename (img_00042.jpg → 42)
- `bbox`: COCO format [x, y, width, height] in pixels
- `category_id`: 0-355 product categories, 356 = unknown_product
- `score`: confidence 0-1

## Submission Limits
- 3 submissions per day per team (resets midnight UTC)
- 2 in-flight at once
- 2 infrastructure-failure freebies per day

## Scoring Details
- detection mAP@0.5: IoU ≥ 0.5, category ignored (70% weight)
- classification mAP@0.5: IoU ≥ 0.5 AND correct category_id (30% weight)
- Public leaderboard uses public test set; final ranking uses private test set

## Current Architecture (training/run.py)
Two-stage pipeline:
1. **YOLOv8x detection** with sliced inference (full image + overlapping 1280px tiles), merged via Weighted Box Fusion
2. **EfficientNet-V2-S classifier** refines product identity on cropped detections

Key parameters: YOLO_CONF=0.10 (low for recall), tile overlap 0.25, WBF merge, class-agnostic NMS, classifier override threshold 0.40, 280s time budget.

## GCP Training VM
Project: `ai-nm26osl-1759`, Zone: `us-central1-b`, VM: `yolo-train`. Use Makefile targets for sync/ssh/pull.

## When editing run.py
- NEVER use blocked imports — the zip will be rejected
- Use `pathlib.Path` for all file operations
- Use `json` not `yaml`
- Keep total inference under 280s (20s buffer from 300s limit)
- GPU is always available — use `torch.cuda.is_available()` for portability
- Weights must be next to run.py at zip root
- PyTorch 2.6 changed `weights_only` default — patch `torch.load` if needed
