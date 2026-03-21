---
name: train
description: Train YOLO detector and/or product classifier on GCP VM or locally
user_invocable: true
---

# Training Workflow

Guide the user through training. There are two models to train:

## 1. YOLO Detector (primary — 70% of score)

**Data preparation** (run once per dataset change):
```bash
cd training/
python prepare_dataset.py              # COCO → YOLO format (90/10 split)
python prepare_tiled_dataset.py --tile-size 1280 --overlap 0.25  # tiled dataset
```

**Training command**:
```bash
python train.py --model yolov8x --epochs 300 --imgsz 1280 --batch 8 --device 0
```
- Uses `dataset_tiled/dataset.yaml` by default (tiled + full images)
- Automatically exports FP16 weights after training
- Key augmentations: mosaic, mixup, copy_paste (configured in train.py)
- Pin `ultralytics==8.1.0` — other versions break weight loading in sandbox

**Resume training**: `python train.py --resume`
**Fine-tune from weights**: `python train.py --weights runs/.../best.pt --epochs 50`

## 2. Product Classifier (secondary — 30% of score)

**Data preparation** (requires annotations + product reference images):
```bash
python prepare_crops.py  # Extract crops from annotations + copy reference images
```

**Training command**:
```bash
python train_classifier.py --epochs 50 --batch 64 --img_size 256 --device 0
```
- EfficientNet-V2-S with ImageNet pretrained backbone
- Differential LR: backbone gets 100x lower LR
- Class-balanced sampling + weighted loss for long-tail distribution
- Exports FP16 weights automatically

## GCP VM Workflow
```bash
make sync          # Upload code to VM
make sync-data     # Upload coco/ and product/ data (first time only)
make ssh           # SSH into VM
make train-yolo    # Train YOLO on VM
make train-classifier  # Train classifier on VM
make pull-weights  # Download trained weights
```

## Important Notes
- YOLO nc=357 (categories 0-356, where 356=unknown_product)
- Train/val split uses seed=42, 90/10 ratio — consistent across all scripts
- Check `training/runs/` for training outputs, metrics, and weights
- FP16 weights must fit under 420 MB total for submission
