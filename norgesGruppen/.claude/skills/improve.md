---
name: improve
description: Suggest and implement improvements to boost competition score
user_invocable: true
---

# Improvement Analysis

Analyze the current approach and suggest targeted improvements. Read the current `training/run.py`, `training/train.py`, and `training/train_classifier.py` first.

## Score Breakdown
Score = 0.7 * detection_mAP + 0.3 * classification_mAP

**Detection improvements (70% impact)** have highest ROI:
- Better YOLO training (augmentation, hyperparams, epochs)
- Tiled inference tuning (tile size, overlap, WBF parameters)
- Test-time augmentation (TTA)
- Confidence threshold tuning
- Larger models (YOLOv8x vs l vs m) if time budget allows
- Multi-scale inference

**Classification improvements (30% impact)**:
- Better classifier architecture or larger backbone
- More training data (reference images, harder augmentations)
- Ensemble multiple classifiers
- Adjust override threshold

## Constraints to Keep in Mind
- 300s total timeout (currently using 280s budget)
- 420 MB max weight size (FP16)
- 8 GB RAM, 24 GB VRAM (L4 GPU)
- Only pre-installed packages available at runtime
- Max 3 weight files in submission
- 3 submissions/day — test locally first with `evaluate.py --val-only`

## Analysis Steps
1. Read current run.py architecture and parameters
2. Check if latest local evaluation scores exist
3. Identify the bottleneck (detection vs classification)
4. Propose specific, implementable changes ranked by expected impact
5. For each change, note if it affects training or inference (or both)

## Common High-Impact Changes
- **Increase YOLO training epochs** if not converged
- **Add more augmentation** (mosaic, copy-paste, multi-scale)
- **Tune NMS/WBF thresholds** for the specific dataset
- **Lower confidence threshold** for recall (detection is 70% of score)
- **Multi-resolution training** (random imgsz during training)
- **Pseudo-labeling** on unlabeled images
- **Test-time augmentation** (horizontal flip, multi-scale)

Always propose changes that respect sandbox constraints and can be validated locally before using a submission slot.
