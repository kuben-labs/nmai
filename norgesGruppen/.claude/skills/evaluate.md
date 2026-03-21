---
name: evaluate
description: Run local evaluation to estimate competition score
user_invocable: true
---

# Local Evaluation

## Quick Evaluate (val set only)
```bash
cd training/
python run.py --input ../coco/train/images --output predictions.json
python evaluate.py --predictions predictions.json --val-only
```

## Full Evaluate (all training images)
```bash
python run.py --input ../coco/train/images --output predictions.json
python evaluate.py --predictions predictions.json
```

## Score Interpretation
Output shows three numbers:
```
Detection mAP@50:       X.XXXX  (× 0.7 = Y.YYYY)    # Did you find products?
Classification mAP@50:  X.XXXX  (× 0.3 = Y.YYYY)    # Did you identify them?
Combined score:         X.XXXX                         # This is your competition score
```

- **Detection mAP**: IoU ≥ 0.5, category ignored. Measures bounding box quality.
- **Classification mAP**: IoU ≥ 0.5 AND correct category_id. Measures product identification.
- **Combined**: 0.7 * detection + 0.3 * classification

## Notes
- Val split is 10% of training data (seed=42) — same split used by all prepare scripts
- Local scores on training data will be higher than competition scores on unseen test images
- Detection-only (category_id=0 for all) gives max 0.70 combined score
- The `--val-only` flag evaluates on held-out images not seen during training — more realistic
