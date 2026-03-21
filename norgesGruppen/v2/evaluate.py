"""
Evaluate predictions against COCO ground truth using the competition scoring.

Score = 0.7 * detection_mAP + 0.3 * classification_mAP

  detection mAP:       IoU >= 0.5, category ignored
  classification mAP:  IoU >= 0.5 AND correct category_id

Usage:
    python evaluate.py --predictions predictions.json
    python evaluate.py --predictions predictions.json --annotations ../coco/train/annotations.json
"""

import argparse
import copy
import json
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


COCO_ANN = Path(__file__).parent.parent / "coco" / "train" / "annotations.json"

# Validation image IDs (same split as prepare_dataset.py with SEED=42)
# We compute these at runtime to stay in sync with the training split.
VAL_RATIO = 0.1
SEED = 42


def get_val_image_ids(coco_gt):
    """Reproduce the same 90/10 split as prepare_dataset.py."""
    import random
    all_ids = sorted(coco_gt.getImgIds())
    rng = random.Random(SEED)
    rng.shuffle(all_ids)
    n_val = max(1, int(len(all_ids) * VAL_RATIO))
    return set(all_ids[:n_val])


def compute_map50(coco_gt, predictions, class_agnostic=False):
    """
    Compute mAP@IoU=0.5 using pycocotools.

    If class_agnostic=True, all categories are collapsed to a single class
    so only localization matters (detection score).
    """
    if not predictions:
        return 0.0

    if class_agnostic:
        # Collapse all categories to class 1
        gt_dataset = copy.deepcopy(coco_gt.dataset)
        for ann in gt_dataset["annotations"]:
            ann["category_id"] = 1
        gt_dataset["categories"] = [
            {"id": 1, "name": "product", "supercategory": "product"}
        ]
        coco_gt_mod = COCO()
        coco_gt_mod.dataset = gt_dataset
        coco_gt_mod.createIndex()

        preds_mod = copy.deepcopy(predictions)
        for p in preds_mod:
            p["category_id"] = 1
    else:
        coco_gt_mod = coco_gt
        preds_mod = predictions

    coco_dt = coco_gt_mod.loadRes(preds_mod)
    coco_eval = COCOeval(coco_gt_mod, coco_dt, "bbox")
    coco_eval.params.iouThrs = [0.5]  # only IoU=0.5
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # stats[0] = AP @ IoU=0.5 (since we only set that threshold)
    return float(coco_eval.stats[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True,
                   help="Path to predictions.json")
    p.add_argument("--annotations", default=str(COCO_ANN),
                   help="Path to COCO annotations.json")
    p.add_argument("--val-only", action="store_true",
                   help="Evaluate only on val split images (default: all)")
    args = p.parse_args()

    print(f"Loading annotations: {args.annotations}")
    coco_gt = COCO(args.annotations)

    with open(args.predictions) as f:
        predictions = json.load(f)

    print(f"Loaded {len(predictions)} predictions")

    # Optionally filter to val images only
    if args.val_only:
        val_ids = get_val_image_ids(coco_gt)
        predictions = [p for p in predictions if p["image_id"] in val_ids]

        # Also filter GT to val images
        gt_dataset = copy.deepcopy(coco_gt.dataset)
        gt_dataset["images"] = [
            img for img in gt_dataset["images"] if img["id"] in val_ids
        ]
        gt_dataset["annotations"] = [
            ann for ann in gt_dataset["annotations"]
            if ann["image_id"] in val_ids
        ]
        coco_gt = COCO()
        coco_gt.dataset = gt_dataset
        coco_gt.createIndex()

        print(f"Filtered to {len(val_ids)} val images, "
              f"{len(predictions)} predictions")

    if not predictions:
        print("No predictions to evaluate!")
        return

    # Ensure predictions have required fields
    pred_img_ids = set(p["image_id"] for p in predictions)
    gt_img_ids = set(coco_gt.getImgIds())
    matched = pred_img_ids & gt_img_ids
    print(f"Images: {len(matched)} matched "
          f"({len(pred_img_ids)} predicted, {len(gt_img_ids)} ground truth)")

    print("\n" + "=" * 60)
    print("DETECTION mAP (class-agnostic, IoU >= 0.5)")
    print("=" * 60)
    det_map = compute_map50(coco_gt, predictions, class_agnostic=True)

    print("\n" + "=" * 60)
    print("CLASSIFICATION mAP (per-class, IoU >= 0.5)")
    print("=" * 60)
    cls_map = compute_map50(coco_gt, predictions, class_agnostic=False)

    combined = 0.7 * det_map + 0.3 * cls_map
    print("\n" + "=" * 60)
    print(f"Detection mAP@50:       {det_map:.4f}  (× 0.7 = {0.7 * det_map:.4f})")
    print(f"Classification mAP@50:  {cls_map:.4f}  (× 0.3 = {0.3 * cls_map:.4f})")
    print(f"Combined score:         {combined:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
