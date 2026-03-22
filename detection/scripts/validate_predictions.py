"""Validate predictions JSON format before submission.

Usage:
    python scripts/validate_predictions.py --predictions output/predictions.json
    python scripts/validate_predictions.py --predictions output/predictions.json \
        --annotations data/raw/train/annotations.json
"""

import argparse
import json
from pathlib import Path


def validate_predictions(
    predictions_path: str, annotations_path: str | None = None
) -> bool:
    """Validate predictions format and optionally compute basic stats against GT."""
    path = Path(predictions_path)
    if not path.exists():
        print(f"FAIL: File not found: {path}")
        return False

    with open(path) as f:
        predictions = json.load(f)

    if not isinstance(predictions, list):
        print(f"FAIL: Expected JSON array, got {type(predictions).__name__}")
        return False

    if len(predictions) == 0:
        print("WARNING: Empty predictions list")
        return True

    print(f"Total predictions: {len(predictions)}")

    errors = []
    required_fields = {"image_id", "category_id", "bbox", "score"}

    image_ids = set()
    category_ids = set()
    scores = []

    for i, pred in enumerate(predictions):
        # Check required fields
        missing = required_fields - set(pred.keys())
        if missing:
            errors.append(f"  Prediction {i}: missing fields {missing}")
            continue

        # Check types
        if not isinstance(pred["image_id"], int):
            errors.append(
                f"  Prediction {i}: image_id must be int, got {type(pred['image_id'])}"
            )

        if not isinstance(pred["category_id"], int):
            errors.append(
                f"  Prediction {i}: category_id must be int, got {type(pred['category_id'])}"
            )

        if not isinstance(pred["bbox"], list) or len(pred["bbox"]) != 4:
            errors.append(f"  Prediction {i}: bbox must be [x, y, w, h] list")
        else:
            x, y, w, h = pred["bbox"]
            if w <= 0 or h <= 0:
                errors.append(f"  Prediction {i}: bbox has non-positive width/height")

        if not isinstance(pred["score"], (int, float)):
            errors.append(f"  Prediction {i}: score must be float")
        elif not (0.0 <= pred["score"] <= 1.0):
            errors.append(f"  Prediction {i}: score {pred['score']} not in [0, 1]")

        image_ids.add(pred["image_id"])
        category_ids.add(pred["category_id"])
        scores.append(pred["score"])

    if errors:
        print(f"\nFOUND {len(errors)} ERRORS:")
        for err in errors[:20]:
            print(err)
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")
        return False

    # Stats
    print(f"Unique images: {len(image_ids)}")
    print(f"Unique categories: {len(category_ids)}")
    print(f"Category ID range: {min(category_ids)} - {max(category_ids)}")
    print(f"Score range: {min(scores):.4f} - {max(scores):.4f}")
    print(f"Avg predictions per image: {len(predictions) / len(image_ids):.1f}")

    # Detection-only check
    if category_ids == {0}:
        print(
            "\nNOTE: All predictions use category_id=0 (detection-only, max score 0.70)"
        )
    elif 0 in category_ids:
        count_zero = sum(1 for p in predictions if p["category_id"] == 0)
        print(f"\nNOTE: {count_zero} predictions use category_id=0 (detection-only)")

    # Compare with ground truth if available
    if annotations_path:
        ann_path = Path(annotations_path)
        if ann_path.exists():
            with open(ann_path) as f:
                gt_data = json.load(f)
            gt_image_ids = {img["id"] for img in gt_data["images"]}
            gt_categories = {cat["id"] for cat in gt_data["categories"]}

            matched_images = image_ids & gt_image_ids
            print(f"\nGT comparison:")
            print(f"  GT images: {len(gt_image_ids)}")
            print(
                f"  Predicted images matching GT: {len(matched_images)}/{len(image_ids)}"
            )
            print(f"  GT categories: {len(gt_categories)}")
            print(
                f"  Predicted categories in GT: {len(category_ids & gt_categories)}/{len(category_ids)}"
            )

    print("\nVALIDATION PASSED")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate predictions JSON format")
    parser.add_argument("--predictions", required=True, help="Path to predictions JSON")
    parser.add_argument(
        "--annotations", default=None, help="Path to GT annotations (optional)"
    )
    args = parser.parse_args()

    success = validate_predictions(args.predictions, args.annotations)
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
