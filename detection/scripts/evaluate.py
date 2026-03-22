"""Local evaluation script that computes the competition score.

Score = 0.7 * detection_mAP@0.5 + 0.3 * classification_mAP@0.5

No pycocotools dependency -- pure Python implementation.

Usage:
    # Evaluate .pt model (ultralytics)
    python scripts/evaluate.py --model runs/full2/weights/best.pt \
        --images data/yolo/images/val --val-only

    # Evaluate ONNX model WITHOUT TTA
    python scripts/evaluate.py --onnx runs/full2/weights/best.onnx \
        --images data/yolo/images/val --val-only

    # Evaluate ONNX model WITH TTA (horizontal flip)
    python scripts/evaluate.py --onnx runs/full2/weights/best.onnx \
        --images data/yolo/images/val --val-only --tta

    # Evaluate from predictions file
    python scripts/evaluate.py --predictions output/predictions.json --val-only
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


# ── mAP computation (pure Python, no dependencies) ──────────────────────────


def compute_iou(box_a: list[float], box_b: list[float]) -> float:
    """Compute IoU between two COCO-format boxes [x, y, w, h]."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    ax1, ay1, ax2, ay2 = ax, ay, ax + aw, ay + ah
    bx1, by1, bx2, by2 = bx, by, bx + bw, by + bh

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    intersection = (ix2 - ix1) * (iy2 - iy1)
    union = aw * ah + bw * bh - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def compute_ap(precisions: list[float], recalls: list[float]) -> float:
    """Compute Average Precision using the 101-point interpolation (COCO style)."""
    if not precisions:
        return 0.0

    precisions = [0.0] + precisions + [0.0]
    recalls = [0.0] + recalls + [1.0]

    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    ap = 0.0
    for t in range(0, 101):
        recall_threshold = t / 100.0
        precision_at_threshold = 0.0
        for r, p in zip(recalls, precisions):
            if r >= recall_threshold:
                precision_at_threshold = max(precision_at_threshold, p)
        ap += precision_at_threshold

    return ap / 101.0


def compute_map_at_iou(
    predictions: list[dict],
    ground_truths: list[dict],
    iou_threshold: float = 0.5,
    category_agnostic: bool = False,
) -> float:
    """Compute mAP@IoU threshold."""
    gt_by_image: dict[int, list[dict]] = defaultdict(list)
    for gt in ground_truths:
        gt_by_image[gt["image_id"]].append(gt)

    if category_agnostic:
        categories = {0}
    else:
        categories = {gt["category_id"] for gt in ground_truths}

    predictions = sorted(predictions, key=lambda p: p["score"], reverse=True)

    per_category_ap = []

    for cat_id in sorted(categories):
        if category_agnostic:
            cat_preds = predictions
            cat_gt_count = sum(len(gts) for gts in gt_by_image.values())
        else:
            cat_preds = [p for p in predictions if p["category_id"] == cat_id]
            cat_gt_count = sum(1 for gt in ground_truths if gt["category_id"] == cat_id)

        if cat_gt_count == 0:
            continue

        matched_gt: set[tuple[int, int]] = set()
        tp_list = []
        fp_list = []

        for pred in cat_preds:
            img_id = pred["image_id"]
            img_gts = gt_by_image.get(img_id, [])

            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(img_gts):
                if not category_agnostic and gt["category_id"] != cat_id:
                    continue

                iou = compute_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and (img_id, best_gt_idx) not in matched_gt:
                tp_list.append(1)
                fp_list.append(0)
                matched_gt.add((img_id, best_gt_idx))
            else:
                tp_list.append(0)
                fp_list.append(1)

        tp_cumsum = 0
        fp_cumsum = 0
        precisions = []
        recalls = []

        for tp, fp in zip(tp_list, fp_list):
            tp_cumsum += tp
            fp_cumsum += fp
            precision = tp_cumsum / (tp_cumsum + fp_cumsum)
            recall = tp_cumsum / cat_gt_count
            precisions.append(precision)
            recalls.append(recall)

        ap = compute_ap(precisions, recalls)
        per_category_ap.append(ap)

    if not per_category_ap:
        return 0.0

    return sum(per_category_ap) / len(per_category_ap)


def compute_competition_score(
    predictions: list[dict],
    annotations_path: str,
    val_image_ids: set[int] | None = None,
) -> dict:
    """Compute the competition score: 0.7 * det_mAP + 0.3 * cls_mAP."""
    with open(annotations_path) as f:
        coco_data = json.load(f)

    ground_truths = coco_data["annotations"]

    if val_image_ids:
        ground_truths = [gt for gt in ground_truths if gt["image_id"] in val_image_ids]
        predictions = [p for p in predictions if p["image_id"] in val_image_ids]

    if not predictions:
        print("No predictions to evaluate!")
        return {"detection_map": 0.0, "classification_map": 0.0, "score": 0.0}

    print(
        f"Evaluating {len(predictions)} predictions against {len(ground_truths)} ground truths"
    )

    detection_map = compute_map_at_iou(
        predictions, ground_truths, iou_threshold=0.5, category_agnostic=True
    )

    classification_map = compute_map_at_iou(
        predictions, ground_truths, iou_threshold=0.5, category_agnostic=False
    )

    score = 0.7 * detection_map + 0.3 * classification_map

    return {
        "detection_map": detection_map,
        "classification_map": classification_map,
        "score": score,
    }


# ── ONNX inference (same code as run.py) ─────────────────────────────────────


def generate_predictions_onnx(
    onnx_path: str,
    images_dir: str,
    use_tta: bool = False,
) -> list[dict]:
    """Generate predictions using ONNX model, optionally with TTA."""
    import cv2
    import numpy as np
    import onnxruntime as ort

    CONF_THRESH = 0.01
    IOU_THRESH = 0.5
    IMGSZ = 1280
    MAX_DET = 300

    def letterbox(img, new_shape=1280):
        h, w = img.shape[:2]
        scale = min(new_shape / h, new_shape / w)
        nh, nw = int(h * scale), int(w * scale)
        img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        pad_h = new_shape - nh
        pad_w = new_shape - nw
        top = pad_h // 2
        left = pad_w // 2
        img_padded = np.full((new_shape, new_shape, 3), 114, dtype=np.uint8)
        img_padded[top : top + nh, left : left + nw] = img_resized
        return img_padded, scale, left, top

    def nms(boxes, scores, class_ids, iou_threshold=0.5):
        if len(boxes) == 0:
            return []
        max_coord = boxes.max()
        offsets = class_ids.astype(np.float32) * (max_coord + 1)
        boxes_offset = boxes.copy()
        boxes_offset[:, 0] += offsets
        boxes_offset[:, 1] += offsets
        boxes_offset[:, 2] += offsets
        boxes_offset[:, 3] += offsets
        x1 = boxes_offset[:, 0]
        y1 = boxes_offset[:, 1]
        x2 = boxes_offset[:, 2]
        y2 = boxes_offset[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            if len(order) == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        return keep

    def infer_single(session, img_bgr, imgsz, flip=False):
        orig_h, orig_w = img_bgr.shape[:2]
        img = img_bgr if not flip else img_bgr[:, ::-1].copy()

        img_padded, scale, pad_left, pad_top = letterbox(img, imgsz)
        blob = img_padded[:, :, ::-1].astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis]

        input_name = session.get_inputs()[0].name
        input_type = session.get_inputs()[0].type
        if "float16" in input_type:
            blob = blob.astype(np.float16)

        outputs = session.run(None, {input_name: blob})
        preds = outputs[0]

        if preds.shape[1] < preds.shape[2]:
            preds = preds.transpose(0, 2, 1)
        preds = preds[0].astype(np.float32)

        boxes_cxcywh = preds[:, :4]
        class_scores = preds[:, 4:]
        class_ids = class_scores.argmax(axis=1)
        scores = class_scores.max(axis=1)

        mask = scores > CONF_THRESH
        boxes_cxcywh = boxes_cxcywh[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        if len(scores) == 0:
            return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)

        boxes_xyxy = np.empty_like(boxes_cxcywh)
        boxes_xyxy[:, 0] = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
        boxes_xyxy[:, 1] = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
        boxes_xyxy[:, 2] = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
        boxes_xyxy[:, 3] = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2

        boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - pad_left) / scale
        boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - pad_top) / scale

        if flip:
            x1_new = orig_w - boxes_xyxy[:, 2]
            x2_new = orig_w - boxes_xyxy[:, 0]
            boxes_xyxy[:, 0] = x1_new
            boxes_xyxy[:, 2] = x2_new

        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, orig_w)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, orig_h)

        return boxes_xyxy, scores, class_ids

    # Load ONNX session
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_path, providers=providers)
    print(f"ONNX model: {onnx_path}")
    print(f"Providers: {session.get_providers()}")
    print(f"TTA: {'ON (horizontal flip)' if use_tta else 'OFF'}")

    predictions = []
    img_dir = Path(images_dir)
    image_files = sorted(
        p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    for idx, img_path in enumerate(image_files):
        image_id = int(img_path.stem.split("_")[-1])
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        if use_tta:
            # Run original + horizontal flip, merge with NMS
            all_boxes, all_scores, all_class_ids = [], [], []
            for flip in [False, True]:
                boxes, scores, class_ids = infer_single(
                    session, img_bgr, IMGSZ, flip=flip
                )
                if len(scores) > 0:
                    all_boxes.append(boxes)
                    all_scores.append(scores)
                    all_class_ids.append(class_ids)

            if len(all_boxes) == 0:
                continue

            all_boxes = np.concatenate(all_boxes)
            all_scores = np.concatenate(all_scores)
            all_class_ids = np.concatenate(all_class_ids)

            keep = nms(all_boxes, all_scores, all_class_ids, IOU_THRESH)
            if len(keep) > MAX_DET:
                keep = keep[:MAX_DET]

            boxes_xyxy = all_boxes[keep]
            scores = all_scores[keep]
            class_ids = all_class_ids[keep]
        else:
            # Single pass, no TTA
            boxes_xyxy, scores, class_ids = infer_single(
                session, img_bgr, IMGSZ, flip=False
            )
            if len(scores) > 0:
                keep = nms(boxes_xyxy, scores, class_ids, IOU_THRESH)
                if len(keep) > MAX_DET:
                    keep = keep[:MAX_DET]
                boxes_xyxy = boxes_xyxy[keep]
                scores = scores[keep]
                class_ids = class_ids[keep]

        for box, score, cat_id in zip(boxes_xyxy, scores, class_ids):
            x1, y1, x2, y2 = box
            predictions.append(
                {
                    "image_id": image_id,
                    "category_id": int(cat_id),
                    "bbox": [
                        round(float(x1), 1),
                        round(float(y1), 1),
                        round(float(x2 - x1), 1),
                        round(float(y2 - y1), 1),
                    ],
                    "score": round(float(score), 4),
                }
            )

        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(image_files)} images")

    print(f"Generated {len(predictions)} predictions")
    return predictions


# ── .pt inference (ultralytics) ──────────────────────────────────────────────


def generate_predictions_pt(model_path: str, images_dir: str) -> list[dict]:
    """Generate predictions using ultralytics .pt model."""
    import torch

    _orig_load = torch.load

    def _safe_load(*a, **kw):
        if "weights_only" not in kw:
            kw["weights_only"] = False
        return _orig_load(*a, **kw)

    torch.load = _safe_load

    from ultralytics import YOLO

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path)
    predictions = []

    img_dir = Path(images_dir)
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        image_id = int(img_path.stem.split("_")[-1])
        results = model(
            str(img_path), device=device, verbose=False, imgsz=1280, conf=0.01
        )
        for result in results:
            if result.boxes is None:
                continue
            for i in range(len(result.boxes)):
                x1, y1, x2, y2 = result.boxes.xyxy[i].tolist()
                predictions.append(
                    {
                        "image_id": image_id,
                        "category_id": int(result.boxes.cls[i].item()),
                        "bbox": [
                            round(x1, 1),
                            round(y1, 1),
                            round(x2 - x1, 1),
                            round(y2 - y1, 1),
                        ],
                        "score": round(float(result.boxes.conf[i].item()), 4),
                    }
                )

    print(f"Generated {len(predictions)} predictions")
    return predictions


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predictions locally")
    parser.add_argument("--predictions", type=str, help="Path to predictions JSON")
    parser.add_argument("--model", type=str, help="Path to .pt weights")
    parser.add_argument("--onnx", type=str, help="Path to .onnx weights")
    parser.add_argument(
        "--images", type=str, help="Images directory (with --model or --onnx)"
    )
    parser.add_argument(
        "--tta", action="store_true", help="Enable TTA (with --onnx only)"
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default="data/raw/train/annotations.json",
        help="Path to COCO annotations",
    )
    parser.add_argument(
        "--val-only",
        action="store_true",
        help="Only evaluate on val split images",
    )
    args = parser.parse_args()

    # Determine val image IDs
    val_image_ids = None
    if args.val_only:
        val_dir = Path("data/yolo/images/val")
        if val_dir.exists():
            val_image_ids = set()
            for img_path in val_dir.iterdir():
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    image_id = int(img_path.stem.split("_")[-1])
                    val_image_ids.add(image_id)
            print(f"Evaluating on {len(val_image_ids)} val images")

    if args.predictions:
        with open(args.predictions) as f:
            predictions = json.load(f)
    elif args.onnx and args.images:
        predictions = generate_predictions_onnx(
            args.onnx, args.images, use_tta=args.tta
        )
    elif args.model and args.images:
        predictions = generate_predictions_pt(args.model, args.images)
    else:
        print("ERROR: Provide --predictions, or --model/--onnx + --images")
        raise SystemExit(1)

    results = compute_competition_score(predictions, args.annotations, val_image_ids)

    print("\n" + "=" * 50)
    print(f"Detection mAP@0.5:       {results['detection_map']:.4f}")
    print(f"Classification mAP@0.5:  {results['classification_map']:.4f}")
    print(f"Competition Score:       {results['score']:.4f}")
    print(
        f"  = 0.7 * {results['detection_map']:.4f} + 0.3 * {results['classification_map']:.4f}"
    )
    print("=" * 50)


if __name__ == "__main__":
    main()
