"""Submission entry point: YOLOv8 ONNX grocery detection."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


SCRIPT_DIR = Path(__file__).resolve().parent
ONNX_PATH = SCRIPT_DIR / "best.onnx"

CONF_THRESH = 0.01
IOU_THRESH = 0.5
IMGSZ = 1280
MAX_DET = 300


def letterbox(img, new_shape=1280):
    """Resize image with letterbox padding to square."""
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
    """Class-aware NMS."""
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


def _infer_once(session, img_bgr, imgsz, conf_thresh):
    """Run single forward pass, return boxes_xyxy, scores, class_ids in original coords."""
    orig_h, orig_w = img_bgr.shape[:2]

    img_padded, scale, pad_left, pad_top = letterbox(img_bgr, imgsz)
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

    mask = scores > conf_thresh
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

    boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, orig_w)
    boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, orig_h)

    return boxes_xyxy, scores, class_ids


def run_yolo_onnx(session, img_bgr, imgsz=1280, conf_thresh=0.01, iou_thresh=0.5):
    """Run YOLOv8 ONNX inference with horizontal flip TTA."""
    orig_h, orig_w = img_bgr.shape[:2]
    all_boxes, all_scores, all_class_ids = [], [], []

    for flip in [False, True]:
        img = img_bgr if not flip else img_bgr[:, ::-1].copy()
        boxes, scores, class_ids = _infer_once(session, img, imgsz, conf_thresh)

        if len(scores) == 0:
            continue

        # Un-flip boxes
        if flip:
            x1_new = orig_w - boxes[:, 2]
            x2_new = orig_w - boxes[:, 0]
            boxes[:, 0] = x1_new
            boxes[:, 2] = x2_new

        all_boxes.append(boxes)
        all_scores.append(scores)
        all_class_ids.append(class_ids)

    if len(all_boxes) == 0:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)

    all_boxes = np.concatenate(all_boxes)
    all_scores = np.concatenate(all_scores)
    all_class_ids = np.concatenate(all_class_ids)

    # NMS across TTA results
    keep = nms(all_boxes, all_scores, all_class_ids, iou_thresh)
    if len(keep) > MAX_DET:
        keep = keep[:MAX_DET]

    return all_boxes[keep], all_scores[keep], all_class_ids[keep]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)

    # Load ONNX model with GPU
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(str(ONNX_PATH), providers=providers)
    print(f"Loaded ONNX model: {ONNX_PATH.name}")
    print(f"Providers: {session.get_providers()}")

    predictions = []
    image_files = sorted(
        p for p in input_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    print(f"Processing {len(image_files)} images...")

    for idx, img_path in enumerate(image_files):
        image_id = int(img_path.stem.split("_")[-1])
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  WARNING: Could not read {img_path.name}")
            continue

        boxes_xyxy, scores, class_ids = run_yolo_onnx(
            session, img_bgr, IMGSZ, CONF_THRESH, IOU_THRESH
        )

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

    print(f"Total predictions: {len(predictions)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(predictions, f)

    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    main()
