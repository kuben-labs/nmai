"""Submission entry point: multi-class YOLO11 (ONNX) detection + classification."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
YOLO_ONNX = SCRIPT_DIR / "yolo.onnx"

# ── Config ────────────────────────────────────────────────────────────────────
CONF_THRESH = 0.15
IOU_THRESH = 0.5
IMGSZ = 1280


def letterbox(img, new_shape=1280):
    """Resize with padding to square."""
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

    # Offset boxes by class_id for class-aware NMS
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


def run_yolo(session, img_bgr, imgsz=1280, conf_thresh=0.15, iou_thresh=0.5):
    """Run YOLO ONNX inference, return boxes, scores, class_ids."""
    orig_h, orig_w = img_bgr.shape[:2]
    img_padded, scale, pad_left, pad_top = letterbox(img_bgr, imgsz)

    # HWC BGR -> CHW RGB normalized
    blob = img_padded[:, :, ::-1].astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis]

    # Check input dtype
    input_name = session.get_inputs()[0].name
    input_type = session.get_inputs()[0].type
    if "float16" in input_type:
        blob = blob.astype(np.float16)

    outputs = session.run(None, {input_name: blob})
    preds = outputs[0]  # [1, 4+nc, num_preds]

    # Transpose if needed
    if preds.shape[1] < preds.shape[2]:
        preds = preds.transpose(0, 2, 1)

    preds = preds[0].astype(np.float32)  # [num_preds, 4+nc]
    boxes_cxcywh = preds[:, :4]
    class_scores = preds[:, 4:]

    # Get best class per prediction
    class_ids = class_scores.argmax(axis=1)
    scores = class_scores.max(axis=1)

    # Filter by confidence
    mask = scores > conf_thresh
    boxes_cxcywh = boxes_cxcywh[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    if len(scores) == 0:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)

    # Convert to xyxy
    boxes_xyxy = np.empty_like(boxes_cxcywh)
    boxes_xyxy[:, 0] = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
    boxes_xyxy[:, 1] = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
    boxes_xyxy[:, 2] = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
    boxes_xyxy[:, 3] = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2

    # NMS
    keep = nms(boxes_xyxy, scores, class_ids, iou_thresh)
    boxes_xyxy = boxes_xyxy[keep]
    scores = scores[keep]
    class_ids = class_ids[keep]

    # Rescale to original image
    boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - pad_left) / scale
    boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - pad_top) / scale

    # Clamp
    boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, orig_w)
    boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, orig_h)

    return boxes_xyxy, scores, class_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)

    # Load YOLO
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(str(YOLO_ONNX), providers=providers)

    predictions = []
    image_files = sorted(input_dir.glob("*.jpg"))

    for img_path in image_files:
        image_id = int(img_path.stem.replace("img_", ""))
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        boxes_xyxy, scores, class_ids = run_yolo(
            session, img_bgr, IMGSZ, CONF_THRESH, IOU_THRESH
        )

        for box, score, cat_id in zip(boxes_xyxy, scores, class_ids):
            x1, y1, x2, y2 = box
            predictions.append({
                "image_id": image_id,
                "category_id": int(cat_id),
                "bbox": [
                    round(float(x1), 1),
                    round(float(y1), 1),
                    round(float(x2 - x1), 1),
                    round(float(y2 - y1), 1),
                ],
                "score": round(float(score), 4),
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(predictions, f)

    print(f"Wrote {len(predictions)} predictions for {len(image_files)} images")


if __name__ == "__main__":
    main()
