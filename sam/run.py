"""
NorgesGruppen Product Detection — Submission Entry Point

Pipeline:
  1. SAM 3 ONNX detector: shelf image → product bounding boxes + confidence scores
  2. EfficientNet classifier ONNX: product crop → category_id

Usage (sandbox):
    python run.py --input /data/images --output /output/predictions.json

Weight files expected alongside this script:
    sam3_detector.onnx   - SAM 3 detector (ONNX, quantized)
    classifier.onnx      - Product classifier (ONNX, FP16)
    text_embed.npy       - Pre-computed text features for SAM 3
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


# --- Constants ---
SAM3_INPUT_SIZE = 1008
CLASSIFIER_INPUT_SIZE = 224
SAM3_MEAN = [0.5, 0.5, 0.5]
SAM3_STD = [0.5, 0.5, 0.5]
CLASSIFIER_MEAN = [0.485, 0.456, 0.406]
CLASSIFIER_STD = [0.229, 0.224, 0.225]
CONFIDENCE_THRESHOLD = 0.3
NMS_IOU_THRESHOLD = 0.5
MIN_BOX_AREA_RATIO = 0.0005
MAX_BOX_AREA_RATIO = 0.25


def load_onnx_session(model_path):
    """Load ONNX model with GPU if available."""
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ort.InferenceSession(str(model_path), providers=providers)


def preprocess_for_sam3(image):
    """Preprocess PIL image for SAM 3 detector."""
    img = image.resize((SAM3_INPUT_SIZE, SAM3_INPUT_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    # Normalize
    for c in range(3):
        arr[:, :, c] = (arr[:, :, c] - SAM3_MEAN[c]) / SAM3_STD[c]
    # HWC → CHW → NCHW
    arr = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]
    return arr.astype(np.float32)


def preprocess_for_classifier(crop):
    """Preprocess PIL crop for classifier."""
    img = crop.resize((CLASSIFIER_INPUT_SIZE, CLASSIFIER_INPUT_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    for c in range(3):
        arr[:, :, c] = (arr[:, :, c] - CLASSIFIER_MEAN[c]) / CLASSIFIER_STD[c]
    arr = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]
    return arr.astype(np.float32)


def nms_boxes(boxes, scores, iou_threshold=0.5):
    """Non-maximum suppression on boxes [N, 4] in xywh format."""
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    areas = boxes[:, 2] * boxes[:, 3]

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

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        mask = iou <= iou_threshold
        order = order[1:][mask]

    return keep


def detect_products_sam3(session, image, orig_w, orig_h):
    """
    Run SAM 3 ONNX detector on an image.
    Returns boxes [N, 4] in xywh COCO format and scores [N].
    """
    input_name = session.get_inputs()[0].name
    img_arr = preprocess_for_sam3(image)

    outputs = session.run(None, {input_name: img_arr})
    # SAM 3 outputs: boxes [N, 4] in cxcywh normalized [0,1], scores [N]
    raw_boxes = outputs[0]  # [N, 4] or [1, N, 4]
    raw_scores = outputs[1]  # [N] or [1, N]

    # Handle batch dimension
    if raw_boxes.ndim == 3:
        raw_boxes = raw_boxes[0]
    if raw_scores.ndim == 2:
        raw_scores = raw_scores[0]

    # Filter by confidence
    mask = raw_scores > CONFIDENCE_THRESHOLD
    boxes_cxcywh = raw_boxes[mask]
    scores = raw_scores[mask]

    if len(boxes_cxcywh) == 0:
        return np.zeros((0, 4)), np.zeros(0)

    # Convert cxcywh normalized → xywh pixel coordinates
    cx = boxes_cxcywh[:, 0] * orig_w
    cy = boxes_cxcywh[:, 1] * orig_h
    w = boxes_cxcywh[:, 2] * orig_w
    h = boxes_cxcywh[:, 3] * orig_h
    x = cx - w / 2
    y = cy - h / 2

    boxes_xywh = np.stack([x, y, w, h], axis=1)

    # Filter by size
    img_area = orig_w * orig_h
    box_areas = w * h
    size_mask = (box_areas / img_area > MIN_BOX_AREA_RATIO) & \
                (box_areas / img_area < MAX_BOX_AREA_RATIO)
    boxes_xywh = boxes_xywh[size_mask]
    scores = scores[size_mask]

    # Filter extreme aspect ratios
    if len(boxes_xywh) > 0:
        aspect = np.maximum(boxes_xywh[:, 2], boxes_xywh[:, 3]) / \
                 (np.minimum(boxes_xywh[:, 2], boxes_xywh[:, 3]) + 1e-6)
        aspect_mask = aspect < 8
        boxes_xywh = boxes_xywh[aspect_mask]
        scores = scores[aspect_mask]

    # NMS
    if len(boxes_xywh) > 0:
        keep = nms_boxes(boxes_xywh, scores, NMS_IOU_THRESHOLD)
        boxes_xywh = boxes_xywh[keep]
        scores = scores[keep]

    return boxes_xywh, scores


def detect_products_sliding_window(image, orig_w, orig_h):
    """
    Fallback detector using sliding window with fixed grid.
    For when SAM 3 ONNX is not available.
    Generates candidate boxes based on typical product sizes on shelves.
    """
    boxes = []
    scores = []

    # Typical product sizes relative to image
    # Products on grocery shelves are roughly 3-8% of image width, 5-15% of height
    box_configs = [
        (0.04, 0.08),  # small products
        (0.06, 0.12),  # medium products
        (0.08, 0.15),  # large products
        (0.05, 0.10),  # typical products
    ]

    for rel_w, rel_h in box_configs:
        bw = int(rel_w * orig_w)
        bh = int(rel_h * orig_h)
        stride_x = max(bw // 2, 1)
        stride_y = max(bh // 2, 1)

        for y in range(0, orig_h - bh, stride_y):
            for x in range(0, orig_w - bw, stride_x):
                boxes.append([x, y, bw, bh])
                scores.append(0.5)

    return np.array(boxes, dtype=np.float32), np.array(scores, dtype=np.float32)


def classify_crops(session, image, boxes, batch_size=32):
    """
    Classify product crops using the ONNX classifier.
    Returns category_ids [N] and classification scores [N].
    """
    if len(boxes) == 0:
        return np.zeros(0, dtype=int), np.zeros(0)

    input_name = session.get_inputs()[0].name
    all_cat_ids = []
    all_scores = []

    for i in range(0, len(boxes), batch_size):
        batch_boxes = boxes[i:i + batch_size]
        batch_inputs = []

        for box in batch_boxes:
            x, y, w, h = box
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(image.width, int(x + w))
            y2 = min(image.height, int(y + h))

            if x2 <= x1 or y2 <= y1:
                batch_inputs.append(np.zeros((1, 3, CLASSIFIER_INPUT_SIZE, CLASSIFIER_INPUT_SIZE), dtype=np.float32))
                continue

            crop = image.crop((x1, y1, x2, y2))
            batch_inputs.append(preprocess_for_classifier(crop))

        batch_arr = np.concatenate(batch_inputs, axis=0)
        logits = session.run(None, {input_name: batch_arr})[0]  # [B, num_classes]

        # Softmax
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        cat_ids = probs.argmax(axis=1)
        cat_scores = probs.max(axis=1)

        all_cat_ids.extend(cat_ids.tolist())
        all_scores.extend(cat_scores.tolist())

    return np.array(all_cat_ids), np.array(all_scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input images directory")
    parser.add_argument("--output", required=True, help="Output predictions JSON path")
    args = parser.parse_args()

    script_dir = Path(__file__).parent

    # Load models
    sam3_path = script_dir / "sam3_detector.onnx"
    classifier_path = script_dir / "classifier.onnx"

    has_sam3 = sam3_path.exists()
    has_classifier = classifier_path.exists()

    if has_sam3:
        print(f"Loading SAM 3 detector: {sam3_path}")
        detector_session = load_onnx_session(sam3_path)
    else:
        print("WARNING: SAM 3 detector not found, using sliding window fallback")
        detector_session = None

    if has_classifier:
        print(f"Loading classifier: {classifier_path}")
        classifier_session = load_onnx_session(classifier_path)
    else:
        print("WARNING: Classifier not found, all predictions will use category_id=0")
        classifier_session = None

    # Process images
    input_dir = Path(args.input)
    image_files = sorted(
        f for f in input_dir.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    print(f"Processing {len(image_files)} images...")

    predictions = []

    for img_path in image_files:
        image_id = int(img_path.stem.split("_")[-1])
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        # 1. Detection
        if detector_session:
            boxes, det_scores = detect_products_sam3(
                detector_session, image, orig_w, orig_h
            )
        else:
            boxes, det_scores = detect_products_sliding_window(
                image, orig_w, orig_h
            )

        if len(boxes) == 0:
            continue

        # 2. Classification
        if classifier_session:
            cat_ids, cls_scores = classify_crops(classifier_session, image, boxes)
            # Combine detection and classification scores
            final_scores = det_scores * 0.5 + cls_scores * 0.5
        else:
            cat_ids = np.zeros(len(boxes), dtype=int)
            final_scores = det_scores

        # 3. Format predictions
        for i in range(len(boxes)):
            predictions.append({
                "image_id": image_id,
                "category_id": int(cat_ids[i]),
                "bbox": [
                    round(float(boxes[i][0]), 1),
                    round(float(boxes[i][1]), 1),
                    round(float(boxes[i][2]), 1),
                    round(float(boxes[i][3]), 1),
                ],
                "score": round(float(final_scores[i]), 3),
            })

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(predictions, f)

    print(f"Wrote {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
