"""
Two-stage shelf product detection:
  Stage 1 — YOLO11 (ONNX) with sliced inference (full image + overlapping tiles)
            merged via Weighted Box Fusion
  Stage 2 — EfficientNet-V2-S classifier refines product identity

Sandbox backend: ONNX (ultralytics 8.1.0 can't load YOLO11 .pt)

Contract:
    python run.py --input /data/images --output /output/predictions.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from torchvision.ops import nms as torchvision_nms
from ensemble_boxes import weighted_boxes_fusion

# ── Paths (weights sit next to run.py in the zip root) ──────────
SCRIPT_DIR = Path(__file__).parent
YOLO_WEIGHTS_ONNX = SCRIPT_DIR / "best.onnx"
YOLO_WEIGHTS_PT = SCRIPT_DIR / "best.pt"
CLASSIFIER_WEIGHTS = SCRIPT_DIR / "best_classifier.pt"

# ── Detection config ────────────────────────────────────────────
YOLO_CONF = 0.08          # slightly lower than v1 (0.10) for better recall
YOLO_IOU = 0.6
YOLO_IMGSZ = 1280
YOLO_MAX_DET = 600        # more detections (shelves are dense)

# ── Sliced inference config ─────────────────────────────────────
TILE_SIZE = 1280
TILE_OVERLAP = 0.3         # increased from 0.25 for better coverage
TILE_MIN_DIM = 1600

# ── WBF config ──────────────────────────────────────────────────
WBF_IOU_THR = 0.55
WBF_SKIP_THR = 0.001
WBF_FULL_WEIGHT = 2
WBF_TILE_WEIGHT = 1

# ── Post-merge NMS ──────────────────────────────────────────────
POST_NMS_IOU = 0.45        # slightly more aggressive than v1 (0.50)

# ── Classifier config ──────────────────────────────────────────
PAD_RATIO = 0.15
CLS_OVERRIDE_THR = 0.40
CLS_BATCH_SIZE = 64
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ── Timing ──────────────────────────────────────────────────────
MAX_TIME = 280

# ── Global state ────────────────────────────────────────────────
_backend = None
_yolo = None
_classifier = None
_cls_transform = None
_device = None

# ── Torch-load patch (PyTorch 2.6 weights_only default) ────────
_orig_torch_load = torch.load


def _patched_torch_load(f, map_location=None, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(f, map_location=map_location, **kwargs)


torch.load = _patched_torch_load


# =====================================================================
#  ONNX Backend
# =====================================================================

def _letterbox(pil_img, target_size=YOLO_IMGSZ):
    """Resize maintaining aspect ratio, center-pad to square."""
    iw, ih = pil_img.size
    scale = min(target_size / iw, target_size / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    resized = pil_img.resize((nw, nh), Image.BILINEAR)

    pad_w = (target_size - nw) // 2
    pad_h = (target_size - nh) // 2
    padded = Image.new("RGB", (target_size, target_size), (114, 114, 114))
    padded.paste(resized, (pad_w, pad_h))

    arr = np.array(padded, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    arr = np.expand_dims(arr, 0)

    return arr, scale, pad_w, pad_h


def _onnx_postprocess(output, scale, pad_w, pad_h, orig_w, orig_h,
                       conf_thr=YOLO_CONF, iou_thr=YOLO_IOU):
    """Decode ONNX output → (boxes_xyxy, scores, class_ids)."""
    pred = output[0]

    if pred.shape[0] > pred.shape[1]:
        pred = pred.T
    pred = pred.T

    boxes_cxcywh = pred[:, :4]
    class_scores = pred[:, 4:]

    max_scores = class_scores.max(axis=1)
    max_classes = class_scores.argmax(axis=1)

    mask = max_scores > conf_thr
    boxes_cxcywh = boxes_cxcywh[mask]
    max_scores = max_scores[mask]
    max_classes = max_classes[mask]

    if len(max_scores) == 0:
        return [], [], []

    cx, cy, w, h = (boxes_cxcywh[:, i] for i in range(4))
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    x1 = (x1 - pad_w) / scale
    y1 = (y1 - pad_h) / scale
    x2 = (x2 - pad_w) / scale
    y2 = (y2 - pad_h) / scale

    x1 = np.clip(x1, 0, orig_w)
    y1 = np.clip(y1, 0, orig_h)
    x2 = np.clip(x2, 0, orig_w)
    y2 = np.clip(y2, 0, orig_h)

    boxes_t = torch.tensor(
        np.stack([x1, y1, x2, y2], axis=1), dtype=torch.float32
    )
    scores_t = torch.tensor(max_scores, dtype=torch.float32)
    classes_t = torch.tensor(max_classes, dtype=torch.int64)

    offsets = classes_t.float() * 4096
    shifted = boxes_t + offsets.unsqueeze(1)
    keep = torchvision_nms(shifted, scores_t, iou_thr)

    if len(keep) > YOLO_MAX_DET:
        keep = keep[:YOLO_MAX_DET]

    keep = keep.tolist()
    return (
        boxes_t[keep].tolist(),
        scores_t[keep].tolist(),
        classes_t[keep].tolist(),
    )


def _onnx_predict(pil_img, augment=False):
    """Run ONNX YOLO on a PIL image. Returns (boxes_xyxy, scores, classes)."""
    iw, ih = pil_img.size
    inp, scale, pad_w, pad_h = _letterbox(pil_img)

    input_name = _yolo.get_inputs()[0].name
    output_name = _yolo.get_outputs()[0].name
    raw = _yolo.run([output_name], {input_name: inp})[0]

    boxes, scores, classes = _onnx_postprocess(
        raw, scale, pad_w, pad_h, iw, ih
    )

    if augment and boxes:
        flipped = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
        inp_f, _, _, _ = _letterbox(flipped)
        raw_f = _yolo.run([output_name], {input_name: inp_f})[0]
        boxes_f, scores_f, classes_f = _onnx_postprocess(
            raw_f, scale, pad_w, pad_h, iw, ih
        )
        for b in boxes_f:
            b[0], b[2] = iw - b[2], iw - b[0]

        boxes.extend(boxes_f)
        scores.extend(scores_f)
        classes.extend(classes_f)

        boxes, scores, classes = class_agnostic_nms(
            boxes, scores, classes, iou_thr=YOLO_IOU
        )

    return boxes, scores, classes


# =====================================================================
#  Ultralytics Backend (fallback for YOLOv8 .pt)
# =====================================================================

def _ultralytics_predict(pil_img, augment=False):
    """Run ultralytics YOLO on a PIL image."""
    results = _yolo.predict(
        pil_img,
        conf=YOLO_CONF,
        iou=YOLO_IOU,
        imgsz=YOLO_IMGSZ,
        max_det=YOLO_MAX_DET,
        verbose=False,
        augment=augment,
    )

    boxes, scores, classes = [], [], []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            boxes.append([x1, y1, x2, y2])
            scores.append(float(box.conf[0]))
            classes.append(int(box.cls[0]))

    return boxes, scores, classes


# =====================================================================
#  Unified predict
# =====================================================================

def yolo_predict(pil_img, augment=False):
    if _backend == "onnx":
        return _onnx_predict(pil_img, augment=augment)
    else:
        return _ultralytics_predict(pil_img, augment=augment)


# =====================================================================
#  Model loading
# =====================================================================

def _load_classifier(path, device):
    ckpt = torch.load(path, map_location=device)
    num_classes = ckpt["num_classes"]
    img_size = ckpt.get("img_size", 224)

    model = models.efficientnet_v2_s(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.0),
        nn.Linear(in_features, num_classes),
    )
    state = {k: v.float() for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state)
    model.to(device).eval()

    xform = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return model, xform


def load_models():
    global _backend, _yolo, _classifier, _cls_transform, _device

    _device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if YOLO_WEIGHTS_ONNX.exists():
        import onnxruntime as ort
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        _yolo = ort.InferenceSession(str(YOLO_WEIGHTS_ONNX), providers=providers)
        _backend = "onnx"
    elif YOLO_WEIGHTS_PT.exists():
        from ultralytics import YOLO
        _yolo = YOLO(str(YOLO_WEIGHTS_PT))
        _backend = "ultralytics"
    else:
        raise FileNotFoundError("No YOLO weights found (best.onnx or best.pt)")

    if CLASSIFIER_WEIGHTS.exists():
        _classifier, _cls_transform = _load_classifier(
            CLASSIFIER_WEIGHTS, _device
        )


# =====================================================================
#  Sliced inference + WBF
# =====================================================================

def get_tile_positions(img_w, img_h):
    stride = int(TILE_SIZE * (1 - TILE_OVERLAP))

    def positions(dim):
        if dim <= TILE_SIZE:
            return [0]
        pos = list(range(0, dim - TILE_SIZE + 1, stride))
        if pos[-1] + TILE_SIZE < dim:
            pos.append(dim - TILE_SIZE)
        return sorted(set(pos))

    return [(x, y) for y in positions(img_h) for x in positions(img_w)]


def class_agnostic_nms(boxes_xyxy, scores, classes, iou_thr=POST_NMS_IOU):
    if not boxes_xyxy:
        return [], [], []

    boxes_t = torch.tensor(boxes_xyxy, dtype=torch.float32)
    scores_t = torch.tensor(scores, dtype=torch.float32)

    keep = torchvision_nms(boxes_t, scores_t, iou_thr)
    keep = keep.tolist()

    return (
        [boxes_xyxy[i] for i in keep],
        [scores[i] for i in keep],
        [classes[i] for i in keep],
    )


def predict_with_slicing(pil_img, use_tiles=True):
    """Run YOLO on full image + tiles, merge with WBF + NMS."""
    iw, ih = pil_img.size

    full_boxes, full_scores, full_classes = yolo_predict(pil_img, augment=True)

    full_boxes_norm = [
        [x1 / iw, y1 / ih, x2 / iw, y2 / ih]
        for x1, y1, x2, y2 in full_boxes
    ]

    tile_boxes_norm = []
    tile_scores = []
    tile_classes = []

    if use_tiles and max(iw, ih) > TILE_MIN_DIM:
        positions = get_tile_positions(iw, ih)

        for tx, ty in positions:
            tw = min(TILE_SIZE, iw - tx)
            th = min(TILE_SIZE, ih - ty)
            tile = pil_img.crop((tx, ty, tx + tw, ty + th))

            t_boxes, t_scores, t_classes = yolo_predict(tile, augment=False)

            for (x1, y1, x2, y2), sc, cl in zip(
                t_boxes, t_scores, t_classes
            ):
                fx1 = max(0.0, min(1.0, (x1 + tx) / iw))
                fy1 = max(0.0, min(1.0, (y1 + ty) / ih))
                fx2 = max(0.0, min(1.0, (x2 + tx) / iw))
                fy2 = max(0.0, min(1.0, (y2 + ty) / ih))

                tile_boxes_norm.append([fx1, fy1, fx2, fy2])
                tile_scores.append(sc)
                tile_classes.append(cl)

    if not full_boxes_norm and not tile_boxes_norm:
        return [], [], []

    boxes_list = []
    scores_list = []
    labels_list = []
    weights = []

    if full_boxes_norm:
        boxes_list.append(full_boxes_norm)
        scores_list.append(full_scores)
        labels_list.append(full_classes)
        weights.append(WBF_FULL_WEIGHT)

    if tile_boxes_norm:
        boxes_list.append(tile_boxes_norm)
        scores_list.append(tile_scores)
        labels_list.append(tile_classes)
        weights.append(WBF_TILE_WEIGHT)

    if len(boxes_list) == 1:
        merged_boxes = np.array(boxes_list[0])
        merged_scores = np.array(scores_list[0])
        merged_labels = np.array(labels_list[0])
    else:
        merged_boxes, merged_scores, merged_labels = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list,
            weights=weights,
            iou_thr=WBF_IOU_THR,
            skip_box_thr=WBF_SKIP_THR,
        )

    result_boxes = []
    for bx in merged_boxes:
        result_boxes.append([
            float(bx[0]) * iw, float(bx[1]) * ih,
            float(bx[2]) * iw, float(bx[3]) * ih,
        ])

    result_scores = (
        merged_scores.tolist()
        if hasattr(merged_scores, "tolist")
        else list(merged_scores)
    )
    result_labels = [int(l) for l in merged_labels]

    result_boxes, result_scores, result_labels = class_agnostic_nms(
        result_boxes, result_scores, result_labels
    )

    return result_boxes, result_scores, result_labels


# =====================================================================
#  Classifier
# =====================================================================

def classify_crops(crops):
    results = []
    for i in range(0, len(crops), CLS_BATCH_SIZE):
        batch = crops[i : i + CLS_BATCH_SIZE]
        tensors = torch.stack([_cls_transform(c) for c in batch]).to(_device)
        with torch.no_grad(), torch.amp.autocast(
            "cuda", enabled=(_device.type == "cuda")
        ):
            logits = _classifier(tensors)
        probs = torch.softmax(logits, dim=1)
        confs, preds = probs.max(1)
        for p, c in zip(preds.cpu().tolist(), confs.cpu().tolist()):
            results.append((p, c))
    return results


# =====================================================================
#  Main prediction pipeline
# =====================================================================

def predict_image(pil_img, image_id, use_tiles=True):
    iw, ih = pil_img.size

    boxes_xyxy, det_scores, det_classes = predict_with_slicing(
        pil_img, use_tiles=use_tiles
    )

    if not boxes_xyxy:
        return []

    crops = []
    for x1, y1, x2, y2 in boxes_xyxy:
        bw, bh = x2 - x1, y2 - y1
        px = int(bw * PAD_RATIO)
        py = int(bh * PAD_RATIO)
        crop = pil_img.crop((
            max(0, int(x1) - px), max(0, int(y1) - py),
            min(iw, int(x2) + px), min(ih, int(y2) + py),
        ))
        crops.append(crop)

    cls_results = None
    if crops and _classifier is not None:
        cls_results = classify_crops(crops)

    predictions = []
    for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
        if cls_results is not None:
            cls_pred, cls_conf = cls_results[i]
            if cls_conf > CLS_OVERRIDE_THR:
                cat_id = cls_pred
                score = 0.5 * cls_conf + 0.5 * det_scores[i]
            else:
                cat_id = det_classes[i]
                score = det_scores[i]
        else:
            cat_id = det_classes[i]
            score = det_scores[i]

        predictions.append({
            "image_id": image_id,
            "category_id": cat_id,
            "bbox": [
                round(float(x1), 1),
                round(float(y1), 1),
                round(float(x2 - x1), 1),
                round(float(y2 - y1), 1),
            ],
            "score": round(float(score), 4),
        })

    return predictions


# =====================================================================
#  Entry point
# =====================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Directory of shelf images")
    p.add_argument("--output", required=True, help="Path to write predictions.json")
    args = p.parse_args()

    load_models()

    input_dir = Path(args.input)
    all_predictions = []

    image_paths = sorted(input_dir.glob("*.jpg"))
    start_time = time.time()
    n_images = len(image_paths)

    for idx, img_path in enumerate(image_paths):
        image_id = int(img_path.stem.split("_")[-1])
        pil_img = Image.open(img_path).convert("RGB")

        elapsed = time.time() - start_time
        remaining_time = MAX_TIME - elapsed
        remaining_images = n_images - idx
        time_per_image = remaining_time / max(1, remaining_images)
        use_tiles = time_per_image > 1.5

        preds = predict_image(pil_img, image_id, use_tiles=use_tiles)
        all_predictions.extend(preds)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_predictions, f)


if __name__ == "__main__":
    main()
