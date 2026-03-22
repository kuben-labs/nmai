"""
Use SAM 3 to generate product detections on training images.

This creates pseudo-labels that can be used to:
1. Augment COCO training annotations
2. Generate additional classifier training crops
3. Serve as a fallback if ONNX export fails (train YOLOv8 on these labels)

Run on the GCP VM:
    python generate_sam3_labels.py --images coco/train --output sam3_detections.json

Output: COCO-format predictions with SAM 3 detected bounding boxes.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def detect_products(processor, image_path, prompts=None):
    """
    Run SAM 3 detection on a single image with multiple text prompts.

    Returns list of dicts with keys: bbox, score, mask_area
    """
    if prompts is None:
        prompts = ["product", "grocery item", "food package"]

    img = Image.open(image_path).convert("RGB")
    width, height = img.size

    all_detections = []

    for prompt in prompts:
        state = processor.set_image(img)
        state = processor.set_text_prompt(prompt=prompt, state=state)

        if "boxes" not in state or len(state["boxes"]) == 0:
            continue

        boxes = state["boxes"].cpu().numpy()  # [N, 4] in xyxy format
        scores = state["scores"].cpu().numpy()  # [N]

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            w, h = x2 - x1, y2 - y1
            # Filter: skip very small or very large detections
            area_ratio = (w * h) / (width * height)
            if area_ratio < 0.001 or area_ratio > 0.3:
                continue
            # Skip extreme aspect ratios (not product-like)
            aspect = max(w, h) / (min(w, h) + 1e-6)
            if aspect > 8:
                continue

            all_detections.append({
                "bbox": [float(x1), float(y1), float(w), float(h)],  # COCO format
                "score": float(scores[i]),
                "prompt": prompt,
            })

    # NMS across prompts
    if len(all_detections) > 0:
        all_detections = nms_detections(all_detections, iou_threshold=0.5)

    return all_detections


def nms_detections(detections, iou_threshold=0.5):
    """Simple NMS on detections."""
    if len(detections) == 0:
        return detections

    boxes = np.array([d["bbox"] for d in detections])
    scores = np.array([d["score"] for d in detections])

    # Convert xywh to xyxy for IoU
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

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return [detections[i] for i in keep]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, help="Path to training images directory")
    parser.add_argument("--output", default="sam3_detections.json", help="Output file")
    parser.add_argument("--checkpoint", default=None, help="SAM 3 checkpoint path")
    parser.add_argument("--confidence", type=float, default=0.3,
                        help="Confidence threshold")
    parser.add_argument("--prompts", nargs="+",
                        default=["product", "grocery item on shelf", "packaged food"],
                        help="Text prompts for detection")
    parser.add_argument("--save-crops", action="store_true",
                        help="Also save detected product crops")
    parser.add_argument("--crops-dir", default="sam3_crops",
                        help="Directory for saved crops")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load SAM 3
    print("Loading SAM 3...")
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model(
        device=device,
        eval_mode=True,
        checkpoint_path=args.checkpoint,
        enable_segmentation=True,
    )
    processor = Sam3Processor(model, confidence_threshold=args.confidence)

    # Find images
    images_dir = Path(args.images)
    image_files = sorted(
        f for f in images_dir.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    print(f"Found {len(image_files)} images")

    if args.save_crops:
        crops_dir = Path(args.crops_dir)
        crops_dir.mkdir(exist_ok=True)

    # Process images
    all_predictions = []
    crop_idx = 0

    for img_path in tqdm(image_files, desc="Detecting products"):
        image_id = int(img_path.stem.split("_")[-1])
        detections = detect_products(processor, img_path, prompts=args.prompts)

        for det in detections:
            pred = {
                "image_id": image_id,
                "category_id": 0,  # Unknown - will be classified later
                "bbox": det["bbox"],
                "score": det["score"],
            }
            all_predictions.append(pred)

            # Optionally save crops
            if args.save_crops:
                img = Image.open(img_path).convert("RGB")
                x, y, w, h = det["bbox"]
                crop = img.crop((int(x), int(y), int(x + w), int(y + h)))
                crop.save(crops_dir / f"crop_{crop_idx:06d}_img{image_id}.jpg")
                crop_idx += 1

    # Save predictions
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_predictions, f)

    print(f"\nSaved {len(all_predictions)} detections to {output_path}")
    if args.save_crops:
        print(f"Saved {crop_idx} crops to {args.crops_dir}/")


if __name__ == "__main__":
    main()
