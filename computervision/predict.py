"""Inference pipeline: YOLO detection + EfficientNet classification on shelf images."""
import argparse
import json
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import timm
from ultralytics import YOLO


def load_classifier(checkpoint_path, device="cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = timm.create_model(
        ckpt["model_name"], pretrained=False, num_classes=ckpt["num_classes"]
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(device)

    imgsz = ckpt.get("imgsz", 224)
    transform = transforms.Compose([
        transforms.Resize(int(imgsz * 1.14)),
        transforms.CenterCrop(imgsz),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return model, transform


def load_class_map(path):
    with open(path) as f:
        data = json.load(f)
    # class_to_idx: name -> idx, we want idx -> name
    if all(isinstance(v, int) for v in data.values()):
        return {v: k for k, v in data.items()}
    return {int(k): v for k, v in data.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to shelf image")
    parser.add_argument("--yolo", default="runs/product_detect/weights/best.pt")
    parser.add_argument("--classifier", default="runs/classify/best.pt")
    parser.add_argument("--class-map", default="runs/classify/class_to_idx.json")
    parser.add_argument("--conf", type=float, default=0.3, help="YOLO confidence threshold")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default=None, help="Output image path")
    parser.add_argument("--product-meta", default="product/metadata.json", help="Product metadata.json path")
    parser.add_argument("--json-out", default=None, help="Output JSON path")
    args = parser.parse_args()

    # Load models
    yolo = YOLO(args.yolo)
    cls_model, cls_transform = load_classifier(args.classifier, args.device)
    idx_to_class = load_class_map(args.class_map)

    # Load product metadata for display names
    meta_path = Path(args.product_meta)
    code_to_name = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        for p in meta["products"]:
            code_to_name[p["product_code"]] = p["product_name"]

    # Detect
    img = Image.open(args.image).convert("RGB")
    results = yolo(args.image, conf=args.conf, verbose=False)

    predictions = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            det_conf = box.conf[0].item()

            # Crop and classify
            crop = img.crop((int(x1), int(y1), int(x2), int(y2)))
            inp = cls_transform(crop).unsqueeze(0).to(args.device)
            with torch.no_grad():
                logits = cls_model(inp)
                probs = torch.softmax(logits, dim=1)
                cls_conf, cls_idx = probs.max(1)

            product_code = idx_to_class.get(cls_idx.item(), "unknown")
            product_name = code_to_name.get(product_code, product_code)

            predictions.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "det_confidence": round(det_conf, 3),
                "product_code": product_code,
                "product_name": product_name,
                "cls_confidence": round(cls_conf.item(), 3),
            })

    # Print results
    print(f"Found {len(predictions)} products:")
    for p in predictions:
        print(f"  {p['product_name']} ({p['product_code']}) "
              f"det={p['det_confidence']:.2f} cls={p['cls_confidence']:.2f} "
              f"bbox={p['bbox']}")

    # Save JSON
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(predictions, f, indent=2)
        print(f"JSON saved to {args.json_out}")

    # Draw annotated image
    output_path = args.output or str(Path(args.image).stem) + "_predicted.jpg"
    draw = ImageDraw.Draw(img)
    for p in predictions:
        x1, y1, x2, y2 = p["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=2)
        label = f"{p['product_name'][:30]} {p['cls_confidence']:.0%}"
        draw.text((x1, max(0, y1 - 12)), label, fill="lime")
    img.save(output_path)
    print(f"Annotated image saved to {output_path}")


if __name__ == "__main__":
    main()
