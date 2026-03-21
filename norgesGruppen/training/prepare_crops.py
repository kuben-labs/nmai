"""
Extract product crops from COCO annotations and organize reference images
for training the product classifier.

Output:
    crops/{class_id}/train_{img_id}_{ann_id}.jpg
    crops/{class_id}/ref_{product_code}_{view}.jpg
    crops/class_names.json

Usage:
    python prepare_crops.py
"""

import json
import random
from pathlib import Path

from PIL import Image

COCO_DIR = Path(__file__).parent.parent / "coco" / "train"
PRODUCT_DIR = Path(__file__).parent.parent / "product"
OUT_DIR = Path(__file__).parent / "crops"
SEED = 42
PAD_RATIO = 0.1  # expand bbox by 10% on each side for context


def main():
    random.seed(SEED)

    with open(COCO_DIR / "annotations.json") as f:
        coco = json.load(f)
    with open(PRODUCT_DIR / "metadata.json") as f:
        meta = json.load(f)

    images_dir = COCO_DIR / "images"
    img_info = {img["id"]: img for img in coco["images"]}

    class_names = [cat["name"] for cat in coco["categories"]]
    num_classes = len(class_names)

    # Category ID == index (sequential 0-355)
    name_to_idx = {}
    for i, cat in enumerate(coco["categories"]):
        name_to_idx[cat["name"].strip().upper()] = i

    print(f"Categories: {num_classes}")

    # Create output dirs
    for i in range(num_classes):
        (OUT_DIR / str(i)).mkdir(parents=True, exist_ok=True)

    # 1. Extract crops from training annotations
    # Cache open images to avoid reopening the same image for each annotation
    img_cache = {}
    crop_count = 0

    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0):
            continue

        img = img_info[ann["image_id"]]
        img_path = images_dir / img["file_name"]
        if not img_path.exists():
            continue

        cls_idx = ann["category_id"]  # sequential, so id == index
        x, y, w, h = ann["bbox"]
        iw, ih = img["width"], img["height"]

        # Skip tiny boxes
        if w < 8 or h < 8:
            continue

        # Add padding
        pad_x = w * PAD_RATIO
        pad_y = h * PAD_RATIO
        x1 = max(0, int(x - pad_x))
        y1 = max(0, int(y - pad_y))
        x2 = min(iw, int(x + w + pad_x))
        y2 = min(ih, int(y + h + pad_y))

        try:
            if ann["image_id"] not in img_cache:
                img_cache[ann["image_id"]] = Image.open(img_path).convert("RGB")
            pil_img = img_cache[ann["image_id"]]
            crop = pil_img.crop((x1, y1, x2, y2))
            out_path = OUT_DIR / str(cls_idx) / f"train_{ann['image_id']}_{ann['id']}.jpg"
            crop.save(out_path, quality=85)
            crop_count += 1
        except Exception as e:
            print(f"  Error cropping ann {ann['id']}: {e}")

        if crop_count % 5000 == 0 and crop_count > 0:
            print(f"  {crop_count} crops extracted...")

        # Keep cache bounded (avoid OOM with large images)
        if len(img_cache) > 20:
            oldest = next(iter(img_cache))
            del img_cache[oldest]

    img_cache.clear()
    print(f"  Total crops: {crop_count}")

    # 2. Copy reference product images
    ref_count = 0
    for prod in meta["products"]:
        if not prod["has_images"]:
            continue
        name = prod["product_name"].strip().upper()
        if name not in name_to_idx:
            continue

        cls_idx = name_to_idx[name]
        code = prod["product_code"]
        folder = PRODUCT_DIR / code

        for view in prod.get("image_types", []):
            img_path = folder / f"{view}.jpg"
            if not img_path.exists():
                continue
            try:
                img = Image.open(img_path).convert("RGB")
                out_path = OUT_DIR / str(cls_idx) / f"ref_{code}_{view}.jpg"
                img.save(out_path, quality=85)
                ref_count += 1
            except Exception:
                pass

    print(f"  Reference images copied: {ref_count}")

    # Save class names
    with open(OUT_DIR / "class_names.json", "w") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    # Print stats
    counts = []
    for i in range(num_classes):
        n = len(list((OUT_DIR / str(i)).glob("*.jpg")))
        counts.append(n)

    print(f"\nClass distribution:")
    print(f"  Total images:  {sum(counts)}")
    print(f"  0 images:      {sum(1 for c in counts if c == 0)}")
    print(f"  <5 images:     {sum(1 for c in counts if c < 5)}")
    print(f"  <10 images:    {sum(1 for c in counts if c < 10)}")
    print(f"  Max per class: {max(counts)}")
    print(f"  Min per class: {min(counts)}")
    print(f"  Mean:          {sum(counts) / len(counts):.1f}")


if __name__ == "__main__":
    main()
