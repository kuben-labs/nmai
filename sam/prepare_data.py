"""
Prepare classifier training data from:
1. Product reference images (product/ directory)
2. Training image crops (from COCO annotations)
3. Optional: SAM 3 generated crops

Run on VM:
    python prepare_data.py --annotations coco/train/annotations.json \
                           --images coco/train \
                           --products product \
                           --output classifier_data
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def build_product_code_to_category(annotations):
    """Build mapping from product_code to category_id using annotations."""
    code_to_cat = {}
    for ann in annotations["annotations"]:
        pc = ann.get("product_code")
        cat_id = ann["category_id"]
        if pc and cat_id != 356:  # 356 = unknown_product
            code_to_cat[str(pc)] = cat_id
    return code_to_cat


def extract_training_crops(annotations, images_dir, output_dir, max_crops_per_cat=200):
    """Extract product crops from COCO training images."""
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    # Build image_id → filename mapping
    id_to_file = {img["id"]: img["file_name"] for img in annotations["images"]}

    # Group annotations by category
    cat_counts = {}
    crop_records = []

    for ann in tqdm(annotations["annotations"], desc="Extracting crops"):
        cat_id = ann["category_id"]
        if cat_id == 356:  # skip unknown
            continue

        cat_counts[cat_id] = cat_counts.get(cat_id, 0) + 1
        if cat_counts[cat_id] > max_crops_per_cat:
            continue

        image_id = ann["image_id"]
        filename = id_to_file.get(image_id)
        if not filename:
            continue

        img_path = images_dir / filename
        if not img_path.exists():
            continue

        x, y, w, h = ann["bbox"]
        if w < 10 or h < 10:
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            # Add small padding
            pad = 5
            x1 = max(0, int(x) - pad)
            y1 = max(0, int(y) - pad)
            x2 = min(img.width, int(x + w) + pad)
            y2 = min(img.height, int(y + h) + pad)
            crop = img.crop((x1, y1, x2, y2))

            crop_name = f"cat{cat_id:03d}_{ann['id']:06d}.jpg"
            crop.save(crops_dir / crop_name, quality=90)
            crop_records.append({
                "filename": crop_name,
                "category_id": cat_id,
                "source": "coco_crop",
                "product_code": ann.get("product_code", ""),
            })
        except Exception:
            continue

    return crop_records


def prepare_reference_images(products_dir, code_to_cat, output_dir):
    """Prepare product reference images with category labels."""
    ref_dir = output_dir / "references"
    ref_dir.mkdir(parents=True, exist_ok=True)

    ref_records = []
    metadata_path = products_dir / "metadata.json"

    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        products = metadata.get("products", [])
    else:
        # Discover products from directory structure
        products = []
        for d in products_dir.iterdir():
            if d.is_dir() and d.name != "__pycache__":
                products.append({"product_code": d.name})

    for prod in tqdm(products, desc="Preparing reference images"):
        pc = str(prod["product_code"])
        cat_id = code_to_cat.get(pc)
        if cat_id is None:
            continue

        prod_dir = products_dir / pc
        if not prod_dir.exists():
            continue

        for img_file in prod_dir.iterdir():
            if img_file.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            try:
                img = Image.open(img_file).convert("RGB")
                ref_name = f"ref_cat{cat_id:03d}_{pc}_{img_file.stem}.jpg"
                img.save(ref_dir / ref_name, quality=90)
                ref_records.append({
                    "filename": ref_name,
                    "category_id": cat_id,
                    "source": "reference",
                    "product_code": pc,
                })
            except Exception:
                continue

    return ref_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", default="coco/train/annotations.json")
    parser.add_argument("--images", default="coco/train")
    parser.add_argument("--products", default="product")
    parser.add_argument("--output", default="classifier_data")
    parser.add_argument("--max-crops-per-cat", type=int, default=200)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    # Load annotations
    ann_path = Path(args.annotations)
    if ann_path.exists():
        print(f"Loading annotations from {ann_path}...")
        with open(ann_path) as f:
            annotations = json.load(f)

        # Build category mapping
        categories = {c["id"]: c["name"] for c in annotations["categories"]}
        code_to_cat = build_product_code_to_category(annotations)
        print(f"  {len(categories)} categories, {len(code_to_cat)} product codes mapped")

        # Save category mapping
        with open(output_dir / "categories.json", "w") as f:
            json.dump(categories, f, ensure_ascii=False, indent=2)

        # Extract crops from training images
        images_dir = Path(args.images)
        if images_dir.exists() and any(images_dir.iterdir()):
            crop_records = extract_training_crops(
                annotations, images_dir, output_dir, args.max_crops_per_cat
            )
            print(f"  Extracted {len(crop_records)} training crops")
        else:
            print(f"  Training images not found at {images_dir}")
            crop_records = []
    else:
        print(f"Annotations not found at {ann_path}")
        print("You need to download the COCO training data first!")
        code_to_cat = {}
        crop_records = []
        categories = {}

    # Prepare reference images
    products_dir = Path(args.products)
    if products_dir.exists():
        ref_records = prepare_reference_images(products_dir, code_to_cat, output_dir)
        print(f"  Prepared {len(ref_records)} reference images")
    else:
        ref_records = []

    # Save manifest
    all_records = crop_records + ref_records
    manifest = {
        "num_categories": len(categories),
        "num_crop_samples": len(crop_records),
        "num_reference_samples": len(ref_records),
        "total_samples": len(all_records),
        "samples": all_records,
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Stats
    cat_dist = {}
    for r in all_records:
        c = r["category_id"]
        cat_dist[c] = cat_dist.get(c, 0) + 1

    print(f"\n=== Dataset Summary ===")
    print(f"Total samples: {len(all_records)}")
    print(f"Categories with data: {len(cat_dist)}")
    if cat_dist:
        counts = list(cat_dist.values())
        print(f"Samples per category: min={min(counts)}, max={max(counts)}, "
              f"median={sorted(counts)[len(counts)//2]}")


if __name__ == "__main__":
    main()
