"""Build classification dataset from product images + COCO crop augmentation."""
import argparse
import json
import shutil
import random
from pathlib import Path
from PIL import Image

MIN_CROP_SIZE = 20


def build_product_code_to_category(coco_data, product_dir):
    with open(product_dir / "metadata.json") as f:
        meta = json.load(f)

    name_to_code = {}
    for p in meta["products"]:
        name_to_code[p["product_name"]] = p["product_code"]

    cat_id_to_code = {}
    for cat in coco_data["categories"]:
        code = name_to_code.get(cat["name"])
        if code:
            cat_id_to_code[cat["id"]] = code
    return cat_id_to_code


def crop_products_from_coco(coco_data, cat_id_to_code, coco_dir, output_dir):
    images = {img["id"]: img for img in coco_data["images"]}

    # Group annotations by image to avoid reopening the same file
    anns_by_img = {}
    for ann in coco_data["annotations"]:
        if cat_id_to_code.get(ann["category_id"]):
            anns_by_img.setdefault(ann["image_id"], []).append(ann)

    count = 0
    total_imgs = len(anns_by_img)
    for i, (img_id, anns) in enumerate(anns_by_img.items()):
        img_info = images[img_id]
        img_path = coco_dir / "images" / img_info["file_name"]
        if not img_path.exists():
            continue

        try:
            img = Image.open(img_path)
            img.load()  # force read into memory once
        except Exception:
            continue

        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w < MIN_CROP_SIZE or h < MIN_CROP_SIZE:
                continue

            code = cat_id_to_code[ann["category_id"]]
            cls_dir = output_dir / code
            cls_dir.mkdir(parents=True, exist_ok=True)

            try:
                crop = img.crop((x, y, x + w, y + h))
                crop_path = cls_dir / f"crop_{ann['id']:06d}.jpg"
                crop.save(crop_path, quality=90)
                count += 1
            except Exception:
                continue

        if (i + 1) % 25 == 0:
            print(f"  Cropping: {i+1}/{total_imgs} images processed, {count} crops so far")

    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--product-dir", default="product", help="Path to product images dir")
    parser.add_argument("--coco-dir", default="coco/train", help="Path to COCO train dir")
    parser.add_argument("--output-dir", default="cls_dataset", help="Output classification dataset dir")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    product_dir = Path(args.product_dir)
    coco_dir = Path(args.coco_dir)
    output_dir = Path(args.output_dir)

    random.seed(args.seed)

    with open(coco_dir / "annotations.json") as f:
        coco_data = json.load(f)
    cat_id_to_code = build_product_code_to_category(coco_data, product_dir)

    # Step 1: Copy product reference images
    with open(product_dir / "metadata.json") as f:
        meta = json.load(f)

    total_ref = 0
    for p in meta["products"]:
        code = p["product_code"]
        src_dir = product_dir / code
        if not src_dir.exists():
            continue

        dst_dir = output_dir / "all" / code
        dst_dir.mkdir(parents=True, exist_ok=True)
        for img_file in src_dir.iterdir():
            if img_file.suffix.lower() in (".jpg", ".jpeg", ".png"):
                shutil.copy2(img_file, dst_dir / img_file.name)
                total_ref += 1

    print(f"Copied {total_ref} reference product images")

    # Step 2: Crop products from shelf images
    crop_count = crop_products_from_coco(
        coco_data, cat_id_to_code, coco_dir, output_dir / "all"
    )
    print(f"Cropped {crop_count} product images from shelf photos")

    # Step 3: Split into train/val
    all_dir = output_dir / "all"
    for split in ("train", "val"):
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    # Only include classes that have at least one valid image
    classes = []
    for d in sorted(all_dir.iterdir()):
        if not d.is_dir():
            continue
        has_images = any(f.suffix.lower() in (".jpg", ".jpeg", ".png") for f in d.iterdir())
        if has_images:
            classes.append(d.name)
        else:
            shutil.rmtree(d)
            print(f"  Skipped empty class: {d.name}")
    class_stats = []

    for cls in classes:
        cls_dir = all_dir / cls
        imgs = [f for f in cls_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        random.shuffle(imgs)

        # Need at least 2 images to split; skip classes with too few
        if len(imgs) < 2:
            print(f"  Skipped class with <2 images: {cls} ({len(imgs)} images)")
            continue

        n_val = max(1, int(len(imgs) * args.val_ratio))
        # Ensure at least 1 in train
        n_val = min(n_val, len(imgs) - 1)
        val_imgs = imgs[:n_val]
        train_imgs = imgs[n_val:]

        for split, split_imgs in [("train", train_imgs), ("val", val_imgs)]:
            dst = output_dir / split / cls
            dst.mkdir(parents=True, exist_ok=True)
            for img_path in split_imgs:
                shutil.copy2(img_path, dst / img_path.name)

        class_stats.append((cls, len(train_imgs), len(val_imgs)))

    class_map = {i: cls for i, cls in enumerate(classes)}
    with open(output_dir / "class_map.json", "w") as f:
        json.dump(class_map, f, indent=2)

    total_train = sum(s[1] for s in class_stats)
    total_val = sum(s[2] for s in class_stats)
    print(f"\nClasses: {len(classes)}")
    print(f"Train: {total_train}, Val: {total_val}")
    print(f"Class map written to {output_dir / 'class_map.json'}")


if __name__ == "__main__":
    main()
