"""Convert COCO-format annotations to YOLO format with train/val split.

Usage:
    python scripts/coco_to_yolo.py \
        --annotations data/raw/train/annotations.json \
        --images data/raw/train/images \
        --output data/yolo \
        --val-ratio 0.15
"""

import argparse
import json
import random
from pathlib import Path


def coco_to_yolo_bbox(
    bbox: list[float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """Convert COCO bbox [x, y, w, h] to YOLO [cx, cy, w, h] normalized."""
    x, y, w, h = bbox
    cx = (x + w / 2) / image_width
    cy = (y + h / 2) / image_height
    nw = w / image_width
    nh = h / image_height

    # Clamp to [0, 1]
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    nw = max(0.0, min(1.0, nw))
    nh = max(0.0, min(1.0, nh))

    return cx, cy, nw, nh


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert COCO to YOLO format")
    parser.add_argument(
        "--annotations",
        type=str,
        default="data/raw/train/annotations.json",
        help="Path to COCO annotations JSON",
    )
    parser.add_argument(
        "--images",
        type=str,
        default="data/raw/train/images",
        help="Path to images directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/yolo",
        help="Output directory for YOLO dataset",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Fraction of images for validation (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible split",
    )
    args = parser.parse_args()

    annotations_path = Path(args.annotations)
    images_dir = Path(args.images)
    output_dir = Path(args.output)

    # Load COCO annotations
    print(f"Loading annotations from {annotations_path}")
    with open(annotations_path) as f:
        coco_data = json.load(f)

    images = coco_data["images"]
    annotations = coco_data["annotations"]
    categories = coco_data["categories"]

    print(f"  Images: {len(images)}")
    print(f"  Annotations: {len(annotations)}")
    print(f"  Categories: {len(categories)}")

    # Build lookup: image_id -> image info
    image_lookup = {img["id"]: img for img in images}

    # Build lookup: image_id -> list of annotations
    image_annotations: dict[int, list[dict]] = {}
    for ann in annotations:
        image_id = ann["image_id"]
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        image_annotations[image_id].append(ann)

    # Train/val split
    random.seed(args.seed)
    image_ids = sorted([img["id"] for img in images])
    random.shuffle(image_ids)
    val_count = max(1, int(len(image_ids) * args.val_ratio))
    val_ids = set(image_ids[:val_count])
    train_ids = set(image_ids[val_count:])

    print(f"  Train images: {len(train_ids)}")
    print(f"  Val images: {len(val_ids)}")

    # Create output directories
    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Process each image
    skipped = 0
    total_labels = 0
    for image_id in image_ids:
        img_info = image_lookup[image_id]
        file_name = img_info["file_name"]
        width = img_info["width"]
        height = img_info["height"]

        # Determine split
        split = "val" if image_id in val_ids else "train"

        # Find source image
        src_image = images_dir / file_name
        if not src_image.exists():
            print(f"  WARNING: Image not found: {src_image}")
            skipped += 1
            continue

        # Create symlink to image (saves disk space)
        dst_image = output_dir / "images" / split / file_name
        if dst_image.is_symlink() or dst_image.exists():
            dst_image.unlink()  # Remove stale/broken symlinks
        dst_image.symlink_to(src_image.resolve())

        # Write YOLO label file
        label_name = Path(file_name).stem + ".txt"
        label_path = output_dir / "labels" / split / label_name

        anns = image_annotations.get(image_id, [])
        lines = []
        for ann in anns:
            category_id = ann["category_id"]
            bbox = ann["bbox"]

            # Skip zero-area boxes
            if bbox[2] <= 0 or bbox[3] <= 0:
                continue

            cx, cy, nw, nh = coco_to_yolo_bbox(bbox, width, height)

            # Skip degenerate boxes
            if nw <= 0 or nh <= 0:
                continue

            lines.append(f"{category_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        with open(label_path, "w") as f:
            f.write("\n".join(lines))
            if lines:
                f.write("\n")

        total_labels += len(lines)

    print(f"\nDone! Output: {output_dir}")
    print(f"  Total label entries: {total_labels}")
    print(f"  Skipped images: {skipped}")

    # Save category names for reference
    names_path = output_dir / "category_names.json"
    cat_names = {cat["id"]: cat["name"] for cat in categories}
    with open(names_path, "w") as f:
        json.dump(cat_names, f, indent=2, ensure_ascii=False)
    print(f"  Category names saved to: {names_path}")

    # Generate dataset.yaml with absolute path for this machine
    configs_dir = Path("configs")
    configs_dir.mkdir(exist_ok=True)
    yaml_path = configs_dir / "dataset.yaml"

    yaml_lines = []
    yaml_lines.append(f"path: {output_dir.resolve()}")
    yaml_lines.append("train: images/train")
    yaml_lines.append("val: images/val")
    yaml_lines.append("")
    yaml_lines.append(f"nc: {len(categories)}")
    yaml_lines.append("")
    yaml_lines.append("names:")
    for cat in sorted(categories, key=lambda c: c["id"]):
        name = cat["name"].replace("'", "''")
        yaml_lines.append(f"  {cat['id']}: '{name}'")

    with open(yaml_path, "w") as f:
        f.write("\n".join(yaml_lines) + "\n")
    print(f"  Dataset config saved to: {yaml_path}")


if __name__ == "__main__":
    main()
