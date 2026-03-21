"""
Create a tiled + full-image training dataset from COCO annotations.

Tiles large shelf images into overlapping 1280x1280 crops, mapping
annotations to each tile.  Includes full images alongside tiles so
YOLO learns both global context and fine-grained detail.

This effectively ~10x multiplies the training data AND lets YOLO see
products at near-native resolution (instead of 2x downscaled).

Output:
    dataset_tiled/images/train/   (full images + tiles)
    dataset_tiled/labels/train/
    dataset_tiled/images/val/     (full images only)
    dataset_tiled/labels/val/
    dataset_tiled/dataset.yaml

Usage:
    python prepare_tiled_dataset.py
    python prepare_tiled_dataset.py --tile-size 1280 --overlap 0.25
"""

import argparse
import json
import random
import shutil
from pathlib import Path

from PIL import Image


COCO_DIR = Path(__file__).parent.parent / "coco" / "train"
OUT_DIR = Path(__file__).parent / "dataset_tiled"
VAL_RATIO = 0.1
SEED = 42


def coco_bbox_to_yolo(bbox, img_w, img_h):
    """Convert COCO [x, y, w, h] to YOLO [cx, cy, w, h] normalized."""
    x, y, w, h = bbox
    cx = max(0.0, min(1.0, (x + w / 2) / img_w))
    cy = max(0.0, min(1.0, (y + h / 2) / img_h))
    nw = max(0.0, min(1.0, w / img_w))
    nh = max(0.0, min(1.0, h / img_h))
    return cx, cy, nw, nh


def get_tile_positions(img_w, img_h, tile_size, overlap_ratio):
    """Generate (x, y) top-left positions for overlapping tiles."""
    stride = int(tile_size * (1 - overlap_ratio))

    def positions(dim):
        if dim <= tile_size:
            return [0]
        pos = list(range(0, dim - tile_size + 1, stride))
        # Ensure last tile reaches the edge
        if pos[-1] + tile_size < dim:
            pos.append(dim - tile_size)
        return sorted(set(pos))

    return [(x, y) for y in positions(img_h) for x in positions(img_w)]


def annotations_in_tile(anns, tx, ty, tile_w, tile_h, min_area=100):
    """
    Get annotations whose CENTER falls within the tile.
    Returns clipped COCO-format bboxes relative to the tile.
    """
    results = []
    for ann in anns:
        x, y, w, h = ann["bbox"]
        cx, cy = x + w / 2, y + h / 2

        # Center must be in tile
        if not (tx <= cx < tx + tile_w and ty <= cy < ty + tile_h):
            continue

        # Clip bbox to tile boundaries
        new_x = max(0, x - tx)
        new_y = max(0, y - ty)
        new_x2 = min(tile_w, x + w - tx)
        new_y2 = min(tile_h, y + h - ty)
        new_w = new_x2 - new_x
        new_h = new_y2 - new_y

        # Skip tiny clipped boxes
        if new_w * new_h < min_area or new_w < 5 or new_h < 5:
            continue

        results.append({
            "category_id": ann["category_id"],
            "bbox": [new_x, new_y, new_w, new_h],
        })
    return results


def write_yolo_labels(anns, img_w, img_h, label_path):
    """Write YOLO-format label file."""
    lines = []
    for ann in anns:
        cx, cy, nw, nh = coco_bbox_to_yolo(ann["bbox"], img_w, img_h)
        lines.append(f"{ann['category_id']} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    label_path.write_text("\n".join(lines))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tile-size", type=int, default=1280,
                   help="Tile size in pixels (default: 1280)")
    p.add_argument("--overlap", type=float, default=0.25,
                   help="Tile overlap ratio 0.0-0.5 (default: 0.25)")
    p.add_argument("--out", type=str, default=str(OUT_DIR),
                   help="Output directory")
    return p.parse_args()


def main():
    args = parse_args()
    tile_size = args.tile_size
    overlap = args.overlap
    out_dir = Path(args.out)

    raw_images = COCO_DIR / "images"
    ann_path = COCO_DIR / "annotations.json"

    print(f"Reading annotations from {ann_path} ...")
    with open(ann_path) as f:
        coco = json.load(f)

    img_info = {img["id"]: img for img in coco["images"]}

    # Group annotations by image
    ann_by_image = {}
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        ann_by_image.setdefault(ann["image_id"], []).append(ann)

    # Only keep images that exist on disk
    available_ids = [
        img_id for img_id, img in img_info.items()
        if (raw_images / img["file_name"]).exists()
    ]

    # Same train/val split as prepare_dataset.py (same seed!)
    random.seed(SEED)
    random.shuffle(available_ids)
    n_val = max(1, int(len(available_ids) * VAL_RATIO))
    val_ids = set(available_ids[:n_val])
    train_ids = set(available_ids[n_val:])

    print(f"Images: {len(available_ids)} ({len(train_ids)} train / {len(val_ids)} val)")
    print(f"Tile size: {tile_size}, overlap: {overlap}")

    total_tiles = 0
    total_full = 0
    total_tile_anns = 0

    for split, ids in [("train", train_ids), ("val", val_ids)]:
        img_out = out_dir / "images" / split
        lbl_out = out_dir / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_id in sorted(ids):
            img = img_info[img_id]
            src_path = raw_images / img["file_name"]
            stem = Path(img["file_name"]).stem
            iw, ih = img["width"], img["height"]
            anns = ann_by_image.get(img_id, [])

            # 1. Always include full image
            shutil.copy2(src_path, img_out / img["file_name"])
            full_anns = [{"category_id": a["category_id"], "bbox": a["bbox"]}
                         for a in anns]
            write_yolo_labels(full_anns, iw, ih, lbl_out / f"{stem}.txt")
            total_full += 1

            # 2. Create tiles (training split only, large images only)
            if split == "train" and (iw > tile_size or ih > tile_size):
                pil_img = Image.open(src_path)
                positions = get_tile_positions(iw, ih, tile_size, overlap)

                # Skip if only 1 tile (same as full image)
                if len(positions) <= 1:
                    pil_img.close()
                    continue

                for tx, ty in positions:
                    tw = min(tile_size, iw - tx)
                    th = min(tile_size, ih - ty)

                    tile_anns = annotations_in_tile(anns, tx, ty, tw, th)

                    # Skip empty tiles
                    if not tile_anns:
                        continue

                    # Crop and save tile
                    tile_name = f"{stem}_t{tx}_{ty}"
                    tile = pil_img.crop((tx, ty, tx + tw, ty + th))
                    tile.save(img_out / f"{tile_name}.jpg", quality=95)

                    write_yolo_labels(tile_anns, tw, th,
                                      lbl_out / f"{tile_name}.txt")
                    total_tiles += 1
                    total_tile_anns += len(tile_anns)

                pil_img.close()

    # Write dataset.yaml
    class_names = [cat["name"] for cat in coco["categories"]]
    yaml_lines = [
        f"path: {out_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        "",
        f"nc: {len(class_names)}",
        "names:",
    ]
    for name in class_names:
        safe = name.replace("\\", "\\\\").replace('"', '\\"')
        yaml_lines.append(f'  - "{safe}"')

    (out_dir / "dataset.yaml").write_text("\n".join(yaml_lines) + "\n")

    # Save category mapping for reference
    cat_map = {i: cat for i, cat in enumerate(coco["categories"])}
    with open(out_dir / "categories.json", "w") as f:
        json.dump(cat_map, f, indent=2, ensure_ascii=False)

    train_images = total_full - len(val_ids) + total_tiles
    print(f"\nDataset written to {out_dir}")
    print(f"  Full images:   {total_full} ({total_full - len(val_ids)} train + {len(val_ids)} val)")
    print(f"  Tiles:         {total_tiles} (with {total_tile_anns} annotations)")
    print(f"  Total train:   {train_images} images")
    print(f"  dataset.yaml:  {out_dir / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
