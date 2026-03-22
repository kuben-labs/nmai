"""
Prepare NorgesGruppen COCO dataset for YOLO11 training.

Improvements over v1:
  - Multi-scale tiling (1280 + 640) for better scale variety
  - Higher overlap (30%) to reduce missed annotations at tile edges
  - Stratified train/val split ensuring rare categories appear in both
  - Optional negative tiles (empty) to reduce false positives
  - Class balancing via oversampling (--balance)

Output:
    dataset_tiled/images/train/   (full images + tiles)
    dataset_tiled/labels/train/
    dataset_tiled/images/val/     (full images only — clean evaluation)
    dataset_tiled/labels/val/
    dataset_tiled/dataset.yaml

Usage:
    python prepare_dataset.py
    python prepare_dataset.py --tile-sizes 1280,960 --overlap 0.3
    python prepare_dataset.py --balance median
"""

import argparse
import json
import os
import random
import shutil
from collections import Counter, defaultdict
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

        if not (tx <= cx < tx + tile_w and ty <= cy < ty + tile_h):
            continue

        new_x = max(0, x - tx)
        new_y = max(0, y - ty)
        new_x2 = min(tile_w, x + w - tx)
        new_y2 = min(tile_h, y + h - ty)
        new_w = new_x2 - new_x
        new_h = new_y2 - new_y

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


def stratified_split(img_ids, ann_by_image, val_ratio, seed):
    """
    Split images into train/val ensuring rare categories appear in both sets.
    Uses a greedy approach: first assign images with rare-only categories to
    both sets, then split the rest randomly.
    """
    rng = random.Random(seed)

    # Count category frequency across images
    cat_to_images = defaultdict(set)
    for img_id in img_ids:
        for ann in ann_by_image.get(img_id, []):
            cat_to_images[ann["category_id"]].add(img_id)

    n_val = max(1, int(len(img_ids) * val_ratio))

    # Identify images containing rare categories (<=3 images total)
    rare_images = set()
    for cat_id, images in cat_to_images.items():
        if len(images) <= 3:
            rare_images.update(images)

    # For rare-category images, ensure at least some go to val
    rare_list = sorted(rare_images)
    rng.shuffle(rare_list)

    val_ids = set()
    train_ids = set()

    # Put ~val_ratio of rare images in val
    n_rare_val = max(1, int(len(rare_list) * val_ratio))
    for img_id in rare_list[:n_rare_val]:
        val_ids.add(img_id)
    for img_id in rare_list[n_rare_val:]:
        train_ids.add(img_id)

    # Distribute remaining images
    remaining = [i for i in img_ids if i not in val_ids and i not in train_ids]
    rng.shuffle(remaining)

    n_val_remaining = n_val - len(val_ids)
    for img_id in remaining[:max(0, n_val_remaining)]:
        val_ids.add(img_id)
    for img_id in remaining[max(0, n_val_remaining):]:
        train_ids.add(img_id)

    return train_ids, val_ids


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tile-sizes", default="1280",
                   help="Comma-separated tile sizes (default: 1280)")
    p.add_argument("--overlap", type=float, default=0.3,
                   help="Tile overlap ratio (default: 0.3)")
    p.add_argument("--out", type=str, default=str(OUT_DIR))
    p.add_argument("--coco-dir", type=str, default=str(COCO_DIR),
                   help="Path to COCO dataset dir (default: coco/train)")
    p.add_argument("--include-negatives", action="store_true",
                   help="Include some empty tiles as negative examples")
    p.add_argument("--neg-ratio", type=float, default=0.05,
                   help="Ratio of negative tiles to keep (default: 0.05)")
    p.add_argument("--balance", default=None,
                   help="Oversample rare classes. Target: 'median', 'p75', or integer count")
    p.add_argument("--max-oversampled", type=int, default=5000,
                   help="Max total train images after oversampling (default: 5000)")
    return p.parse_args()


def main():
    args = parse_args()
    tile_sizes = [int(s) for s in args.tile_sizes.split(",")]
    overlap = args.overlap
    out_dir = Path(args.out)

    coco_dir = Path(args.coco_dir)
    raw_images = coco_dir / "images"
    ann_path = coco_dir / "annotations.json"

    print(f"Reading annotations from {ann_path} ...")
    with open(ann_path) as f:
        coco = json.load(f)

    img_info = {img["id"]: img for img in coco["images"]}

    ann_by_image = {}
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        ann_by_image.setdefault(ann["image_id"], []).append(ann)

    available_ids = [
        img_id for img_id, img in img_info.items()
        if (raw_images / img["file_name"]).exists()
    ]

    # Stratified split for better rare-category coverage
    train_ids, val_ids = stratified_split(
        available_ids, ann_by_image, VAL_RATIO, SEED
    )

    print(f"Images: {len(available_ids)} ({len(train_ids)} train / {len(val_ids)} val)")
    print(f"Tile sizes: {tile_sizes}, overlap: {overlap}")

    rng = random.Random(SEED)
    total_tiles = 0
    total_full = 0
    total_neg = 0

    for split, ids in [("train", train_ids), ("val", val_ids)]:
        img_out = out_dir / "images" / split
        lbl_out = out_dir / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for idx, img_id in enumerate(sorted(ids)):
            if idx % 50 == 0:
                print(f"  {split}: {idx}/{len(ids)} images processed ...")
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

            # 2. Create tiles (training split only, for each tile size)
            if split != "train":
                continue

            pil_img = None

            for tile_size in tile_sizes:
                if iw <= tile_size and ih <= tile_size:
                    continue

                if pil_img is None:
                    pil_img = Image.open(src_path)

                positions = get_tile_positions(iw, ih, tile_size, overlap)

                if len(positions) <= 1:
                    continue

                for tx, ty in positions:
                    tw = min(tile_size, iw - tx)
                    th = min(tile_size, ih - ty)

                    tile_anns = annotations_in_tile(anns, tx, ty, tw, th)

                    if not tile_anns:
                        # Optionally keep some empty tiles as negative examples
                        if args.include_negatives and rng.random() < args.neg_ratio:
                            tile_name = f"{stem}_t{tile_size}_{tx}_{ty}"
                            tile = pil_img.crop((tx, ty, tx + tw, ty + th))
                            tile.save(img_out / f"{tile_name}.jpg", quality=95)
                            (lbl_out / f"{tile_name}.txt").write_text("")
                            total_neg += 1
                        continue

                    tile_name = f"{stem}_t{tile_size}_{tx}_{ty}"
                    tile = pil_img.crop((tx, ty, tx + tw, ty + th))
                    tile.save(img_out / f"{tile_name}.jpg", quality=95)

                    write_yolo_labels(tile_anns, tw, th,
                                      lbl_out / f"{tile_name}.txt")
                    total_tiles += 1

            if pil_img is not None:
                pil_img.close()

    # ── Class balancing via oversampling ─────────────────────────────
    if args.balance:
        img_out = out_dir / "images" / "train"
        lbl_out = out_dir / "labels" / "train"

        # Collect all train label files and their class counts
        train_labels = sorted(lbl_out.glob("*.txt"))
        file_classes = {}  # filename_stem -> list of category_ids
        cat_counts = Counter()
        for lbl_path in train_labels:
            cats = []
            for line in lbl_path.read_text().strip().split("\n"):
                if line.strip():
                    cat_id = int(line.split()[0])
                    cats.append(cat_id)
                    cat_counts[cat_id] += 1
            file_classes[lbl_path.stem] = cats

        # Determine target
        counts = sorted(cat_counts.values())
        if args.balance == "median":
            target = counts[len(counts) // 2]
        elif args.balance == "p75":
            target = counts[int(len(counts) * 0.75)]
        else:
            target = int(args.balance)

        below = {cid: target - cnt for cid, cnt in cat_counts.items() if cnt < target}
        print(f"\nOversampling: target={target}, {len(below)} categories below target")

        # For each file, count how many instances of each below-target class it has
        file_rare_counts = {}
        for stem, cats in file_classes.items():
            rare_count = sum(1 for c in cats if c in below)
            if rare_count > 0:
                file_rare_counts[stem] = Counter(c for c in cats if c in below)

        dup_id = 0
        n_oversampled = 0
        max_dup = args.max_oversampled - len(train_labels)

        while below and n_oversampled < max_dup:
            # Pick the rarest category
            rarest = min(below, key=lambda c: cat_counts[c])

            # Find the file with the most instances of this category
            best_stem = None
            best_count = 0
            for stem, rare_cc in file_rare_counts.items():
                if rare_cc.get(rarest, 0) > best_count:
                    best_count = rare_cc[rarest]
                    best_stem = stem

            if best_stem is None or best_count == 0:
                del below[rarest]
                continue

            # How many copies needed
            n_copies = min(
                (below[rarest] + best_count - 1) // best_count,
                max_dup - n_oversampled,
            )

            src_img_candidates = [
                p for p in img_out.iterdir()
                if p.stem == best_stem and p.suffix in ('.jpg', '.jpeg', '.png')
            ]
            if not src_img_candidates:
                del below[rarest]
                continue

            src_img_path = src_img_candidates[0]
            src_lbl_path = lbl_out / f"{best_stem}.txt"

            for i in range(n_copies):
                dup_id += 1
                dup_stem = f"{best_stem}_os{dup_id}"
                # Symlink image, copy label
                dst_img = img_out / f"{dup_stem}{src_img_path.suffix}"
                dst_lbl = lbl_out / f"{dup_stem}.txt"
                os.symlink(src_img_path.resolve(), dst_img)
                shutil.copy2(src_lbl_path, dst_lbl)
                n_oversampled += 1

            # Update counts
            for cat_id in file_classes[best_stem]:
                cat_counts[cat_id] += n_copies
            below = {cid: target - cnt for cid, cnt in cat_counts.items() if cnt < target}

        new_total = len(train_labels) + n_oversampled
        print(f"  Added {n_oversampled} oversampled images (symlinks)")
        print(f"  Total train images: {new_total}")
        new_counts = sorted(cat_counts.values())
        print(f"  Min/Median/Max class counts: {new_counts[0]} / {new_counts[len(new_counts)//2]} / {new_counts[-1]}")

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

    cat_map = {i: cat for i, cat in enumerate(coco["categories"])}
    with open(out_dir / "categories.json", "w") as f:
        json.dump(cat_map, f, indent=2, ensure_ascii=False)

    train_images = total_full - len(val_ids) + total_tiles + total_neg
    print(f"\nDataset written to {out_dir}")
    print(f"  Full images:    {total_full} ({total_full - len(val_ids)} train + {len(val_ids)} val)")
    print(f"  Tiles:          {total_tiles}")
    if total_neg:
        print(f"  Negative tiles: {total_neg}")
    print(f"  Total train:    {train_images} images")
    print(f"  dataset.yaml:   {out_dir / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
