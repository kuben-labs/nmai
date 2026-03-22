"""
Balance COCO dataset by oversampling images containing rare classes.

Algorithm:
  1. Count instances per category
  2. While any category is below the target count:
     a. Pick the category with the fewest instances
     b. Find the image containing the most instances of that category
     c. Duplicate that image (copy file with new name, add new annotations)
     d. Update all category counts
  3. Write balanced annotations.json + copy/symlink images

Usage:
    python balance_dataset.py                          # target = median count
    python balance_dataset.py --target 50              # explicit target
    python balance_dataset.py --target median --dry-run  # preview only
"""

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path


COCO_DIR = Path(__file__).parent.parent / "coco" / "train"
OUT_DIR = Path(__file__).parent.parent / "coco" / "train_balanced"


def balance(coco, target, max_total_images=2000):
    """
    Returns a new COCO dict with duplicated images/annotations to balance classes.
    Images are not literally duplicated here — just new entries pointing to copies.
    """
    images = list(coco["images"])
    annotations = list(coco["annotations"])
    categories = coco["categories"]

    # Build indexes
    ann_by_image = defaultdict(list)
    for ann in annotations:
        ann_by_image[ann["image_id"]].append(ann)

    img_by_id = {img["id"]: img for img in images}

    # Current category counts
    cat_counts = Counter(ann["category_id"] for ann in annotations)

    # For each image, count instances per category
    img_cat_counts = {}
    for img_id, anns in ann_by_image.items():
        img_cat_counts[img_id] = Counter(a["category_id"] for a in anns)

    next_img_id = max(img["id"] for img in images) + 1
    next_ann_id = max(ann["id"] for ann in annotations) + 1

    new_images = []
    new_annotations = []
    total_duplicated = 0
    dup_count = Counter()  # track how many times each image was duplicated

    # Iteratively boost the rarest category
    while True:
        # Find categories still below target
        below = {cid: target - cnt for cid, cnt in cat_counts.items() if cnt < target}
        if not below:
            break

        if len(images) + len(new_images) >= max_total_images:
            print(f"  Stopping: reached max {max_total_images} total images")
            break

        # Pick the rarest category
        rarest_cat = min(below, key=lambda c: cat_counts[c])
        deficit = below[rarest_cat]

        # Find the image with the most instances of this category
        best_img_id = None
        best_count = 0
        for img_id, cc in img_cat_counts.items():
            if cc.get(rarest_cat, 0) > best_count:
                best_count = cc[rarest_cat]
                best_img_id = img_id

        if best_img_id is None or best_count == 0:
            # No image contains this category at all — skip
            cat_counts[rarest_cat] = target  # mark as done
            continue

        # How many times to duplicate this image to reach target for rarest_cat
        n_copies = min(
            (deficit + best_count - 1) // best_count,  # ceil division
            max_total_images - len(images) - len(new_images),
        )
        if n_copies <= 0:
            cat_counts[rarest_cat] = target
            continue

        src_img = img_by_id[best_img_id]
        src_anns = ann_by_image[best_img_id]
        src_stem = Path(src_img["file_name"]).stem
        src_ext = Path(src_img["file_name"]).suffix

        for i in range(n_copies):
            dup_count[best_img_id] += 1
            dup_idx = dup_count[best_img_id]

            new_file_name = f"{src_stem}_dup{dup_idx}{src_ext}"
            new_img = {
                "id": next_img_id,
                "file_name": new_file_name,
                "width": src_img["width"],
                "height": src_img["height"],
                "source_image_id": best_img_id,  # track provenance
            }
            new_images.append(new_img)
            img_by_id[next_img_id] = new_img

            new_anns_for_img = []
            for ann in src_anns:
                new_ann = {
                    "id": next_ann_id,
                    "image_id": next_img_id,
                    "category_id": ann["category_id"],
                    "bbox": list(ann["bbox"]),
                    "area": ann.get("area", ann["bbox"][2] * ann["bbox"][3]),
                    "iscrowd": 0,
                }
                new_annotations.append(new_ann)
                new_anns_for_img.append(new_ann)
                next_ann_id += 1

            ann_by_image[next_img_id] = new_anns_for_img
            img_cat_counts[next_img_id] = Counter(a["category_id"] for a in src_anns)

            # Update global counts
            for a in src_anns:
                cat_counts[a["category_id"]] += 1

            next_img_id += 1
            total_duplicated += 1

    balanced_coco = {
        "images": images + new_images,
        "annotations": annotations + new_annotations,
        "categories": categories,
    }
    return balanced_coco, total_duplicated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="median",
                        help="Target count per class: integer or 'median'/'p75' (default: median)")
    parser.add_argument("--max-images", type=int, default=2000,
                        help="Max total images including duplicates (default: 2000)")
    parser.add_argument("--out", type=str, default=str(OUT_DIR))
    parser.add_argument("--dry-run", action="store_true",
                        help="Only print stats, don't write files")
    parser.add_argument("--symlink", action="store_true",
                        help="Use symlinks instead of copying images")
    args = parser.parse_args()

    ann_path = COCO_DIR / "annotations.json"
    print(f"Reading {ann_path} ...")
    with open(ann_path) as f:
        coco = json.load(f)

    cat_counts = Counter(a["category_id"] for a in coco["annotations"])
    counts = sorted(cat_counts.values())

    # Determine target
    if args.target == "median":
        target = counts[len(counts) // 2]
    elif args.target == "p75":
        target = counts[int(len(counts) * 0.75)]
    else:
        target = int(args.target)

    print(f"\nOriginal stats:")
    print(f"  Images: {len(coco['images'])}")
    print(f"  Annotations: {len(coco['annotations'])}")
    print(f"  Categories: {len(coco['categories'])}")
    print(f"  Min/Median/Max counts: {counts[0]} / {counts[len(counts)//2]} / {counts[-1]}")
    print(f"  Target count: {target}")
    print(f"  Categories below target: {sum(1 for c in counts if c < target)}")

    balanced, n_dup = balance(coco, target, args.max_images)

    new_counts = Counter(a["category_id"] for a in balanced["annotations"])
    new_vals = sorted(new_counts.values())

    print(f"\nBalanced stats:")
    print(f"  Images: {len(balanced['images'])} (+{n_dup} duplicated)")
    print(f"  Annotations: {len(balanced['annotations'])}")
    print(f"  Min/Median/Max counts: {new_vals[0]} / {new_vals[len(new_vals)//2]} / {new_vals[-1]}")
    print(f"  Categories still below target: {sum(1 for c in new_vals if c < target)}")

    if args.dry_run:
        print("\nDry run — no files written.")
        return

    out_dir = Path(args.out)
    img_out = out_dir / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    # Copy/symlink images
    src_images = COCO_DIR / "images"
    copied, linked = 0, 0

    for img in balanced["images"]:
        dst = img_out / img["file_name"]
        if dst.exists():
            continue

        if "source_image_id" in img:
            # This is a duplicate — find source file
            src_img_id = img["source_image_id"]
            src_file = None
            for orig in coco["images"]:
                if orig["id"] == src_img_id:
                    src_file = orig["file_name"]
                    break
            src_path = src_images / src_file
        else:
            src_path = src_images / img["file_name"]

        if args.symlink:
            dst.symlink_to(src_path.resolve())
            linked += 1
        else:
            shutil.copy2(src_path, dst)
            copied += 1

    # Write balanced annotations
    ann_out = out_dir / "annotations.json"
    with open(ann_out, "w") as f:
        json.dump(balanced, f, ensure_ascii=False)

    print(f"\nWritten to {out_dir}")
    print(f"  Images: {copied} copied, {linked} linked")
    print(f"  Annotations: {ann_out}")


if __name__ == "__main__":
    main()
