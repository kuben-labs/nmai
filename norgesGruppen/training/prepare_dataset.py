"""
Prepare NorgesGruppen COCO dataset for YOLOv8 training.

Reads from the already-extracted coco/train/ directory, splits into
train/val (90/10), and converts COCO JSON annotations to YOLO .txt format.

YOLO label format per line:
  <class_id> <x_center> <y_center> <width> <height>   (all normalized 0-1)
"""

import json
import random
import shutil
from pathlib import Path


COCO_DIR = Path(__file__).parent.parent / "coco" / "train"
OUT_DIR = Path(__file__).parent / "dataset"
VAL_RATIO = 0.1
SEED = 42


def coco_bbox_to_yolo(bbox, img_w, img_h):
    """Convert COCO [x, y, w, h] to YOLO [cx, cy, w, h] normalized."""
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return cx, cy, nw, nh


def main():
    raw_images = COCO_DIR / "images"
    ann_path = COCO_DIR / "annotations.json"

    print(f"Reading annotations from {ann_path} ...")

    with open(ann_path) as f:
        coco = json.load(f)

    # Build lookup structures
    img_info = {img["id"]: img for img in coco["images"]}
    cat_id_to_idx = {cat["id"]: i for i, cat in enumerate(coco["categories"])}

    # Group annotations by image
    ann_by_image: dict[int, list] = {}
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        ann_by_image.setdefault(ann["image_id"], []).append(ann)

    # Only keep images that actually exist on disk
    available_ids = [
        img_id for img_id, img in img_info.items()
        if (raw_images / img["file_name"]).exists()
    ]

    print(f"Found {len(available_ids)} images with annotations")

    # Train / val split
    random.seed(SEED)
    random.shuffle(available_ids)
    n_val = max(1, int(len(available_ids) * VAL_RATIO))
    val_ids = set(available_ids[:n_val])
    train_ids = set(available_ids[n_val:])
    print(f"Split: {len(train_ids)} train / {len(val_ids)} val")

    for split, ids in [("train", train_ids), ("val", val_ids)]:
        img_out = OUT_DIR / "images" / split
        lbl_out = OUT_DIR / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_id in ids:
            img = img_info[img_id]
            src = raw_images / img["file_name"]
            dst = img_out / img["file_name"]
            shutil.copy2(src, dst)

            anns = ann_by_image.get(img_id, [])
            label_lines = []
            for ann in anns:
                cls = cat_id_to_idx[ann["category_id"]]
                cx, cy, nw, nh = coco_bbox_to_yolo(
                    ann["bbox"], img["width"], img["height"]
                )
                # clamp to [0, 1]
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                nw = max(0.0, min(1.0, nw))
                nh = max(0.0, min(1.0, nh))
                label_lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            stem = Path(img["file_name"]).stem
            (lbl_out / f"{stem}.txt").write_text("\n".join(label_lines))

    # Write dataset.yaml using double-quoted strings to handle special chars
    class_names = [cat["name"] for cat in coco["categories"]]
    yaml_content = f"""\
path: {OUT_DIR.resolve()}
train: images/train
val: images/val

nc: {len(class_names)}
names:
"""
    for name in class_names:
        # Use double quotes, escape any double quotes inside the name
        safe = name.replace("\\", "\\\\").replace('"', '\\"')
        yaml_content += f'  - "{safe}"\n'

    yaml_path = OUT_DIR / "dataset.yaml"
    yaml_path.write_text(yaml_content)

    # Save category mapping for inference
    cat_map = {i: cat for i, cat in enumerate(coco["categories"])}
    with open(OUT_DIR / "categories.json", "w") as f:
        json.dump(cat_map, f, indent=2, ensure_ascii=False)

    print(f"\nDataset ready at: {OUT_DIR}")
    print(f"  dataset.yaml : {yaml_path}")
    print(f"  classes      : {len(class_names)}")


if __name__ == "__main__":
    main()
