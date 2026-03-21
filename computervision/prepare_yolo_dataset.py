"""Convert COCO annotations to YOLO format with all 357 product categories."""
import argparse
import json
import shutil
import random
from pathlib import Path


def coco_to_yolo_bbox(bbox, img_w, img_h):
    """Convert COCO bbox [x,y,w,h] to YOLO [cx,cy,w,h] normalized."""
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return cx, cy, nw, nh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-dir", default="coco/train", help="Path to COCO train dir")
    parser.add_argument("--output-dir", default="yolo_dataset", help="Output YOLO dataset dir")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    coco_dir = Path(args.coco_dir)
    output_dir = Path(args.output_dir)

    with open(coco_dir / "annotations.json") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}
    num_classes = max(categories.keys()) + 1

    print(f"Categories: {len(categories)} (max_id={max(categories.keys())}, nc={num_classes})")

    # Group annotations by image
    ann_by_img = {}
    for ann in coco["annotations"]:
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    # Split into train/val
    img_ids = list(images.keys())
    random.seed(args.seed)
    random.shuffle(img_ids)
    n_val = max(1, int(len(img_ids) * args.val_ratio))
    val_ids = set(img_ids[:n_val])
    train_ids = set(img_ids[n_val:])

    for split, ids in [("train", train_ids), ("val", val_ids)]:
        img_dir = output_dir / "images" / split
        lbl_dir = output_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_id in ids:
            img_info = images[img_id]
            fname = img_info["file_name"]
            src = coco_dir / "images" / fname
            if not src.exists():
                continue
            shutil.copy2(src, img_dir / fname)

            anns = ann_by_img.get(img_id, [])
            label_path = lbl_dir / (Path(fname).stem + ".txt")
            with open(label_path, "w") as lf:
                for ann in anns:
                    cat_id = ann["category_id"]
                    cx, cy, nw, nh = coco_to_yolo_bbox(
                        ann["bbox"], img_info["width"], img_info["height"]
                    )
                    cx = max(0, min(1, cx))
                    cy = max(0, min(1, cy))
                    nw = max(0, min(1, nw))
                    nh = max(0, min(1, nh))
                    lf.write(f"{cat_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

    # Build names as JSON dict (safe from YAML special char issues)
    names_dict = {cat_id: categories[cat_id] for cat_id in sorted(categories.keys())}
    names_json = json.dumps(names_dict, ensure_ascii=False)

    yaml_content = f"""path: {output_dir.resolve()}
train: images/train
val: images/val

nc: {num_classes}
names: {names_json}
"""
    with open(output_dir / "dataset.yaml", "w") as f:
        f.write(yaml_content)

    print(f"Train: {len(train_ids)} images, Val: {len(val_ids)} images")
    print(f"Total annotations: {len(coco['annotations'])}")
    print(f"Dataset written to {output_dir}")


if __name__ == "__main__":
    main()
