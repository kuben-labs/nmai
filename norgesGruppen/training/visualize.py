"""
Visualize annotated images from either COCO or YOLO format datasets.

Usage:
    python visualize.py                   # random image from augmented dataset
    python visualize.py --source coco     # original COCO images
    python visualize.py --source aug      # augmented YOLO dataset (default)
    python visualize.py --source orig     # prepared (non-augmented) dataset
    python visualize.py --image img_00001_aug0.jpg
    python visualize.py --count 4         # grid of 4 random images
"""

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

COCO_DIR    = Path(__file__).parent.parent / "coco" / "train"
AUG_DIR     = Path(__file__).parent / "dataset_aug"
DATASET_DIR = Path(__file__).parent / "dataset"
OUT_DIR     = Path(__file__).parent / "viz"

COLORS = [
    "#FF3333", "#33FF57", "#3399FF", "#FF9933", "#CC33FF",
    "#FF33CC", "#33FFEE", "#FFFF33", "#FF6633", "#33CCFF",
]


def get_font(size=14):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_yolo_labels(img_path: Path, label_path: Path, class_names: list) -> Image.Image:
    """Draw boxes from a YOLO .txt label file onto the image."""
    img  = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = get_font(13)
    iw, ih = img.size

    if not label_path.exists():
        return img

    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls = int(parts[0])
        cx, cy, nw, nh = map(float, parts[1:])
        x1 = int((cx - nw / 2) * iw)
        y1 = int((cy - nh / 2) * ih)
        x2 = int((cx + nw / 2) * iw)
        y2 = int((cy + nh / 2) * ih)

        color = COLORS[cls % len(COLORS)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        label = class_names[cls][:28] if cls < len(class_names) else str(cls)
        bbox_text = draw.textbbox((0, 0), label, font=font)
        tw = bbox_text[2] - bbox_text[0]
        th = bbox_text[3] - bbox_text[1]
        ty = y1 - th - 4 if y1 - th - 4 > 0 else y2
        draw.rectangle([x1, ty, x1 + tw + 4, ty + th + 4], fill=color)
        draw.text((x1 + 2, ty + 2), label, fill="white", font=font)

    return img


def run(images_dir: Path, labels_dir: Path, class_names: list,
        image_name, count: int, out):
    images = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.jpeg"))
    if not images:
        print(f"No images found in {images_dir}")
        return

    if image_name:
        selected = [f for f in images if f.name == image_name]
        if not selected:
            print(f"Image '{image_name}' not found in {images_dir}")
            return
    else:
        selected = random.sample(images, min(count, len(images)))

    OUT_DIR.mkdir(exist_ok=True)

    if len(selected) == 1:
        img_path   = selected[0]
        label_path = labels_dir / (img_path.stem + ".txt")
        result     = draw_yolo_labels(img_path, label_path, class_names)
        out_path   = Path(out) if out else OUT_DIR / f"viz_{img_path.name}"
        result.save(out_path)
        n = len(label_path.read_text().strip().splitlines()) if label_path.exists() else 0
        print(f"Saved: {out_path}  ({n} boxes)")
    else:
        imgs = []
        for img_path in selected:
            label_path = labels_dir / (img_path.stem + ".txt")
            drawn      = draw_yolo_labels(img_path, label_path, class_names)
            drawn.thumbnail((800, 600))
            n = len(label_path.read_text().strip().splitlines()) if label_path.exists() else 0
            imgs.append((drawn, img_path.name, n))

        cols   = 2
        rows   = (len(imgs) + 1) // cols
        cw, ch = imgs[0][0].size
        grid   = Image.new("RGB", (cw * cols, ch * rows), (30, 30, 30))
        for idx, (im, _, _) in enumerate(imgs):
            r, c = divmod(idx, cols)
            grid.paste(im, (c * cw, r * ch))

        out_path = Path(out) if out else OUT_DIR / "viz_grid.jpg"
        grid.save(out_path)
        print(f"Saved grid: {out_path}")
        for _, fname, n in imgs:
            print(f"  {fname}: {n} boxes")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="aug", choices=["coco", "aug", "orig"],
                   help="aug=augmented, orig=prepared dataset, coco=original images")
    p.add_argument("--image", default=None)
    p.add_argument("--count", type=int, default=1)
    p.add_argument("--out",   default=None)
    args = p.parse_args()

    with open(COCO_DIR / "annotations.json") as f:
        coco = json.load(f)
    class_names = [c["name"] for c in coco["categories"]]

    if args.source == "aug":
        images_dir = AUG_DIR / "images" / "train"
        labels_dir = AUG_DIR / "labels" / "train"
    elif args.source == "orig":
        images_dir = DATASET_DIR / "images" / "train"
        labels_dir = DATASET_DIR / "labels" / "train"
    else:  # coco
        images_dir = COCO_DIR / "images"
        labels_dir = DATASET_DIR / "labels" / "train"

    run(images_dir, labels_dir, class_names, args.image, args.count, args.out)


if __name__ == "__main__":
    main()
