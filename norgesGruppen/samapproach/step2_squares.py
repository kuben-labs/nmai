"""
Step 2: Convert SAM3 masks to bounding squares.

For each mask, computes the bounding box then expands to a square
(using the longer side), clamped to image bounds.

Outputs:
  - output/squares/{image_name}.json  — list of {'box': [x1,y1,x2,y2], 'square': [x1,y1,x2,y2], 'score': float}
  - output/vis_squares/{image_name}   — visualization of squares on image
"""

import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from PIL import Image
from pathlib import Path
from tqdm import tqdm

import config


def mask_to_square(mask: np.ndarray, image_w: int, image_h: int) -> list[int]:
    """Convert a binary mask to a bounding square [x1, y1, x2, y2]."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())

    w = x2 - x1
    h = y2 - y1
    side = max(w, h)

    # Center the square on the bounding box center
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    sq_x1 = int(cx - side / 2)
    sq_y1 = int(cy - side / 2)
    sq_x2 = sq_x1 + side
    sq_y2 = sq_y1 + side

    # Clamp to image bounds
    sq_x1 = max(0, sq_x1)
    sq_y1 = max(0, sq_y1)
    sq_x2 = min(image_w, sq_x2)
    sq_y2 = min(image_h, sq_y2)

    return [sq_x1, sq_y1, sq_x2, sq_y2]


def box_to_square(box: list[float], image_w: int, image_h: int) -> list[int]:
    """Convert a bounding box [x1,y1,x2,y2] to a square."""
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    side = max(w, h)

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    sq_x1 = int(cx - side / 2)
    sq_y1 = int(cy - side / 2)
    sq_x2 = sq_x1 + int(side)
    sq_y2 = sq_y1 + int(side)

    sq_x1 = max(0, sq_x1)
    sq_y1 = max(0, sq_y1)
    sq_x2 = min(image_w, sq_x2)
    sq_y2 = min(image_h, sq_y2)

    return [sq_x1, sq_y1, sq_x2, sq_y2]


def process_image(mask_file: Path, image_path: Path) -> list[dict]:
    """Load masks for one image and compute squares."""
    with open(mask_file, "rb") as f:
        results = pickle.load(f)

    image = Image.open(image_path)
    w, h = image.size

    squares = []
    for r in results:
        # Prefer mask-based square (more accurate), fallback to box
        if r["mask"] is not None and r["mask"].any():
            sq = mask_to_square(r["mask"], w, h)
        else:
            sq = box_to_square(r["box"], w, h)

        squares.append({
            "box": r["box"],
            "square": sq,
            "score": r["score"],
        })

    return squares


def visualize_squares(image_path: Path, squares: list[dict], save_path: Path):
    """Draw bounding squares on the shelf image."""
    image = np.array(Image.open(image_path).convert("RGB"))
    fig, axes = plt.subplots(1, 2, figsize=(28, 12))

    # Left: original boxes from SAM3
    axes[0].imshow(image)
    axes[0].set_title("SAM3 bounding boxes", fontsize=14)
    for s in squares:
        x1, y1, x2, y2 = s["box"]
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1.2, edgecolor="lime", facecolor="none"
        )
        axes[0].add_patch(rect)
    axes[0].axis("off")

    # Right: expanded squares
    axes[1].imshow(image)
    axes[1].set_title(f"Bounding squares ({len(squares)} items)", fontsize=14)
    rng = np.random.default_rng(42)
    for s in squares:
        x1, y1, x2, y2 = s["square"]
        color = rng.random(3)
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1.5, edgecolor=color, facecolor=(*color, 0.1)
        )
        axes[1].add_patch(rect)
        axes[1].text(x1, y1 - 3, f'{s["score"]:.2f}',
                     fontsize=6, color="white",
                     bbox=dict(boxstyle="round,pad=0.15", facecolor=color, alpha=0.7))
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def run():
    """Process all mask files and generate squares."""
    config.ensure_dirs()

    mask_files = sorted(config.MASKS_DIR.glob("*.pkl"))
    if not mask_files:
        print("No mask files found. Run step1_segment.py first.")
        return

    print(f"Processing {len(mask_files)} mask files into squares...")
    for mask_file in tqdm(mask_files, desc="Computing squares"):
        stem = mask_file.stem

        # Find corresponding image
        image_path = None
        for ext in [".jpg", ".jpeg"]:
            candidate = config.COCO_IMAGES / f"{stem}{ext}"
            if candidate.exists():
                image_path = candidate
                break

        if image_path is None:
            print(f"  Warning: no image found for {stem}, skipping")
            continue

        squares = process_image(mask_file, image_path)

        # Save squares as JSON
        out_file = config.SQUARES_DIR / f"{stem}.json"
        with open(out_file, "w") as f:
            json.dump(squares, f, indent=2)

        # Visualize
        vis_file = config.VIS_SQUARES / f"{stem}.jpg"
        visualize_squares(image_path, squares, vis_file)

    print(f"\nDone! Squares saved to {config.SQUARES_DIR}")
    print(f"Visualizations saved to {config.VIS_SQUARES}")


if __name__ == "__main__":
    run()
