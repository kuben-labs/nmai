"""
Step 1: Segment shelf images using SAM 3 with text prompt "product".

Outputs:
  - output/masks/{image_name}.pkl        — list of dicts with 'mask', 'box', 'score'
  - output/vis_segmentation/{image_name}  — visualization of all masks overlaid
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

import config


def load_model():
    """Load SAM3 image model and processor."""
    print("Loading SAM3 model...")
    model = build_sam3_image_model()
    # Sam3Processor already filters by confidence_threshold internally
    processor = Sam3Processor(
        model,
        confidence_threshold=config.SAM3_SCORE_THRESHOLD,
    )
    print("SAM3 model loaded.")
    return processor


def segment_image(processor, image_path: Path) -> list[dict]:
    """
    Run SAM3 on a single shelf image.
    Returns list of dicts: {'mask': np.ndarray, 'box': [x1,y1,x2,y2], 'score': float}
    """
    image = Image.open(image_path).convert("RGB")
    state = processor.set_image(image)
    # set_text_prompt(prompt, state) — note: prompt is first arg
    state = processor.set_text_prompt(config.SAM3_TEXT_PROMPT, state)

    # state["masks"] shape: (N, 1, H, W) bool tensors
    # state["boxes"] shape: (N, 4) in [x1, y1, x2, y2] pixel coords
    # state["scores"] shape: (N,) confidence scores
    masks = state["masks"].cpu().numpy()
    boxes = state["boxes"].cpu().numpy()
    scores = state["scores"].cpu().numpy()

    results = []
    for i in range(len(scores)):
        # masks is (N, 1, H, W) — squeeze the channel dim
        mask = masks[i, 0]
        box = boxes[i].tolist()
        results.append({
            "mask": mask.astype(bool),
            "box": box,
            "score": float(scores[i]),
        })

    print(f"  {image_path.name}: {len(results)} products detected")
    return results


def visualize_masks(image_path: Path, results: list[dict], save_path: Path):
    """Overlay all masks on the image with random colors and bounding boxes."""
    image = np.array(Image.open(image_path).convert("RGB"))
    fig, ax = plt.subplots(1, 1, figsize=(20, 14))
    ax.imshow(image)

    rng = np.random.default_rng(42)
    for r in results:
        mask = r["mask"]
        color = rng.random(3)

        # Semi-transparent mask overlay
        overlay = np.zeros((*mask.shape, 4))
        overlay[mask] = [*color, 0.4]
        ax.imshow(overlay)

        # Bounding box
        x1, y1, x2, y2 = r["box"]
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1.5, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)

        # Score label
        ax.text(x1, y1 - 4, f'{r["score"]:.2f}',
                fontsize=7, color="white",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.7))

    ax.set_title(f"{image_path.name} — {len(results)} products", fontsize=14)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def run(max_images=None):
    """Run segmentation on all shelf images."""
    config.ensure_dirs()
    processor = load_model()

    image_paths = sorted(
        list(config.COCO_IMAGES.glob("*.jpg")) +
        list(config.COCO_IMAGES.glob("*.jpeg"))
    )
    if max_images:
        image_paths = image_paths[:max_images]

    print(f"\nSegmenting {len(image_paths)} shelf images...")
    for img_path in tqdm(image_paths, desc="Segmenting"):
        stem = img_path.stem

        results = segment_image(processor, img_path)

        # Save masks
        mask_file = config.MASKS_DIR / f"{stem}.pkl"
        with open(mask_file, "wb") as f:
            pickle.dump(results, f)

        # Visualize
        vis_file = config.VIS_SEGMENT / f"{stem}.jpg"
        visualize_masks(img_path, results, vis_file)

    print(f"\nDone! Masks saved to {config.MASKS_DIR}")
    print(f"Visualizations saved to {config.VIS_SEGMENT}")


if __name__ == "__main__":
    run(max_images=config.MAX_IMAGES)
