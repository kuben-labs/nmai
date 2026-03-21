"""
Step 4: Crop shelf segments, embed with DINOv2, match against product DB.

For each shelf image:
  1. Load the square coordinates from step 2
  2. Crop each square from the original image
  3. Embed each crop with DINOv2
  4. Cosine similarity against all product embeddings
  5. Filter out non-products (similarity below threshold)
  6. Visualize: show each crop with its top match and score

Outputs:
  - output/vis_matches/{image_name}.jpg  — visualization of matched products
  - output/matches/{image_name}.json     — per-crop match results
"""

import json
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import patches
from PIL import Image
from pathlib import Path
from tqdm import tqdm

import config


def load_product_db():
    """Load pre-computed product embeddings."""
    emb_file = config.EMBEDDINGS_DIR / "product_embeddings.pkl"
    with open(emb_file, "rb") as f:
        product_embeddings = pickle.load(f)

    # Build matrix for fast cosine similarity
    skus = sorted(product_embeddings.keys())
    emb_matrix = np.stack([product_embeddings[sku] for sku in skus])  # (N_products, embed_dim)
    return skus, emb_matrix


def load_dinov2():
    """Load DINOv2 model."""
    print(f"Loading DINOv2 ({config.DINOV2_MODEL})...")
    model = torch.hub.load("facebookresearch/dinov2", config.DINOV2_MODEL)
    model = model.to(config.DEVICE)
    model.eval()
    return model


def get_transform():
    """DINOv2 preprocessing."""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(config.EMBED_IMAGE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(config.EMBED_IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def embed_crops(model, crops: list[Image.Image], transform) -> np.ndarray:
    """Embed a list of PIL crops. Returns (N, embed_dim)."""
    if not crops:
        return np.zeros((0, config.EMBED_DIM))

    embeddings = []
    for i in range(0, len(crops), config.EMBED_BATCH_SIZE):
        batch = crops[i : i + config.EMBED_BATCH_SIZE]
        tensors = torch.stack([transform(c) for c in batch]).to(config.DEVICE)
        feats = model(tensors).cpu().numpy()
        embeddings.append(feats)

    emb = np.concatenate(embeddings, axis=0)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return emb / norms


def match_crops(crop_embeddings: np.ndarray, product_skus: list[str],
                product_matrix: np.ndarray) -> list[dict]:
    """
    Cosine similarity between each crop and all products.
    Returns list of match results per crop.
    """
    if len(crop_embeddings) == 0:
        return []

    # Cosine similarity: crop_emb @ product_matrix.T
    # Both are already L2-normalized
    similarities = crop_embeddings @ product_matrix.T  # (N_crops, N_products)

    results = []
    for i in range(len(crop_embeddings)):
        sims = similarities[i]
        top_indices = np.argsort(sims)[::-1][: config.TOP_K]

        top_matches = []
        for idx in top_indices:
            top_matches.append({
                "sku": product_skus[idx],
                "similarity": float(sims[idx]),
            })

        best_sim = float(sims[top_indices[0]])
        is_product = best_sim >= config.SIMILARITY_THRESHOLD

        results.append({
            "crop_index": i,
            "is_product": is_product,
            "best_similarity": best_sim,
            "best_sku": product_skus[top_indices[0]],
            "top_matches": top_matches,
        })

    return results


def visualize_matches(image_path: Path, squares: list[dict],
                      match_results: list[dict], save_path: Path):
    """
    Draw squares on shelf image, color-coded:
    - Green = matched product (above threshold)
    - Red = filtered out (non-product / low similarity)
    Show the best SKU match label on each green square.
    """
    image = np.array(Image.open(image_path).convert("RGB"))

    n_products = sum(1 for m in match_results if m["is_product"])
    n_filtered = sum(1 for m in match_results if not m["is_product"])

    fig, ax = plt.subplots(1, 1, figsize=(22, 14))
    ax.imshow(image)

    for sq, match in zip(squares, match_results):
        x1, y1, x2, y2 = sq["square"]

        if match["is_product"]:
            color = "lime"
            label = f'SKU:{match["best_sku"]}\n{match["best_similarity"]:.3f}'
        else:
            color = "red"
            label = f'X {match["best_similarity"]:.3f}'

        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1.5, edgecolor=color,
            facecolor=(*plt.cm.colors.to_rgba(color)[:3], 0.08)
        )
        ax.add_patch(rect)

        fontsize = 5 if len(match_results) > 40 else 7
        ax.text(x1 + 2, y1 + 12, label, fontsize=fontsize, color="white",
                bbox=dict(boxstyle="round,pad=0.15", facecolor=color, alpha=0.6))

    ax.set_title(
        f"{image_path.name} — {n_products} products matched, "
        f"{n_filtered} filtered out (threshold={config.SIMILARITY_THRESHOLD})",
        fontsize=13
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def run():
    """Match all shelf crops against product database."""
    config.ensure_dirs()

    # Create matches output dir
    matches_dir = config.OUTPUT_DIR / "matches"
    matches_dir.mkdir(parents=True, exist_ok=True)

    # Load product embeddings
    print("Loading product embedding database...")
    product_skus, product_matrix = load_product_db()
    print(f"Loaded {len(product_skus)} product SKUs")

    # Load DINOv2
    model = load_dinov2()
    transform = get_transform()

    # Process each shelf image
    square_files = sorted(config.SQUARES_DIR.glob("*.json"))
    if not square_files:
        print("No square files found. Run step2_squares.py first.")
        return

    total_products = 0
    total_filtered = 0

    print(f"\nMatching {len(square_files)} shelf images against product DB...")
    for sq_file in tqdm(square_files, desc="Matching"):
        stem = sq_file.stem

        # Load squares
        with open(sq_file) as f:
            squares = json.load(f)

        # Find image
        image_path = None
        for ext in [".jpg", ".jpeg"]:
            candidate = config.COCO_IMAGES / f"{stem}{ext}"
            if candidate.exists():
                image_path = candidate
                break
        if image_path is None:
            continue

        # Crop squares from image
        image = Image.open(image_path).convert("RGB")
        crops = []
        for sq in squares:
            x1, y1, x2, y2 = sq["square"]
            crop = image.crop((x1, y1, x2, y2))
            # Ensure minimum size
            if crop.width < 4 or crop.height < 4:
                crop = Image.new("RGB", (config.EMBED_IMAGE_SIZE, config.EMBED_IMAGE_SIZE))
            crops.append(crop)

        # Embed crops
        crop_embeddings = embed_crops(model, crops, transform)

        # Match against products
        match_results = match_crops(crop_embeddings, product_skus, product_matrix)

        n_products = sum(1 for m in match_results if m["is_product"])
        n_filtered = sum(1 for m in match_results if not m["is_product"])
        total_products += n_products
        total_filtered += n_filtered

        # Save match results
        out_file = matches_dir / f"{stem}.json"
        with open(out_file, "w") as f:
            json.dump(match_results, f, indent=2)

        # Visualize
        vis_file = config.VIS_MATCHES / f"{stem}.jpg"
        visualize_matches(image_path, squares, match_results, vis_file)

    print(f"\nDone!")
    print(f"Total products matched: {total_products}")
    print(f"Total non-products filtered: {total_filtered}")
    print(f"Match results saved to {matches_dir}")
    print(f"Visualizations saved to {config.VIS_MATCHES}")


if __name__ == "__main__":
    run()
