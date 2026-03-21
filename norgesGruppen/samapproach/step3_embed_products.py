"""
Step 3: Pre-embed all product images using DINOv2.

For each product SKU folder, embeds every angle image and averages
the embeddings into a single per-SKU vector.

Outputs:
  - output/embeddings/product_embeddings.pkl — dict: {sku: np.ndarray (embed_dim,)}
  - output/embeddings/product_metadata.json  — dict: {sku: [list of image files]}
"""

import pickle
import json
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm

import config


def load_dinov2():
    """Load DINOv2 model from torch hub."""
    print(f"Loading DINOv2 ({config.DINOV2_MODEL})...")
    model = torch.hub.load("facebookresearch/dinov2", config.DINOV2_MODEL)
    model = model.to(config.DEVICE)
    model.eval()
    print("DINOv2 loaded.")
    return model


def get_transform():
    """Standard DINOv2 preprocessing transform."""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(config.EMBED_IMAGE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(config.EMBED_IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def embed_images(model, image_paths: list[Path], transform) -> np.ndarray:
    """Embed a list of images, returns (N, embed_dim) array."""
    embeddings = []

    for i in range(0, len(image_paths), config.EMBED_BATCH_SIZE):
        batch_paths = image_paths[i : i + config.EMBED_BATCH_SIZE]
        batch_tensors = []

        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                batch_tensors.append(transform(img))
            except Exception as e:
                print(f"  Warning: failed to load {p}: {e}")
                # Use a zero tensor as placeholder
                batch_tensors.append(torch.zeros(3, config.EMBED_IMAGE_SIZE, config.EMBED_IMAGE_SIZE))

        batch = torch.stack(batch_tensors).to(config.DEVICE)
        feats = model(batch)  # (B, embed_dim)
        embeddings.append(feats.cpu().numpy())

    return np.concatenate(embeddings, axis=0)


def run():
    """Embed all product images and save per-SKU average embeddings."""
    config.ensure_dirs()
    model = load_dinov2()
    transform = get_transform()

    # Discover all product SKU folders
    sku_dirs = sorted([d for d in config.PRODUCT_DIR.iterdir() if d.is_dir()])
    print(f"\nFound {len(sku_dirs)} product SKUs")

    product_embeddings = {}
    product_metadata = {}

    for sku_dir in tqdm(sku_dirs, desc="Embedding products"):
        sku = sku_dir.name
        image_files = sorted(
            list(sku_dir.glob("*.jpg")) +
            list(sku_dir.glob("*.jpeg")) +
            list(sku_dir.glob("*.png"))
        )

        if not image_files:
            continue

        # Embed all angles for this SKU
        embeddings = embed_images(model, image_files, transform)

        # L2 normalize each embedding, then average
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        embeddings = embeddings / norms
        avg_embedding = embeddings.mean(axis=0)

        # Normalize the average
        avg_norm = np.linalg.norm(avg_embedding)
        if avg_norm > 1e-8:
            avg_embedding = avg_embedding / avg_norm

        product_embeddings[sku] = avg_embedding
        product_metadata[sku] = [str(f.relative_to(config.PRODUCT_DIR)) for f in image_files]

    # Save
    emb_file = config.EMBEDDINGS_DIR / "product_embeddings.pkl"
    with open(emb_file, "wb") as f:
        pickle.dump(product_embeddings, f)

    meta_file = config.EMBEDDINGS_DIR / "product_metadata.json"
    with open(meta_file, "w") as f:
        json.dump(product_metadata, f, indent=2)

    print(f"\nDone! Embedded {len(product_embeddings)} SKUs")
    print(f"Embeddings saved to {emb_file}")
    print(f"Metadata saved to {meta_file}")


if __name__ == "__main__":
    run()
