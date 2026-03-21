"""
Central configuration for the SAM3 + DINOv2 shelf matching pipeline.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
COCO_IMAGES = ROOT / "coco" / "train" / "images"
PRODUCT_DIR = ROOT / "product"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# Subdirectories created at runtime
VIS_SEGMENT = OUTPUT_DIR / "vis_segmentation"
VIS_SQUARES = OUTPUT_DIR / "vis_squares"
VIS_MATCHES = OUTPUT_DIR / "vis_matches"
MASKS_DIR = OUTPUT_DIR / "masks"
SQUARES_DIR = OUTPUT_DIR / "squares"
EMBEDDINGS_DIR = OUTPUT_DIR / "embeddings"

# ── SAM3 ───────────────────────────────────────────────────────────────
SAM3_TEXT_PROMPT = "product"  # text prompt for exhaustive product segmentation
SAM3_SCORE_THRESHOLD = 0.3   # minimum SAM3 confidence to keep a mask

# ── DINOv2 ─────────────────────────────────────────────────────────────
DINOV2_MODEL = "dinov2_vitb14"  # options: dinov2_vits14, dinov2_vitb14, dinov2_vitl14
EMBED_DIM = 768                 # vitb14 = 768, vits14 = 384, vitl14 = 1024
EMBED_BATCH_SIZE = 32
EMBED_IMAGE_SIZE = 224          # DINOv2 input resolution

# ── Matching ───────────────────────────────────────────────────────────
SIMILARITY_THRESHOLD = 0.35  # below this cosine similarity → not a real product
TOP_K = 5                    # top-K product matches to return per crop

# ── General ────────────────────────────────────────────────────────────
DEVICE = "cuda"  # SAM3 requires GPU
MAX_IMAGES = None  # set to an int to limit shelf images processed (None = all)


def ensure_dirs():
    """Create all output directories."""
    for d in [VIS_SEGMENT, VIS_SQUARES, VIS_MATCHES, MASKS_DIR, SQUARES_DIR, EMBEDDINGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
