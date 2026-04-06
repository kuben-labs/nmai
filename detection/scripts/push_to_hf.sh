#!/usr/bin/env bash
# Push dataset and model repos to HuggingFace.
# Run from the detection/ directory.
#
# Prerequisites:
#   pip install huggingface_hub
#   huggingface-cli login
#
# Usage:
#   ./scripts/push_to_hf.sh          # push both
#   ./scripts/push_to_hf.sh dataset  # push dataset only
#   ./scripts/push_to_hf.sh models   # push models only

set -euo pipefail

HF_USER="valiantlynxz"
DATASET_REPO="norwegian-grocery"
MODELS_REPO="norwegian-grocery-detector"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

push_dataset() {
    echo "==> Pushing dataset to ${HF_USER}/${DATASET_REPO}..."

    local tmpdir="${PROJECT_DIR}/_hf_push_dataset"
    rm -rf "$tmpdir"

    # Create the HF repo (idempotent)
    huggingface-cli repo create "$DATASET_REPO" --type dataset 2>/dev/null || true

    git clone "https://huggingface.co/datasets/${HF_USER}/${DATASET_REPO}" "$tmpdir"
    cd "$tmpdir"

    git lfs install
    git lfs track "*.jpg" "*.jpeg" "*.png" "*.zip"
    git add .gitattributes

    # Copy dataset README + loading script
    cp "${PROJECT_DIR}/hf_dataset/README.md" .
    cp "${PROJECT_DIR}/hf_dataset/norwegian_grocery.py" .

    # Copy raw data
    cp -r "${PROJECT_DIR}/data/raw/train" ./train
    cp -r "${PROJECT_DIR}/data/raw/NM_NGD_product_images" ./NM_NGD_product_images

    # Copy the conversion script (for users who want local YOLO files)
    mkdir -p scripts
    cp "${PROJECT_DIR}/scripts/coco_to_yolo.py" scripts/

    git add .
    git commit -m "Initial dataset upload: 248 images, 22731 annotations, 356 categories"
    git push

    cd "$PROJECT_DIR"
    rm -rf "$tmpdir"
    echo "==> Dataset pushed: https://huggingface.co/datasets/${HF_USER}/${DATASET_REPO}"
}

push_models() {
    echo "==> Pushing models to ${HF_USER}/${MODELS_REPO}..."

    local tmpdir="${PROJECT_DIR}/_hf_push_models"
    rm -rf "$tmpdir"

    # Create the HF repo (idempotent)
    huggingface-cli repo create "$MODELS_REPO" --type model 2>/dev/null || true

    git clone "https://huggingface.co/${HF_USER}/${MODELS_REPO}" "$tmpdir"
    cd "$tmpdir"

    git lfs install
    git lfs track "*.onnx"
    git add .gitattributes

    # Copy model README + config
    cp "${PROJECT_DIR}/hf_models/README.md" .
    cp "${PROJECT_DIR}/hf_models/config.json" .

    # Copy all submission variants
    for dir in "${PROJECT_DIR}/submissions/submission"*; do
        [ -d "$dir" ] || continue
        name="$(basename "$dir")"
        cp -r "$dir" "./$name"
    done

    git add .
    git commit -m "Initial model upload: 5 variants (yolov8m, yolov8l, yolo12n, yolo12x, ensemble)"
    git push

    cd "$PROJECT_DIR"
    rm -rf "$tmpdir"
    echo "==> Models pushed: https://huggingface.co/${HF_USER}/${MODELS_REPO}"
}

# ── Main ────────────────────────────────────────────────────────────────

cd "$PROJECT_DIR"

case "${1:-both}" in
    dataset) push_dataset ;;
    models)  push_models ;;
    both)    push_dataset; push_models ;;
    *)       echo "Usage: $0 [dataset|models|both]"; exit 1 ;;
esac
