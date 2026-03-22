#!/bin/bash
# Setup script for SAM 3 + training dependencies on GCP VM
# Run this on the yolo-train VM (L4 GPU, CUDA 12.6+)

set -e

echo "=== Setting up SAM 3 environment ==="

# Create conda environment
conda create -n sam3 python=3.12 -y
conda activate sam3

# Install PyTorch 2.7+ with CUDA 12.6
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Clone and install SAM 3
if [ ! -d "sam3_repo" ]; then
    git clone https://github.com/facebookresearch/sam3.git sam3_repo
fi
cd sam3_repo
pip install -e .
cd ..

# Install training and export dependencies
pip install \
    timm==0.9.12 \
    onnx \
    onnxruntime-gpu \
    onnxruntime-extensions \
    albumentations \
    scikit-learn \
    pycocotools \
    pillow \
    tqdm

# For ONNX quantization
pip install onnxruntime-tools neural-compressor

# Login to HuggingFace for SAM 3 checkpoint access
echo ""
echo "=== HuggingFace Authentication ==="
echo "SAM 3 checkpoints require access approval."
echo "1. Request access at: https://huggingface.co/facebook/sam3"
echo "2. Run: huggingface-cli login"
echo ""

# Download training data if not present
if [ ! -f "coco/train/annotations.json" ]; then
    echo "=== Training data not found ==="
    echo "Download from the competition website:"
    echo "  1. NM_NGD_coco_dataset.zip → extract to coco/train/"
    echo "  2. NM_NGD_product_images.zip → extract to product/"
    echo ""
fi

echo "=== Setup complete ==="
echo "Activate with: conda activate sam3"
echo ""
echo "Next steps:"
echo "  1. Download training data (if not done)"
echo "  2. python export_sam3.py          # Export SAM 3 to ONNX"
echo "  3. python prepare_data.py         # Prepare classifier training data"
echo "  4. python train_classifier.py     # Train product classifier"
echo "  5. python build_submission.py     # Build submission zip"
