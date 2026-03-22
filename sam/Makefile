ZONE=us-central1-f
PROJECT=ai-nm26osl-1759
VM=yolo-train

REMOTE_DIR=$(HOME)/samapproach
REMOTE_COCO=$(HOME)/coco/train
REMOTE_IMAGES=$(REMOTE_COCO)/images
REMOTE_PRODUCT=$(HOME)/product
PYTHON=python3

export PYTHONPATH := $(HOME)/sam3_repo:$(PYTHONPATH)

# ---- GCP (run locally) ----

sync:
	gcloud compute scp *.py *.sh Makefile $(VM):~/samapproach/ --project=$(PROJECT) --zone=$(ZONE)

pull-weights:
	mkdir -p weights
	gcloud compute scp $(VM):~/samapproach/weights/* weights/ --project=$(PROJECT) --zone=$(ZONE)

ssh:
	gcloud compute ssh $(VM) --project=$(PROJECT) --zone=$(ZONE)

# ---- VM targets (run on VM) ----

install-sam3:
	pip install "timm>=1.0.17" ftfy==6.1.1 regex iopath huggingface_hub
	@if [ ! -d "$(HOME)/sam3_repo" ]; then \
		git clone https://github.com/facebookresearch/sam3.git $(HOME)/sam3_repo; \
	fi
	@grep -q 'sam3_repo' $(HOME)/.bashrc || echo 'export PYTHONPATH=$$HOME/sam3_repo:$$PYTHONPATH' >> $(HOME)/.bashrc
	@echo "Done. Run: source ~/.bashrc"

test-sam3:
	$(PYTHON) -c "\
	from sam3.model_builder import build_sam3_image_model; \
	from sam3.model.sam3_image_processor import Sam3Processor; \
	from PIL import Image; \
	model = build_sam3_image_model(); \
	processor = Sam3Processor(model, confidence_threshold=0.3); \
	import glob; imgs = sorted(glob.glob('$(REMOTE_IMAGES)/*.jpg')); \
	image = Image.open(imgs[0]); \
	state = processor.set_image(image); \
	state = processor.set_text_prompt(state=state, prompt='product'); \
	print(f'Image: {imgs[0]}'); \
	print(f'Detected {len(state[\"boxes\"])} products'); \
	print(f'Scores: {state[\"scores\"][:5]}'); \
	"

prepare-data:
	$(PYTHON) prepare_data.py \
		--annotations $(REMOTE_COCO)/annotations.json \
		--images $(REMOTE_IMAGES) \
		--products $(REMOTE_PRODUCT) \
		--output classifier_data

generate-labels:
	$(PYTHON) generate_sam3_labels.py \
		--images $(REMOTE_IMAGES) \
		--output sam3_detections.json \
		--save-crops

train-classifier:
	$(PYTHON) train_classifier.py \
		--data classifier_data \
		--epochs 30 \
		--output weights/classifier.onnx

export-sam3:
	mkdir -p weights
	$(PYTHON) export_sam3.py --quantize int8 --output-dir weights

build-zip:
	$(PYTHON) build_submission.py

# Full pipeline
all: prepare-data generate-labels train-classifier export-sam3 build-zip
