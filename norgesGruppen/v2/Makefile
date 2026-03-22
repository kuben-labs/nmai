ZONE=us-central1-f
PROJECT=ai-nm26osl-1759
VM=yolo-train-a100

# ---- GCP ----

sync:
	gcloud compute scp *.py Makefile $(VM):~/v2/ --project=$(PROJECT) --zone=$(ZONE)

sync-all:
	gcloud compute scp *.py Makefile $(VM):~/v2/ --project=$(PROJECT) --zone=$(ZONE)
	gcloud compute scp --recurse ../coco ../product $(VM):~/  --project=$(PROJECT) --zone=$(ZONE)

sync-data:
	gcloud compute scp --recurse ../coco ../product $(VM):~/  --project=$(PROJECT) --zone=$(ZONE)

ssh:
	gcloud compute ssh $(VM) --project=$(PROJECT) --zone=$(ZONE)

pull-weights:
	gcloud compute scp $(VM):~/v2/runs/yolo11x_ngd/weights/best.onnx runs/yolo11x_ngd/weights/ --project=$(PROJECT) --zone=$(ZONE)
	gcloud compute scp $(VM):~/v2/runs/classifier/best_classifier_fp16.pt runs/classifier/ --project=$(PROJECT) --zone=$(ZONE)

pull-zip:
	gcloud compute scp $(VM):~/v2/submission.zip . --project=$(PROJECT) --zone=$(ZONE)

# ---- Data preparation ----

prepare:
	python3 prepare_dataset.py

prepare-multi:
	python3 prepare_dataset.py --tile-sizes 1280,960 --overlap 0.3

prepare-crops:
	python3 prepare_crops.py

prepare-all: prepare prepare-crops

# ---- Training ----

install-ultralytics:
	pip install ultralytics --upgrade

# Single GPU training
train:
	python3 train.py --model yolo11x --epochs 300 --imgsz 1280 --batch 8 --device 0

# Multi-GPU DDP (single model across all GPUs)
NGPUS ?= 16

train-ddp:
	torchrun --nproc_per_node=$(NGPUS) train_ddp.py --batch 8

# Multi-GPU: launch parallel experiments (1 per GPU)
experiments:
	python3 launch_experiments.py

experiments-status:
	python3 launch_experiments.py --status

experiments-results:
	python3 launch_experiments.py --results

experiments-stop:
	python3 launch_experiments.py --stop

experiments-dry:
	python3 launch_experiments.py --dry-run

train-classifier:
	python3 train_classifier.py --epochs 50 --batch 64 --img_size 256 --device 0

train-all: train train-classifier

train-large-batch:
	python3 train.py --model yolo11x --epochs 300 --imgsz 1280 --batch 8 --device 0

export-onnx:
	python3 train.py --export-only runs/yolo11x_ngd/weights/best.pt

# ---- Evaluation ----

evaluate:
	python3 run.py --input ../coco/train/images --output predictions.json
	python3 evaluate.py --predictions predictions.json --val-only

# ---- Packaging ----

package:
	python3 package.py \
		--yolo runs/yolo11x_ngd/weights/best.onnx \
		--classifier runs/classifier/best_classifier_fp16.pt

# ---- Quick test ----

test:
	python3 run.py --input ../coco/train/images --output test_predictions.json
