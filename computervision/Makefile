ZONE=us-central1-f
PROJECT=ai-nm26osl-1759
VM=yolo-train

# Remote paths on VM
REMOTE_TRAINING=~/computervision
REMOTE_COCO=~/coco/train
REMOTE_PRODUCT=~/product

# ---- GCP ----

sync:
	gcloud compute scp *.py *.sh requirements.txt Makefile $(VM):$(REMOTE_TRAINING)/ --project=$(PROJECT) --zone=$(ZONE)

sync-data:
	gcloud compute scp --recurse coco product $(VM):~/ --project=$(PROJECT) --zone=$(ZONE)

ssh:
	gcloud compute ssh $(VM) --project=$(PROJECT) --zone=$(ZONE)

pull-weights:
	gcloud compute scp --recurse $(VM):$(REMOTE_TRAINING)/runs/ ./runs/ --project=$(PROJECT) --zone=$(ZONE)

pull-submission:
	gcloud compute scp $(VM):$(REMOTE_TRAINING)/submission.zip ./submission.zip --project=$(PROJECT) --zone=$(ZONE)

# ---- Training pipeline (run on VM) ----

.PHONY: install prepare train train-parallel compare predict export-onnx package submit all

install:
	pip install -r requirements.txt

prepare:
	rm -rf yolo_dataset
	python3 prepare_yolo_dataset.py --coco-dir $(REMOTE_COCO) --output-dir yolo_dataset

train:
	python3 train_yolo.py --epochs 200 --imgsz 1280 --device 0

train-parallel:
	bash train_parallel.sh

resume:
	python3 resume.py --weights $(RESUME_FROM)

find-weights:
	find runs -name 'best.pt' -o -name 'last.pt' | sort

compare:
	python3 compare_runs.py

# Full pipeline
all: install prepare train

# ---- Submission (run on VM after training) ----
# Set WEIGHTS to the actual best.pt path printed after training
WEIGHTS=runs/detect/runs/exp_11x_1280/weights/best.pt

export-onnx:
	python3 export_onnx.py --weights $(WEIGHTS)

package:
	python3 build_submission.py --yolo-onnx $(WEIGHTS:.pt=.onnx)

submit: export-onnx package
