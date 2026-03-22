ZONE=us-central1-f
PROJECT=ai-nm26osl-1759
VM=yolo-train-a100

# ---- GCP ----

sync:
	gcloud compute scp *.py requirements.txt Makefile $(VM):~/training/ --project=$(PROJECT) --zone=$(ZONE)

sync-data:
	gcloud compute scp --recurse ../coco ../product $(VM):~/  --project=$(PROJECT) --zone=$(ZONE)

ssh:
	gcloud compute ssh $(VM) --project=$(PROJECT) --zone=$(ZONE)

pull-weights:
	gcloud compute scp $(VM):~/training/runs/yolov8x_ngd/weights/best.pt runs/yolov8x_ngd/weights/ --project=$(PROJECT) --zone=$(ZONE)
	gcloud compute scp $(VM):~/training/runs/classifier/best_classifier_fp16.pt runs/classifier/ --project=$(PROJECT) --zone=$(ZONE)

pull-zip:
	gcloud compute scp $(VM):~/training/submission.zip . --project=$(PROJECT) --zone=$(ZONE)

# ---- Data preparation ----

prepare:
	python3 prepare_dataset.py

prepare-tiled:
	python3 prepare_tiled_dataset.py --tile-size 1280 --overlap 0.25

prepare-crops:
	python3 prepare_crops.py

prepare-all: prepare prepare-tiled prepare-crops

# ---- Training ----

train-yolo:
	python3 train.py --model yolov8x --epochs 300 --imgsz 1280 --batch 8 --device 0

train-yolo11:
	pip install ultralytics --upgrade
	python3 train_yolo11.py --model yolo11x --epochs 300 --imgsz 1280 --batch 8 --device 0

train-v2:
	pip install ultralytics --upgrade
	python3 train_yolo11.py --model yolo11x --epochs 300 --imgsz 1280 --batch -1 --device 0 \
		--name yolo11x_v2 --patience 30

train-classifier:
	python3 train_classifier.py --epochs 50 --batch 64 --img_size 256 --device 0

train-all: train-yolo train-classifier

# ---- Evaluation ----

evaluate:
	python3 run.py --input ../coco/train/images --output predictions.json
	python3 evaluate.py --predictions predictions.json --val-only

# ---- Packaging ----

package:
	python3 package.py \
		--yolo runs/yolov8x_ngd/weights/best.pt \
		--classifier runs/classifier/best_classifier_fp16.pt

package-onnx:
	python3 package.py \
		--yolo runs/yolo11x_ngd/weights/best.onnx \
		--classifier runs/classifier/best_classifier_fp16.pt

# ---- Quick test ----

test:
	python3 run.py --input ../coco/train/images --output test_predictions.json
