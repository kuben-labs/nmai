#!/bin/bash
# Run multiple YOLO training experiments in parallel on separate GPUs.
# Each experiment uses different hyperparams/models to find the best config.

# Experiment 1: YOLO11x @ 1280, strong augmentation (GPU 0)
python3 train_yolo.py --model yolo11x.pt --imgsz 1280 --epochs 200 --device 0 --name exp_11x_1280 &

# Experiment 2: YOLO11x @ 960, more batch (GPU 1)
python3 train_yolo.py --model yolo11x.pt --imgsz 960 --epochs 200 --device 1 --name exp_11x_960 &

# Experiment 3: YOLO11l @ 1280, lighter model (GPU 2)
python3 train_yolo.py --model yolo11l.pt --imgsz 1280 --epochs 200 --device 2 --name exp_11l_1280 &

# Experiment 4: YOLO11x @ 1280, batch 8 forced (GPU 3)
python3 train_yolo.py --model yolo11x.pt --imgsz 1280 --epochs 200 --batch 8 --device 3 --name exp_11x_1280_b8 &

echo "Started 4 experiments on GPUs 0-3. Monitor with: tail -f runs/*/train.log"
echo "Check progress: ls runs/*/weights/best.pt"
wait
echo "All experiments finished!"
