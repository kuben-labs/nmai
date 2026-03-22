oder@coder-gormerykw-nmai-7b97d9758-fgnl8:~$ history
    1  nvtop
    2  pip install nvtop
    3  ls
    4  ls -a
    5  history
    6  history -a
    7  nvidia-smi
    8  clear
    9  nvidia-smi
   10  nvidia-smi
   11  nvidia-smi
   12  pip install ultralytics==8.1.0
   13  pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
   14  pip install pycocotools==2.0.7 albumentations==1.3.1
   15  clear
   16  python scripts/train.py --model yolov8m.pt --imgsz 640 --epochs 50 --name baseline
   17  cd detection/
   18  python scripts/train.py --model yolov8m.pt --imgsz 640 --epochs 50 --name baseline
   19  python scripts/coco_to_yolo.py
   20  python scripts/train.py --model yolov8m.pt --imgsz 640 --epochs 50 --name baseline
   21  python scripts/evaluate.py --model runs/baseline/weights/best.pt     --images data/yolo/images/val --val-only
   22  python scripts/evaluate.py --model runs/baseline/weights/best.pt     --images data/yolo/images/val --val-only
   23  python scripts/package_submission.py --weights runs/full/weights/best.pt
   24  python scripts/package_submission.py --weights runs/baseline/weights/best.pt
   25  clear
   26  ls | tee train.log
   27  c
   28  python scripts/coco_to_yolo.py
   29  python scripts/coco_to_yolo.py | tee train.py
   30  python scripts/coco_to_yolo.py | tee train.log
   31  python scripts/coco_to_yolo.py | tee train.log
   32  python scripts/coco_to_yolo.py | tee -a train.log
   33  c
   34  python scripts/coco_to_yolo.py | tee -a train.log
   35  c
   36  clear
   37  python scripts/train.py --model yolov8m.pt --imgsz 1280 --epochs 150 --name full
   38  python scripts/train.py --model yolov8m.pt --imgsz 1280 --epochs 150 --name full | tee -a train.log
   39  pip install --force-reinstall pycocotools==2.0.7 | tee -a train.log 
   40  python scripts/evaluate.py --model runs/full2/weights/best.pt --images data/yolo/images/val --val-only | tee -a train.log 
   41  python scripts/evaluate.py --model runs/full2/weights/best.pt --images data/yolo/images/val --val-only | tee -a train.log 
   42  python scripts/package_submission.py --weights runs/full2/weights/best.pt | tee -a train.log
   43  pip install numpy==2.2.6 && pip install --force-reinstall --no-cache-dir pycocotools==2.0.7
   44  python scripts/evaluate.py --model runs/full2/weights/best.pt --images data/yolo/images/val --val-only | tee -a train.log 
   45  clear
   46  python scripts/package_submission.py --weights runs/full2/weights/best.pt | tee -a train.log
   47  python scripts/package_submission.py --weights runs/full2/weights/best.pt | tee -a train.log
   48  c
   49  python scripts/coco_to_yolo.py | tee -a train.log
   50  python scripts/evaluate.py --model runs/full2/weights/best.pt     --images data/yolo/images/val --val-only
   51  python scripts/package_submission.py --weights runs/full2/weights/best.pt | tee -a train.log
   52  cd submission
   53  python scripts/run.py --input data/raw/train/images --output output/predictions.json
   54  python scripts/run.py --input ../data/raw/train/images --output output/predictions.json
   55  python scripts/run.py --input ../data/raw/train/images --output output/predictions.json
   56  python run.py --input ../data/raw/train/images --output output/predictions.json
   57  cd ..
   58  rm -rf submission/ train.log 
   59  python scripts/export_onnx.py --weights runs/full2/weights/best.pt --imgsz 1280
   60  # 1. Export the trained model to ONNX
   61  python scripts/export_onnx.py --weights runs/full2/weights/best.pt --imgsz 1280
   62  # 2. The ONNX file will be at runs/full2/weights/best.onnx
   63  # Package it for submission
   64  python scripts/package_submission.py --weights runs/full2/weights/best.onnx
   65  c
   66  clear
   67  python scripts/coco_to_yolo.py | tee -a train.log
   68  python scripts/train.py --model yolov8l.pt --imgsz 1280 --epochs 150 --name large | tee -a train.py
   69  pip install numpy==1.26.4
   70  python scripts/train.py --model yolov8l.pt --imgsz 1280 --epochs 150 --name large | tee -a train.log
   71  python scripts/train.py --model yolov8l.pt --imgsz 1280 --epochs 150 --name large | tee -a train.log
   72  python scripts/evaluate.py --model runs/large/weights/best.pt     --images data/yolo/images/val --val-only
   73  python scripts/evaluate.py --model runs/full2/weights/best.pt     --images data/yolo/images/val --val-only
   74  c
   75  clear
   76  # 1. ONNX without TTA (baseline, same as what you submitted)
   77  python scripts/evaluate.py --onnx runs/full2/weights/best.onnx     --images data/yolo/images/val --val-only
   78  python scripts/evaluate.py --onnx runs/full2/weights/best.onnx     --images data/yolo/images/val --val-only
   79  # 1. ONNX without TTA (baseline, same as what you submitted)
   80  python scripts/evaluate.py --onnx runs/full2/weights/best.onnx     --images data/yolo/images/val --val-only
   81  # 2. ONNX with TTA (horizontal flip)
   82  python scripts/evaluate.py --onnx runs/full2/weights/best.onnx     --images data/yolo/images/val --val-only --tta
   83  python scripts/coco_to_yolo.py
   84  c
   85  clear
   86  python scripts/evaluate.py --onnx runs/full2/weights/best.onnx     --images data/yolo/images/val --val-only
   87  python scripts/evaluate.py --onnx runs/full2/weights/best.onnx     --images data/yolo/images/val --val-only --tta
   88  python scripts/export_onnx.py --weights runs/large/weights/best.pt --imgsz 1280
   89  python scripts/evaluate.py --onnx runs/large/weights/best.onnx     --images data/yolo/images/val --val-only
   90  python scripts/evaluate.py --onnx runs/large/weights/best.onnx     --images data/yolo/images/val --val-only --tta
   91  c
   92  clear
   93  python scripts/coco_to_yolo.py
   94  python scripts/package_submission.py --weights runs/large/weights/best.onnx
   95  python scripts/train.py --model yolov12m.pt --imgsz 1280 --epochs 150 --name large | tee -a train.log
   96  python scripts/train.py --model yolov12n.pt --imgsz 1280 --epochs 150 --name large | tee -a train.log
   97  python scripts/train.py --model yolo11n.pt --imgsz 1280 --epochs 150 --name large | tee -a train.log
   98  python scripts/train.py --model yolo11m.pt --imgsz 1280 --epochs 150 --name large | tee -a train.log
   99  pip install --upgrade ultralytics
  100  python scripts/train.py --model yolo11n.pt --imgsz 1280 --epochs 150 --name yolo11n
  101  python scripts/train.py --model yolov12x.pt --imgsz 1280 --epochs 150 --name yolov12x
  102  python scripts/train.py --model yolov12l.pt --imgsz 1280 --epochs 150 --name yolov12l
  103  python scripts/train.py --model yolo11x.pt --imgsz 1280 --epochs 150 --name yolo11x
  104  tmux
  105  pip install tmux
  106  pip install -U ultralytics
  107  c
  108  clear
  109  apt install tmux
  110  sudo apt install tmux
  111  apt-get update
  112  sudo apt-get update
  113  sudo apt-get upgrade
  114  clear
  115  sudo apt install tmux
  116  c
  117  tmux new-session -t detection
  118  c
  119  clear
  120  tmux attach-session -t detection
  121  clear
  122  l
  123  nvtop
  124  sudo apt install nvto
  125  sudo apt install nvtop
  126  nvtop
  127  python scripts/export_onnx.py --weights runs/yolo12x/weights/best.pt --imgsz 1280
  128  cd detection/
  129  python scripts/export_onnx.py --weights runs/yolo12x/weights/best.pt --imgsz 1280
  130  python scripts/evaluate.py --model runs/yolo12x/weights/best.pt     --images data/yolo/images/val --val-only
  131  python scripts/evaluate.py --onnx runs/yolo12x/weights/best.onnx     --images data/yolo/images/val --val-only
  132  python scripts/evaluate.py --onnx runs/yolo12x/weights/best.onnx     --images data/yolo/images/val --val-only --tta
  133  python scripts/package_submission.py --weights runs/yolo12x/weights/best.pt | tee -a train.log
  134  cd detection/
  135  python scripts/package_submission.py --weights runs/yolo12x/weights/best.pt | tee -a train.log
  136  python run.py --input ../data/raw/train/images --output output/predictions.json
  137  cd submission
  138  python run.py --input ../data/raw/train/images --output output/predictions.json
  139  python scripts/package_submission.py --weights runs/yolo12x/weights/best.onnx | tee -a train.log
  140  cd ../
  141  python scripts/package_submission.py --weights runs/yolo12x/weights/best.onnx | tee -a train.log
  142  cd submission
  143  python run.py --input ../data/raw/train/images --output output/predictions.json
  144  history
coder@coder-gormerykw-nmai-7b97d9758-fgnl8:~$ 