NorgesGruppen Data: Examples & Tips
Random Baseline
Minimal run.py that generates random predictions (use to verify your setup):
import argparse
import json
import random
from pathlib import Path
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
 
    predictions = []
    for img in sorted(Path(args.input).iterdir()):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        image_id = int(img.stem.split("_")[-1])
        for _ in range(random.randint(5, 20)):
            predictions.append({
                "image_id": image_id,
                "category_id": random.randint(0, 356),
                "bbox": [
                    round(random.uniform(0, 1500), 1),
                    round(random.uniform(0, 800), 1),
                    round(random.uniform(20, 200), 1),
                    round(random.uniform(20, 200), 1),
                ],
                "score": round(random.uniform(0.01, 1.0), 3),
            })
 
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)
 
if __name__ == "__main__":
    main()
YOLOv8 Example
Using YOLOv8n with GPU auto-detection. Important: The pretrained COCO model outputs COCO class IDs (0-79), not product IDs (0-355). For correct product classification, fine-tune on the competition training data with nc=357. Detection-only submissions (wrong category_ids) still score up to 70%.
import argparse
import json
from pathlib import Path
import torch
from ultralytics import YOLO
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO("yolov8n.pt")
    predictions = []
 
    for img in sorted(Path(args.input).iterdir()):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        image_id = int(img.stem.split("_")[-1])
        results = model(str(img), device=device, verbose=False)
        for r in results:
            if r.boxes is None:
                continue
            for i in range(len(r.boxes)):
                x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
                predictions.append({
                    "image_id": image_id,
                    "category_id": int(r.boxes.cls[i].item()),
                    "bbox": [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)],
                    "score": round(float(r.boxes.conf[i].item()), 3),
                })
 
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)
 
if __name__ == "__main__":
    main()
Include yolov8n.pt in your zip. This pretrained COCO model serves as a baseline — fine-tune on the competition training data for better results. With GPU available, larger models like YOLOv8m/l/x are also feasible within the timeout.
ONNX Inference Example
ONNX works with any model framework. Use CUDAExecutionProvider for GPU acceleration:
Export (on your training machine):
# From ultralytics:
from ultralytics import YOLO
model = YOLO("best.pt")
model.export(format="onnx", imgsz=640, opset=17)
 
# From any PyTorch model:
import torch
model = ...  # your trained model
dummy = torch.randn(1, 3, 640, 640)
torch.onnx.export(model, dummy, "model.onnx", opset_version=17)
Inference (in your run.py):
import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image
import onnxruntime as ort
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
 
    session = ort.InferenceSession("model.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    predictions = []
 
    for img_path in sorted(Path(args.input).iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        image_id = int(img_path.stem.split("_")[-1])
 
        img = Image.open(img_path).convert("RGB").resize((640, 640))
        arr = np.array(img).astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]
 
        outputs = session.run(None, {input_name: arr})
        # Process outputs based on your model's output format
        # ...
 
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)
 
if __name__ == "__main__":
    main()
Common Errors



Error
Fix




run.py not found at zip root
Zip the contents, not the folder. See "Creating Your Zip" in submission docs.


Disallowed file type: __MACOSX/...
macOS Finder resource forks. Use terminal: zip -r ../sub.zip . -x ".*" "__MACOSX/*"


Disallowed file type: .bin
Rename .bin → .pt (same format) or convert to .safetensors


Security scan found violations
Remove imports of subprocess, socket, os, etc. Use pathlib instead.


No predictions.json in output
Make sure run.py writes to the --output path


Timed out after 300s
Ensure GPU is used (model.to("cuda")), or use a smaller model


Exit code 137
Out of memory (8 GB limit). Reduce batch size or use FP16


Exit code 139
Segfault — likely model weight version mismatch. Re-export with matching package version or use ONNX.


ModuleNotFoundError
Package not in sandbox. Export model to ONNX or include model code in your .py files.


KeyError / RuntimeError on model load
Version mismatch. Pin exact sandbox versions or export to ONNX.



Tips

Start with the random baseline to verify your setup works
GPU is available — larger models (YOLOv8m/l/x, custom transformers) are feasible within the 300s timeout
Use torch.cuda.is_available() to write code that works both locally (CPU) and on the server (GPU)
FP16 quantization is recommended — smaller weights, faster GPU inference
ONNX with CUDAExecutionProvider gives good GPU performance for any framework
Process images one at a time to stay within memory limits
Use torch.no_grad() during inference
Test your code locally before uploading
You don't need all sandbox packages for training — only match what you use
