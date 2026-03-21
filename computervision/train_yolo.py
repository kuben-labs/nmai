"""Train YOLO11 for multi-class product detection + classification on shelf images."""
import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo11x.pt", help="Base model")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=-1, help="Batch size, -1 for auto")
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs")
    parser.add_argument("--name", default="multiclass")
    args = parser.parse_args()

    dataset_yaml = Path(__file__).parent / "yolo_dataset" / "dataset.yaml"

    model = YOLO(args.model)
    model.train(
        data=str(dataset_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=30,
        save=True,
        save_period=10,
        plots=True,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        degrees=5.0,
        translate=0.15,
        scale=0.4,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.2,
        weight_decay=0.0005,
        dropout=0.1,
        warmup_epochs=5,
        warmup_momentum=0.5,
        close_mosaic=20,
    )

    print(f"Best weights: {args.project}/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
