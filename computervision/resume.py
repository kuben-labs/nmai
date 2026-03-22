"""Resume training from a checkpoint."""
import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="runs/detect/runs/exp_11x_1280/weights/last.pt")
    args = parser.parse_args()

    model = YOLO(args.weights)
    model.train(resume=True)


if __name__ == "__main__":
    main()
