"""
Multi-GPU DDP training via torchrun.

The key trick: monkey-patch dist.init_process_group to force gloo backend
BEFORE ultralytics imports, then let ultralytics handle all DDP logic
(sampler, validation gather, dataloader rebuild, etc.) — we just override
the backend since nccl has driver issues on this VM.

Usage:
    torchrun --nproc_per_node=16 train_ddp.py
    torchrun --nproc_per_node=4  train_ddp.py --epochs 200
    torchrun --nproc_per_node=16 train_ddp.py --batch 8
"""

import argparse
import datetime
import os
from pathlib import Path

import torch
import torch.distributed as dist

# ── Force gloo backend ─────────────────────────────────────────────
# Must happen BEFORE ultralytics is imported. When ultralytics detects
# RANK env var (set by torchrun), it calls dist.init_process_group
# with backend="nccl". Our patch intercepts that and forces gloo.
_orig_init_pg = dist.init_process_group


def _force_gloo_init(*args, **kwargs):
    # Override backend to gloo, keep everything else
    kwargs["backend"] = "gloo"
    # Give gloo more time — 16 GPUs need patience
    if "timeout" not in kwargs:
        kwargs["timeout"] = datetime.timedelta(minutes=30)
    if args:
        # If backend was passed positionally, replace it
        args = ("gloo",) + args[1:]
    return _orig_init_pg(*args, **kwargs)


dist.init_process_group = _force_gloo_init
# ────────────────────────────────────────────────────────────────────

DATASET_YAML = Path(__file__).parent / "dataset_tiled" / "dataset.yaml"
RUNS_DIR = Path(__file__).parent / "runs"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolo11x",
                   choices=["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"])
    p.add_argument("--weights", default=None)
    p.add_argument("--data", default=str(DATASET_YAML))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=8, help="Batch size PER GPU")
    p.add_argument("--workers", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Set default CUDA device for this process
    torch.cuda.set_device(local_rank)

    if rank == 0:
        print(f"DDP: {world_size} GPUs (gloo forced), batch {args.batch}/GPU, "
              f"{args.batch * world_size} effective")

    from ultralytics import YOLO

    weights = args.weights if args.weights else f"{args.model}.pt"
    model = YOLO(weights)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=local_rank,
        project=str(RUNS_DIR),
        name=f"{args.model}_ddp",

        # ── Augmentation ────────────────────────────────────────
        hsv_h=0.015,
        hsv_s=0.3,
        hsv_v=0.3,
        degrees=3.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0001,
        fliplr=0.5,
        flipud=0.0,
        mosaic=1.0,
        close_mosaic=20,
        mixup=0.0,
        copy_paste=0.0,

        # ── Schedule ────────────────────────────────────────────
        cos_lr=True,
        patience=80,
        warmup_epochs=5,
        warmup_momentum=0.5,

        # ── Loss ────────────────────────────────────────────────
        cls=1.5,
        box=7.5,
        dfl=1.5,

        # ── Regularization ──────────────────────────────────────
        dropout=0.1,

        # ── Other ───────────────────────────────────────────────
        deterministic=False,
        save=True,
        plots=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        nbs=64,
    )

    if rank == 0:
        best_pts = sorted(RUNS_DIR.glob(f"**/{args.model}_ddp*/weights/best.pt"))
        if best_pts:
            model_exp = YOLO(str(best_pts[-1]))
            model_exp.export(
                format="onnx", imgsz=args.imgsz,
                half=True, opset=17, simplify=True,
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
