#!/usr/bin/env python3
"""
Launch multiple YOLO training experiments, one per GPU.

Usage:
    python launch_experiments.py                    # launch all experiments
    python launch_experiments.py --status           # check running experiments
    python launch_experiments.py --stop             # kill all experiments
    python launch_experiments.py --stop exp03       # kill specific experiment
    python launch_experiments.py --gpus 0,1,2,3     # use only these GPUs
    python launch_experiments.py --dry-run           # show what would run
"""

import argparse
import json
import signal
import subprocess
import time
from pathlib import Path


RUNS_DIR = Path(__file__).parent / "runs"
STATUS_FILE = Path(__file__).parent / "experiments_status.json"


# ── Experiment definitions ───────────────────────────────────────────
# Each dict overrides defaults from train_single.py.
# Experiments are assigned to GPUs in order.

EXPERIMENTS = [
    # All yolo11x — vary training techniques only

    # --- Baseline ---
    {
        "name": "exp00_baseline",
        "batch": 4,
    },

    # --- Learning rate ---
    {
        "name": "exp01_lr_high",
        "batch": 4,
        "lr0": 0.002,
    },
    {
        "name": "exp02_lr_low",
        "batch": 4,
        "lr0": 0.0005,
    },
    {
        "name": "exp03_lr_vlow",
        "batch": 4,
        "lr0": 0.0002,
        "lrf": 0.05,
    },

    # --- Augmentation strength ---
    {
        "name": "exp04_strong_aug",
        "batch": 4,
        "hsv_s": 0.5,
        "hsv_v": 0.4,
        "degrees": 5.0,
        "scale": 0.6,
        "shear": 3.0,
        "translate": 0.15,
    },
    {
        "name": "exp05_minimal_aug",
        "batch": 4,
        "hsv_s": 0.15,
        "hsv_v": 0.15,
        "degrees": 1.0,
        "scale": 0.3,
        "shear": 0.5,
        "mosaic": 0.5,
    },
    {
        "name": "exp06_no_mosaic",
        "batch": 4,
        "mosaic": 0.0,
        "close_mosaic": 0,
    },
    {
        "name": "exp07_mosaic_early_close",
        "batch": 4,
        "close_mosaic": 20,
    },

    # --- Loss weights (classification = 30% of competition score) ---
    {
        "name": "exp08_cls_heavy",
        "batch": 4,
        "cls": 3.0,
    },
    {
        "name": "exp09_cls_very_heavy",
        "batch": 4,
        "cls": 5.0,
    },
    {
        "name": "exp10_box_heavy",
        "batch": 4,
        "box": 10.0,
        "cls": 1.0,
    },

    # --- Regularization ---
    {
        "name": "exp11_dropout_high",
        "batch": 4,
        "dropout": 0.2,
    },
    {
        "name": "exp12_no_dropout",
        "batch": 4,
        "dropout": 0.0,
    },

    # --- Optimizer ---
    {
        "name": "exp13_sgd",
        "batch": 4,
        "optimizer": "SGD",
        "lr0": 0.01,
        "lrf": 0.01,
    },

    # --- Image size ---
    {
        "name": "exp14_imgsz_1600",
        "batch": 2,
        "imgsz": 1600,
    },

    # --- Bigger batch (gradient accumulation via nbs) ---
    {
        "name": "exp15_nbs128",
        "batch": 4,
        "nbs": 128,
        "lr0": 0.002,
    },
]


def get_available_gpus():
    """Get list of available GPU IDs."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        )
        return [int(line.strip()) for line in result.stdout.strip().split("\n")]
    except Exception:
        return [0]


def build_cmd(experiment):
    """Build the command line for train_single.py from experiment config."""
    cmd = ["python3", "train_single.py"]
    for key, val in experiment.items():
        cmd.extend([f"--{key}", str(val)])
    return cmd


def save_status(status):
    with open(STATUS_FILE, "w") as f:
        json.dump(status, f, indent=2)


def load_status():
    if STATUS_FILE.exists():
        with open(STATUS_FILE) as f:
            return json.load(f)
    return {}


def parse_best_metrics(name):
    """Parse results.csv and return best mAP50 + current epoch count."""
    import csv
    csv_files = sorted(RUNS_DIR.glob(f"**/{name}*/results.csv"))
    if not csv_files:
        return None

    try:
        with open(csv_files[-1]) as f:
            reader = csv.reader(f)
            header = [h.strip() for h in next(reader)]

            map50_idx = None
            map50_95_idx = None
            for i, h in enumerate(header):
                if "metrics/mAP50(B)" in h:
                    map50_idx = i
                elif "metrics/mAP50-95(B)" in h:
                    map50_95_idx = i

            if map50_idx is None:
                return None

            best_map50 = 0.0
            best_map50_95 = 0.0
            best_epoch = 0
            n_epochs = 0
            last_map50 = 0.0

            for row in reader:
                if len(row) <= map50_idx:
                    continue
                n_epochs += 1
                try:
                    m50 = float(row[map50_idx].strip())
                    last_map50 = m50
                    if m50 > best_map50:
                        best_map50 = m50
                        best_epoch = n_epochs
                        if map50_95_idx and len(row) > map50_95_idx:
                            best_map50_95 = float(row[map50_95_idx].strip())
                except (ValueError, IndexError):
                    continue

            return {
                "map50": best_map50,
                "map50_95": best_map50_95,
                "best_epoch": best_epoch,
                "n_epochs": n_epochs,
                "last_map50": last_map50,
            }
    except Exception:
        return None


def show_status():
    """Show status of all experiments with live metrics."""
    import os
    status = load_status()
    if not status:
        print("No experiments tracked yet.")
        return

    rows = []
    for name, info in sorted(status.items()):
        pid = info.get("pid", "?")
        gpu = info.get("gpu", "?")
        state = "unknown"

        if pid != "?":
            try:
                os.kill(int(pid), 0)
                state = "running"
            except (ProcessLookupError, PermissionError, ValueError):
                state = "stopped"

        run_dirs = sorted(RUNS_DIR.glob(f"**/{name}*/weights/best.pt"))
        if run_dirs:
            if state != "running":
                state = "done"

        metrics = parse_best_metrics(name)
        rows.append((name, gpu, pid, state, metrics))

    # Sort by best mAP50 descending (experiments with metrics first)
    rows.sort(key=lambda r: r[4]["map50"] if r[4] else -1, reverse=True)

    print(f"\n{'#':<3} {'Name':<28} {'GPU':>3} {'Status':<8} {'Epoch':>6} "
          f"{'Best mAP50':>10} {'@ Ep':>5} {'Last mAP50':>10} {'mAP50-95':>9}")
    print("-" * 95)

    for i, (name, gpu, pid, state, m) in enumerate(rows):
        if m:
            print(f"{i+1:<3} {name:<28} {gpu:>3} {state:<8} {m['n_epochs']:>5}/100 "
                  f"{m['map50']:>10.4f} {m['best_epoch']:>5} {m['last_map50']:>10.4f} {m['map50_95']:>9.4f}")
        else:
            print(f"{i+1:<3} {name:<28} {gpu:>3} {state:<8}   {'—':>4}       {'—':>6}  {'—':>4}       {'—':>6}      {'—':>5}")

    # Print winner
    with_metrics = [r for r in rows if r[4]]
    if with_metrics:
        best = with_metrics[0]
        print(f"\nCurrent best: {best[0]} — mAP50={best[4]['map50']:.4f} (epoch {best[4]['best_epoch']})")


def show_results():
    """Parse results.csv from each experiment and rank by best mAP50."""
    import csv

    results = []

    for exp in EXPERIMENTS:
        name = exp["name"]
        csv_files = sorted(RUNS_DIR.glob(f"**/{name}*/results.csv"))
        if not csv_files:
            continue

        try:
            with open(csv_files[-1]) as f:
                reader = csv.reader(f)
                header = [h.strip() for h in next(reader)]

                # Find relevant columns — ultralytics uses these header names
                map50_idx = None
                map50_95_idx = None
                prec_idx = None
                recall_idx = None
                for i, h in enumerate(header):
                    if "metrics/mAP50(B)" in h:
                        map50_idx = i
                    elif "metrics/mAP50-95(B)" in h:
                        map50_95_idx = i
                    elif "metrics/precision(B)" in h:
                        prec_idx = i
                    elif "metrics/recall(B)" in h:
                        recall_idx = i

                if map50_idx is None:
                    continue

                # Read all rows, track best mAP50 and last epoch
                best_map50 = 0.0
                best_map50_95 = 0.0
                best_prec = 0.0
                best_recall = 0.0
                best_epoch = 0
                n_epochs = 0

                for row in reader:
                    if len(row) <= map50_idx:
                        continue
                    n_epochs += 1
                    try:
                        m50 = float(row[map50_idx].strip())
                        if m50 > best_map50:
                            best_map50 = m50
                            best_epoch = n_epochs
                            if map50_95_idx and len(row) > map50_95_idx:
                                best_map50_95 = float(row[map50_95_idx].strip())
                            if prec_idx and len(row) > prec_idx:
                                best_prec = float(row[prec_idx].strip())
                            if recall_idx and len(row) > recall_idx:
                                best_recall = float(row[recall_idx].strip())
                    except (ValueError, IndexError):
                        continue

                results.append({
                    "name": name,
                    "map50": best_map50,
                    "map50_95": best_map50_95,
                    "precision": best_prec,
                    "recall": best_recall,
                    "best_epoch": best_epoch,
                    "total_epochs": n_epochs,
                })
        except Exception as e:
            print(f"Error reading {name}: {e}")

    if not results:
        print("No results yet. Wait for experiments to complete at least 1 epoch.")
        return

    # Sort by mAP50 descending
    results.sort(key=lambda r: r["map50"], reverse=True)

    print(f"\n{'Rank':<5} {'Name':<28} {'mAP50':>7} {'mAP50-95':>9} {'Prec':>7} {'Recall':>7} {'Best@':>6} {'Epochs':>7}")
    print("-" * 85)

    for i, r in enumerate(results):
        print(f"{i+1:<5} {r['name']:<28} {r['map50']:>7.4f} {r['map50_95']:>9.4f} "
              f"{r['precision']:>7.4f} {r['recall']:>7.4f} {r['best_epoch']:>6} {r['total_epochs']:>7}")

    print("-" * 85)
    winner = results[0]
    print(f"\nBest: {winner['name']} — mAP50={winner['map50']:.4f} at epoch {winner['best_epoch']}")

    # Show path to best weights
    best_pts = sorted(RUNS_DIR.glob(f"**/{winner['name']}*/weights/best.pt"))
    if best_pts:
        print(f"Weights: {best_pts[-1]}")


def stop_experiments(name_filter=None):
    """Stop running experiments."""
    import os
    status = load_status()
    stopped = 0

    for name, info in status.items():
        if name_filter and name_filter not in name:
            continue
        pid = info.get("pid")
        if pid:
            try:
                os.kill(int(pid), signal.SIGTERM)
                print(f"Stopped {name} (PID {pid})")
                stopped += 1
            except (ProcessLookupError, PermissionError):
                pass

    print(f"Stopped {stopped} experiment(s)")


def launch(gpus, dry_run=False):
    """Launch experiments across available GPUs."""
    import os

    if len(EXPERIMENTS) > len(gpus):
        print(f"WARNING: {len(EXPERIMENTS)} experiments but only {len(gpus)} GPUs.")
        print(f"  First {len(gpus)} experiments will be launched.")
        print(f"  Re-run after some finish to launch the rest.\n")

    status = load_status()
    launched = 0
    script_dir = Path(__file__).parent

    for i, exp in enumerate(EXPERIMENTS):
        if i >= len(gpus):
            break

        gpu = gpus[i]
        name = exp["name"]

        # Skip if already running
        if name in status:
            pid = status[name].get("pid")
            if pid:
                try:
                    os.kill(int(pid), 0)
                    print(f"SKIP {name} — already running (PID {pid}, GPU {status[name].get('gpu')})")
                    continue
                except (ProcessLookupError, PermissionError, ValueError):
                    pass  # Process dead, can relaunch

        # Skip if already completed (has best.pt)
        best_pts = sorted(RUNS_DIR.glob(f"**/{name}*/weights/best.pt"))
        if best_pts:
            print(f"SKIP {name} — already completed ({best_pts[-1]})")
            continue

        cmd = build_cmd(exp)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
        log_file = script_dir / f"logs_{name}.txt"

        if dry_run:
            print(f"[DRY RUN] GPU {gpu}: CUDA_VISIBLE_DEVICES={gpu} {' '.join(cmd)}")
            print(f"          Log: {log_file}")
            continue

        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(log_file, "w")

        proc = subprocess.Popen(
            cmd,
            cwd=str(script_dir),
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # detach from terminal
        )

        status[name] = {
            "pid": proc.pid,
            "gpu": gpu,
            "cmd": " ".join(cmd),
            "log": str(log_file),
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        print(f"LAUNCHED {name} on GPU {gpu} (PID {proc.pid})")
        print(f"  Log: {log_file}")
        launched += 1

    save_status(status)
    print(f"\nLaunched {launched} experiment(s). Use --status to check progress.")
    print(f"Logs: tail -f logs_<name>.txt")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--status", action="store_true", help="Show experiment status")
    p.add_argument("--results", action="store_true", help="Compare results, ranked by mAP50")
    p.add_argument("--stop", nargs="?", const="", default=None,
                   help="Stop experiments (optionally filter by name)")
    p.add_argument("--gpus", default=None,
                   help="Comma-separated GPU IDs (default: all available)")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would be launched without starting")
    args = p.parse_args()

    if args.results:
        show_results()
        return

    if args.status:
        show_status()
        return

    if args.stop is not None:
        stop_experiments(args.stop if args.stop else None)
        return

    gpus = [int(g) for g in args.gpus.split(",")] if args.gpus else get_available_gpus()
    print(f"Available GPUs: {gpus}")
    print(f"Experiments defined: {len(EXPERIMENTS)}\n")
    launch(gpus, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
