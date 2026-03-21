"""Compare YOLO training runs and print best results."""
import csv
from pathlib import Path


def read_results(results_csv):
    """Read metrics from YOLO results.csv, return best epoch by mAP50."""
    best = None
    with open(results_csv) as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            try:
                # ultralytics results.csv column names vary, try common ones
                map50 = None
                for key in row:
                    k = key.strip()
                    if "mAP50(B)" in k or "mAP50" in k and "95" not in k:
                        map50 = float(row[key])
                        break
                if map50 is None:
                    # Fall back to 5th metrics column
                    vals = list(row.values())
                    map50 = float(vals[6].strip()) if len(vals) > 6 else None

                if map50 is not None and (best is None or map50 > best["mAP50"]):
                    epoch = int(float(row.get("epoch", row.get("                  epoch", "0")).strip()))
                    map50_95 = None
                    for key in row:
                        k = key.strip()
                        if "mAP50-95(B)" in k or "mAP50-95" in k:
                            map50_95 = float(row[key])
                            break
                    best = {"epoch": epoch, "mAP50": map50, "mAP50-95": map50_95}
            except (ValueError, IndexError):
                continue
    return best


def main():
    runs_dir = Path("runs")
    results = []

    for exp_dir in sorted(runs_dir.rglob("results.csv")):
        name = exp_dir.parent.name
        best = read_results(exp_dir)
        if best:
            weights = exp_dir.parent / "weights" / "best.pt"
            results.append((name, best, weights.exists()))

    if not results:
        print("No results found. Training still running?")
        return

    # Sort by mAP50 descending
    results.sort(key=lambda x: x[1]["mAP50"], reverse=True)

    print(f"{'Experiment':<25} {'mAP50':>8} {'mAP50-95':>10} {'Epoch':>7} {'Weights':>8}")
    print("-" * 65)
    for name, best, has_weights in results:
        m95 = f"{best['mAP50-95']:.4f}" if best["mAP50-95"] is not None else "N/A"
        w = "yes" if has_weights else "no"
        print(f"{name:<25} {best['mAP50']:>8.4f} {m95:>10} {best['epoch']:>7} {w:>8}")

    winner = results[0]
    print(f"\nBest: {winner[0]} (mAP50={winner[1]['mAP50']:.4f})")
    print(f"Weights: runs/{winner[0]}/weights/best.pt")
    print(f"\nTo submit: make submit WEIGHTS=runs/{winner[0]}/weights/best.pt")


if __name__ == "__main__":
    main()
