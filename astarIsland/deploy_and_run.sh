#!/bin/bash
# Deploy to GCP VM and run astar island solver
# Usage: ./deploy_and_run.sh [backtest|solve|study|info|scores]
set -e

ZONE=us-central1-f
PROJECT=ai-nm26osl-1759
VM=yolo-train
REMOTE_DIR=~/astarIsland
CMD="${1:-backtest}"

SSH="gcloud compute ssh $VM --project=$PROJECT --zone=$ZONE --"
SCP="gcloud compute scp --project=$PROJECT --zone=$ZONE"

echo "=== Creating remote directory ==="
$SSH "mkdir -p $REMOTE_DIR/.cache"

echo "=== Syncing solve.py and .env ==="
$SCP solve.py .env $VM:$REMOTE_DIR/

echo "=== Syncing cache files ==="
$SCP .cache/*.json $VM:$REMOTE_DIR/.cache/ 2>/dev/null || true

echo "=== Installing dependencies on VM ==="
$SSH "pip install -q lightgbm scikit-learn numpy requests 2>&1 | tail -1"

echo "=== Checking GPU ==="
$SSH "nvidia-smi | head -4"

case "$CMD" in
  backtest)
    echo "=== Running backtest on GPU ==="
    $SSH "cd $REMOTE_DIR && python3 solve.py --backtest --gpu" 2>&1 | tee backtest_results.txt
    ;;
  solve)
    echo "=== Running study + solve on GPU ==="
    $SSH "cd $REMOTE_DIR && python3 solve.py --all --gpu"
    echo "=== Pulling updated cache ==="
    $SCP --recurse $VM:$REMOTE_DIR/.cache/ .cache/
    ;;
  study)
    echo "=== Running study ==="
    $SSH "cd $REMOTE_DIR && python3 solve.py --study"
    $SCP --recurse $VM:$REMOTE_DIR/.cache/ .cache/
    ;;
  info)
    $SSH "cd $REMOTE_DIR && python3 solve.py --info"
    ;;
  scores)
    $SSH "cd $REMOTE_DIR && python3 solve.py --scores"
    ;;
  *)
    echo "Unknown command: $CMD"
    echo "Usage: $0 [backtest|solve|study|info|scores]"
    exit 1
    ;;
esac

echo "=== Done ==="
