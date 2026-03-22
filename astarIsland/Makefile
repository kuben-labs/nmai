ZONE=us-central1-f
PROJECT=ai-nm26osl-1759
VM=yolo-train
REMOTE_DIR=~/astarIsland

# ---- Sync ----

sync:
	gcloud compute scp solve.py .env Makefile $(VM):$(REMOTE_DIR)/ --project=$(PROJECT) --zone=$(ZONE)

sync-cache:
	gcloud compute ssh $(VM) --project=$(PROJECT) --zone=$(ZONE) -- "mkdir -p $(REMOTE_DIR)/.cache"
	gcloud compute scp --recurse .cache/ $(VM):$(REMOTE_DIR)/.cache/ --project=$(PROJECT) --zone=$(ZONE)

sync-all: sync sync-cache

# ---- Setup ----

setup:
	gcloud compute ssh $(VM) --project=$(PROJECT) --zone=$(ZONE) -- "\
		cd $(REMOTE_DIR) && \
		pip install -q lightgbm scikit-learn numpy requests"

setup-gpu:
	gcloud compute ssh $(VM) --project=$(PROJECT) --zone=$(ZONE) -- "\
		cd $(REMOTE_DIR) && \
		pip install -q 'lightgbm>=4.0' scikit-learn numpy requests && \
		python3 -c 'import lightgbm; print(\"LightGBM:\", lightgbm.__version__)' && \
		nvidia-smi | head -4"

# ---- Remote Run ----

study:
	gcloud compute ssh $(VM) --project=$(PROJECT) --zone=$(ZONE) -- "\
		cd $(REMOTE_DIR) && python3 solve.py --study"

backtest:
	gcloud compute ssh $(VM) --project=$(PROJECT) --zone=$(ZONE) -- "\
		cd $(REMOTE_DIR) && python3 solve.py --backtest --gpu" 2>&1 | tee backtest_results.txt

solve:
	gcloud compute ssh $(VM) --project=$(PROJECT) --zone=$(ZONE) -- "\
		cd $(REMOTE_DIR) && python3 solve.py --all --gpu"

solve-only:
	gcloud compute ssh $(VM) --project=$(PROJECT) --zone=$(ZONE) -- "\
		cd $(REMOTE_DIR) && python3 solve.py --solve --gpu"

info:
	gcloud compute ssh $(VM) --project=$(PROJECT) --zone=$(ZONE) -- "\
		cd $(REMOTE_DIR) && python3 solve.py --info"

scores:
	gcloud compute ssh $(VM) --project=$(PROJECT) --zone=$(ZONE) -- "\
		cd $(REMOTE_DIR) && python3 solve.py --scores"

# ---- Pull Results ----

pull-cache:
	gcloud compute scp --recurse $(VM):$(REMOTE_DIR)/.cache/ .cache/ --project=$(PROJECT) --zone=$(ZONE)

# ---- SSH ----

ssh:
	gcloud compute ssh $(VM) --project=$(PROJECT) --zone=$(ZONE)

# ---- Full Pipeline ----

deploy: sync-all setup-gpu
	@echo "Deployed and ready. Run 'make solve' or 'make backtest'"

run-all: sync-all solve pull-cache
	@echo "Done! Results pulled to local .cache/"
