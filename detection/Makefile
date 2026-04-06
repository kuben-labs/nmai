# NorgesGruppen Grocery Detection - Data & Model Management
#
# Usage:
#   make setup          - Download everything and prepare YOLO format
#   make data           - Download dataset from HuggingFace
#   make yolo           - Generate YOLO format from COCO annotations
#   make models         - Download all model submissions from HuggingFace
#   make clean-data     - Remove downloaded data
#   make clean-models   - Remove downloaded models
#   make clean          - Remove all downloaded/generated artifacts

HF_DATASET_REPO := valiantlynxz/norwegian-grocery
HF_MODELS_REPO  := valiantlynxz/norwegian-grocery-detector

DATA_DIR     := data/raw
YOLO_DIR     := data/yolo
CONFIGS_DIR  := configs
SUBMISSIONS  := submissions

.PHONY: setup data yolo models clean clean-data clean-models clean-yolo help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Full setup ──────────────────────────────────────────────────────────

setup: data yolo models ## Download data + models, generate YOLO format

# ── Dataset ─────────────────────────────────────────────────────────────

data: $(DATA_DIR)/train/annotations.json ## Download dataset from HuggingFace

$(DATA_DIR)/train/annotations.json:
	@echo "==> Cloning dataset from $(HF_DATASET_REPO)..."
	@mkdir -p $(DATA_DIR)
	git lfs install --skip-smudge 2>/dev/null || true
	git clone https://huggingface.co/datasets/$(HF_DATASET_REPO) $(DATA_DIR)/_hf_dataset
	@cd $(DATA_DIR)/_hf_dataset && git lfs pull
	@# Move files into expected structure
	@mv $(DATA_DIR)/_hf_dataset/train $(DATA_DIR)/train
	@mv $(DATA_DIR)/_hf_dataset/NM_NGD_product_images $(DATA_DIR)/NM_NGD_product_images
	@# Keep the conversion script
	@mkdir -p scripts
	@cp -n $(DATA_DIR)/_hf_dataset/scripts/coco_to_yolo.py scripts/coco_to_yolo.py 2>/dev/null || true
	@rm -rf $(DATA_DIR)/_hf_dataset
	@echo "==> Dataset ready at $(DATA_DIR)/"

# ── YOLO conversion ────────────────────────────────────────────────────

yolo: $(YOLO_DIR)/category_names.json ## Generate YOLO format from COCO annotations

$(YOLO_DIR)/category_names.json: $(DATA_DIR)/train/annotations.json
	@echo "==> Converting COCO to YOLO format..."
	python scripts/coco_to_yolo.py \
		--annotations $(DATA_DIR)/train/annotations.json \
		--images $(DATA_DIR)/train/images \
		--output $(YOLO_DIR) \
		--val-ratio 0.15
	@echo "==> YOLO dataset ready at $(YOLO_DIR)/"
	@echo "==> Dataset config at $(CONFIGS_DIR)/dataset.yaml"

# ── Models ──────────────────────────────────────────────────────────────

models: $(SUBMISSIONS)/.pulled ## Download all models from HuggingFace

$(SUBMISSIONS)/.pulled:
	@echo "==> Cloning models from $(HF_MODELS_REPO)..."
	@mkdir -p $(SUBMISSIONS)
	git lfs install --skip-smudge 2>/dev/null || true
	git clone https://huggingface.co/$(HF_MODELS_REPO) $(SUBMISSIONS)/_hf_models
	@cd $(SUBMISSIONS)/_hf_models && git lfs pull
	@# Move submission directories into place
	@for dir in $(SUBMISSIONS)/_hf_models/submission*; do \
		name=$$(basename "$$dir"); \
		rm -rf $(SUBMISSIONS)/$$name; \
		mv "$$dir" $(SUBMISSIONS)/$$name; \
	done
	@rm -rf $(SUBMISSIONS)/_hf_models
	@touch $(SUBMISSIONS)/.pulled
	@echo "==> Models ready at $(SUBMISSIONS)/"

# Individual model targets for convenience
models-%: ## Download a specific submission variant (e.g., make models-submission_ensemble)
	@echo "==> This requires full clone; use 'make models' instead"
	@$(MAKE) models

# ── Cleanup ─────────────────────────────────────────────────────────────

clean-data: ## Remove downloaded dataset
	rm -rf $(DATA_DIR)/train $(DATA_DIR)/NM_NGD_product_images $(DATA_DIR)/_hf_dataset

clean-yolo: ## Remove generated YOLO dataset and config
	rm -rf $(YOLO_DIR) $(CONFIGS_DIR)/dataset.yaml

clean-models: ## Remove downloaded models
	rm -rf $(SUBMISSIONS)/submission* $(SUBMISSIONS)/.pulled $(SUBMISSIONS)/_hf_models

clean: clean-data clean-yolo clean-models ## Remove all downloaded/generated artifacts
	@echo "==> Cleaned all data, YOLO output, and models"
