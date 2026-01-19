.PHONY: install install-dev data train eval reproduce test lint format clean help

# Default target
.DEFAULT_GOAL := help

# ════════════════════════════════════════════════════════════════════════════════
# Installation
# ════════════════════════════════════════════════════════════════════════════════

install: ## Install package in production mode
	pip install -e .

install-dev: ## Install package with development dependencies
	pip install -e ".[dev,notebooks]"
	pre-commit install

# ════════════════════════════════════════════════════════════════════════════════
# Data Preparation
# ════════════════════════════════════════════════════════════════════════════════

data: ## Download and preprocess all datasets
	bash scripts/download_data.sh
	@echo "✓ Data preparation complete"

# ════════════════════════════════════════════════════════════════════════════════
# Training
# ════════════════════════════════════════════════════════════════════════════════

train: ## Run training with default config
	python train.py

train-coalt: ## Run training with CoalT protocol
	python train.py protocol=coalt

train-cot: ## Run training with Vanilla CoT protocol
	python train.py protocol=vanilla_cot

train-standard: ## Run training with Standard protocol
	python train.py protocol=standard

# ════════════════════════════════════════════════════════════════════════════════
# Evaluation
# ════════════════════════════════════════════════════════════════════════════════

eval: ## Evaluate saved results
	python evaluate.py --results_dir outputs/

# ════════════════════════════════════════════════════════════════════════════════
# Reproduction
# ════════════════════════════════════════════════════════════════════════════════

reproduce: ## Reproduce all paper results
	bash scripts/reproduce_all.sh

reproduce-table3: ## Reproduce main results
	python train.py --config-name=experiment/reproduce_table3 protocol=coalt seed=42
	python train.py --config-name=experiment/reproduce_table3 protocol=vanilla_cot seed=42
	python train.py --config-name=experiment/reproduce_table3 protocol=standard seed=42

reproduce-table4: ## Reproduce ablations
	python train.py --config-name=experiment/ablation protocol=coalt seed=42
	python train.py --config-name=experiment/ablation protocol=coalt_no_complement seed=42
	python train.py --config-name=experiment/ablation protocol=coalt_no_value seed=42

# ════════════════════════════════════════════════════════════════════════════════
# Testing & Quality
# ════════════════════════════════════════════════════════════════════════════════

test: ## Run unit tests
	pytest tests/ -v --tb=short

test-cov: ## Run tests with coverage
	pytest tests/ --cov=src/coalition_llm --cov-report=html --cov-report=term

test-all: ## Run all tests including slow ones
	pytest tests/ -v --tb=short -m ""

lint: ## Run linting
	ruff check src/ tests/
	mypy src/

format: ## Format code
	ruff format src/ tests/
	ruff check --fix src/ tests/

# ════════════════════════════════════════════════════════════════════════════════
# Cleaning
# ════════════════════════════════════════════════════════════════════════════════

clean: ## Clean build artifacts
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/
	rm -rf htmlcov/ .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

clean-outputs: ## Clean experiment outputs
	rm -rf outputs/ logs/

clean-all: clean clean-outputs ## Clean everything

# ════════════════════════════════════════════════════════════════════════════════
# Help
# ════════════════════════════════════════════════════════════════════════════════

help: ## Show this help message
	@echo "Coalition Formation in LLM Agent Networks"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
