## Basic project Makefile with Conda environment management
## Assumes Conda is already installed locally.

.PHONY: help env install install-torch hooks notebook jupyter shell print-activate remove-env clean

# ---- Configuration ----
CONDA ?= conda
CONDA_ENV_NAME ?= mlp
PYTHON_VERSION ?= 3.12

# Repo/data locations (override if your layout differs)
REPO_ROOT ?= $(HOME)/repos/mlp
ML_DATA_DIR ?= $(REPO_ROOT)/data

# Helper to run commands inside the environment without interactive activation
CONDA_RUN = $(CONDA) run -n $(CONDA_ENV_NAME)
PIP = $(CONDA_RUN) python -m pip

help:
	@echo "Targets:"
	@echo "  env              Create Conda env $(CONDA_ENV_NAME) if missing (python $(PYTHON_VERSION))"
	@echo "  install          Install project and tools into env (pip install -e .)"
	@echo "  install-torch    Install PyTorch CPU-only build into env via conda"
	@echo "  hooks            Install env activate hooks to set ML_DATA_DIR=$(ML_DATA_DIR)"
	@echo "  notebook         Launch Jupyter Notebook in the env"
	@echo "  jupyter          Alias of 'notebook'"
	@echo "  shell            Print a command to activate the env and open a shell"
	@echo "  print-activate   Print the 'conda activate' command"
	@echo "  remove-env       Remove the Conda env (DANGEROUS)"
	@echo "  clean            No-op placeholder (kept for convention)"

# Create the env if it doesn't exist
env:
	@set -e; \
	if $(CONDA) env list | awk '{print $$1}' | grep -qx '$(CONDA_ENV_NAME)'; then \
	  echo "Conda env '$(CONDA_ENV_NAME)' already exists."; \
	else \
	  echo "Creating conda env '$(CONDA_ENV_NAME)' with python $(PYTHON_VERSION)..."; \
	  $(CONDA) create -y -n $(CONDA_ENV_NAME) python=$(PYTHON_VERSION); \
	fi

# Install everything into the env
install: env
	@set -e; \
	$(PIP) install --upgrade pip setuptools wheel; \
	# Install project in editable mode (installs deps from pyproject.toml)
	$(PIP) install -e .; \
	# Ensure Jupyter is available in the environment
	$(CONDA) install -y -n $(CONDA_ENV_NAME) jupyter >/dev/null 2>&1 || true; \
	echo "Install complete in env '$(CONDA_ENV_NAME)'."

# Optional: CPU-only PyTorch via the pytorch channel
install-torch: env
	@echo "Installing PyTorch (CPU-only) into env '$(CONDA_ENV_NAME)'..."
	@$(CONDA) install -y -n $(CONDA_ENV_NAME) torch torchvision torchaudio cpuonly -c torch
	@echo "PyTorch install complete."

# Install activation hooks to set ML_DATA_DIR/MLP_DATA_DIR on conda activate
hooks: env
	@set -e; \
	ENV_PREFIX="$$($(CONDA_RUN) python -c 'import os; print(os.environ.get("CONDA_PREFIX",""))')"; \
	if [ -z "$$ENV_PREFIX" ]; then echo "Could not determine CONDA_PREFIX"; exit 1; fi; \
	mkdir -p "$$ENV_PREFIX/etc/conda/activate.d" "$$ENV_PREFIX/etc/conda/deactivate.d"; \
	ACT_FILE="$$ENV_PREFIX/etc/conda/activate.d/mlp_env.sh"; \
	DEACT_FILE="$$ENV_PREFIX/etc/conda/deactivate.d/mlp_env.sh"; \
	printf '#!/bin/sh\nexport ML_DATA_DIR="%s"\nexport MLP_DATA_DIR="%s"\n' "$(ML_DATA_DIR)" "$(ML_DATA_DIR)" > "$$ACT_FILE"; \
	printf '#!/bin/sh\nunset ML_DATA_DIR\nunset MLP_DATA_DIR\n' > "$$DEACT_FILE"; \
	chmod +x "$$ACT_FILE" "$$DEACT_FILE"; \
	echo "Installed ML_DATA_DIR activation hooks in $$ENV_PREFIX (ML_DATA_DIR=$(ML_DATA_DIR))."

# Launch Jupyter Notebook from the env
notebook: env
	@echo "Starting Jupyter Notebook in env '$(CONDA_ENV_NAME)'..."
	@$(CONDA_RUN) jupyter notebook

jupyter: notebook

# Helpful: show how to activate the env in the user's shell
print-activate:
	@echo "Run: conda activate $(CONDA_ENV_NAME)"

shell: print-activate

# Danger: remove the env entirely
remove-env:
	@echo "Removing conda env '$(CONDA_ENV_NAME)'..."
	@$(CONDA) remove -y -n $(CONDA_ENV_NAME) --all
	@echo "Environment removed."

clean:
	@true
