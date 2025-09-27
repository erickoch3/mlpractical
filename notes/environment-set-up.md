# Environment Setup

This guide assumes Conda is already installed locally. We provide Makefile targets to create and manage the environment, install dependencies, and run common tools.

This course uses Python 3 with NumPy, SciPy, and Jupyter. We use project‑specific Conda environments so dependencies are isolated from your system Python.

## Creating the Conda Environment

Create the environment (if it doesn’t exist) and install project dependencies:

```bash
make env
make install
```

- The environment is named `mlp` by default (see `Makefile`).
- `make install` installs the project in editable mode (`pip install -e .`) inside the environment and ensures Jupyter is available.

Optional: install PyTorch (CPU‑only) via Conda:

```bash
make install-torch
```

To use the environment interactively in your shell:

```bash
conda activate mlp
```

## Getting the Course Code

The course code is available in a Git repository on GitHub: https://github.com/cortu01/mlpractical

Git is a version control system, and GitHub hosts Git repositories. We use Git to distribute code for labs and assignments. For Git beginners, see this guide: http://rogerdudler.github.io/git-guide/

### Cloning the Repository

Advanced Git users may create a private fork of `cortu01/mlpractical` for syncing work between machines. Do NOT create a public fork.

Clone the repository to your home directory:

```bash
git clone https://github.com/cortu01/mlpractical.git ~/mlpractical
cd ~/mlpractical
```

List contents (Windows: `dir /a`):

```bash
ls -a
```

You’ll see:
- `data`: Data files for labs and assignments
- `mlp`: The custom Python package for this course
- `notebooks`: Jupyter notebook files for each lab
- `.git`: Git repository data (don’t modify directly)

Configure Git with your details (optional):

```bash
git config --global user.name "Your Name"
git config --global user.email "your-email@sms.ed.ac.uk"
```

### Understanding Branches

We use Git branches to organize course content. Each lab has its own branch (e.g., `mlp2025-26/lab1`, `mlp2025-26/lab2`).

Check current branch:

```bash
git status
```

Switch to the first lab branch:

```bash
git checkout mlp2025-26/lab1
```

## Installing the MLP Python Package

The `mlp` directory contains a custom NumPy‑based neural network framework for this course. Install it into the Conda environment with:

```bash
make install
```

This installs the package in editable mode, so code changes are immediately available without reinstalling.

## Setting Up the Data Directory

The `data` directory contains files used in labs and assignments. You can have `MLP_DATA_DIR` set automatically whenever the environment is activated by installing a small activation hook:

```bash
make hooks
```

This creates activation scripts in the environment so `MLP_DATA_DIR` points to `~/mlpractical/data` when the `mlp` env is active.

## Starting Jupyter Notebooks

Your environment is now ready! To start working with the lab notebooks:

1. Make sure you're in the `mlpractical` directory
2. Launch Jupyter via the environment:

```bash
make notebook
```

3. In the browser interface, navigate to the `notebooks` directory
4. Open `01_Introduction.ipynb` to start the first lab

## Minimal Setup (Quick Start)

1. Clone the repository and switch to the appropriate lab branch:
```bash
git clone https://github.com/cortu01/mlpractical.git ~/mlpractical
cd ~/mlpractical
git checkout mlp2025-26/lab1
```

2. Create the env and install the package and tools:
```bash
make env
make install
```

3. Optional: install PyTorch (CPU‑only):
```bash
make install-torch
```

4. Optional: set up data directory activation hooks:
```bash
make hooks
```

5. Start Jupyter Notebook:
```bash
make notebook
```

Or activate the env for interactive work:
```bash
conda activate mlp
```

