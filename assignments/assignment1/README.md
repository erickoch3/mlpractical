# Coursework 1 Experimentation Stack

This directory hosts a self-contained workflow for **MLP Coursework 1** (spec released on 13 October 2025 and due at 12:00 on 24 October 2025) built around Hydra-driven configuration, DVC-managed synthetic data generation, and experiment tracking with Weights & Biases (W&B). The official brief and report template live in [`docs/`](docs/) for quick reference while experimenting.

## Project layout

- [`docs/`](docs/) – original coursework specification (`MLP_2025_CW1_Spec.pdf`) and report template (`MLP2025_26_CW1_Template.pdf`).
- [`conf/`](conf/) – Hydra configuration tree organised by data, model, training, logging, and Hydra runtime overrides.
- [`src/assignment1/`](src/assignment1/) – Python package containing data pipelines (`data.py`), model definitions (`models.py`), utilities (`utils.py`), and the Hydra entry point (`train.py`).
- [`scripts/generate_synthetic_emnist.py`](scripts/generate_synthetic_emnist.py) – torchvision-based augmentation pipeline orchestrated via DVC to expand the EMNIST training set.
- [`dvc.yaml`](dvc.yaml) / [`params.yaml`](params.yaml) – declare the synthetic data stage and its hyperparameters.
- [`pyproject.toml`](pyproject.toml) – dependency specification (Hydra, Torch, torchvision, W&B, DVC, Matplotlib, TensorBoard, etc.).

## Getting started

1. **Create / activate the course conda environment** (as recommended in the spec) or any Python ≥ 3.9 environment.
2. Install the tooling in editable mode:
   ```bash
   pip install -e assignments/assignment1[dev]
   ```
3. Authenticate W&B if you intend to log runs:
   ```bash
   wandb login
   ```
4. Verify Hydra is available:
   ```bash
   python -c "import hydra; print(hydra.__version__)"
   ```

## Data management with DVC + torchvision

- Synthetic data parameters are in `params.yaml` under the `synth_emnist` key (seed, augmentations, per-class budget).
- Generate the augmented dataset (balanced over the 47-class EMNIST label set) with:
  ```bash
  cd assignments/assignment1
  dvc repro synth_emnist
  ```
  The command calls `scripts/generate_synthetic_emnist.py`, which downloads EMNIST (if required), applies torchvision augmentations, and writes `data/synthetic_emnist/raw/synthetic_emnist.pt` along with metadata for provenance.
- `dvc.yaml` tracks the resulting directory so that derived artifacts remain reproducible and shareable without pushing the tensor data to Git.

> **Note:** DVC is declared as a dependency but not pre-installed on the lab machines; run `pip install dvc` inside your environment before invoking the stage.

## Running experiments with Hydra

All experiments are controlled via `src/assignment1/train.py`:

```bash
python assignments/assignment1/src/assignment1/train.py
```

Key Hydra features enabled out-of-the-box:

- **Parameter sweeps**: optimise hidden widths and dropout, for example
  ```bash
  python assignments/assignment1/src/assignment1/train.py -m \
    model.hidden_dims='[256,256,128]' \
    model.dropout=0.2,0.4 \
    training.optimizer.params.lr=0.001,0.0003
  ```
- **Optuna sweeper**: supply a search space via overrides, e.g.
  ```bash
  python assignments/assignment1/src/assignment1/train.py \
    hydra/sweeper=optuna \
    'model.hidden_dims=[128,256,512]' \
    'model.dropout=tag(log,0.1,0.5)' \
    training.optimizer.params.lr='tag(log,1e-4,5e-3)'
  ```
  Hydra stores single-run outputs in `runs/<timestamp>` and sweeps under `multirun/`, both excluded from Git but ready for inspection.

## Experiment tracking and visualisation

- **Weights & Biases** (enabled by default): the train script initializes a run named after `experiment_name`, logs metrics (`train/`, `validation/`, `test/` namespaces), and attaches sample grids. Set `WANDB_MODE=offline` to buffer results locally.
- **TensorBoard**: scalars are mirrored to `${hydra.run.dir}/tensorboard`. Launch TensorBoard with `tensorboard --logdir runs` from inside `assignments/assignment1`.
- **Matplotlib figures**: the first training batch is exported to `figures/samples_epoch0.png`; extend this pattern for coursework report visuals as you explore dropout, penalties, and custom activations.

## Integrating coursework tasks

- **Task 1 (Problem identification)**: reproduce the baseline curves by running the default config (EMNIST Balanced split, three hidden-layer MLP, AdamW) and logging results to both W&B and TensorBoard for comparison against Figure 1 in the spec.
- **Task 2 (Regularization study)**: sweep over dropout rates (`model.dropout`), weight decay (`training.optimizer.params.weight_decay`), and label smoothing (`training.label_smoothing`). The synthetic dataset pipeline enables ablations contrasting augmented vs. vanilla EMNIST.
- **Task 3 (Report)**: use the template in `docs/MLP2025_26_CW1_Template.pdf` and archive key plots/metrics from Hydra outputs under `runs/` to streamline report assembly.

## Next steps

1. Confirm DVC (`dvc repro`) and the training entrypoint both execute in your environment.
2. Extend the Hydra config tree with specific experiments for dropout, L1/L2 penalties, and the provided custom activation layer.
3. Begin drafting the report, citing the logged metrics/figures generated through this stack.

For further detail, consult the coursework specification (`docs/MLP_2025_CW1_Spec.pdf`, pages 1–11) alongside Lab 5/6 materials referenced in Section 1 of the brief.
