"""Utility helpers for training orchestration."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn

log = logging.getLogger(__name__)


@dataclass
class StepOutput:
    loss: float
    accuracy: float


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    preds = predictions.argmax(dim=1)
    correct = (preds == targets).float().sum().item()
    return correct / targets.numel()


def save_matplotlib_grid(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    log.info("Saved figure to %s", path)


def step(
    model: nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    max_grad_norm: Optional[float] = None,
) -> StepOutput:
    inputs, targets = batch
    inputs = inputs.to(device)
    targets = targets.to(device)

    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    with torch.cuda.amp.autocast(enabled=scaler is not None):
        outputs = model(inputs)
        loss = criterion(outputs, targets)

    if optimizer is not None:
        if scaler is not None:
            scaler.scale(loss).backward()
            if max_grad_norm:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

    acc = accuracy(outputs.detach(), targets)
    return StepOutput(loss=float(loss.detach().cpu()), accuracy=acc)


def maybe_log_to_wandb(metrics: Dict[str, float], step: int) -> None:
    try:
        import wandb

        if wandb.run is not None:
            wandb.log(metrics, step=step)
    except ImportError:
        pass


