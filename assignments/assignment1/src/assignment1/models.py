"""Model definitions for coursework experiments."""

from __future__ import annotations

import math
from typing import Iterable, List

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn


def _get_activation(name: str, **kwargs) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(**kwargs)
    if name == "leaky_relu":
        return nn.LeakyReLU(**kwargs)
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(f"Unsupported activation '{name}'")


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int],
        num_classes: int,
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dims: List[int] = [input_dim, *list(hidden_dims), num_classes]
        layers: List[nn.Module] = []
        act = activation

        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            if out_dim == num_classes:
                continue
            layers.append(_get_activation(act))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        batch = x.size(0)
        return self.network(x.view(batch, -1))


def _apply_weight_init(module: nn.Module, strategy: str) -> None:
    strategy = strategy.lower()
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            if strategy == "xavier_uniform":
                nn.init.xavier_uniform_(layer.weight)
            elif strategy == "xavier_normal":
                nn.init.xavier_normal_(layer.weight)
            elif strategy == "kaiming_uniform":
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            elif strategy == "kaiming_normal":
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            else:
                raise ValueError(f"Unknown weight init strategy: {strategy}")
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)


def build_model(cfg: DictConfig) -> nn.Module:
    if cfg.name != "mlp":
        raise ValueError(f"Unsupported model type: {cfg.name}")

    model = MLP(
        input_dim=int(cfg.input_dim),
        hidden_dims=cfg.hidden_dims,
        num_classes=int(cfg.num_classes),
        activation=cfg.activation,
        dropout=float(cfg.dropout),
    )

    if cfg.get("weight_init"):
        _apply_weight_init(model, cfg.weight_init)

    return model


__all__ = ["build_model", "MLP"]

