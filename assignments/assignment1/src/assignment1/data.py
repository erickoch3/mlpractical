"""Data loading utilities for EMNIST coursework experiments."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch
from omegaconf import DictConfig
from torch.utils.data import ConcatDataset, DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

log = logging.getLogger(__name__)


def _normalize_transform(mean: float, std: float) -> transforms.Normalize:
    return transforms.Normalize(mean=(mean,), std=(std,))


def _augmentation_pipeline(cfg: DictConfig, mean: float, std: float) -> transforms.Compose:
    ops: list = []
    aug = cfg.get("augmentations", {}) if cfg is not None else {}

    rotation = float(aug.get("rotation", 0.0))
    if rotation > 0:
        ops.append(transforms.RandomRotation(rotation))

    translate = float(aug.get("translate", 0.0))
    if translate > 0:
        frac = translate / 28.0
        ops.append(transforms.RandomAffine(degrees=0, translate=(frac, frac)))

    shear = float(aug.get("shear", 0.0))
    if shear > 0:
        ops.append(transforms.RandomAffine(degrees=0, shear=shear))

    if aug.get("random_resized_crop"):
        scale = aug.get("random_resized_crop_scale", (0.8, 1.0))
        ops.append(transforms.RandomResizedCrop(28, scale=tuple(scale)))

    if aug.get("horizontal_flip", False):
        ops.append(transforms.RandomHorizontalFlip())

    ops.append(transforms.ToTensor())

    noise = float(aug.get("gaussian_noise", 0.0))
    if noise > 0:
        ops.append(AdditiveGaussianNoise(noise))

    ops.append(_normalize_transform(mean, std))
    return transforms.Compose(ops)


def _evaluation_pipeline(mean: float, std: float) -> transforms.Compose:
    return transforms.Compose([transforms.ToTensor(), _normalize_transform(mean, std)])


class AdditiveGaussianNoise(torch.nn.Module):
    def __init__(self, std: float = 0.05) -> None:
        super().__init__()
        self.std = std

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        noise = torch.randn_like(tensor) * self.std
        return torch.clamp(tensor + noise, 0.0, 1.0)


def _split_indices(length: int, val_split: float, seed: int) -> Tuple[list[int], list[int]]:
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(length, generator=generator).tolist()
    val_length = int(length * val_split)
    val_indices = indices[:val_length]
    train_indices = indices[val_length:]
    return train_indices, val_indices


def _load_synthetic_dataset(path: Path, mean: float, std: float) -> Optional[TensorDataset]:
    if not path.exists():
        log.warning("Synthetic dataset not found at %s; continuing without it.", path)
        return None

    artifact = torch.load(path, map_location="cpu")
    images = artifact["images"].float()
    labels = artifact["labels"].long()
    if images.ndim == 3:
        images = images.unsqueeze(1)
    # Ensure normalisation is consistent
    norm = _normalize_transform(mean, std)
    images = norm(images)
    log.info("Loaded %d synthetic samples from %s", len(labels), path)
    return TensorDataset(images, labels)


def prepare_dataloaders(cfg: DictConfig) -> Dict[str, DataLoader]:
    data_cfg = cfg.data
    training_cfg = cfg.training

    mean = float(data_cfg.normalization.mean)
    std = float(data_cfg.normalization.std)

    train_transform = _augmentation_pipeline(data_cfg, mean, std)
    eval_transform = _evaluation_pipeline(mean, std)

    root = Path(data_cfg.root).expanduser()
    root.mkdir(parents=True, exist_ok=True)

    base_dataset = datasets.EMNIST(
        root=str(root),
        split="balanced",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    train_indices, val_indices = _split_indices(len(base_dataset), data_cfg.val_split, training_cfg.seed)

    train_dataset = Subset(
        datasets.EMNIST(
            root=str(root),
            split="balanced",
            train=True,
            download=False,
            transform=train_transform,
        ),
        train_indices,
    )

    val_dataset = Subset(
        datasets.EMNIST(
            root=str(root),
            split="balanced",
            train=True,
            download=False,
            transform=eval_transform,
        ),
        val_indices,
    )

    test_dataset = datasets.EMNIST(
        root=str(root),
        split="balanced",
        train=False,
        download=True,
        transform=eval_transform,
    )

    if bool(data_cfg.use_synthetic):
        synthetic_path = Path(data_cfg.synthetic_artifact)
        synthetic_dataset = _load_synthetic_dataset(synthetic_path, mean, std)
        if synthetic_dataset is not None:
            train_dataset = ConcatDataset([train_dataset, synthetic_dataset])

    loader_kwargs = {
        "batch_size": int(data_cfg.batch_size),
        "num_workers": int(data_cfg.num_workers),
        "pin_memory": bool(data_cfg.pin_memory),
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=bool(data_cfg.shuffle),
        drop_last=True,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}


__all__ = ["prepare_dataloaders"]

