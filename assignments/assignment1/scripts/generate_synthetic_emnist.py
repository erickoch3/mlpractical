#!/usr/bin/env python
"""Generate an augmented EMNIST dataset for experimentation.

This script is meant to be orchestrated by DVC. It reads hyperparameters
from a params.yaml file, synthesises additional balanced samples using
Torchvision transforms, and saves the result under data/synthetic_emnist.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch
from torchvision import datasets, transforms
import yaml


def _build_transform(cfg: Dict[str, float]) -> transforms.Compose:
    ops: list = []
    rotation = float(cfg.get("rotation", 0.0))
    if rotation > 0:
        ops.append(transforms.RandomRotation(degrees=rotation))

    translate = float(cfg.get("translate", 0.0))
    if translate > 0:
        # torchvision expects values in range [0, 1] relative to image size
        frac = translate / 28.0
        ops.append(transforms.RandomAffine(degrees=0, translate=(frac, frac)))

    shear = float(cfg.get("shear", 0.0))
    if shear > 0:
        ops.append(transforms.RandomAffine(degrees=0, shear=shear))

    scale = cfg.get("scale", None)
    if isinstance(scale, (list, tuple)) and len(scale) == 2:
        ops.append(transforms.RandomAffine(degrees=0, scale=tuple(scale)))

    if cfg.get("horizontal_flip", False):
        ops.append(transforms.RandomHorizontalFlip())

    if cfg.get("random_resized_crop", False):
        ops.append(
            transforms.RandomResizedCrop(
                size=28,
                scale=tuple(cfg.get("random_resized_crop_scale", (0.8, 1.0))),
            )
        )

    ops.append(transforms.ToTensor())

    noise = float(cfg.get("gaussian_noise", 0.0))
    if noise > 0:
        ops.append(AdditiveGaussianNoise(std=noise))

    ops.append(
        transforms.Normalize(
            mean=(0.1307,),
            std=(0.3081,),
        )
    )

    return transforms.Compose(ops)


class AdditiveGaussianNoise(torch.nn.Module):
    def __init__(self, std: float = 0.05) -> None:
        super().__init__()
        self.std = std

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if not self.training:
            return tensor
        noise = torch.randn_like(tensor) * self.std
        return torch.clamp(tensor + noise, 0.0, 1.0)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_params(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"params file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        params = yaml.safe_load(handle)
    return params or {}


def sample_indices(class_indices: Dict[int, Iterable[int]], count: int) -> Iterable[int]:
    for key, indices in class_indices.items():
        if len(indices) == 0:
            raise RuntimeError(f"No samples available for class {key}.")
        yield from np.random.choice(indices, size=count, replace=True)


def main(args: argparse.Namespace) -> None:
    params = load_params(Path(args.params))
    cfg = params.get("synth_emnist", {})

    seed = int(cfg.get("seed", 42))
    seed_everything(seed)

    root = Path(cfg.get("raw_root", args.raw_root)).expanduser().resolve()
    output_dir = Path(cfg.get("output_dir", args.output_dir)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    num_variants = int(cfg.get("variants_per_sample", 1))
    samples_per_class = int(cfg.get("num_samples_per_class", 1000))
    transform_cfg = cfg.get("transforms", {})

    print(f"Loading EMNIST Balanced split from {root}")
    dataset = datasets.EMNIST(
        root=str(root),
        split="balanced",
        train=True,
        download=cfg.get("download", True),
    )

    transform = _build_transform(transform_cfg)

    class_indices: Dict[int, list[int]] = {}
    for idx in range(len(dataset)):
        _, target = dataset[idx]
        class_indices.setdefault(int(target), []).append(idx)

    images: list[torch.Tensor] = []
    labels: list[int] = []

    print("Generating augmented samples ...")
    for label, indices in class_indices.items():
        chosen = np.random.choice(indices, size=samples_per_class, replace=True)
        for idx in chosen:
            image, _ = dataset[idx]
            for _ in range(num_variants):
                augmented = transform(image)
                images.append(augmented)
                labels.append(label)

    tensor_images = torch.stack(images)
    tensor_labels = torch.tensor(labels, dtype=torch.long)

    artifact_path = output_dir / "synthetic_emnist.pt"
    metadata_path = output_dir / "metadata.yaml"

    torch.save({"images": tensor_images, "labels": tensor_labels}, artifact_path)

    with metadata_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "seed": seed,
                "variants_per_sample": num_variants,
                "num_samples_per_class": samples_per_class,
                "transform": transform_cfg,
                "total_samples": int(tensor_labels.numel()),
            },
            handle,
        )

    print(f"Saved augmented dataset to {artifact_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic EMNIST data via torchvision.")
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    parser.add_argument("--raw-root", default="~/.cache/torch/emnist", help="Torchvision EMNIST root")
    parser.add_argument("--output-dir", default="data/synthetic_emnist/raw", help="Output directory")
    args = parser.parse_args()
    main(args)
