# -*- coding: utf-8 -*-
"""Utility helpers for the MLP package."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Tuple


def _normalise_required(required_files: Iterable[str] | None) -> Tuple[str, ...]:
    if required_files is None:
        return tuple()
    if isinstance(required_files, (str, bytes)):
        return (str(required_files),)
    return tuple(str(name) for name in required_files)


def _has_required_files(data_dir: Path, required_files: Tuple[str, ...]) -> bool:
    return all((data_dir / name).exists() for name in required_files)


def ensure_mlp_data_dir(required_files: Iterable[str] | None = None) -> Path:
    """Locate the MLP data directory and ensure ``MLP_DATA_DIR`` is set.

    Args:
        required_files: Optional collection of file names that must exist
            inside the data directory. If provided, the directory is only
            considered valid if all of these files are present.

    Returns:
        The resolved ``Path`` to the data directory.

    Raises:
        RuntimeError: If no suitable data directory can be located.
    """
    required = _normalise_required(required_files)

    # Check whether the environment variable is already defined and valid.
    env_value = os.environ.get("MLP_DATA_DIR")
    if env_value:
        candidate = Path(env_value).expanduser().resolve()
        if candidate.is_dir() and _has_required_files(candidate, required):
            return candidate

    # Potential starting points: current working directory and this module.
    module_path = Path(__file__).resolve()
    search_roots = {
        Path.cwd().resolve(),
        module_path.parent,       # .../mlp
        module_path.parents[1],   # repository root (mlp/..)
    }

    for root in search_roots:
        for parent in (root,) + tuple(root.parents):
            candidate = parent / "data"
            if not candidate.is_dir():
                continue
            if required and not _has_required_files(candidate, required):
                continue
            os.environ["MLP_DATA_DIR"] = str(candidate)
            return candidate

    msg = (
        "Could not locate MLP data directory. Set the MLP_DATA_DIR environment "
        "variable or ensure the repository's data/ directory is present."
    )
    raise RuntimeError(msg)


__all__ = ["ensure_mlp_data_dir"]
