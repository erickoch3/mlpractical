import os
from pathlib import Path

import numpy as np
import pytest
import sys


def _data_dir() -> str:
    """Return absolute path to the repository's data directory."""
    # tests/ -> repo root -> data
    return str(Path(__file__).resolve().parents[1] / "data")


@pytest.fixture(autouse=True)
def set_mlp_data_dir(monkeypatch):
    """Ensure MLP_DATA_DIR is set for all tests."""
    data_path = _data_dir()
    if not os.path.isdir(data_path):
        pytest.skip(f"Data directory not found at {data_path}")
    monkeypatch.setenv("MLP_DATA_DIR", data_path)


class TestMNISTDataProvider:
    def test_one_hot_encoding_properties(self):
        # Ensure repository root on path for package import
        repo_root = str(Path(__file__).resolve().parents[1])
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        import mlp.data_providers as data_providers

        batch_size = 5
        max_batches = 2
        dp = data_providers.MNISTDataProvider(
            which_set="valid",
            batch_size=batch_size,
            max_num_batches=max_batches,
            shuffle_order=False,
        )

        seen_batches = 0
        for inputs, targets in dp:
            # Inputs should be 2D (batch, features)
            assert inputs.shape[0] == batch_size
            assert inputs.ndim == 2

            # One-hot targets checks (from introduction notebook)
            # - values are 0 or 1
            assert np.all(np.logical_or(targets == 0.0, targets == 1.0)), (
                "Targets should only contain 0s and 1s"
            )
            # - each row sums to 1
            row_sums = targets.sum(axis=1)
            assert np.allclose(row_sums, 1.0), (
                f"Each row should sum to 1, got sums: {row_sums}"
            )
            # - 2D array with 10 classes
            assert targets.ndim == 2, (
                f"Targets should be 2D array, got shape: {targets.shape}"
            )
            assert targets.shape[1] == 10, (
                f"Should have 10 classes for MNIST, got: {targets.shape[1]}"
            )

            seen_batches += 1

        assert seen_batches == max_batches


class TestMetOfficeDataProvider:
    @pytest.mark.parametrize("window_size", [2, 5, 10])
    def test_window_and_batch_shapes(self, window_size):
        # Ensure repository root on path for package import
        repo_root = str(Path(__file__).resolve().parents[1])
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        import mlp.data_providers as data_providers

        batch_size = 3
        max_batches = 2

        dp = data_providers.MetOfficeDataProvider(
            window_size=window_size,
            batch_size=batch_size,
            max_num_batches=max_batches,
            shuffle_order=False,
        )

        seen_batches = 0
        for inputs, targets in dp:
            expected_input_shape = (batch_size, window_size - 1)
            expected_target_shape = (batch_size,)

            assert inputs.shape == expected_input_shape, (
                f"Input shape mismatch: got {inputs.shape}, expected {expected_input_shape}"
            )
            assert targets.shape == expected_target_shape, (
                f"Target shape mismatch: got {targets.shape}, expected {expected_target_shape}"
            )

            # Basic dtype sanity
            assert np.issubdtype(inputs.dtype, np.floating)
            assert np.issubdtype(targets.dtype, np.floating)

            seen_batches += 1

        assert seen_batches == max_batches
