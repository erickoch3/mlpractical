from __future__ import annotations

import os
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # repo root for 'mlp' package

from mlp import data_providers
from mlp.hebbian import ensure_mlp_data_dir, scale01, poisson_encode, TwoLayerHebbianSNN


def main():
    ensure_mlp_data_dir()
    BATCH = 64
    TRAIN_BATCHES = 30  # quicker pass
    VALID_BATCHES = 10
    input_dim = 28 * 28
    hidden_dim = 256
    num_classes = 10
    T = 12

    train_dp = data_providers.MNISTDataProvider('train', batch_size=BATCH, max_num_batches=TRAIN_BATCHES, shuffle_order=True)
    valid_dp = data_providers.MNISTDataProvider('valid', batch_size=BATCH, max_num_batches=VALID_BATCHES, shuffle_order=False)

    snn = TwoLayerHebbianSNN(input_dim, hidden_dim, num_classes, dt=0.005, tau_h=0.02, tau_o=0.02,
                             vth_h=1.0, vth_o=1.0, seed=3, k_hidden=16, k_output=1)

    rng = np.random.RandomState(123)
    EPOCHS = 2

    for ep in range(1, EPOCHS + 1):
        for xb, yb in train_dp:
            xb = scale01(xb)
            y_int = yb.argmax(axis=1).astype(int)
            spikes = poisson_encode(xb, T=T, rate_hz=30.0, dt=snn.dt, rng=rng)
            snn.train_step(spikes, y_int, eta1=0.01, eta2=0.04, eta2_neg=0.02,
                           teacher_rate_hz=100.0, decay1=1e-4, decay2=1e-4, rng=rng)

        # Validate
        correct, total = 0, 0
        for xb, yb in valid_dp:
            xb = scale01(xb)
            y_int = yb.argmax(axis=1).astype(int)
            spikes = poisson_encode(xb, T=T, rate_hz=30.0, dt=snn.dt, rng=rng)
            preds = snn.predict(spikes)
            correct += (preds == y_int).sum()
            total += xb.shape[0]
        acc = correct / max(1, total)
        print(f"Epoch {ep:02d} | valid acc: {acc:.3f}")


if __name__ == "__main__":
    main()
