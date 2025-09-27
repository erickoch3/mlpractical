from __future__ import annotations

import os
from pathlib import Path
import numpy as np


def ensure_mlp_data_dir() -> Path:
    if os.environ.get('MLP_DATA_DIR'):
        return Path(os.environ['MLP_DATA_DIR'])
    here = Path.cwd().resolve()
    for p in [here] + list(here.parents):
        candidate = p / 'data'
        if (candidate / 'mnist-train.npz').exists():
            os.environ['MLP_DATA_DIR'] = str(candidate)
            return candidate
    raise RuntimeError('Could not locate data directory with mnist-*.npz')


def scale01(x: np.ndarray) -> np.ndarray:
    xmax = np.max(x)
    if xmax > 1.5:
        return (x / 255.0).astype(np.float32)
    return x.astype(np.float32)


def poisson_encode(inputs: np.ndarray, T: int, rate_hz: float = 30.0, dt: float = 0.005,
                   rng: np.random.RandomState | None = None) -> np.ndarray:
    if rng is None:
        rng = np.random.RandomState(0)
    B, D = inputs.shape
    lam = np.clip(inputs, 0.0, 1.0) * rate_hz * dt
    spikes = rng.rand(B, T, D) < lam[:, None, :]
    return spikes.astype(np.float32)


def k_wta(v: np.ndarray, k: int | None) -> np.ndarray:
    if k is None or k <= 0:
        return np.zeros_like(v, dtype=np.float32)
    k = min(k, v.shape[0])
    idx = np.argpartition(v, -k)[-k:]
    spikes = np.zeros_like(v, dtype=np.float32)
    spikes[idx] = (v[idx] > 0).astype(np.float32)
    return spikes


class TwoLayerHebbianSNN:
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int,
                 dt: float = 0.005, tau_h: float = 0.02, tau_o: float = 0.02,
                 vth_h: float = 1.0, vth_o: float = 1.0, seed: int = 0,
                 k_hidden: int | None = 32, k_output: int | None = 1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dt = dt
        self.alpha_h = float(np.exp(-dt / tau_h))
        self.alpha_o = float(np.exp(-dt / tau_o))
        self.vth_h = vth_h
        self.vth_o = vth_o
        self.k_hidden = k_hidden
        self.k_output = k_output
        rng = np.random.RandomState(seed)
        self.W1 = (0.01 * rng.randn(input_dim, hidden_dim)).astype(np.float32)
        self.W2 = (0.01 * rng.randn(hidden_dim, num_classes)).astype(np.float32)

    def forward_counts(self, spikes: np.ndarray) -> np.ndarray:
        B, T, D = spikes.shape
        assert D == self.input_dim
        counts = np.zeros((B, self.num_classes), dtype=np.float32)
        for b in range(B):
            v_h = np.zeros(self.hidden_dim, dtype=np.float32)
            v_o = np.zeros(self.num_classes, dtype=np.float32)
            for _t in range(T):
                x_t = spikes[b, _t]
                v_h = self.alpha_h * v_h + x_t @ self.W1
                s_h = k_wta(v_h, self.k_hidden)
                v_h = v_h - self.vth_h * s_h
                v_o = self.alpha_o * v_o + s_h @ self.W2
                s_o = k_wta(v_o, self.k_output)
                v_o = v_o - self.vth_o * s_o
                counts[b] += s_o
        return counts

    def predict(self, spikes: np.ndarray) -> np.ndarray:
        counts = self.forward_counts(spikes)
        return counts.argmax(axis=1)

    def train_step(self, spikes: np.ndarray, y_int: np.ndarray,
                   eta1: float = 0.01, eta2: float = 0.02, eta2_neg: float = 0.01,
                   teacher_rate_hz: float = 60.0, decay1: float = 5e-5, decay2: float = 5e-5,
                   rng: np.random.RandomState | None = None):
        if rng is None:
            rng = np.random.RandomState(0)
        B, T, D = spikes.shape
        C = self.num_classes
        p_teacher = min(1.0, teacher_rate_hz * self.dt)
        for b in range(B):
            v_h = np.zeros(self.hidden_dim, dtype=np.float32)
            v_o = np.zeros(C, dtype=np.float32)
            yb = int(y_int[b])
            for _t in range(T):
                x_t = spikes[b, _t]  # (D,)
                # Hidden dynamics
                v_h = self.alpha_h * v_h + x_t @ self.W1
                s_h = k_wta(v_h, self.k_hidden)
                v_h = v_h - self.vth_h * s_h
                # Output dynamics
                v_o = self.alpha_o * v_o + s_h @ self.W2
                s_o_nat = k_wta(v_o, self.k_output)
                v_o = v_o - self.vth_o * s_o_nat
                # Teacher spike
                teach = 1.0 if rng.rand() < p_teacher else 0.0
                y_vec = np.zeros(C, dtype=np.float32); y_vec[yb] = teach
                post_pos = np.maximum(s_o_nat, y_vec)
                post_neg = s_o_nat.copy(); post_neg[yb] = 0.0
                # Oja's rule for W1 (stabilizes growth)
                if s_h.any():
                    J = np.where(s_h > 0)[0]
                    x_col = x_t.astype(np.float32)[:, None]
                    sj = s_h[J][None, :]
                    self.W1[:, J] += eta1 * (x_col @ sj - (sj * sj) * self.W1[:, J])
                # Supervised Hebbian and anti-Hebbian for W2
                self.W2 += eta2 * np.outer(s_h, post_pos).astype(np.float32)
                if eta2_neg > 0:
                    self.W2 -= eta2_neg * np.outer(s_h, post_neg).astype(np.float32)
                # Weight decay
                if decay1 > 0: self.W1 *= (1.0 - decay1)
                if decay2 > 0: self.W2 *= (1.0 - decay2)
            # Column-wise normalization each pattern
            def norm_cols(W):
                n = np.linalg.norm(W, axis=0, keepdims=True) + 1e-6
                W /= n
            norm_cols(self.W1); norm_cols(self.W2)
        # Clip
        np.clip(self.W1, -1.0, 1.0, out=self.W1)
        np.clip(self.W2, -1.0, 1.0, out=self.W2)

