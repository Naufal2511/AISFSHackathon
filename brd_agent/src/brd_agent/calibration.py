"""Calibration helpers. The gate's thresholds assume tier-1 probabilities are calibrated.
Fit a temperature on a held-out labeled split, then apply it to tier-1 logits before building Briefs."""
from __future__ import annotations

import numpy as np


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(logits: np.ndarray, y: np.ndarray, grid=np.linspace(0.3, 5.0, 95)) -> float:
    best, bt = 1e9, 1.0
    for t in grid:
        p = _softmax(logits / t)[np.arange(len(y)), y]
        nll = -np.log(np.clip(p, 1e-9, 1)).mean()
        if nll < best:
            best, bt = nll, float(t)
    return bt


def ece(probs: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    conf, pred = probs.max(1), probs.argmax(1)
    acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(total)
