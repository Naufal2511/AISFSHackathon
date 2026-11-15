"""One device choice for every model: CUDA if there is an NVIDIA GPU, else the Apple GPU (MPS) on a Mac, else CPU.
PEN6_DEVICE=cpu|mps|cuda forces it (for example if MPS misbehaves on one laptop)."""
from __future__ import annotations

import os


def pick_device() -> str:
    import torch
    want = (os.environ.get("PEN6_DEVICE") or "").strip().lower()
    if want in ("cpu", "mps", "cuda"):
        if want == "cuda" and not torch.cuda.is_available():
            return "cpu"
        if want == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            return "cpu"
        return want
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
