"""Pick the best available torch device: CUDA (NVIDIA), MPS (Apple Silicon) or CPU."""
from __future__ import annotations

import torch


def best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
