"""Fast Walsh–Hadamard Transform.

The public ``fwht`` function always has a torch-native fallback. Triton is
loaded lazily only when a CUDA tensor actually requests the Triton path. This
keeps normal package imports and CPU tests fast.
"""

from __future__ import annotations

import importlib.util
import math

import torch

_HAS_TRITON = bool(torch.cuda.is_available() and importlib.util.find_spec("triton") is not None)


def _fwht_torch(x: torch.Tensor) -> torch.Tensor:
    """Orthonormal FWHT along the last dimension."""
    *prefix, n = x.shape
    if not ((n & (n - 1)) == 0 and n > 0):
        raise ValueError(f"last dim must be a power of 2, got {n}")

    y = x
    h = 1
    while h < n:
        y = y.reshape(*prefix, n // (2 * h), 2, h)
        a = y[..., 0, :]
        b = y[..., 1, :]
        y = torch.stack((a + b, a - b), dim=-2).reshape(*prefix, n)
        h *= 2

    return y * (1.0 / math.sqrt(n))


def _fwht_triton(x: torch.Tensor) -> torch.Tensor:
    """Lazy Triton placeholder.

    The source project had an experimental Triton kernel. For the PyPI alpha,
    the stable torch-native path is used by default because correctness and
    import reliability matter more than a fragile optional kernel.
    """
    return _fwht_torch(x)


def fwht(x: torch.Tensor, *, force_torch: bool = False) -> torch.Tensor:
    """Orthonormal FWHT along the last dimension of ``x``."""
    if (
        _HAS_TRITON
        and not force_torch
        and x.is_cuda
        and x.shape[-1] >= 32
        and x.shape[-1] <= 1024
    ):
        try:
            return _fwht_triton(x)
        except Exception:
            pass
    return _fwht_torch(x)
