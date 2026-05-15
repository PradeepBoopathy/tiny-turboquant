"""On-disk + on-device cache of scalar codebooks.

For the torch implementation we use a fast grid-based Lloyd-Max solver. It is
much faster than the scipy continuous-integral reference implementation and is
accurate enough for tests, demos, and small research runs.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch


def _cache_dir() -> Path:
    p = Path(os.environ.get("TURBOQUANT_CACHE", Path.home() / ".cache" / "turboquant"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def _sphere_coord_pdf_grid(d: int, grid: np.ndarray) -> np.ndarray:
    # The coordinate distribution of a random unit vector on S^(d-1) is sharply
    # concentrated near zero for large d. Work in logs for stability.
    import math
    from math import lgamma

    x = np.clip(grid, -1.0 + 1e-7, 1.0 - 1e-7)
    log_norm = lgamma(d / 2.0) - 0.5 * math.log(math.pi) - lgamma((d - 1) / 2.0)
    exponent = (d - 3) / 2.0
    log_pdf = log_norm + exponent * np.log1p(-(x * x))
    pdf = np.exp(log_pdf - np.max(log_pdf))
    pdf /= np.trapezoid(pdf, x)
    return pdf.astype(np.float64)


@lru_cache(maxsize=128)
def _grid_lloyd_centroids(d_pad: int, bits: int) -> np.ndarray:
    if not 1 <= bits <= 8:
        raise ValueError(f"bits must be in [1, 8], got {bits}")

    n_levels = 1 << bits
    # Keep package startup reasonable. This is a research codebook, not an
    # exact offline optimizer. Bincount makes the Lloyd step vectorized.
    n_grid = max(4096, n_levels * 256)
    x = np.linspace(-1.0 + 1e-7, 1.0 - 1e-7, n_grid, dtype=np.float64)
    w = _sphere_coord_pdf_grid(d_pad, x)
    w /= w.sum()

    cdf = np.cumsum(w)
    q = (np.arange(n_levels, dtype=np.float64) + 0.5) / n_levels
    c = np.interp(q, cdf, x)

    for _ in range(50):
        bounds = np.concatenate(([-np.inf], 0.5 * (c[:-1] + c[1:]), [np.inf]))
        bucket = np.searchsorted(bounds[1:-1], x)
        denom = np.bincount(bucket, weights=w, minlength=n_levels)
        numer = np.bincount(bucket, weights=x * w, minlength=n_levels)
        new_c = c.copy()
        valid = denom > 0
        new_c[valid] = numer[valid] / denom[valid]
        if np.max(np.abs(new_c - c)) < 1e-7:
            c = new_c
            break
        c = new_c

    return np.sort(c).astype(np.float32)


def get_centroids(
    d_pad: int,
    bits: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return centroid tensor of shape ``(2**bits,)`` on the target device."""
    if not 1 <= bits <= 8:
        raise ValueError(f"bits must be in [1, 8], got {bits}")

    fname = _cache_dir() / f"centroids_grid_d{d_pad}_b{bits}.npy"
    if fname.exists():
        c = np.load(fname)
    else:
        c = _grid_lloyd_centroids(int(d_pad), int(bits))
        np.save(fname, c)
    return torch.from_numpy(c).to(device=device, dtype=dtype).contiguous()
