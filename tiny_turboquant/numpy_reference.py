"""
Reference implementation of TurboQuant.

Paper: Zandieh, Daliri, Hadian, Mirrokni (2025).
  "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
  arXiv:2504.19874

This file implements both:
    - TurboQuantMSE   : Algorithm 1, optimised for L2/MSE distortion.
    - TurboQuantProd  : Algorithm 2, unbiased inner-product estimator
                        (MSE quantizer at b-1 bits + 1-bit QJL on residual).

The random rotation is realised by a randomized Fast Walsh-Hadamard
Transform (sign-flip -> Hadamard -> permutation -> sign-flip -> Hadamard).
This is O(d log d) instead of O(d^2) for a dense Gaussian rotation, and
empirically gives the same coordinate-wise Beta concentration that the
paper relies on.

Inputs are assumed (or rescaled) to unit L2 norm; norms are stored
separately at fp16/fp32 as the paper recommends.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad
from scipy.special import gammaln


# ---------------------------------------------------------------------------
# 1. Beta density of a coordinate of a uniform point on S^{d-1}
# ---------------------------------------------------------------------------

def sphere_coord_pdf(d: int):
    """f_X(x) for a coordinate of a random point on the unit sphere in R^d."""
    log_norm = gammaln(d / 2.0) - 0.5 * math.log(math.pi) - gammaln((d - 1) / 2.0)
    norm = math.exp(log_norm)
    exponent = (d - 3) / 2.0

    def pdf(x: float) -> float:
        if x <= -1.0 or x >= 1.0:
            return 0.0
        return norm * (1.0 - x * x) ** exponent

    return pdf


# ---------------------------------------------------------------------------
# 2. Lloyd-Max scalar quantizer for an arbitrary 1-D pdf on [-1, 1]
# ---------------------------------------------------------------------------

def lloyd_max_centroids(
    pdf,
    n_levels: int,
    support: tuple[float, float] = (-1.0, 1.0),
    n_iter: int = 200,
    tol: float = 1e-10,
) -> np.ndarray:
    """Solve the continuous k-means / Lloyd-Max problem on `support`."""
    lo, hi = support
    # Initialise centroids at quantiles of the pdf (works far better than uniform).
    grid = np.linspace(lo, hi, 4096)
    px = np.array([pdf(x) for x in grid])
    cdf = np.cumsum(px)
    cdf /= cdf[-1]
    qs = (np.arange(n_levels) + 0.5) / n_levels
    centroids = np.interp(qs, cdf, grid)

    prev_cost = np.inf
    for _ in range(n_iter):
        # Voronoi boundaries are midpoints between sorted centroids.
        c = np.sort(centroids)
        bounds = np.concatenate(([lo], 0.5 * (c[:-1] + c[1:]), [hi]))

        new_centroids = np.empty_like(c)
        cost = 0.0
        for i in range(n_levels):
            a, b = bounds[i], bounds[i + 1]
            num, _ = quad(lambda x: x * pdf(x), a, b, limit=200)
            den, _ = quad(pdf, a, b, limit=200)
            new_centroids[i] = num / den if den > 0 else 0.5 * (a + b)
            mse, _ = quad(
                lambda x, ci=new_centroids[i]: (x - ci) ** 2 * pdf(x),
                a, b, limit=200,
            )
            cost += mse

        if abs(prev_cost - cost) < tol:
            centroids = new_centroids
            break
        prev_cost = cost
        centroids = new_centroids

    return np.sort(centroids)


# ---------------------------------------------------------------------------
# 3. Randomized Fast Walsh-Hadamard Transform as the random rotation
# ---------------------------------------------------------------------------

def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _fwht(a: np.ndarray) -> np.ndarray:
    """Normalised Fast Walsh-Hadamard Transform along the last axis (orthonormal).

    Length must be a power of two. The matrix realised is H / sqrt(n) where H
    is the unnormalised Hadamard matrix, so the transform is its own inverse.
    """
    a = a.astype(np.float64, copy=True)
    n = a.shape[-1]
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            x = a[..., i : i + h].copy()
            y = a[..., i + h : i + 2 * h].copy()
            a[..., i : i + h] = x + y
            a[..., i + h : i + 2 * h] = x - y
        h *= 2
    return a / math.sqrt(n)


@dataclass
class RandomRotation:
    """Random orthonormal rotation built from sign-flips and FWHTs."""

    d: int
    d_pad: int
    s1: np.ndarray  # +-1
    s2: np.ndarray  # +-1
    perm: np.ndarray
    inv_perm: np.ndarray

    @classmethod
    def make(cls, d: int, rng: np.random.Generator) -> "RandomRotation":
        d_pad = _next_pow2(d)
        s1 = rng.choice([-1.0, 1.0], size=d_pad)
        s2 = rng.choice([-1.0, 1.0], size=d_pad)
        perm = rng.permutation(d_pad)
        inv_perm = np.argsort(perm)
        return cls(d=d, d_pad=d_pad, s1=s1, s2=s2, perm=perm, inv_perm=inv_perm)

    def _pad(self, x: np.ndarray) -> np.ndarray:
        if self.d == self.d_pad:
            return x
        pad_shape = list(x.shape)
        pad_shape[-1] = self.d_pad - self.d
        return np.concatenate([x, np.zeros(pad_shape)], axis=-1)

    def apply(self, x: np.ndarray) -> np.ndarray:
        # M = H S2 P H S1  (each H is the orthonormal FWHT)
        y = self._pad(x) * self.s1
        y = _fwht(y)
        y = y[..., self.perm] * self.s2
        y = _fwht(y)
        return y

    def apply_T(self, y: np.ndarray) -> np.ndarray:
        # M^T = S1 H P^T S2 H
        z = _fwht(y) * self.s2
        z_unperm = np.empty_like(z)
        z_unperm[..., self.perm] = z
        z = _fwht(z_unperm) * self.s1
        return z[..., : self.d]


# ---------------------------------------------------------------------------
# 4. TurboQuant-MSE  (Algorithm 1)
# ---------------------------------------------------------------------------

class TurboQuantMSE:
    def __init__(self, d: int, bits: int, seed: int = 0):
        if bits < 1:
            raise ValueError("bits must be >= 1")
        self.d = d
        self.bits = bits
        rng = np.random.default_rng(seed)
        self.rot = RandomRotation.make(d, rng)
        # Beta density for the rotated coordinates lives on [-1, 1].
        # In the padded space it's effectively scaled by sqrt(d/d_pad);
        # the difference is negligible for our purposes.
        pdf = sphere_coord_pdf(self.rot.d_pad)
        self.centroids = lloyd_max_centroids(pdf, n_levels=2 ** bits)

    def quant(self, x: np.ndarray) -> np.ndarray:
        """Return integer indices in [0, 2^bits) of shape (..., d_pad)."""
        y = self.rot.apply(x)
        # Nearest-centroid lookup. Centroids are sorted, so use searchsorted.
        c = self.centroids
        boundaries = 0.5 * (c[:-1] + c[1:])
        idx = np.searchsorted(boundaries, y)
        return idx.astype(np.int32)

    def dequant(self, idx: np.ndarray) -> np.ndarray:
        y_hat = self.centroids[idx]
        return self.rot.apply_T(y_hat)


# ---------------------------------------------------------------------------
# 5. TurboQuant-Prod  (Algorithm 2)
# ---------------------------------------------------------------------------

class TurboQuantProd:
    def __init__(self, d: int, bits: int, seed: int = 0):
        if bits < 2:
            raise ValueError("bits must be >= 2 (1 bit goes to QJL)")
        self.d = d
        self.bits = bits
        self.mse = TurboQuantMSE(d=d, bits=bits - 1, seed=seed)
        rng = np.random.default_rng(seed + 1)
        # Independent QJL projection. Dense Gaussian; could be replaced by a
        # second randomized FWHT for speed.
        self.S = rng.standard_normal((d, d)).astype(np.float64)

    def quant(self, x: np.ndarray):
        idx = self.mse.quant(x)
        x_mse = self.mse.dequant(idx)
        r = x - x_mse
        gamma = np.linalg.norm(r, axis=-1, keepdims=True)
        signs = np.sign(r @ self.S.T)
        signs[signs == 0] = 1.0
        return idx, signs.astype(np.int8), gamma.astype(np.float32)

    def dequant(self, idx: np.ndarray, signs: np.ndarray, gamma: np.ndarray):
        x_mse = self.mse.dequant(idx)
        # QJL inverse: sqrt(pi/2) / d * S^T z
        qjl_recon = (math.sqrt(math.pi / 2.0) / self.d) * (signs.astype(np.float64) @ self.S)
        return x_mse + gamma * qjl_recon


# ---------------------------------------------------------------------------
# 6. Smoke test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    d, n = 256, 2000

    X = rng.standard_normal((n, d))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Y = rng.standard_normal((n, d))
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)

    print(f"dim={d}, n={n}\n")

    for b in (1, 2, 3, 4):
        q = TurboQuantMSE(d=d, bits=b, seed=0)
        idx = q.quant(X)
        Xh = q.dequant(idx)
        mse = np.mean(np.sum((X - Xh) ** 2, axis=1))
        # Theoretical small-b MSE bounds from Theorem 1 of the paper.
        ref = {1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009}[b]
        print(f"MSE  b={b}: empirical={mse:.4f}  paper~={ref}")

    print()
    for b in (2, 3, 4):
        q = TurboQuantProd(d=d, bits=b, seed=0)
        idx, signs, gamma = q.quant(X)
        Xh = q.dequant(idx, signs, gamma)
        ip_true = np.sum(X * Y, axis=1)
        ip_est = np.sum(Xh * Y, axis=1)
        bias = float(np.mean(ip_est - ip_true))
        var = float(np.mean((ip_est - ip_true) ** 2))
        ref = {2: 1.57 / d, 3: 0.56 / d, 4: 0.18 / d}[b]
        print(
            f"PROD b={b}: bias={bias:+.4f}  inner-prod-MSE={var:.5f}  paper~={ref:.5f}"
        )
