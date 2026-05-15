"""Torch TurboQuant — MSE and inner-product-oriented variants.

This file intentionally keeps the quantizer simple: ``quant`` returns low-range
uint8 indices; physical bit-packing is done by ``tiny_turboquant.bitpack`` because
cache/index formats need shape metadata.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .codebooks import get_centroids
from .rotation import RandomRotation


def _validate_bits(bits: int) -> None:
    if not 1 <= int(bits) <= 8:
        raise ValueError(f"bits must be in [1, 8], got {bits}")


@dataclass
class TurboQuantMSE:
    d: int
    bits: int
    rotation: RandomRotation
    centroids: torch.Tensor
    boundaries: torch.Tensor

    @classmethod
    def build(
        cls,
        d: int,
        bits: int,
        device: torch.device | str = "cpu",
        seed: int = 0,
        dtype: torch.dtype = torch.float32,
    ) -> "TurboQuantMSE":
        _validate_bits(bits)
        rot = RandomRotation.make(d, device=device, seed=seed, dtype=dtype)
        centroids = get_centroids(rot.d_pad, bits, device=device, dtype=dtype)
        boundaries = 0.5 * (centroids[:-1] + centroids[1:])
        return cls(d=d, bits=bits, rotation=rot, centroids=centroids, boundaries=boundaries)

    @property
    def device(self) -> torch.device:
        return self.centroids.device

    @property
    def d_pad(self) -> int:
        return self.rotation.d_pad

    def to(self, device: torch.device | str) -> "TurboQuantMSE":
        return TurboQuantMSE(
            d=self.d,
            bits=self.bits,
            rotation=self.rotation.to(device),
            centroids=self.centroids.to(device),
            boundaries=self.boundaries.to(device),
        )

    def quant(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize ``x`` of shape ``(..., d)`` to uint8 indices ``(..., d_pad)``."""
        y = self.rotation.apply(x)
        idx = torch.searchsorted(self.boundaries, y.contiguous())
        return idx.to(torch.uint8)

    def dequant(self, idx: torch.Tensor) -> torch.Tensor:
        y_hat = self.centroids[idx.long()]
        return self.rotation.apply_T(y_hat)


@dataclass
class TurboQuantProd:
    """Two-stage inner-product-oriented quantizer.

    Stage 1 uses TurboQuantMSE at ``bits - 1``. Stage 2 stores one-bit QJL-style
    residual signs. The dequantized vector is useful for simple experiments; a
    production ANN/KV path should estimate inner products without reconstructing
    dense vectors every query.
    """

    d: int
    bits: int
    mse: TurboQuantMSE
    S: torch.Tensor

    @classmethod
    def build(
        cls,
        d: int,
        bits: int,
        device: torch.device | str = "cpu",
        seed: int = 0,
        dtype: torch.dtype = torch.float32,
    ) -> "TurboQuantProd":
        if not 2 <= int(bits) <= 8:
            raise ValueError(f"bits must be in [2, 8], got {bits}")
        mse = TurboQuantMSE.build(d, bits - 1, device=device, seed=seed, dtype=dtype)
        g = torch.Generator(device="cpu").manual_seed(seed + 1)
        S = torch.randn(d, d, generator=g).to(device=device, dtype=dtype)
        return cls(d=d, bits=bits, mse=mse, S=S)

    def quant(self, x: torch.Tensor):
        idx = self.mse.quant(x)
        x_mse = self.mse.dequant(idx)
        r = x - x_mse
        gamma = r.norm(dim=-1, keepdim=True)
        signs = torch.sign(r @ self.S.T)
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        return idx, signs.to(torch.int8), gamma

    def dequant(self, idx: torch.Tensor, signs: torch.Tensor, gamma: torch.Tensor):
        x_mse = self.mse.dequant(idx)
        scale = math.sqrt(math.pi / 2.0) / self.d
        qjl_recon = scale * (signs.to(self.S.dtype) @ self.S)
        return x_mse + gamma * qjl_recon
