"""Randomised orthogonal rotation built from sign-flips, FWHT, permutations."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .fwht import fwht


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


@dataclass
class RandomRotation:
    """Orthonormal rotation: x  ↦  H · S2 · P · H · S1 · pad(x)."""

    d: int
    d_pad: int
    s1: torch.Tensor          # +-1, shape (d_pad,)
    s2: torch.Tensor
    perm: torch.Tensor        # int64, shape (d_pad,)
    inv_perm: torch.Tensor

    @classmethod
    def make(
        cls,
        d: int,
        device: torch.device | str = "cpu",
        seed: int = 0,
        dtype: torch.dtype = torch.float32,
    ) -> "RandomRotation":
        d_pad = _next_pow2(d)
        g = torch.Generator(device="cpu").manual_seed(seed)
        s1 = (torch.randint(0, 2, (d_pad,), generator=g, dtype=torch.int8) * 2 - 1).to(dtype)
        s2 = (torch.randint(0, 2, (d_pad,), generator=g, dtype=torch.int8) * 2 - 1).to(dtype)
        perm = torch.randperm(d_pad, generator=g)
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(d_pad)
        return cls(
            d=d,
            d_pad=d_pad,
            s1=s1.to(device),
            s2=s2.to(device),
            perm=perm.to(device),
            inv_perm=inv_perm.to(device),
        )

    def to(self, device: torch.device | str) -> "RandomRotation":
        return RandomRotation(
            d=self.d,
            d_pad=self.d_pad,
            s1=self.s1.to(device),
            s2=self.s2.to(device),
            perm=self.perm.to(device),
            inv_perm=self.inv_perm.to(device),
        )

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        if self.d == self.d_pad:
            return x
        pad_shape = list(x.shape)
        pad_shape[-1] = self.d_pad - self.d
        zeros = x.new_zeros(pad_shape)
        return torch.cat((x, zeros), dim=-1)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        y = self._pad(x) * self.s1
        y = fwht(y)
        y = y.index_select(-1, self.perm) * self.s2
        y = fwht(y)
        return y

    def apply_T(self, y: torch.Tensor) -> torch.Tensor:
        z = fwht(y) * self.s2
        z = z.index_select(-1, self.inv_perm)
        z = fwht(z) * self.s1
        return z[..., : self.d]
