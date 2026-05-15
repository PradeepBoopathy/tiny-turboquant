"""Outlier-channel split for K/V or embedding vectors."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .quantizer import TurboQuantMSE


@dataclass
class OutlierSplitTurboQuant:
    d: int
    n_out: int
    bits_out: int
    bits_reg: int
    out_idx: torch.Tensor
    reg_idx: torch.Tensor
    q_out: TurboQuantMSE
    q_reg: TurboQuantMSE

    @property
    def effective_bits(self) -> float:
        n_reg = self.d - self.n_out
        return (self.n_out * self.bits_out + n_reg * self.bits_reg) / self.d

    @property
    def device(self) -> torch.device:
        return self.out_idx.device

    @classmethod
    def calibrate(
        cls,
        samples: torch.Tensor,
        n_out: int = 32,
        bits_out: int = 3,
        bits_reg: int = 2,
        seed: int = 0,
    ) -> "OutlierSplitTurboQuant":
        if samples.ndim != 2:
            raise ValueError(f"samples must be 2-D (N, d), got shape {tuple(samples.shape)}")
        _, d = samples.shape
        n_out = int(max(0, min(n_out, d)))
        if n_out == d:
            raise ValueError("n_out must be smaller than d so that regular channels remain")

        device, dtype = samples.device, samples.dtype
        if n_out == 0:
            out_idx = torch.empty(0, device=device, dtype=torch.long)
        else:
            score = samples.abs().amax(dim=0)
            out_idx = torch.topk(score, n_out).indices.sort().values

        full = torch.arange(d, device=device)
        mask = torch.ones(d, dtype=torch.bool, device=device)
        mask[out_idx] = False
        reg_idx = full[mask]

        q_out = TurboQuantMSE.build(max(n_out, 1), bits_out, device=device, seed=seed, dtype=dtype)
        q_reg = TurboQuantMSE.build(d - n_out, bits_reg, device=device, seed=seed + 1, dtype=dtype)
        return cls(
            d=d,
            n_out=n_out,
            bits_out=bits_out,
            bits_reg=bits_reg,
            out_idx=out_idx,
            reg_idx=reg_idx,
            q_out=q_out,
            q_reg=q_reg,
        )

    def quant(self, x: torch.Tensor):
        x_reg = x.index_select(-1, self.reg_idx)
        norm_reg = x_reg.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        idx_reg = self.q_reg.quant(x_reg / norm_reg)

        if self.n_out == 0:
            idx_out = torch.empty(*x.shape[:-1], 0, device=x.device, dtype=torch.uint8)
            norm_out = torch.empty(*x.shape[:-1], 1, device=x.device, dtype=x.dtype)
        else:
            x_out = x.index_select(-1, self.out_idx)
            norm_out = x_out.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            idx_out = self.q_out.quant(x_out / norm_out)
        return idx_out, idx_reg, norm_out, norm_reg

    def dequant(self, idx_out, idx_reg, norm_out, norm_reg):
        x_reg = self.q_reg.dequant(idx_reg) * norm_reg
        out = torch.empty(*x_reg.shape[:-1], self.d, device=x_reg.device, dtype=x_reg.dtype)
        out.index_copy_(-1, self.reg_idx, x_reg)

        if self.n_out > 0:
            x_out = self.q_out.dequant(idx_out) * norm_out
            out.index_copy_(-1, self.out_idx, x_out)
        return out
