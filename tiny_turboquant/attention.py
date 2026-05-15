"""Attention utilities for tiny-turboquant research workflows.

The functions here are intentionally framework-light. They are useful for
measuring quality without reconstructing one giant dense cache tensor.
They are not production fused CUDA/Triton kernels.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F


def dense_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Standard scaled dot-product attention for tensors shaped (B, H, Q/S, D)."""
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query/key/value must be shaped (B, H, Q_or_S, D)")
    if scale is None:
        scale = 1.0 / math.sqrt(query.shape[-1])
    scores = torch.matmul(query, key.transpose(-1, -2)) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, value)



def sdpa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """PyTorch scaled-dot-product attention wrapper for (B, H, Q/S, D).

    This uses PyTorch's available SDPA backend. It is a useful reference path
    for performance research, but it is not a tiny-turboquant compressed-cache
    kernel.
    """
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query/key/value must be shaped (B, H, Q_or_S, D)")
    return F.scaled_dot_product_attention(query, key, value, scale=scale, is_causal=False)

def streaming_paged_attention(
    query: torch.Tensor,
    kv_pages: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    *,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Compute attention over K/V pages with an online softmax.

    This avoids concatenating all K/V pages into one dense tensor before the
    attention computation. It still dequantizes each compressed page before it
    is consumed, so it is a memory-quality research utility, not a fused kernel.

    Args:
        query: Tensor shaped (B, H, Q, D).
        kv_pages: Iterable of ``(key_page, value_page)`` tensors, each shaped
            (B, H, S_page, D).
        scale: Optional attention scale. Defaults to ``1 / sqrt(D)``.

    Returns:
        Tensor shaped (B, H, Q, D).
    """
    if query.ndim != 4:
        raise ValueError("query must be shaped (B, H, Q, D)")
    if scale is None:
        scale = 1.0 / math.sqrt(query.shape[-1])

    work_dtype = torch.float32 if query.dtype in (torch.float16, torch.bfloat16) else query.dtype
    q = query.to(work_dtype)

    m = torch.full((*q.shape[:-1], 1), -float("inf"), device=q.device, dtype=work_dtype)
    l = torch.zeros((*q.shape[:-1], 1), device=q.device, dtype=work_dtype)
    acc = torch.zeros_like(q, dtype=work_dtype)
    saw_page = False

    for key, value in kv_pages:
        if key.numel() == 0:
            continue
        saw_page = True
        k = key.to(device=q.device, dtype=work_dtype)
        v = value.to(device=q.device, dtype=work_dtype)
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale
        page_m = scores.max(dim=-1, keepdim=True).values
        m_new = torch.maximum(m, page_m)
        alpha = torch.exp(m - m_new)
        p = torch.exp(scores - m_new)
        acc = acc * alpha + torch.matmul(p, v)
        l = l * alpha + p.sum(dim=-1, keepdim=True)
        m = m_new

    if not saw_page:
        raise ValueError("kv_pages is empty")

    out = acc / l.clamp_min(torch.finfo(work_dtype).tiny)
    return out.to(query.dtype)


def attention_similarity(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    """Return simple attention-output quality metrics."""
    ref = reference.float().flatten()
    cand = candidate.float().flatten()
    rel_err = ((ref - cand).norm() / ref.norm().clamp_min(1e-12)).item()
    cos = F.cosine_similarity(ref, cand, dim=0).item()
    return {"relative_error": float(rel_err), "cosine_similarity": float(cos)}
