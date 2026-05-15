"""Performance-research utilities for page-wise attention experiments.

This module is intentionally honest about scope. It benchmarks dense-concat
attention, PyTorch SDPA when available, and streaming page-wise attention over
K/V pages. It does not implement a production fused CUDA/Triton kernel.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .attention import attention_similarity, dense_attention, sdpa_attention, streaming_paged_attention


@dataclass
class PageAttentionBenchConfig:
    batch_size: int = 1
    heads: int = 8
    query_len: int = 1
    seq_len: int = 512
    head_dim: int = 64
    page_size: int = 128
    device: str = "auto"
    dtype: str = "auto"
    warmup: int = 2
    repeats: int = 5
    # Backward-compatible alias accepted by earlier demo scripts.
    # If supplied, it overrides repeats during __post_init__.
    iters: Optional[int] = None
    seed: int = 123

    def __post_init__(self) -> None:
        if self.iters is not None:
            self.repeats = int(self.iters)


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        # v0.7.1: demo-safe fallback instead of crashing on CPU-only PyTorch builds.
        return "cpu"
    return device


def _runtime_warnings(requested_device: str, resolved_device: str, requested_dtype: str, resolved_dtype: torch.dtype) -> List[str]:
    warnings: List[str] = []
    if requested_device == "cuda" and resolved_device == "cpu":
        warnings.append(
            "device='cuda' was requested but CUDA is not available; falling back to CPU. "
            "Install a CUDA-enabled PyTorch build or use --device cpu."
        )
    if requested_dtype in {"float16", "fp16", "bfloat16", "bf16"} and resolved_device == "cpu" and resolved_dtype == torch.float32:
        warnings.append(
            f"dtype={requested_dtype!r} was requested on CPU; using float32 for stable CPU execution."
        )
    return warnings


def _resolve_dtype(dtype: str, device: str) -> torch.dtype:
    if dtype == "auto":
        return torch.float16 if device == "cuda" else torch.float32
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype not in mapping:
        raise ValueError(f"unsupported dtype {dtype!r}")
    if device == "cpu" and mapping[dtype] == torch.float16:
        # Many CPU ops support fp16 poorly. Keep demos stable.
        return torch.float32
    if device == "cpu" and mapping[dtype] == torch.bfloat16:
        # Keep CPU demos stable across machines and PyTorch builds.
        return torch.float32
    return mapping[dtype]


def _sync(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_call(fn, *, device: str, warmup: int, repeats: int) -> Tuple[torch.Tensor, float]:
    out = None
    for _ in range(max(0, warmup)):
        out = fn()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(max(1, repeats)):
        out = fn()
    _sync(device)
    elapsed = (time.perf_counter() - t0) / max(1, repeats)
    assert out is not None
    return out, float(elapsed)


def make_kv_pages(
    key: torch.Tensor,
    value: torch.Tensor,
    page_size: int,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Split dense K/V tensors into token pages.

    Args:
        key/value: tensors shaped ``(B, H, S, D)``.
        page_size: positive number of tokens per page.
    """
    if key.shape != value.shape:
        raise ValueError("key and value must have the same shape")
    if key.ndim != 4:
        raise ValueError("key/value must be shaped (B, H, S, D)")
    page_size = int(max(1, page_size))
    pages = []
    for start in range(0, int(key.shape[2]), page_size):
        end = min(start + page_size, int(key.shape[2]))
        pages.append((key[:, :, start:end, :].contiguous(), value[:, :, start:end, :].contiguous()))
    return pages


def triton_status() -> Dict[str, Any]:
    """Return optional Triton availability without importing it eagerly."""
    try:
        import importlib.util

        has_triton = importlib.util.find_spec("triton") is not None
    except Exception:
        has_triton = False
    return {
        "available": bool(has_triton and torch.cuda.is_available()),
        "package_importable": bool(has_triton),
        "cuda_available": bool(torch.cuda.is_available()),
        "status": (
            "Triton package and CUDA available; tiny-turboquant can attempt experimental fused kernels when the config is supported."
            if has_triton and torch.cuda.is_available()
            else "Triton fused kernels are not active; using PyTorch research backends."
        ),
    }


def run_page_attention_benchmark(config: PageAttentionBenchConfig) -> Dict[str, Any]:
    """Benchmark dense attention vs page-wise streaming attention.

    Returns a JSON-serializable dictionary. Timing is diagnostic and depends on
    hardware, page size, dtype, and PyTorch version.
    """
    requested_device = config.device
    requested_dtype = config.dtype
    device = _resolve_device(requested_device)
    dtype = _resolve_dtype(requested_dtype, device)
    warnings = _runtime_warnings(requested_device, device, requested_dtype, dtype)
    torch.manual_seed(int(config.seed))

    q = torch.randn(
        config.batch_size,
        config.heads,
        config.query_len,
        config.head_dim,
        device=device,
        dtype=dtype,
    )
    k = torch.randn(
        config.batch_size,
        config.heads,
        config.seq_len,
        config.head_dim,
        device=device,
        dtype=dtype,
    )
    v = torch.randn_like(k)
    pages = make_kv_pages(k, v, config.page_size)

    dense_out, dense_seconds = _time_call(
        lambda: dense_attention(q, k, v),
        device=device,
        warmup=config.warmup,
        repeats=config.repeats,
    )
    stream_out, stream_seconds = _time_call(
        lambda: streaming_paged_attention(q, pages),
        device=device,
        warmup=config.warmup,
        repeats=config.repeats,
    )

    sdpa_report: Optional[Dict[str, Any]] = None
    try:
        sdpa_out, sdpa_seconds = _time_call(
            lambda: sdpa_attention(q, k, v),
            device=device,
            warmup=config.warmup,
            repeats=config.repeats,
        )
        sdpa_report = {
            "seconds": float(sdpa_seconds),
            "similarity_to_dense": attention_similarity(dense_out, sdpa_out),
        }
    except Exception as exc:  # pragma: no cover - backend dependent
        sdpa_report = {"seconds": None, "error": str(exc), "similarity_to_dense": None}

    dense_kv_bytes = int(k.numel() * k.element_size() + v.numel() * v.element_size())
    largest_page_tokens = max(int(page[0].shape[2]) for page in pages)
    largest_page_bytes = int(
        config.batch_size
        * config.heads
        * largest_page_tokens
        * config.head_dim
        * 2
        * torch.empty((), dtype=dtype).element_size()
    )

    return {
        "config": {
            **asdict(config),
            "device": device,
            "dtype": str(dtype).replace("torch.", ""),
            "resolved_device": device,
            "resolved_dtype": str(dtype).replace("torch.", ""),
            "requested_device": requested_device,
            "requested_dtype": requested_dtype,
            "iters": int(config.repeats),
        },
        "warnings": warnings,
        "page_count": len(pages),
        "tokens_per_page": int(config.page_size),
        "memory": {
            "dense_kv_bytes": dense_kv_bytes,
            "largest_page_kv_bytes": largest_page_bytes,
            "largest_page_fraction_of_dense": largest_page_bytes / max(dense_kv_bytes, 1),
            "note": (
                "Page-wise attention avoids allocating an additional full concat buffer during research evaluation; "
                "it does not reduce the stored K/V payload by itself."
            ),
        },
        "timing": {
            "dense_manual_seconds": float(dense_seconds),
            "streaming_paged_seconds": float(stream_seconds),
            "sdpa": sdpa_report,
            "note": "Timing is diagnostic only; no production acceleration claim is made.",
        },
        "quality": {
            "streaming_vs_dense": attention_similarity(dense_out, stream_out),
        },
        "triton": triton_status(),
        "interpretation": (
            "This benchmark compares attention evaluation paths for research. "
            "Production acceleration still requires fused dequant + attention kernels and serving-engine validation."
        ),
    }


def save_page_attention_json(report: Dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def page_attention_markdown_report(report: Dict[str, Any]) -> str:
    cfg = report["config"]
    mem = report["memory"]
    t = report["timing"]
    q = report["quality"]["streaming_vs_dense"]
    sdpa = t.get("sdpa") or {}
    sdpa_seconds = sdpa.get("seconds")
    sdpa_text = "not available" if sdpa_seconds is None else f"{sdpa_seconds:.6f}s"
    warnings = report.get("warnings") or []
    warnings_md = "\n".join(f"- {w}" for w in warnings) if warnings else "- None"
    return f"""# tiny-turboquant page-attention benchmark

## Setup

- Batch size: `{cfg['batch_size']}`
- Heads: `{cfg['heads']}`
- Query length: `{cfg['query_len']}`
- Sequence length: `{cfg['seq_len']}`
- Head dim: `{cfg['head_dim']}`
- Page size: `{cfg['page_size']}`
- Device: `{cfg['device']}`
- dtype: `{cfg['dtype']}`
- Requested device: `{cfg.get('requested_device', cfg['device'])}`
- Requested dtype: `{cfg.get('requested_dtype', cfg['dtype'])}`

## Warnings

{warnings_md}

## Memory

| Metric | Value |
|---|---:|
| Dense K/V bytes | {mem['dense_kv_bytes']} |
| Largest page K/V bytes | {mem['largest_page_kv_bytes']} |
| Largest page fraction | {mem['largest_page_fraction_of_dense']:.4f} |

## Timing

| Path | Avg seconds |
|---|---:|
| Dense manual attention | {t['dense_manual_seconds']:.6f} |
| Streaming paged attention | {t['streaming_paged_seconds']:.6f} |
| PyTorch SDPA | {sdpa_text} |

## Quality

| Metric | Value |
|---|---:|
| Relative error | {q['relative_error']:.8f} |
| Cosine similarity | {q['cosine_similarity']:.8f} |

> Timing is diagnostic only. This benchmark does not claim production inference acceleration.
"""


def save_page_attention_markdown(report: Dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(page_attention_markdown_report(report), encoding="utf-8")


__all__ = [
    "PageAttentionBenchConfig",
    "make_kv_pages",
    "triton_status",
    "run_page_attention_benchmark",
    "save_page_attention_json",
    "page_attention_markdown_report",
    "save_page_attention_markdown",
]
