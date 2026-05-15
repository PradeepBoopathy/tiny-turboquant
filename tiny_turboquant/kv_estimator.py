"""KV-cache memory estimation utilities.

The estimator is intentionally transparent. It is useful for explaining why
KV-cache memory grows with context length, layers, heads, and concurrency before
running a real model benchmark.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional, Union


OutlierSpec = Union[int, str]


def auto_outlier_count(value: OutlierSpec, head_dim: int, *, kind: str) -> int:
    """Resolve an outlier count.

    Rules:
        key:   min(32, head_dim // 2)
        value: min(16, head_dim // 4)

    Counts are clamped so at least one regular channel remains.
    """
    d = int(head_dim)
    if isinstance(value, str):
        if value.lower() != "auto":
            raise ValueError(f"unsupported outlier value {value!r}; use int or 'auto'")
        if kind == "key":
            raw = min(32, max(0, d // 2))
        elif kind == "value":
            raw = min(16, max(0, d // 4))
        else:
            raw = min(16, max(0, d // 4))
    else:
        raw = int(value)
    return int(max(0, min(raw, max(0, d - 2))))


@dataclass(frozen=True)
class KVCacheMemoryEstimate:
    layers: int
    kv_heads: int
    head_dim: int
    seq_len: int
    batch_size: int
    dtype_bytes: int
    fp16_bytes: int
    compressed_estimated_bytes: int
    compression_ratio: float
    memory_saved_pct: float
    key_recent_window: int
    value_recent_window: int
    key_bits: int
    value_bits: int
    key_outlier_bits: int
    value_outlier_bits: int
    n_key_outliers: int
    n_value_outliers: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _tensor_estimate_bytes(
    *,
    layers: int,
    kv_heads: int,
    head_dim: int,
    seq_len: int,
    batch_size: int,
    dtype_bytes: int,
    bits: int,
    outlier_bits: Optional[int],
    n_outliers: int,
    recent_window: int,
    norm_bytes: int,
) -> int:
    recent = min(max(0, int(recent_window)), int(seq_len))
    compressed = max(0, int(seq_len) - recent)
    dense_bytes = batch_size * layers * kv_heads * recent * head_dim * dtype_bytes
    if compressed == 0:
        return int(dense_bytes)
    out_bits = bits if outlier_bits is None else int(outlier_bits)
    out = max(0, min(int(n_outliers), max(0, int(head_dim) - 1)))
    reg = int(head_dim) - out
    bits_per_vector = reg * int(bits) + out * out_bits
    payload_bytes = (batch_size * layers * kv_heads * compressed * bits_per_vector + 7) // 8
    # Store one vector norm per K/V vector. Outlier split stores extra norms in
    # the current prototype; keep this estimate conservative but simple.
    norm_count = batch_size * layers * kv_heads * compressed
    norm_payload = norm_count * int(norm_bytes)
    return int(dense_bytes + payload_bytes + norm_payload)


def estimate_kv_cache_memory(
    *,
    layers: int,
    kv_heads: int,
    head_dim: int,
    seq_len: int,
    batch_size: int = 1,
    dtype_bytes: int = 2,
    key_bits: int = 6,
    value_bits: int = 4,
    key_outlier_bits: Optional[int] = 8,
    value_outlier_bits: Optional[int] = 8,
    n_key_outliers: OutlierSpec = "auto",
    n_value_outliers: OutlierSpec = "auto",
    key_recent_window: int = 128,
    value_recent_window: int = 64,
    norm_bytes: int = 2,
) -> KVCacheMemoryEstimate:
    """Estimate baseline and hybrid-compressed KV-cache memory.

    Baseline assumes dense fp16/bf16-style K+V cache using ``dtype_bytes``.
    Compressed estimate assumes dense recent windows and low-bit older cache.
    """
    layers = int(layers)
    kv_heads = int(kv_heads)
    head_dim = int(head_dim)
    seq_len = int(seq_len)
    batch_size = int(batch_size)
    dtype_bytes = int(dtype_bytes)

    fp16_bytes = batch_size * seq_len * layers * 2 * kv_heads * head_dim * dtype_bytes

    nk = auto_outlier_count(n_key_outliers, head_dim, kind="key")
    nv = auto_outlier_count(n_value_outliers, head_dim, kind="value")
    kob = key_bits if key_outlier_bits is None else int(key_outlier_bits)
    vob = value_bits if value_outlier_bits is None else int(value_outlier_bits)

    key_bytes = _tensor_estimate_bytes(
        layers=layers,
        kv_heads=kv_heads,
        head_dim=head_dim,
        seq_len=seq_len,
        batch_size=batch_size,
        dtype_bytes=dtype_bytes,
        bits=key_bits,
        outlier_bits=kob,
        n_outliers=nk,
        recent_window=key_recent_window,
        norm_bytes=norm_bytes,
    )
    value_bytes = _tensor_estimate_bytes(
        layers=layers,
        kv_heads=kv_heads,
        head_dim=head_dim,
        seq_len=seq_len,
        batch_size=batch_size,
        dtype_bytes=dtype_bytes,
        bits=value_bits,
        outlier_bits=vob,
        n_outliers=nv,
        recent_window=value_recent_window,
        norm_bytes=norm_bytes,
    )
    compressed = int(key_bytes + value_bytes)
    ratio = fp16_bytes / max(compressed, 1)
    saved = (1.0 - compressed / max(fp16_bytes, 1)) * 100.0
    return KVCacheMemoryEstimate(
        layers=layers,
        kv_heads=kv_heads,
        head_dim=head_dim,
        seq_len=seq_len,
        batch_size=batch_size,
        dtype_bytes=dtype_bytes,
        fp16_bytes=int(fp16_bytes),
        compressed_estimated_bytes=int(compressed),
        compression_ratio=float(ratio),
        memory_saved_pct=float(saved),
        key_recent_window=int(key_recent_window),
        value_recent_window=int(value_recent_window),
        key_bits=int(key_bits),
        value_bits=int(value_bits),
        key_outlier_bits=int(kob),
        value_outlier_bits=int(vob),
        n_key_outliers=int(nk),
        n_value_outliers=int(nv),
    )


__all__ = ["KVCacheMemoryEstimate", "auto_outlier_count", "estimate_kv_cache_memory"]
