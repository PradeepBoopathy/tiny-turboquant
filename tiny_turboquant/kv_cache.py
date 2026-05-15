"""HuggingFace-compatible TurboQuant KV-cache implementations.

Two cache classes are provided:

``TurboQuantKVCache``
    Backward-compatible all-packed cache. It physically bit-packs all K/V
    entries but returns dense dequantized K/V to Hugging Face attention.

``HybridTurboQuantKVCache``
    Quality-aware cache. It keeps a configurable recent-token window in dense
    precision and compresses older tokens. It supports separate K/V bit widths,
    separate outlier settings, optional per-layer calibration, and optional
    per-head calibration.

Important limitation: these classes implement compressed *storage* and
memory-quality experiments. They are not production fused compressed-attention
kernels. Hugging Face attention still receives dense K/V tensors.
"""

from __future__ import annotations

from math import prod
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch

from .bitpack import pack_indices, tensor_nbytes, unpack_indices
from .outlier_split import OutlierSplitTurboQuant
from .quantizer import TurboQuantMSE
from .kv_estimator import auto_outlier_count
from .kv_presets import resolve_kv_cache_preset

try:  # pragma: no cover - transformers is optional for core unit tests
    from transformers.cache_utils import Cache as _HFCacheBase
except Exception:  # pragma: no cover
    class _HFCacheBase:  # minimal protocol for testing without transformers
        pass


PackedPage = Dict[str, Any]
QuantizerLike = Any


def _validate_bits(name: str, bits: int) -> int:
    bits = int(bits)
    if not 1 <= bits <= 8:
        raise ValueError(f"{name} must be in [1, 8], got {bits}")
    return bits


def _call_hf_cache_init(obj: object) -> None:
    try:
        super(type(obj), obj).__init__(layers=[])
    except TypeError:
        super(type(obj), obj).__init__()


def _safe_n_outliers(n_outliers: int | str, d: int, *, kind: str = "generic") -> int:
    """Clamp outlier count so at least one regular channel remains."""
    if isinstance(n_outliers, str):
        if n_outliers.lower() != "auto":
            raise ValueError(f"unsupported outlier count {n_outliers!r}; use int or 'auto'")
        if kind == "key":
            raw = min(32, max(0, int(d) // 2))
        elif kind == "value":
            raw = min(16, max(0, int(d) // 4))
        else:
            raw = min(16, max(0, int(d) // 4))
    else:
        raw = int(n_outliers)
    return int(max(0, min(raw, max(0, int(d) - 1))))


def _make_quantizer(
    sample: torch.Tensor,
    *,
    bits: int,
    outlier_bits: Optional[int],
    n_outliers: int,
    seed: int,
) -> QuantizerLike:
    """Create a uniform or outlier-split quantizer from a K/V sample."""
    if sample.ndim != 4:
        raise ValueError(f"sample must be shaped (B, H, S, D), got {tuple(sample.shape)}")

    bits = _validate_bits("bits", bits)
    if outlier_bits is None:
        outlier_bits = bits
    outlier_bits = _validate_bits("outlier_bits", outlier_bits)

    d = int(sample.shape[-1])
    device, dtype = sample.device, sample.dtype
    n_out = _safe_n_outliers(n_outliers, d)

    if outlier_bits == bits or n_out <= 0:
        return TurboQuantMSE.build(d, bits, device=device, seed=seed, dtype=dtype)

    return OutlierSplitTurboQuant.calibrate(
        sample.reshape(-1, d),
        n_out=n_out,
        bits_out=outlier_bits,
        bits_reg=bits,
        seed=seed,
    )


def _make_headwise_quantizers(
    sample: torch.Tensor,
    *,
    bits: int,
    outlier_bits: Optional[int],
    n_outliers: int,
    seed: int,
) -> List[QuantizerLike]:
    """Build one quantizer per attention head.

    This is a quality-hardening feature. Heads can have different K/V
    distributions, so a single shared calibration can be too crude.
    """
    if sample.ndim != 4:
        raise ValueError(f"sample must be shaped (B, H, S, D), got {tuple(sample.shape)}")
    n_heads = int(sample.shape[1])
    return [
        _make_quantizer(
            sample[:, h : h + 1, :, :].contiguous(),
            bits=bits,
            outlier_bits=outlier_bits,
            n_outliers=n_outliers,
            seed=seed + h * 104729,
        )
        for h in range(n_heads)
    ]


def _is_headwise(q: QuantizerLike) -> bool:
    return isinstance(q, list)


def _pack_uniform(x: torch.Tensor, q: TurboQuantMSE, norm_dtype: torch.dtype) -> PackedPage:
    b, h, s, d = x.shape
    flat = x.reshape(-1, d)
    norm = flat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    idx = q.quant(flat / norm).reshape(b, h, s, -1)
    idx_packed, idx_shape = pack_indices(idx, q.bits)
    return {
        "type": "uniform",
        "bits": q.bits,
        "idx": idx_packed,
        "idx_shape": idx_shape,
        "norm": norm.reshape(b, h, s, 1).to(norm_dtype),
        "seq_len": s,
        "dense_shape": (b, h, s, d),
    }


def _unpack_uniform(page: PackedPage, q: TurboQuantMSE, head_dim: int) -> torch.Tensor:
    idx = unpack_indices(page["idx"], page["bits"], page["idx_shape"])
    b, h, s, _ = idx.shape
    x_unit = q.dequant(idx.reshape(-1, idx.shape[-1])).reshape(b, h, s, head_dim)
    return x_unit * page["norm"].to(dtype=x_unit.dtype)


def _pack_outlier(
    x: torch.Tensor,
    q: OutlierSplitTurboQuant,
    norm_dtype: torch.dtype,
) -> PackedPage:
    b, h, s, d = x.shape
    flat = x.reshape(-1, d)
    norm = flat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    idx_out, idx_reg, n_out, n_reg = q.quant(flat / norm)
    idx_out = idx_out.reshape(b, h, s, -1)
    idx_reg = idx_reg.reshape(b, h, s, -1)
    packed_out, shape_out = pack_indices(idx_out, q.bits_out)
    packed_reg, shape_reg = pack_indices(idx_reg, q.bits_reg)
    return {
        "type": "outlier_split",
        "bits_out": q.bits_out,
        "bits_reg": q.bits_reg,
        "idx_out": packed_out,
        "idx_out_shape": shape_out,
        "idx_reg": packed_reg,
        "idx_reg_shape": shape_reg,
        "n_out": n_out.reshape(b, h, s, 1).to(norm_dtype),
        "n_reg": n_reg.reshape(b, h, s, 1).to(norm_dtype),
        "norm": norm.reshape(b, h, s, 1).to(norm_dtype),
        "seq_len": s,
        "dense_shape": (b, h, s, d),
    }


def _unpack_outlier(
    page: PackedPage,
    q: OutlierSplitTurboQuant,
    head_dim: int,
) -> torch.Tensor:
    idx_out = unpack_indices(page["idx_out"], page["bits_out"], page["idx_out_shape"])
    idx_reg = unpack_indices(page["idx_reg"], page["bits_reg"], page["idx_reg_shape"])
    b, h, s, _ = idx_reg.shape
    norm_dtype = page["norm"].dtype
    x_unit = q.dequant(
        idx_out.reshape(-1, idx_out.shape[-1]),
        idx_reg.reshape(-1, idx_reg.shape[-1]),
        page["n_out"].reshape(-1, 1).to(dtype=norm_dtype),
        page["n_reg"].reshape(-1, 1).to(dtype=norm_dtype),
    ).reshape(b, h, s, head_dim)
    return x_unit * page["norm"].to(dtype=x_unit.dtype)


def _pack_headwise(x: torch.Tensor, qs: Sequence[QuantizerLike], norm_dtype: torch.dtype) -> PackedPage:
    _, h, s, _ = x.shape
    if h != len(qs):
        raise ValueError(f"head count mismatch: tensor has {h}, quantizers has {len(qs)}")
    subpages = [
        _pack_tensor(x[:, head : head + 1, :, :].contiguous(), qs[head], norm_dtype)
        for head in range(h)
    ]
    return {
        "type": "headwise",
        "subpages": subpages,
        "seq_len": s,
        "num_heads": h,
        "dense_shape": tuple(x.shape),
    }


def _unpack_headwise(page: PackedPage, qs: Sequence[QuantizerLike], head_dim: int) -> torch.Tensor:
    subpages = page["subpages"]
    if len(subpages) != len(qs):
        raise ValueError("headwise page/quantizer mismatch")
    xs = [_unpack_page(p, qs[i], head_dim) for i, p in enumerate(subpages)]
    return torch.cat(xs, dim=1)


def _pack_tensor(x: torch.Tensor, q: QuantizerLike, norm_dtype: torch.dtype) -> PackedPage:
    if _is_headwise(q):
        return _pack_headwise(x, q, norm_dtype)
    if isinstance(q, TurboQuantMSE):
        return _pack_uniform(x, q, norm_dtype)
    return _pack_outlier(x, q, norm_dtype)


def _unpack_page(page: PackedPage, q: QuantizerLike, head_dim: int) -> torch.Tensor:
    if page["type"] == "headwise":
        return _unpack_headwise(page, q, head_dim)
    if page["type"] == "uniform":
        return _unpack_uniform(page, q, head_dim)
    return _unpack_outlier(page, q, head_dim)


def _unpack_pages(pages: List[PackedPage], q: QuantizerLike, head_dim: int) -> Optional[torch.Tensor]:
    if not pages:
        return None
    xs = [_unpack_page(page, q, head_dim) for page in pages]
    return xs[0] if len(xs) == 1 else torch.cat(xs, dim=2)


def _page_tensor_bytes(page: PackedPage) -> int:
    if page["type"] == "headwise":
        return int(sum(_page_tensor_bytes(p) for p in page["subpages"]))
    total = 0
    for value in page.values():
        if torch.is_tensor(value):
            total += tensor_nbytes(value)
    return int(total)


def _packed_payload_bytes(shape: Tuple[int, ...], bits: int) -> int:
    return (int(prod(shape)) * int(bits) + 7) // 8


def _page_theoretical_bytes(page: PackedPage, norm_dtype: torch.dtype) -> int:
    if page["type"] == "headwise":
        return int(sum(_page_theoretical_bytes(p, norm_dtype) for p in page["subpages"]))

    norm_bytes = torch.empty((), dtype=norm_dtype).element_size()
    if page["type"] == "uniform":
        return int(
            _packed_payload_bytes(page["idx_shape"], page["bits"])
            + int(prod(page["norm"].shape)) * norm_bytes
        )
    return int(
        _packed_payload_bytes(page["idx_out_shape"], page["bits_out"])
        + _packed_payload_bytes(page["idx_reg_shape"], page["bits_reg"])
        + (
            int(prod(page["n_out"].shape))
            + int(prod(page["n_reg"].shape))
            + int(prod(page["norm"].shape))
        )
        * norm_bytes
    )


def _page_dense_shape(page: PackedPage) -> Optional[Tuple[int, int, int, int]]:
    dense_shape = page.get("dense_shape")
    if dense_shape is not None:
        return tuple(int(x) for x in dense_shape)
    shape = page.get("idx_shape") or page.get("idx_reg_shape")
    if shape is None:
        return None
    b, h, s = int(shape[0]), int(shape[1]), int(shape[2])
    d = int(shape[-1])
    return b, h, s, d


def page_device(page: PackedPage) -> torch.device:
    if page["type"] == "headwise":
        return page_device(page["subpages"][0])
    for value in page.values():
        if torch.is_tensor(value):
            return value.device
    return torch.device("cpu")


def _dense_seq_len(x: Optional[torch.Tensor]) -> int:
    return 0 if x is None else int(x.shape[-2])


class TurboQuantKVCache(_HFCacheBase):
    """KV cache that stores all Keys and Values as packed low-bit pages.

    This is the v0.1-compatible cache. It is useful for measuring packed memory
    compression, but generation quality may drift because every token is stored
    in compressed form.
    """

    def __init__(
        self,
        bits: int = 4,
        bits_outlier: Optional[int] = None,
        n_outlier: int = 32,
        seed: int = 0,
        norm_dtype: torch.dtype = torch.float16,
    ):
        _call_hf_cache_init(self)

        self.bits = _validate_bits("bits", bits)
        self.bits_outlier = self.bits if bits_outlier is None else _validate_bits("bits_outlier", bits_outlier)
        self.n_outlier = int(n_outlier)
        self.seed = int(seed)
        self.norm_dtype = norm_dtype

        self._q_key: Optional[Any] = None
        self._q_value: Optional[Any] = None
        self._head_dim: Optional[int] = None
        self._dtype: Optional[torch.dtype] = None

        self._key_pages: List[List[PackedPage]] = []
        self._value_pages: List[List[PackedPage]] = []
        self._seen_tokens = 0

    def _build_quantizers(self, key_sample: torch.Tensor, value_sample: torch.Tensor) -> None:
        if key_sample.shape[-1] != value_sample.shape[-1]:
            raise ValueError("key/value head dimensions differ")
        self._head_dim = int(key_sample.shape[-1])
        self._dtype = key_sample.dtype
        self._q_key = _make_quantizer(
            key_sample,
            bits=self.bits,
            outlier_bits=self.bits_outlier,
            n_outliers=self.n_outlier,
            seed=self.seed,
        )
        self._q_value = _make_quantizer(
            value_sample,
            bits=self.bits,
            outlier_bits=self.bits_outlier,
            n_outliers=self.n_outlier,
            seed=self.seed + 7,
        )

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if key_states.ndim != 4 or value_states.ndim != 4:
            raise ValueError("key_states and value_states must be shaped (B, H, S, D)")
        if self._q_key is None:
            self._build_quantizers(key_states, value_states)
        assert self._q_key is not None and self._q_value is not None and self._head_dim is not None

        while len(self._key_pages) <= layer_idx:
            self._key_pages.append([])
            self._value_pages.append([])

        self._key_pages[layer_idx].append(_pack_tensor(key_states, self._q_key, self.norm_dtype))
        self._value_pages[layer_idx].append(_pack_tensor(value_states, self._q_value, self.norm_dtype))

        if layer_idx == 0:
            self._seen_tokens += int(key_states.shape[-2])

        keys = _unpack_pages(self._key_pages[layer_idx], self._q_key, self._head_dim)
        values = _unpack_pages(self._value_pages[layer_idx], self._q_value, self._head_dim)
        assert keys is not None and values is not None
        return keys, values

    def iter_dequantized_kv_pages(self, layer_idx: int = 0) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        if layer_idx >= len(self._key_pages):
            return
        assert self._q_key is not None and self._q_value is not None and self._head_dim is not None
        for kp, vp in zip(self._key_pages[layer_idx], self._value_pages[layer_idx]):
            yield _unpack_page(kp, self._q_key, self._head_dim), _unpack_page(vp, self._q_value, self._head_dim)

    def paged_attention(self, query_states: torch.Tensor, layer_idx: int = 0, scale: Optional[float] = None) -> torch.Tensor:
        from .attention import streaming_paged_attention

        return streaming_paged_attention(query_states, self.iter_dequantized_kv_pages(layer_idx), scale=scale)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self._key_pages):
            return 0
        return int(sum(page["seq_len"] for page in self._key_pages[layer_idx]))

    def get_max_length(self) -> Optional[int]:
        return None

    def get_max_cache_shape(self) -> Optional[int]:
        return None

    def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        return self.get_seq_length(layer_idx)

    def __len__(self) -> int:
        return len(self._key_pages)

    @property
    def is_initialized(self) -> bool:  # type: ignore[override]
        return any(len(pages) > 0 for pages in self._key_pages)

    @property
    def is_sliding(self) -> list:  # type: ignore[override]
        return [False] * len(self._key_pages)

    def get_mask_sizes(self, query_length, layer_idx: int) -> Tuple[int, int]:
        q = int(query_length.shape[-1]) if torch.is_tensor(query_length) and query_length.ndim >= 1 else int(query_length)
        return self.get_seq_length(layer_idx) + q, 0

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        for layer_idx in range(len(self._key_pages)):
            for store, q in ((self._key_pages, self._q_key), (self._value_pages, self._q_value)):
                if q is None or self._head_dim is None:
                    continue
                new_pages: List[PackedPage] = []
                for page in store[layer_idx]:
                    dense = _unpack_page(page, q, self._head_dim).index_select(0, beam_idx.to(page_device(page)))
                    new_pages.append(_pack_tensor(dense, q, self.norm_dtype))
                store[layer_idx] = new_pages

    def actual_memory_bytes(self) -> int:
        return int(sum(_page_tensor_bytes(page) for pages in (*self._key_pages, *self._value_pages) for page in pages))

    def theoretical_memory_bytes(self) -> int:
        return int(sum(_page_theoretical_bytes(page, self.norm_dtype) for pages in (*self._key_pages, *self._value_pages) for page in pages))

    def memory_bytes(self) -> int:
        return self.actual_memory_bytes()

    def fp16_baseline_bytes(self) -> int:
        total = 0
        for pages in (*self._key_pages, *self._value_pages):
            for page in pages:
                shape = _page_dense_shape(page)
                if shape is not None:
                    b, h, s, d = shape
                    total += b * h * s * d * 2
        return int(total)


class HybridTurboQuantKVCache(_HFCacheBase):
    """Quality-aware hybrid KV cache.

    Recent tokens are kept dense; older tokens are packed using low-bit
    quantization. This is meant to reduce early generation drift compared with
    compressing the entire cache.

    v0.3 adds optional per-head calibration and paged-attention utilities.
    """

    def __init__(
        self,
        bits: int = 4,
        *,
        key_bits: Optional[int] = None,
        value_bits: Optional[int] = None,
        recent_window: int = 128,
        key_recent_window: Optional[int] = None,
        value_recent_window: Optional[int] = None,
        key_outlier_bits: Optional[int] = None,
        value_outlier_bits: Optional[int] = None,
        n_key_outliers: int | str = 64,
        n_value_outliers: int | str = 32,
        per_layer_calibration: bool = True,
        per_head_calibration: bool = False,
        seed: int = 0,
        norm_dtype: torch.dtype = torch.float16,
        recent_dtype: Optional[torch.dtype] = None,
    ):
        _call_hf_cache_init(self)

        self.bits = _validate_bits("bits", bits)
        self.key_bits = _validate_bits("key_bits", self.bits if key_bits is None else key_bits)
        self.value_bits = _validate_bits("value_bits", self.bits if value_bits is None else value_bits)
        self.key_outlier_bits = self.key_bits if key_outlier_bits is None else _validate_bits("key_outlier_bits", key_outlier_bits)
        self.value_outlier_bits = self.value_bits if value_outlier_bits is None else _validate_bits("value_outlier_bits", value_outlier_bits)
        self.n_key_outliers = n_key_outliers
        self.n_value_outliers = n_value_outliers
        self.recent_window = int(max(0, recent_window))
        self.key_recent_window = self.recent_window if key_recent_window is None else int(max(0, key_recent_window))
        self.value_recent_window = self.recent_window if value_recent_window is None else int(max(0, value_recent_window))
        self.per_layer_calibration = bool(per_layer_calibration)
        self.per_head_calibration = bool(per_head_calibration)
        self.seed = int(seed)
        self.norm_dtype = norm_dtype
        self.recent_dtype = recent_dtype

        self._head_dim: Optional[int] = None
        self._dtype: Optional[torch.dtype] = None

        self._key_pages: List[List[PackedPage]] = []
        self._value_pages: List[List[PackedPage]] = []
        self._key_recent: List[Optional[torch.Tensor]] = []
        self._value_recent: List[Optional[torch.Tensor]] = []

        self._q_key_by_layer: List[Optional[Any]] = []
        self._q_value_by_layer: List[Optional[Any]] = []
        self._q_key_global: Optional[Any] = None
        self._q_value_global: Optional[Any] = None
        self._seen_tokens = 0

    @classmethod
    def from_preset(cls, preset: str = "balanced", **overrides: Any) -> "HybridTurboQuantKVCache":
        """Construct a hybrid cache from a safe named preset.

        Presets are research defaults for benchmarking:
        ``safe``, ``balanced``, ``aggressive``, and ``quality-headwise``.
        Override any constructor parameter by passing it as a keyword argument.
        """
        cfg = resolve_kv_cache_preset(preset, **overrides)
        return cls(**cfg)

    def resolved_outlier_counts(self, head_dim: Optional[int] = None) -> Dict[str, int]:
        """Return resolved Key/Value outlier counts for a given head dimension."""
        d = int(head_dim if head_dim is not None else (self._head_dim or 0))
        if d <= 0:
            return {"key": 0, "value": 0}
        return {
            "key": auto_outlier_count(self.n_key_outliers, d, kind="key"),
            "value": auto_outlier_count(self.n_value_outliers, d, kind="value"),
        }

    def _ensure_layer(self, layer_idx: int) -> None:
        while len(self._key_pages) <= layer_idx:
            self._key_pages.append([])
            self._value_pages.append([])
            self._key_recent.append(None)
            self._value_recent.append(None)
            self._q_key_by_layer.append(None)
            self._q_value_by_layer.append(None)

    def _build_q(
        self,
        sample: torch.Tensor,
        *,
        bits: int,
        outlier_bits: int,
        n_outliers: int,
        seed: int,
    ) -> QuantizerLike:
        if self.per_head_calibration:
            return _make_headwise_quantizers(
                sample,
                bits=bits,
                outlier_bits=outlier_bits,
                n_outliers=n_outliers,
                seed=seed,
            )
        return _make_quantizer(
            sample,
            bits=bits,
            outlier_bits=outlier_bits,
            n_outliers=n_outliers,
            seed=seed,
        )

    def _get_or_build_quantizers(
        self,
        key_sample: torch.Tensor,
        value_sample: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[Any, Any]:
        if key_sample.shape[-1] != value_sample.shape[-1]:
            raise ValueError("key/value head dimensions differ")
        if self._head_dim is None:
            self._head_dim = int(key_sample.shape[-1])
            self._dtype = key_sample.dtype

        if self.per_layer_calibration:
            qk = self._q_key_by_layer[layer_idx]
            qv = self._q_value_by_layer[layer_idx]
            if qk is None:
                qk = self._build_q(
                    key_sample,
                    bits=self.key_bits,
                    outlier_bits=self.key_outlier_bits,
                    n_outliers=auto_outlier_count(self.n_key_outliers, int(key_sample.shape[-1]), kind="key"),
                    seed=self.seed + layer_idx * 1009,
                )
                self._q_key_by_layer[layer_idx] = qk
            if qv is None:
                qv = self._build_q(
                    value_sample,
                    bits=self.value_bits,
                    outlier_bits=self.value_outlier_bits,
                    n_outliers=auto_outlier_count(self.n_value_outliers, int(value_sample.shape[-1]), kind="value"),
                    seed=self.seed + 7 + layer_idx * 1009,
                )
                self._q_value_by_layer[layer_idx] = qv
            return qk, qv

        if self._q_key_global is None:
            self._q_key_global = self._build_q(
                key_sample,
                bits=self.key_bits,
                outlier_bits=self.key_outlier_bits,
                n_outliers=auto_outlier_count(self.n_key_outliers, int(key_sample.shape[-1]), kind="key"),
                seed=self.seed,
            )
        if self._q_value_global is None:
            self._q_value_global = self._build_q(
                value_sample,
                bits=self.value_bits,
                outlier_bits=self.value_outlier_bits,
                n_outliers=auto_outlier_count(self.n_value_outliers, int(value_sample.shape[-1]), kind="value"),
                seed=self.seed + 7,
            )
        return self._q_key_global, self._q_value_global

    def _layer_quantizers(self, layer_idx: int) -> Tuple[Optional[Any], Optional[Any]]:
        if self.per_layer_calibration:
            if layer_idx >= len(self._q_key_by_layer):
                return None, None
            return self._q_key_by_layer[layer_idx], self._q_value_by_layer[layer_idx]
        return self._q_key_global, self._q_value_global

    def _append_hybrid_layer(
        self,
        *,
        store_pages: List[List[PackedPage]],
        store_recent: List[Optional[torch.Tensor]],
        dense: torch.Tensor,
        q: Any,
        layer_idx: int,
        recent_window: int,
    ) -> None:
        dense_for_recent = dense.to(self.recent_dtype) if self.recent_dtype is not None else dense
        current = store_recent[layer_idx]
        joined = dense_for_recent if current is None else torch.cat([current, dense_for_recent], dim=2)

        if recent_window == 0:
            compress_len = int(joined.shape[2])
        else:
            compress_len = max(0, int(joined.shape[2]) - recent_window)

        if compress_len > 0:
            to_compress = joined[:, :, :compress_len, :]
            store_pages[layer_idx].append(_pack_tensor(to_compress.to(dense.dtype), q, self.norm_dtype))
            store_recent[layer_idx] = joined[:, :, compress_len:, :].contiguous()
        else:
            store_recent[layer_idx] = joined.contiguous()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if key_states.ndim != 4 or value_states.ndim != 4:
            raise ValueError("key_states and value_states must be shaped (B, H, S, D)")

        self._ensure_layer(layer_idx)
        qk, qv = self._get_or_build_quantizers(key_states, value_states, layer_idx)
        assert self._head_dim is not None

        self._append_hybrid_layer(
            store_pages=self._key_pages,
            store_recent=self._key_recent,
            dense=key_states,
            q=qk,
            layer_idx=layer_idx,
            recent_window=self.key_recent_window,
        )
        self._append_hybrid_layer(
            store_pages=self._value_pages,
            store_recent=self._value_recent,
            dense=value_states,
            q=qv,
            layer_idx=layer_idx,
            recent_window=self.value_recent_window,
        )

        if layer_idx == 0:
            self._seen_tokens += int(key_states.shape[-2])

        old_k = _unpack_pages(self._key_pages[layer_idx], qk, self._head_dim)
        old_v = _unpack_pages(self._value_pages[layer_idx], qv, self._head_dim)
        recent_k = self._key_recent[layer_idx]
        recent_v = self._value_recent[layer_idx]

        if old_k is None:
            assert recent_k is not None and recent_v is not None
            return recent_k.to(key_states.dtype), recent_v.to(value_states.dtype)

        if recent_k is None or recent_k.shape[2] == 0:
            return old_k.to(key_states.dtype), old_v.to(value_states.dtype)

        return (
            torch.cat([old_k.to(key_states.dtype), recent_k.to(key_states.dtype)], dim=2),
            torch.cat([old_v.to(value_states.dtype), recent_v.to(value_states.dtype)], dim=2),
        )

    def _reconstruct_layer_kv(self, layer_idx: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Reconstruct aligned full-layer K/V tensors for research utilities.

        Hybrid mode can use different recent windows for K and V. That means the
        physical compressed page boundaries for K and V are not guaranteed to
        match. Attention, however, requires each K page and V page to contain the
        same token range. This helper reconstructs the logical full K/V streams
        and lets ``iter_dequantized_kv_pages`` re-slice them into aligned chunks.
        """
        if layer_idx >= len(self._key_pages):
            return None, None

        qk, qv = self._layer_quantizers(layer_idx)
        if qk is None or qv is None or self._head_dim is None:
            return None, None

        old_k = _unpack_pages(self._key_pages[layer_idx], qk, self._head_dim)
        old_v = _unpack_pages(self._value_pages[layer_idx], qv, self._head_dim)
        recent_k = self._key_recent[layer_idx]
        recent_v = self._value_recent[layer_idx]

        parts_k = []
        parts_v = []
        if old_k is not None:
            parts_k.append(old_k)
        if old_v is not None:
            parts_v.append(old_v)
        if recent_k is not None and recent_k.shape[2] > 0:
            parts_k.append(recent_k)
        if recent_v is not None and recent_v.shape[2] > 0:
            parts_v.append(recent_v)

        if not parts_k or not parts_v:
            return None, None

        full_k = parts_k[0] if len(parts_k) == 1 else torch.cat(parts_k, dim=2)
        full_v = parts_v[0] if len(parts_v) == 1 else torch.cat(parts_v, dim=2)

        if full_k.shape[2] != full_v.shape[2]:
            raise RuntimeError(
                "Hybrid KV cache has mismatched logical K/V lengths: "
                f"K={full_k.shape[2]}, V={full_v.shape[2]}. This indicates a cache update bug."
            )

        return full_k, full_v

    def iter_dequantized_kv_pages(
        self,
        layer_idx: int = 0,
        *,
        page_size: Optional[int] = None,
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Yield aligned dequantized K/V chunks for research attention utilities.

        v0.3.1 fixes a v0.3.0 edge case where different K/V recent windows
        produced unequal physical page lengths. This iterator now slices the
        logical reconstructed K/V streams into equal token ranges before yielding
        them to streaming attention.
        """
        full_k, full_v = self._reconstruct_layer_kv(layer_idx)
        if full_k is None or full_v is None:
            return

        if page_size is None:
            # Keep chunks modest while avoiding too many tiny Python iterations.
            page_size = max(1, min(128, int(full_k.shape[2])))
        page_size = int(max(1, page_size))

        for start in range(0, int(full_k.shape[2]), page_size):
            end = min(start + page_size, int(full_k.shape[2]))
            yield full_k[:, :, start:end, :], full_v[:, :, start:end, :]

    def paged_attention(
        self,
        query_states: torch.Tensor,
        layer_idx: int = 0,
        scale: Optional[float] = None,
        page_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute attention over aligned cache chunks.

        This is a research utility, not a fused CUDA/Triton production kernel.
        It uses online softmax over chunks and is useful for checking quality.
        Because hybrid K/V can have different recent windows, v0.3.1 aligns the
        logical K/V token ranges before streaming over them.
        """
        from .attention import streaming_paged_attention

        return streaming_paged_attention(
            query_states,
            self.iter_dequantized_kv_pages(layer_idx, page_size=page_size),
            scale=scale,
        )

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self._key_pages):
            return 0
        compressed = int(sum(page["seq_len"] for page in self._key_pages[layer_idx]))
        return compressed + _dense_seq_len(self._key_recent[layer_idx])

    def compressed_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self._key_pages):
            return 0
        return int(sum(page["seq_len"] for page in self._key_pages[layer_idx]))

    def recent_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self._key_recent):
            return 0
        return _dense_seq_len(self._key_recent[layer_idx])

    def get_max_length(self) -> Optional[int]:
        return None

    def get_max_cache_shape(self) -> Optional[int]:
        return None

    def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        return self.get_seq_length(layer_idx)

    def __len__(self) -> int:
        return len(self._key_pages)

    @property
    def is_initialized(self) -> bool:  # type: ignore[override]
        return any(self.get_seq_length(i) > 0 for i in range(len(self._key_pages)))

    @property
    def is_sliding(self) -> list:  # type: ignore[override]
        return [False] * len(self._key_pages)

    def get_mask_sizes(self, query_length, layer_idx: int) -> Tuple[int, int]:
        q = int(query_length.shape[-1]) if torch.is_tensor(query_length) and query_length.ndim >= 1 else int(query_length)
        return self.get_seq_length(layer_idx) + q, 0

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        for layer_idx in range(len(self._key_pages)):
            qk, qv = self._layer_quantizers(layer_idx)
            if self._key_recent[layer_idx] is not None:
                self._key_recent[layer_idx] = self._key_recent[layer_idx].index_select(
                    0, beam_idx.to(self._key_recent[layer_idx].device)
                )
            if self._value_recent[layer_idx] is not None:
                self._value_recent[layer_idx] = self._value_recent[layer_idx].index_select(
                    0, beam_idx.to(self._value_recent[layer_idx].device)
                )
            for store, q in ((self._key_pages, qk), (self._value_pages, qv)):
                if q is None or self._head_dim is None:
                    continue
                new_pages: List[PackedPage] = []
                for page in store[layer_idx]:
                    dense = _unpack_page(page, q, self._head_dim).index_select(0, beam_idx.to(page_device(page)))
                    new_pages.append(_pack_tensor(dense, q, self.norm_dtype))
                store[layer_idx] = new_pages

    def actual_memory_bytes(self) -> int:
        total = 0
        for pages in (*self._key_pages, *self._value_pages):
            for page in pages:
                total += _page_tensor_bytes(page)
        for dense in (*self._key_recent, *self._value_recent):
            if dense is not None:
                total += tensor_nbytes(dense)
        return int(total)

    def theoretical_memory_bytes(self) -> int:
        total = 0
        for pages in (*self._key_pages, *self._value_pages):
            for page in pages:
                total += _page_theoretical_bytes(page, self.norm_dtype)
        for dense in (*self._key_recent, *self._value_recent):
            if dense is not None:
                total += tensor_nbytes(dense)
        return int(total)

    def memory_bytes(self) -> int:
        return self.actual_memory_bytes()

    def fp16_baseline_bytes(self) -> int:
        total = 0
        for pages in (*self._key_pages, *self._value_pages):
            for page in pages:
                shape = _page_dense_shape(page)
                if shape is not None:
                    b, h, s, d = shape
                    total += b * h * s * d * 2
        for dense in (*self._key_recent, *self._value_recent):
            if dense is not None:
                b, h, s, d = dense.shape
                total += int(b) * int(h) * int(s) * int(d) * 2
        return int(total)
