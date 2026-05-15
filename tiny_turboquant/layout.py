"""Kernel-ready compressed KV page layout research utilities.

v0.10.3 adds TurboQuant-style residual-correction experiments on top of the affine
per-page/per-channel compressed KV layout.  The residual path stores a cheap
1-bit sign correction plus a per-page/per-channel residual magnitude so lower-bit
layouts can be compared against the safer affine layouts using inner-product,
softmax, and attention-output diagnostics.

This is an experimental research path inspired by the public TurboQuant problem
setting.  It is not a Google TurboQuant implementation and does not claim
production inference acceleration.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .attention import attention_similarity, dense_attention, streaming_paged_attention
from .attention_perf import _resolve_device, _resolve_dtype, _runtime_warnings, _sync
from .bitpack import pack_indices, tensor_nbytes, unpack_indices
from .kv_presets import resolve_kv_cache_preset
from .quantizer import TurboQuantMSE


LAYOUT_QUANTIZATION_MODES = ("affine", "residual-affine", "codebook")


def _as_float(x: torch.Tensor) -> torch.Tensor:
    return x.float()


def _safe_scale(min_v: torch.Tensor, max_v: torch.Tensor, max_idx: int) -> torch.Tensor:
    scale = (max_v - min_v) / max(max_idx, 1)
    # If a channel is constant, keep a dummy scale so dequant works.
    return torch.where(scale.abs() < 1e-12, torch.ones_like(scale), scale)


def _affine_quantize(x: torch.Tensor, bits: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-page/per-channel affine quantization.

    ``x`` is shaped ``(B, H, L, D)``.  Min/scale are stored per ``(B,H,D)`` and
    shared across the page's token dimension.  This is still simple, but it is
    much more quality-stable for attention than the older global scalar
    codebook path.
    """
    bits = int(bits)
    if not 1 <= bits <= 8:
        raise ValueError(f"bits must be in [1, 8], got {bits}")
    x_f = x.float()
    max_idx = (1 << bits) - 1
    min_v = x_f.amin(dim=2, keepdim=True)
    max_v = x_f.amax(dim=2, keepdim=True)
    scale = _safe_scale(min_v, max_v, max_idx)
    idx = torch.round((x_f - min_v) / scale).clamp(0, max_idx).to(torch.uint8)
    return idx, min_v.contiguous(), scale.contiguous()


def _affine_dequantize(idx: torch.Tensor, min_v: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return idx.float() * scale + min_v


def _residual_quantize_sign(residual: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return a 1-bit sign tensor and per-page/per-channel residual magnitude.

    ``residual`` is shaped ``(B,H,L,D)``.  The sign bit stores whether the
    residual is non-negative.  The residual magnitude is the mean absolute
    residual per ``(B,H,D)`` and is intentionally tiny metadata.  This is not the
    full QJL residual from the TurboQuant paper; it is a practical
    TurboQuant-style correction probe for this research package.
    """
    r = residual.float()
    sign = (r >= 0).to(torch.uint8)
    magnitude = r.abs().mean(dim=2, keepdim=True).contiguous()
    return sign, magnitude


def _residual_dequantize_sign(sign_idx: torch.Tensor, magnitude: torch.Tensor) -> torch.Tensor:
    sign = sign_idx.float().mul(2.0).sub(1.0)
    return sign * magnitude


def _quality_summary(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    diff = a.float() - b.float()
    denom = a.float().norm().clamp_min(1e-12)
    rel = float(diff.norm() / denom)
    cos = float(F.cosine_similarity(a.float().reshape(1, -1), b.float().reshape(1, -1)).item())
    mse = float((diff ** 2).mean().item())
    max_abs = float(diff.abs().max().item())
    return {
        "relative_error": rel,
        "cosine_similarity": cos,
        "mse": mse,
        "max_abs_error": max_abs,
    }


def _attention_score_diagnostics(query: torch.Tensor, key: torch.Tensor, key_hat: torch.Tensor) -> Dict[str, float]:
    q = query.float()
    k = key.float()
    kh = key_hat.float()
    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    scores_hat = torch.matmul(q, kh.transpose(-2, -1)) * scale
    p = torch.softmax(scores, dim=-1)
    ph = torch.softmax(scores_hat, dim=-1)
    kl = torch.sum(p * (torch.log(p.clamp_min(1e-12)) - torch.log(ph.clamp_min(1e-12))), dim=-1)
    score_diff = scores_hat - scores
    return {
        "score_relative_error": float((scores - scores_hat).norm() / scores.norm().clamp_min(1e-12)),
        "score_cosine_similarity": float(F.cosine_similarity(scores.reshape(1, -1), scores_hat.reshape(1, -1)).item()),
        "inner_product_bias_mean": float(score_diff.mean().item()),
        "inner_product_bias_abs_mean": float(score_diff.abs().mean().item()),
        "inner_product_bias_max_abs": float(score_diff.abs().max().item()),
        "softmax_kl_mean": float(kl.mean().item()),
        "softmax_kl_max": float(kl.max().item()),
    }


def _attention_quality_warning(q: Dict[str, float]) -> str:
    cos = float(q.get("cosine_similarity", 0.0))
    rel = float(q.get("relative_error", 999.0))
    if cos >= 0.99 and rel <= 0.10:
        return "ok: compressed-page attention is close to dense in this benchmark."
    if cos >= 0.95:
        return "caution: compressed-page attention is directionally close, but error is still material."
    return "weak: compressed-page attention quality is not yet safe; use quality-layout settings or improve calibration before fused kernels."


@dataclass
class CompressedKVPage:
    """A single compressed K/V page.

    ``quantization_mode`` is either ``affine`` or ``codebook``.

    Affine mode stores per-page/per-channel min/scale tensors.  Key indices
    represent *rotated* K values, so a future attention kernel can rotate Q once
    and consume rotated key values directly.
    """

    layer_id: int
    page_id: int
    page_start: int
    page_length: int
    key_packed: torch.Tensor
    value_packed: torch.Tensor
    key_index_shape: Tuple[int, ...]
    value_index_shape: Tuple[int, ...]
    key_bits: int
    value_bits: int
    quantization_mode: str = "affine"
    key_min: Optional[torch.Tensor] = None
    key_scale: Optional[torch.Tensor] = None
    value_min: Optional[torch.Tensor] = None
    value_scale: Optional[torch.Tensor] = None
    key_residual_packed: Optional[torch.Tensor] = None
    value_residual_packed: Optional[torch.Tensor] = None
    key_residual_shape: Optional[Tuple[int, ...]] = None
    value_residual_shape: Optional[Tuple[int, ...]] = None
    key_residual_scale: Optional[torch.Tensor] = None
    value_residual_scale: Optional[torch.Tensor] = None

    @property
    def residual_payload_bytes(self) -> int:
        total = 0
        if self.key_residual_packed is not None:
            total += tensor_nbytes(self.key_residual_packed)
        if self.value_residual_packed is not None:
            total += tensor_nbytes(self.value_residual_packed)
        return int(total)

    @property
    def residual_metadata_bytes(self) -> int:
        total = 0
        if self.key_residual_scale is not None:
            total += tensor_nbytes(self.key_residual_scale)
        if self.value_residual_scale is not None:
            total += tensor_nbytes(self.value_residual_scale)
        return int(total)

    @property
    def payload_bytes(self) -> int:
        return int(tensor_nbytes(self.key_packed) + tensor_nbytes(self.value_packed) + self.residual_payload_bytes)

    @property
    def key_payload_bytes(self) -> int:
        extra = tensor_nbytes(self.key_residual_packed) if self.key_residual_packed is not None else 0
        return int(tensor_nbytes(self.key_packed) + extra)

    @property
    def value_payload_bytes(self) -> int:
        extra = tensor_nbytes(self.value_residual_packed) if self.value_residual_packed is not None else 0
        return int(tensor_nbytes(self.value_packed) + extra)

    @property
    def quant_metadata_bytes(self) -> int:
        total = 0
        for t in (self.key_min, self.key_scale, self.value_min, self.value_scale):
            if t is not None:
                total += tensor_nbytes(t)
        total += self.residual_metadata_bytes
        return int(total)

    def to_dict(self, include_payload_tensors: bool = False) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "layer_id": self.layer_id,
            "page_id": self.page_id,
            "page_start": self.page_start,
            "page_length": self.page_length,
            "key_index_shape": list(self.key_index_shape),
            "value_index_shape": list(self.value_index_shape),
            "key_bits": self.key_bits,
            "value_bits": self.value_bits,
            "quantization_mode": self.quantization_mode,
            "key_payload_bytes": self.key_payload_bytes,
            "value_payload_bytes": self.value_payload_bytes,
            "payload_bytes": self.payload_bytes,
            "residual_payload_bytes": self.residual_payload_bytes,
            "residual_metadata_bytes": self.residual_metadata_bytes,
            "quant_metadata_bytes": self.quant_metadata_bytes,
        }
        if include_payload_tensors:
            d["key_packed"] = self.key_packed.detach().cpu().tolist()
            d["value_packed"] = self.value_packed.detach().cpu().tolist()
        return d


@dataclass
class LayoutMemoryReport:
    """Memory accounting for a compressed KV page layout."""

    fp16_kv_bytes: int
    compressed_payload_bytes: int
    key_payload_bytes: int
    value_payload_bytes: int
    quant_metadata_bytes: int
    codebook_bytes: int
    rotation_metadata_bytes: int
    page_table_bytes: int
    actual_total_bytes: int
    effective_compression_ratio: float
    payload_compression_ratio: float
    effective_memory_saved_pct: float
    payload_memory_saved_pct: float
    page_count: int
    page_size: int
    quantization_mode: str = "affine"
    note: str = (
        "Effective memory includes payload, quant metadata, codebooks if used, rotation metadata, "
        "and a conservative page-table estimate. This is layout accounting, not a production allocator measurement."
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CompressedKVPageTable:
    """Kernel-ready compressed page table for one layer of K/V tensors."""

    pages: List[CompressedKVPage]
    key_quantizer: TurboQuantMSE
    value_quantizer: TurboQuantMSE
    dense_shape: Tuple[int, int, int, int]
    page_size: int
    layer_id: int = 0
    dtype_bytes: int = 2
    frozen_calibration: bool = True
    quantization_mode: str = "affine"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dense(
        cls,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        key_bits: int = 8,
        value_bits: int = 6,
        page_size: int = 128,
        layer_id: int = 0,
        seed: int = 0,
        dtype_bytes: int = 2,
        quantization_mode: str = "affine",
    ) -> "CompressedKVPageTable":
        if key.shape != value.shape:
            raise ValueError("key and value must have the same shape")
        if key.ndim != 4:
            raise ValueError("key/value must be shaped (B, H, S, D)")
        if quantization_mode not in LAYOUT_QUANTIZATION_MODES:
            raise ValueError(f"quantization_mode must be one of {LAYOUT_QUANTIZATION_MODES}")
        page_size = int(max(1, page_size))
        b, h, s, d = [int(x) for x in key.shape]
        # The key quantizer supplies the orthogonal rotation.  In affine mode its
        # scalar centroids are not used for the payload.
        q_key = TurboQuantMSE.build(d, bits=max(1, min(8, int(key_bits))), device=key.device, seed=seed, dtype=torch.float32)
        q_value = TurboQuantMSE.build(d, bits=max(1, min(8, int(value_bits))), device=value.device, seed=seed + 997, dtype=torch.float32)

        key_f = key.float()
        value_f = value.float()
        pages: List[CompressedKVPage] = []
        for page_id, start in enumerate(range(0, s, page_size)):
            end = min(start + page_size, s)
            k_page = key_f[:, :, start:end, :].contiguous()
            v_page = value_f[:, :, start:end, :].contiguous()
            k_res_packed = v_res_packed = None
            k_res_shape = v_res_shape = None
            k_res_scale = v_res_scale = None
            if quantization_mode in {"affine", "residual-affine"}:
                k_rot = q_key.rotation.apply(k_page).contiguous()
                k_idx, k_min, k_scale = _affine_quantize(k_rot, key_bits)
                v_idx, v_min, v_scale = _affine_quantize(v_page, value_bits)
                key_min, key_scale = k_min, k_scale
                value_min, value_scale = v_min, v_scale
                if quantization_mode == "residual-affine":
                    k_base = _affine_dequantize(k_idx, k_min, k_scale)
                    v_base = _affine_dequantize(v_idx, v_min, v_scale)
                    k_sign, k_res_scale = _residual_quantize_sign(k_rot - k_base)
                    v_sign, v_res_scale = _residual_quantize_sign(v_page - v_base)
                    k_res_packed, k_res_shape = pack_indices(k_sign, 1)
                    v_res_packed, v_res_shape = pack_indices(v_sign, 1)
            else:
                k_idx = q_key.quant(k_page)
                v_idx = q_value.quant(v_page)
                key_min = key_scale = value_min = value_scale = None
            k_packed, k_shape = pack_indices(k_idx, key_bits)
            v_packed, v_shape = pack_indices(v_idx, value_bits)
            pages.append(
                CompressedKVPage(
                    layer_id=layer_id,
                    page_id=page_id,
                    page_start=start,
                    page_length=end - start,
                    key_packed=k_packed.contiguous(),
                    value_packed=v_packed.contiguous(),
                    key_index_shape=tuple(k_shape),
                    value_index_shape=tuple(v_shape),
                    key_bits=int(key_bits),
                    value_bits=int(value_bits),
                    quantization_mode=quantization_mode,
                    key_min=key_min,
                    key_scale=key_scale,
                    value_min=value_min,
                    value_scale=value_scale,
                    key_residual_packed=k_res_packed.contiguous() if k_res_packed is not None else None,
                    value_residual_packed=v_res_packed.contiguous() if v_res_packed is not None else None,
                    key_residual_shape=tuple(k_res_shape) if k_res_shape is not None else None,
                    value_residual_shape=tuple(v_res_shape) if v_res_shape is not None else None,
                    key_residual_scale=k_res_scale,
                    value_residual_scale=v_res_scale,
                )
            )
        return cls(
            pages=pages,
            key_quantizer=q_key,
            value_quantizer=q_value,
            dense_shape=(b, h, s, d),
            page_size=page_size,
            layer_id=int(layer_id),
            dtype_bytes=int(dtype_bytes),
            frozen_calibration=True,
            quantization_mode=quantization_mode,
            metadata={
                "layout": "rotated-key-packed-pages",
                "key_bits": int(key_bits),
                "value_bits": int(value_bits),
                "quantization_mode": quantization_mode,
                "seed": int(seed),
            },
        )

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def seq_len(self) -> int:
        return int(self.dense_shape[2])

    @property
    def head_dim(self) -> int:
        return int(self.dense_shape[3])

    def freeze_calibration(self) -> None:
        self.frozen_calibration = True

    def calibrate_prefill(self, key: torch.Tensor, value: torch.Tensor) -> None:
        if key.shape[-1] != self.head_dim or value.shape[-1] != self.head_dim:
            raise ValueError("prefill tensors must match the page-table head dimension")
        self.frozen_calibration = True

    def _unpack_indices(self, page: CompressedKVPage, kind: str) -> torch.Tensor:
        if kind == "key":
            return unpack_indices(page.key_packed, page.key_bits, page.key_index_shape)
        if kind == "value":
            return unpack_indices(page.value_packed, page.value_bits, page.value_index_shape)
        raise ValueError("kind must be 'key' or 'value'")

    def dequant_key_page(self, page: CompressedKVPage, *, rotated: bool = False) -> torch.Tensor:
        idx = self._unpack_indices(page, "key")
        if page.quantization_mode in {"affine", "residual-affine"}:
            assert page.key_min is not None and page.key_scale is not None
            k_rot = _affine_dequantize(idx, page.key_min, page.key_scale)
            if page.quantization_mode == "residual-affine" and page.key_residual_packed is not None:
                assert page.key_residual_shape is not None and page.key_residual_scale is not None
                sign_idx = unpack_indices(page.key_residual_packed, 1, page.key_residual_shape)
                k_rot = k_rot + _residual_dequantize_sign(sign_idx, page.key_residual_scale)
            if rotated:
                return k_rot
            return self.key_quantizer.rotation.apply_T(k_rot)
        if rotated:
            return self.key_quantizer.centroids[idx.long()]
        return self.key_quantizer.dequant(idx)

    def dequant_value_page(self, page: CompressedKVPage) -> torch.Tensor:
        idx = self._unpack_indices(page, "value")
        if page.quantization_mode in {"affine", "residual-affine"}:
            assert page.value_min is not None and page.value_scale is not None
            v = _affine_dequantize(idx, page.value_min, page.value_scale)
            if page.quantization_mode == "residual-affine" and page.value_residual_packed is not None:
                assert page.value_residual_shape is not None and page.value_residual_scale is not None
                sign_idx = unpack_indices(page.value_residual_packed, 1, page.value_residual_shape)
                v = v + _residual_dequantize_sign(sign_idx, page.value_residual_scale)
            return v
        return self.value_quantizer.dequant(idx)

    def iter_dequantized_pages(self, *, rotated_keys: bool = False) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        for page in self.pages:
            yield self.dequant_key_page(page, rotated=rotated_keys), self.dequant_value_page(page)

    def to_dense(self) -> Tuple[torch.Tensor, torch.Tensor]:
        ks, vs = [], []
        for k, v in self.iter_dequantized_pages(rotated_keys=False):
            ks.append(k)
            vs.append(v)
        return torch.cat(ks, dim=2), torch.cat(vs, dim=2)

    def to_rotated_key_dense(self) -> torch.Tensor:
        ks = [self.dequant_key_page(p, rotated=True) for p in self.pages]
        return torch.cat(ks, dim=2)

    def rotate_query(self, query: torch.Tensor) -> torch.Tensor:
        return self.key_quantizer.rotation.apply(query.float()).to(query.dtype)

    def memory_report(self) -> LayoutMemoryReport:
        b, h, s, d = self.dense_shape
        fp16_kv_bytes = int(b * h * s * d * 2 * self.dtype_bytes)
        key_payload = int(sum(p.key_payload_bytes for p in self.pages))
        value_payload = int(sum(p.value_payload_bytes for p in self.pages))
        payload = key_payload + value_payload
        quant_meta = int(sum(p.quant_metadata_bytes for p in self.pages))
        if self.quantization_mode == "codebook":
            qk = self.key_quantizer
            qv = self.value_quantizer
            codebook = int(
                tensor_nbytes(qk.centroids)
                + tensor_nbytes(qk.boundaries)
                + tensor_nbytes(qv.centroids)
                + tensor_nbytes(qv.boundaries)
            )
        else:
            codebook = 0
        qk = self.key_quantizer
        rotation = int(
            tensor_nbytes(qk.rotation.s1)
            + tensor_nbytes(qk.rotation.s2)
            + tensor_nbytes(qk.rotation.perm)
            + tensor_nbytes(qk.rotation.inv_perm)
        )
        page_table = int(len(self.pages) * 64)
        actual = int(payload + quant_meta + codebook + rotation + page_table)
        return LayoutMemoryReport(
            fp16_kv_bytes=fp16_kv_bytes,
            compressed_payload_bytes=payload,
            key_payload_bytes=key_payload,
            value_payload_bytes=value_payload,
            quant_metadata_bytes=quant_meta,
            codebook_bytes=codebook,
            rotation_metadata_bytes=rotation,
            page_table_bytes=page_table,
            actual_total_bytes=actual,
            effective_compression_ratio=fp16_kv_bytes / max(actual, 1),
            payload_compression_ratio=fp16_kv_bytes / max(payload, 1),
            effective_memory_saved_pct=(1.0 - actual / max(fp16_kv_bytes, 1)) * 100.0,
            payload_memory_saved_pct=(1.0 - payload / max(fp16_kv_bytes, 1)) * 100.0,
            page_count=len(self.pages),
            page_size=int(self.page_size),
            quantization_mode=self.quantization_mode,
        )

    def to_dict(self, include_pages: bool = False) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "dense_shape": list(self.dense_shape),
            "page_size": self.page_size,
            "page_count": self.page_count,
            "layer_id": self.layer_id,
            "frozen_calibration": self.frozen_calibration,
            "quantization_mode": self.quantization_mode,
            "metadata": dict(self.metadata),
            "memory": self.memory_report().to_dict(),
        }
        if include_pages:
            d["pages"] = [p.to_dict() for p in self.pages]
        return d


class RotatedCompressedKVCache:
    """Minimal prefill/freeze wrapper around the compressed page table."""

    def __init__(
        self,
        *,
        key_bits: int = 8,
        value_bits: int = 6,
        page_size: int = 128,
        preset: Optional[str] = None,
        seed: int = 0,
        quantization_mode: str = "affine",
    ) -> None:
        if preset is not None:
            cfg = resolve_kv_cache_preset(preset)
            key_bits = int(cfg.get("key_bits", key_bits))
            value_bits = int(cfg.get("value_bits", value_bits))
        self.key_bits = int(key_bits)
        self.value_bits = int(value_bits)
        self.page_size = int(page_size)
        self.seed = int(seed)
        self.quantization_mode = quantization_mode
        self.tables: Dict[int, CompressedKVPageTable] = {}
        self._frozen = False

    @classmethod
    def from_preset(cls, preset: str = "quality", **overrides: Any) -> "RotatedCompressedKVCache":
        if preset in LAYOUT_PRESETS:
            cfg = dict(LAYOUT_PRESETS[preset])
            cfg.update(overrides)
            return cls(**cfg)
        cfg = resolve_kv_cache_preset(preset)
        return cls(
            key_bits=int(overrides.get("key_bits", cfg["key_bits"])),
            value_bits=int(overrides.get("value_bits", cfg["value_bits"])),
            page_size=int(overrides.get("page_size", 128)),
            preset=None,
            seed=int(overrides.get("seed", 0)),
            quantization_mode=str(overrides.get("quantization_mode", "affine")),
        )

    def calibrate_prefill(self, key: torch.Tensor, value: torch.Tensor, layer_idx: int = 0) -> CompressedKVPageTable:
        if self._frozen and layer_idx in self.tables:
            raise RuntimeError("cache calibration is frozen; allocate a new cache or call before freeze_calibration")
        table = CompressedKVPageTable.from_dense(
            key,
            value,
            key_bits=self.key_bits,
            value_bits=self.value_bits,
            page_size=self.page_size,
            layer_id=layer_idx,
            seed=self.seed + layer_idx * 1009,
            dtype_bytes=2 if key.dtype in {torch.float16, torch.bfloat16} else key.element_size(),
            quantization_mode=self.quantization_mode,
        )
        self.tables[int(layer_idx)] = table
        return table

    def freeze_calibration(self) -> None:
        self._frozen = True
        for table in self.tables.values():
            table.freeze_calibration()

    def table(self, layer_idx: int = 0) -> CompressedKVPageTable:
        return self.tables[int(layer_idx)]

    def memory_report(self, layer_idx: int = 0) -> LayoutMemoryReport:
        return self.table(layer_idx).memory_report()

    def attention(self, query: torch.Tensor, layer_idx: int = 0) -> torch.Tensor:
        return compressed_page_attention_reference(query, self.table(layer_idx), rotate_query=True)


LAYOUT_PRESETS: Dict[str, Dict[str, Any]] = {
    # Quality-first preset for correctness before fused kernels.
    "quality": {"key_bits": 8, "value_bits": 8, "quantization_mode": "affine"},
    # Good default for v0.8.1 layout hardening: preserve K strongly, compress V.
    "safe-layout": {"key_bits": 8, "value_bits": 6, "quantization_mode": "affine"},
    "balanced-layout": {"key_bits": 8, "value_bits": 4, "quantization_mode": "affine"},
    "aggressive-layout": {"key_bits": 6, "value_bits": 4, "quantization_mode": "affine"},
    "residual-safe": {"key_bits": 8, "value_bits": 6, "quantization_mode": "residual-affine"},
    "residual-balanced": {"key_bits": 6, "value_bits": 4, "quantization_mode": "residual-affine"},
    "residual-aggressive": {"key_bits": 4, "value_bits": 4, "quantization_mode": "residual-affine"},
    "legacy-codebook": {"key_bits": 6, "value_bits": 4, "quantization_mode": "codebook"},
}


def available_layout_presets() -> List[str]:
    return sorted(LAYOUT_PRESETS)


def resolve_layout_preset(name: str) -> Dict[str, Any]:
    if name not in LAYOUT_PRESETS:
        raise ValueError(f"unknown layout preset: {name!r}. Available: {available_layout_presets()}")
    return dict(LAYOUT_PRESETS[name])


@dataclass
class LayoutBenchConfig:
    batch_size: int = 1
    heads: int = 8
    query_len: int = 1
    seq_len: int = 1024
    head_dim: int = 64
    page_size: int = 128
    key_bits: int = 8
    value_bits: int = 6
    preset: Optional[str] = "safe-layout"
    quantization_mode: str = "affine"
    device: str = "auto"
    dtype: str = "auto"
    warmup: int = 1
    repeats: int = 3
    iters: Optional[int] = None
    seed: int = 123

    def __post_init__(self) -> None:
        if self.iters is not None:
            self.repeats = int(self.iters)
        if self.preset:
            if self.preset in LAYOUT_PRESETS:
                cfg = resolve_layout_preset(self.preset)
            else:
                cfg = resolve_kv_cache_preset(self.preset)
            self.key_bits = int(cfg["key_bits"])
            self.value_bits = int(cfg["value_bits"])
            self.quantization_mode = str(cfg.get("quantization_mode", self.quantization_mode))
        if self.quantization_mode not in LAYOUT_QUANTIZATION_MODES:
            raise ValueError(f"quantization_mode must be one of {LAYOUT_QUANTIZATION_MODES}")


@dataclass
class LayoutSweepConfig:
    batch_size: int = 1
    heads: int = 8
    query_len: int = 1
    seq_len: int = 512
    head_dim: int = 64
    page_size: int = 128
    bit_pairs: Sequence[str] = ("8,8", "8,6", "8,4", "6,4", "4,4")
    quantization_mode: str = "affine"
    device: str = "auto"
    dtype: str = "auto"
    seed: int = 123


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


def compressed_page_attention_reference(
    query: torch.Tensor,
    page_table: CompressedKVPageTable,
    *,
    rotate_query: bool = True,
) -> torch.Tensor:
    """Reference attention over the compressed page layout."""
    if rotate_query:
        q = page_table.rotate_query(query)
        pages = list(page_table.iter_dequantized_pages(rotated_keys=True))
        return streaming_paged_attention(q, pages)
    pages = list(page_table.iter_dequantized_pages(rotated_keys=False))
    return streaming_paged_attention(query, pages)


def rotate_q_attention_reference(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *, seed: int = 0) -> torch.Tensor:
    """Dense rotate-Q equivalence path without quantization."""
    d = int(key.shape[-1])
    rot = TurboQuantMSE.build(d, bits=8, device=key.device, seed=seed, dtype=torch.float32).rotation
    q_rot = rot.apply(query.float()).to(query.dtype)
    k_rot = rot.apply(key.float()).to(key.dtype)
    return dense_attention(q_rot, k_rot, value)


def run_rotate_q_check(
    *,
    batch_size: int = 1,
    heads: int = 8,
    query_len: int = 1,
    seq_len: int = 1024,
    head_dim: int = 64,
    device: str = "auto",
    dtype: str = "auto",
    seed: int = 123,
) -> Dict[str, Any]:
    requested_device = device
    requested_dtype = dtype
    resolved_device = _resolve_device(device)
    resolved_dtype = _resolve_dtype(dtype, resolved_device)
    warnings = _runtime_warnings(requested_device, resolved_device, requested_dtype, resolved_dtype)
    torch.manual_seed(int(seed))
    q = torch.randn(batch_size, heads, query_len, head_dim, device=resolved_device, dtype=resolved_dtype)
    k = torch.randn(batch_size, heads, seq_len, head_dim, device=resolved_device, dtype=resolved_dtype)
    v = torch.randn_like(k)
    dense = dense_attention(q, k, v)
    rotated = rotate_q_attention_reference(q, k, v, seed=seed)
    return {
        "version": "rotate-q-check",
        "config": {
            "batch_size": batch_size,
            "heads": heads,
            "query_len": query_len,
            "seq_len": seq_len,
            "head_dim": head_dim,
            "requested_device": requested_device,
            "device": resolved_device,
            "requested_dtype": requested_dtype,
            "dtype": str(resolved_dtype).replace("torch.", ""),
            "seed": seed,
        },
        "warnings": warnings,
        "quality": attention_similarity(dense, rotated),
        "interpretation": "Rotate-Q validates that K can be stored in rotated form and Q rotated once before attention.",
    }


def _build_random_qkv(config: LayoutBenchConfig | LayoutSweepConfig, device: str, dtype: torch.dtype):
    torch.manual_seed(int(config.seed))
    q = torch.randn(config.batch_size, config.heads, config.query_len, config.head_dim, device=device, dtype=dtype)
    k = torch.randn(config.batch_size, config.heads, config.seq_len, config.head_dim, device=device, dtype=dtype)
    v = torch.randn_like(k)
    return q, k, v


def _layout_quality_details(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, table: CompressedKVPageTable, dense_out: torch.Tensor, ref_out: torch.Tensor) -> Dict[str, Any]:
    k_hat, v_hat = table.to_dense()
    q_rot = table.rotate_query(q)
    k_rot = table.key_quantizer.rotation.apply(k.float()).to(q.dtype)
    k_rot_hat = table.to_rotated_key_dense().to(q.dtype)
    return {
        "compressed_page_vs_dense": attention_similarity(dense_out, ref_out),
        "key_reconstruction": _quality_summary(k, k_hat.to(k.dtype)),
        "rotated_key_reconstruction": _quality_summary(k_rot, k_rot_hat.to(k_rot.dtype)),
        "value_reconstruction": _quality_summary(v, v_hat.to(v.dtype)),
        "attention_scores": _attention_score_diagnostics(q_rot, k_rot, k_rot_hat),
    }


def run_layout_benchmark(config: LayoutBenchConfig) -> Dict[str, Any]:
    requested_device = config.device
    requested_dtype = config.dtype
    device = _resolve_device(requested_device)
    dtype = _resolve_dtype(requested_dtype, device)
    warnings = _runtime_warnings(requested_device, device, requested_dtype, dtype)
    q, k, v = _build_random_qkv(config, device, dtype)

    t0 = time.perf_counter()
    table = CompressedKVPageTable.from_dense(
        k,
        v,
        key_bits=config.key_bits,
        value_bits=config.value_bits,
        page_size=config.page_size,
        seed=config.seed,
        dtype_bytes=torch.empty((), dtype=dtype).element_size(),
        quantization_mode=config.quantization_mode,
    )
    build_seconds = time.perf_counter() - t0

    dense_out, dense_seconds = _time_call(
        lambda: dense_attention(q, k, v),
        device=device,
        warmup=config.warmup,
        repeats=config.repeats,
    )
    ref_out, ref_seconds = _time_call(
        lambda: compressed_page_attention_reference(q, table, rotate_query=True),
        device=device,
        warmup=config.warmup,
        repeats=config.repeats,
    )
    nonrot_out, nonrot_seconds = _time_call(
        lambda: compressed_page_attention_reference(q, table, rotate_query=False),
        device=device,
        warmup=0,
        repeats=1,
    )
    rotate_check = run_rotate_q_check(
        batch_size=config.batch_size,
        heads=config.heads,
        query_len=config.query_len,
        seq_len=config.seq_len,
        head_dim=config.head_dim,
        device=device,
        dtype=str(dtype).replace("torch.", ""),
        seed=config.seed,
    )
    mem = table.memory_report().to_dict()
    quality_details = _layout_quality_details(q, k, v, table, dense_out, ref_out)
    quality_details["nonrot_compressed_page_vs_dense"] = attention_similarity(dense_out, nonrot_out)
    quality_details["rotate_q_dense_equivalence"] = rotate_check["quality"]
    warning = _attention_quality_warning(quality_details["compressed_page_vs_dense"])
    quality_details["quality_warning"] = warning
    if warning.startswith("weak"):
        warnings = list(warnings) + [warning]
    return {
        "version": "0.10.3",
        "config": {
            **asdict(config),
            "device": device,
            "dtype": str(dtype).replace("torch.", ""),
            "requested_device": requested_device,
            "requested_dtype": requested_dtype,
            "iters": int(config.repeats),
        },
        "warnings": warnings,
        "layout": table.to_dict(include_pages=False),
        "memory": mem,
        "timing": {
            "layout_build_seconds": float(build_seconds),
            "dense_attention_seconds": float(dense_seconds),
            "compressed_page_reference_seconds": float(ref_seconds),
            "compressed_page_nonrot_reference_seconds": float(nonrot_seconds),
            "note": "Reference path is Python/PyTorch and is expected to be slower than dense attention until fused kernels exist.",
        },
        "quality": quality_details,
        "interpretation": (
            "v0.8.1 hardens the compressed page layout with affine per-page calibration, richer quality diagnostics, "
            "quality warnings, and v0.10.3 residual-affine correction diagnostics. It is TurboQuant-style research, not a Google TurboQuant implementation or production acceleration claim."
        ),
    }


def _parse_bit_pair(s: str) -> Tuple[int, int]:
    a, b = s.replace("/", ",").split(",", 1)
    return int(a), int(b)


def run_layout_quality_sweep(config: LayoutSweepConfig) -> Dict[str, Any]:
    requested_device = config.device
    requested_dtype = config.dtype
    device = _resolve_device(requested_device)
    dtype = _resolve_dtype(requested_dtype, device)
    warnings = _runtime_warnings(requested_device, device, requested_dtype, dtype)
    rows = []
    for pair in config.bit_pairs:
        kb, vb = _parse_bit_pair(pair)
        bench = LayoutBenchConfig(
            batch_size=config.batch_size,
            heads=config.heads,
            query_len=config.query_len,
            seq_len=config.seq_len,
            head_dim=config.head_dim,
            page_size=config.page_size,
            key_bits=kb,
            value_bits=vb,
            preset=None,
            quantization_mode=config.quantization_mode,
            device=device,
            dtype=str(dtype).replace("torch.", ""),
            warmup=0,
            repeats=1,
            seed=config.seed,
        )
        r = run_layout_benchmark(bench)
        rows.append({
            "key_bits": kb,
            "value_bits": vb,
            "quantization_mode": config.quantization_mode,
            "effective_compression_ratio": r["memory"]["effective_compression_ratio"],
            "effective_memory_saved_pct": r["memory"]["effective_memory_saved_pct"],
            "attention_relative_error": r["quality"]["compressed_page_vs_dense"]["relative_error"],
            "attention_cosine_similarity": r["quality"]["compressed_page_vs_dense"]["cosine_similarity"],
            "key_cosine_similarity": r["quality"]["key_reconstruction"]["cosine_similarity"],
            "value_cosine_similarity": r["quality"]["value_reconstruction"]["cosine_similarity"],
            "score_relative_error": r["quality"]["attention_scores"]["score_relative_error"],
            "inner_product_bias_abs_mean": r["quality"]["attention_scores"]["inner_product_bias_abs_mean"],
            "softmax_kl_mean": r["quality"]["attention_scores"]["softmax_kl_mean"],
            "quality_warning": r["quality"]["quality_warning"],
        })
    return {
        "version": "layout-quality-sweep-v0.10.3",
        "config": {
            **asdict(config),
            "device": device,
            "dtype": str(dtype).replace("torch.", ""),
            "requested_device": requested_device,
            "requested_dtype": requested_dtype,
        },
        "warnings": warnings,
        "rows": rows,
        "interpretation": "Use the sweep to choose a quality-safe compressed page layout. residual-affine is a TurboQuant-style sign-residual experiment, not a Google TurboQuant implementation.",
    }




@dataclass
class ResidualSweepConfig:
    batch_size: int = 1
    heads: int = 8
    query_len: int = 1
    seq_len: int = 1024
    head_dim: int = 64
    page_size: int = 128
    bit_pairs: Sequence[str] = ("8,6", "6,4", "4,4")
    modes: Sequence[str] = ("affine", "residual-affine")
    device: str = "auto"
    dtype: str = "auto"
    seed: int = 123


def run_residual_correction_sweep(config: ResidualSweepConfig) -> Dict[str, Any]:
    requested_device = config.device
    requested_dtype = config.dtype
    device = _resolve_device(requested_device)
    dtype = _resolve_dtype(requested_dtype, device)
    warnings = _runtime_warnings(requested_device, device, requested_dtype, dtype)
    rows = []
    grouped: Dict[str, Dict[str, Any]] = {}
    for pair in config.bit_pairs:
        kb, vb = _parse_bit_pair(pair)
        grouped[pair] = {}
        for mode in config.modes:
            if mode not in LAYOUT_QUANTIZATION_MODES:
                raise ValueError(f"unknown quantization mode {mode!r}")
            bench = LayoutBenchConfig(
                batch_size=config.batch_size,
                heads=config.heads,
                query_len=config.query_len,
                seq_len=config.seq_len,
                head_dim=config.head_dim,
                page_size=config.page_size,
                key_bits=kb,
                value_bits=vb,
                preset=None,
                quantization_mode=mode,
                device=device,
                dtype=str(dtype).replace("torch.", ""),
                warmup=0,
                repeats=1,
                seed=config.seed,
            )
            r = run_layout_benchmark(bench)
            row = {
                "key_bits": kb,
                "value_bits": vb,
                "bit_pair": f"{kb},{vb}",
                "quantization_mode": mode,
                "effective_compression_ratio": r["memory"]["effective_compression_ratio"],
                "effective_memory_saved_pct": r["memory"]["effective_memory_saved_pct"],
                "attention_relative_error": r["quality"]["compressed_page_vs_dense"]["relative_error"],
                "attention_cosine_similarity": r["quality"]["compressed_page_vs_dense"]["cosine_similarity"],
                "key_cosine_similarity": r["quality"]["key_reconstruction"]["cosine_similarity"],
                "value_cosine_similarity": r["quality"]["value_reconstruction"]["cosine_similarity"],
                "score_relative_error": r["quality"]["attention_scores"]["score_relative_error"],
                "inner_product_bias_abs_mean": r["quality"]["attention_scores"]["inner_product_bias_abs_mean"],
                "softmax_kl_mean": r["quality"]["attention_scores"]["softmax_kl_mean"],
                "quality_warning": r["quality"]["quality_warning"],
            }
            rows.append(row)
            grouped[pair][mode] = row
    comparisons = []
    for pair, by_mode in grouped.items():
        if "affine" in by_mode and "residual-affine" in by_mode:
            base = by_mode["affine"]
            res = by_mode["residual-affine"]
            comparisons.append({
                "bit_pair": pair,
                "attention_relative_error_delta": res["attention_relative_error"] - base["attention_relative_error"],
                "attention_cosine_delta": res["attention_cosine_similarity"] - base["attention_cosine_similarity"],
                "score_relative_error_delta": res["score_relative_error"] - base["score_relative_error"],
                "softmax_kl_delta": res["softmax_kl_mean"] - base["softmax_kl_mean"],
                "compression_ratio_delta": res["effective_compression_ratio"] - base["effective_compression_ratio"],
                "interpretation": "negative error/KL deltas mean residual correction improved the metric; compression can drop because residual signs add payload.",
            })
    return {
        "version": "residual-correction-sweep-v0.10.3",
        "config": {
            **asdict(config),
            "device": device,
            "dtype": str(dtype).replace("torch.", ""),
            "requested_device": requested_device,
            "requested_dtype": requested_dtype,
        },
        "warnings": warnings,
        "rows": rows,
        "comparisons": comparisons,
        "interpretation": (
            "v0.10.3 compares affine layouts against a TurboQuant-style 1-bit sign residual correction. "
            "This probes whether lower-bit KV layouts can preserve QK inner products and attention output better. "
            "It is not a Google TurboQuant implementation and does not claim production inference acceleration."
        ),
    }


def layout_markdown_report(report: Dict[str, Any]) -> str:
    mem = report["memory"]
    timing = report["timing"]
    quality = report["quality"]
    lines = [
        "# tiny-turboquant layout benchmark report",
        "",
        "## Memory",
        f"- FP16 K/V bytes: {mem['fp16_kv_bytes']}",
        f"- Compressed payload bytes: {mem['compressed_payload_bytes']}",
        f"- Quant metadata bytes: {mem.get('quant_metadata_bytes', 0)}",
        f"- Actual total bytes: {mem['actual_total_bytes']}",
        f"- Payload compression ratio: {mem['payload_compression_ratio']:.4f}x",
        f"- Effective compression ratio: {mem['effective_compression_ratio']:.4f}x",
        f"- Effective memory saved: {mem['effective_memory_saved_pct']:.2f}%",
        "",
        "## Timing",
        f"- Layout build seconds: {timing['layout_build_seconds']:.6f}",
        f"- Dense attention seconds: {timing['dense_attention_seconds']:.6f}",
        f"- Compressed page reference seconds: {timing['compressed_page_reference_seconds']:.6f}",
        "",
        "## Quality",
        f"- Compressed page vs dense: {quality['compressed_page_vs_dense']}",
        f"- Key reconstruction: {quality.get('key_reconstruction')}",
        f"- Value reconstruction: {quality.get('value_reconstruction')}",
        f"- Attention scores: {quality.get('attention_scores')}",
        f"- Quality warning: {quality.get('quality_warning')}",
        f"- Rotate-Q dense equivalence: {quality['rotate_q_dense_equivalence']}",
        "",
        "## Boundary",
        "This report validates layout and correctness for future fused kernels. It does not claim production inference acceleration.",
    ]
    return "\n".join(lines) + "\n"


def layout_sweep_markdown_report(report: Dict[str, Any]) -> str:
    lines = [
        "# tiny-turboquant layout quality sweep",
        "",
        "| K bits | V bits | Compression | Saved % | Attention cos | Rel err | K cos | V cos | Softmax KL | Warning |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in report["rows"]:
        lines.append(
            f"| {r['key_bits']} | {r['value_bits']} | {r['effective_compression_ratio']:.3f}x | "
            f"{r['effective_memory_saved_pct']:.2f} | {r['attention_cosine_similarity']:.6f} | "
            f"{r['attention_relative_error']:.6f} | {r['key_cosine_similarity']:.6f} | "
            f"{r['value_cosine_similarity']:.6f} | {r['softmax_kl_mean']:.6f} | {r['quality_warning']} |"
        )
    lines.append("\nThis sweep is diagnostic and does not claim production inference acceleration.\n")
    return "\n".join(lines)




def residual_sweep_markdown_report(report: Dict[str, Any]) -> str:
    lines = [
        "# tiny-turboquant v0.10.3 residual-correction sweep",
        "",
        "This is a TurboQuant-style residual-correction diagnostic, not a Google TurboQuant implementation and not a production acceleration claim.",
        "",
        "| K bits | V bits | Mode | Compression | Saved % | Attention cos | Rel err | Score rel err | Bias abs mean | Softmax KL | Warning |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in report["rows"]:
        lines.append(
            f"| {r['key_bits']} | {r['value_bits']} | {r['quantization_mode']} | "
            f"{r['effective_compression_ratio']:.3f}x | {r['effective_memory_saved_pct']:.2f} | "
            f"{r['attention_cosine_similarity']:.6f} | {r['attention_relative_error']:.6f} | "
            f"{r['score_relative_error']:.6f} | {r['inner_product_bias_abs_mean']:.6f} | "
            f"{r['softmax_kl_mean']:.6f} | {r['quality_warning']} |"
        )
    if report.get("comparisons"):
        lines.extend(["", "## Residual deltas", "", "| Bits | Attn rel-err delta | Attn cos delta | Score rel-err delta | Softmax KL delta | Compression delta |", "|---|---:|---:|---:|---:|---:|"])
        for c in report["comparisons"]:
            lines.append(
                f"| {c['bit_pair']} | {c['attention_relative_error_delta']:.6f} | "
                f"{c['attention_cosine_delta']:.6f} | {c['score_relative_error_delta']:.6f} | "
                f"{c['softmax_kl_delta']:.6f} | {c['compression_ratio_delta']:.6f} |"
            )
    lines.append("\nBoundary: residual correction is a research diagnostic and does not imply production inference acceleration.\n")
    return "\n".join(lines)


def save_layout_json(report: Dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")


def save_layout_markdown(report: Dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(layout_markdown_report(report), encoding="utf-8")


def save_layout_sweep_markdown(report: Dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(layout_sweep_markdown_report(report), encoding="utf-8")


def save_residual_sweep_markdown(report: Dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(residual_sweep_markdown_report(report), encoding="utf-8")
