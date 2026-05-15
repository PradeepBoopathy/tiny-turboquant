"""Experimental fused compressed decode-attention research utilities.

v0.10.7 extends the Triton fused compressed decode-attention prototype with
CUDA Graph replay diagnostics on top of page-size/BLOCK_M sweep support for the affine safe-layout path.  The supported fast path remains intentionally narrow:
query_len=1, batch_size=1, key_bits=8, affine per-page/per-channel K, and
value_bits in {4, 6, 8}.  Unsupported configurations fall back to the
quality-safe PyTorch compressed-page reference path and label that mode
explicitly.  v0.10.7 reports normal fused timing, optional CUDA Graph replay timing, kernel block size, optional page-size sweep results, and speed/quality gaps
against dense attention, SDPA, and the Python compressed-page reference path.

This module does not claim production inference acceleration.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import torch

try:  # optional dependency; Windows/CPU users should still import the package.
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - depends on runtime environment
    triton = None
    tl = None

from .attention import attention_similarity, dense_attention, sdpa_attention
from .attention_perf import _resolve_device, _resolve_dtype, _runtime_warnings, _sync, triton_status
from .layout import (
    CompressedKVPageTable,
    LayoutBenchConfig,
    compressed_page_attention_reference,
    resolve_layout_preset,
)


def _time_call(fn, *, device: str, warmup: int, repeats: int) -> Tuple[torch.Tensor, float]:
    out = None
    for _ in range(max(0, int(warmup))):
        out = fn()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(max(1, int(repeats))):
        out = fn()
    _sync(device)
    elapsed = (time.perf_counter() - t0) / max(1, int(repeats))
    assert out is not None
    return out, float(elapsed)


def _time_cuda_graph_call(fn, *, device: str, warmup: int, graph_replays: int) -> Tuple[Optional[torch.Tensor], Optional[float], Dict[str, Any]]:
    """Try to capture and replay a fixed-shape CUDA decode call.

    CUDA Graph replay is a benchmark mode, not a different algorithm. It can
    fail if the runtime, allocator, or called operators are not graph-safe. In
    that case the caller should keep the normal timing and report the reason.
    """
    if device != "cuda" or not torch.cuda.is_available():
        return None, None, {
            "attempted": False,
            "captured": False,
            "reason": "CUDA Graph replay requires a CUDA device",
        }
    if not hasattr(torch.cuda, "CUDAGraph"):
        return None, None, {
            "attempted": False,
            "captured": False,
            "reason": "torch.cuda.CUDAGraph is unavailable in this PyTorch build",
        }

    replays = max(1, int(graph_replays))
    try:
        out = None
        for _ in range(max(1, int(warmup))):
            out = fn()
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        # Capture allocations into the graph memory pool. The captured output
        # tensor is reused on replay; the benchmark is valid for fixed-shape
        # decode microbenchmarks.
        with torch.cuda.graph(graph):
            out = fn()
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(replays):
            graph.replay()
        torch.cuda.synchronize()
        seconds = (time.perf_counter() - t0) / replays
        assert out is not None
        return out, float(seconds), {
            "attempted": True,
            "captured": True,
            "replays": replays,
            "reason": "CUDA Graph capture and replay succeeded",
        }
    except Exception as exc:  # pragma: no cover - CUDA/runtime dependent
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        return None, None, {
            "attempted": True,
            "captured": False,
            "replays": replays,
            "reason": f"CUDA Graph capture/replay failed: {exc!r}",
        }


@dataclass
class FusedDecodeBenchConfig:
    """Configuration for experimental compressed decode-attention benchmarks."""

    batch_size: int = 1
    heads: int = 8
    query_len: int = 1
    seq_len: int = 2048
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
    prefer_triton: bool = True
    kernel_block_m: int = 64
    kernel_num_warps: int = 4
    tune_kernel: bool = False
    tune_block_m_values: Tuple[int, ...] = (8, 16, 32, 64, 128)
    tune_num_warps: bool = False
    tune_num_warps_values: Tuple[int, ...] = (1, 2, 4, 8)
    tune_page_size: bool = False
    tune_page_size_values: Tuple[int, ...] = (64, 128, 256)
    cuda_graph: bool = False
    graph_replays: int = 100
    competitiveness_target: str = "sdpa"

    def __post_init__(self) -> None:
        if self.iters is not None:
            self.repeats = int(self.iters)
        if self.preset:
            cfg = resolve_layout_preset(self.preset)
            self.key_bits = int(cfg.get("key_bits", self.key_bits))
            self.value_bits = int(cfg.get("value_bits", self.value_bits))
            self.quantization_mode = str(cfg.get("quantization_mode", self.quantization_mode))
        if self.query_len != 1:
            # The v0.9 path is explicitly decode-only.  Keep this strict so
            # users do not confuse it with prefill attention.
            raise ValueError("FusedDecodeBenchConfig is decode-only: query_len must be 1")


def _triton_runtime_status() -> Dict[str, Any]:
    status = triton_status()
    # Keep a normalized field stable for reports.  v0.9.0 accidentally looked
    # for ``triton_available`` while triton_status() exposes ``available``.
    return {
        "available": bool(status.get("available", False)),
        "package_importable": bool(status.get("package_importable", False)),
        "cuda_available": bool(status.get("cuda_available", False)),
        "detail": status,
    }




if triton is not None and tl is not None:  # pragma: no cover - requires CUDA+Triton
    @triton.jit
    def _ttq_affine_decode_kernel(
        q_ptr,
        key_packed_ptr,
        value_packed_ptr,
        key_min_ptr,
        key_scale_ptr,
        value_min_ptr,
        value_scale_ptr,
        key_residual_packed_ptr,
        value_residual_packed_ptr,
        key_residual_scale_ptr,
        value_residual_scale_ptr,
        out_ptr,
        H: tl.constexpr,
        S: tl.constexpr,
        D: tl.constexpr,
        PAGE: tl.constexpr,
        PAGES: tl.constexpr,
        KEY_BITS: tl.constexpr,
        VALUE_BITS: tl.constexpr,
        HAS_RESIDUAL: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
        SCALE: tl.constexpr,
    ):
        # One program computes one head for batch=1/query_len=1.  v0.10.7
        # extends the narrow Triton prototype to residual-affine pages and
        # low-bit K as well as V.  This is still an experimental microkernel,
        # not a general serving kernel.
        h = tl.program_id(0)
        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        q = tl.load(q_ptr + h * D + offs_d, mask=mask_d, other=0.0).to(tl.float32)

        m = tl.full((), -3.4028234663852886e38, tl.float32)
        l = tl.full((), 0.0, tl.float32)
        acc = tl.zeros((BLOCK_D,), tl.float32)
        offs_m = tl.arange(0, BLOCK_M)

        values_per_page = H * PAGE * D
        key_bytes_per_page = (values_per_page * KEY_BITS + 7) // 8
        value_bytes_per_page = (values_per_page * VALUE_BITS + 7) // 8
        residual_bytes_per_page = (values_per_page + 7) // 8

        for p in range(0, PAGES):
            k_min = tl.load(key_min_ptr + (p * H + h) * D + offs_d, mask=mask_d, other=0.0).to(tl.float32)
            k_scale = tl.load(key_scale_ptr + (p * H + h) * D + offs_d, mask=mask_d, other=1.0).to(tl.float32)
            v_min = tl.load(value_min_ptr + (p * H + h) * D + offs_d, mask=mask_d, other=0.0).to(tl.float32)
            v_scale = tl.load(value_scale_ptr + (p * H + h) * D + offs_d, mask=mask_d, other=1.0).to(tl.float32)

            if HAS_RESIDUAL:
                k_res_scale = tl.load(key_residual_scale_ptr + (p * H + h) * D + offs_d, mask=mask_d, other=0.0).to(tl.float32)
                v_res_scale = tl.load(value_residual_scale_ptr + (p * H + h) * D + offs_d, mask=mask_d, other=0.0).to(tl.float32)

            for start in range(0, PAGE, BLOCK_M):
                t = start + offs_m
                token_mask = t < PAGE
                flat = h * PAGE * D + t[:, None] * D + offs_d[None, :]
                mask = token_mask[:, None] & mask_d[None, :]

                if KEY_BITS == 8:
                    k_idx = tl.load(
                        key_packed_ptr + p * key_bytes_per_page + flat,
                        mask=mask,
                        other=0,
                    ).to(tl.float32)
                else:
                    k_bit_pos = flat * KEY_BITS
                    k_byte_pos = k_bit_pos // 8
                    k_shift = k_bit_pos - k_byte_pos * 8
                    kb0 = tl.load(
                        key_packed_ptr + p * key_bytes_per_page + k_byte_pos,
                        mask=mask,
                        other=0,
                    ).to(tl.uint32)
                    kb1 = tl.load(
                        key_packed_ptr + p * key_bytes_per_page + k_byte_pos + 1,
                        mask=mask & ((k_byte_pos + 1) < key_bytes_per_page),
                        other=0,
                    ).to(tl.uint32)
                    k_raw = kb0 | (kb1 << 8)
                    k_idx_i = (k_raw >> k_shift) & ((1 << KEY_BITS) - 1)
                    k_idx = k_idx_i.to(tl.float32)

                k_val = k_idx * k_scale[None, :] + k_min[None, :]

                if HAS_RESIDUAL:
                    sign_bit_pos = flat
                    sign_byte_pos = sign_bit_pos // 8
                    sign_shift = sign_bit_pos - sign_byte_pos * 8
                    ksb = tl.load(
                        key_residual_packed_ptr + p * residual_bytes_per_page + sign_byte_pos,
                        mask=mask,
                        other=0,
                    ).to(tl.uint32)
                    k_sign = ((ksb >> sign_shift) & 1).to(tl.float32) * 2.0 - 1.0
                    k_val = k_val + k_sign * k_res_scale[None, :]

                scores = tl.sum(k_val * q[None, :], axis=1) * SCALE
                scores = tl.where(token_mask, scores, -3.4028234663852886e38)
                block_m = tl.max(scores, axis=0)
                m_new = tl.maximum(m, block_m)
                alpha = tl.exp(m - m_new)
                probs = tl.exp(scores - m_new)

                if VALUE_BITS == 8:
                    v_idx = tl.load(
                        value_packed_ptr + p * value_bytes_per_page + flat,
                        mask=mask,
                        other=0,
                    ).to(tl.float32)
                else:
                    bit_pos = flat * VALUE_BITS
                    byte_pos = bit_pos // 8
                    shift = bit_pos - byte_pos * 8
                    b0 = tl.load(
                        value_packed_ptr + p * value_bytes_per_page + byte_pos,
                        mask=mask,
                        other=0,
                    ).to(tl.uint32)
                    b1 = tl.load(
                        value_packed_ptr + p * value_bytes_per_page + byte_pos + 1,
                        mask=mask & ((byte_pos + 1) < value_bytes_per_page),
                        other=0,
                    ).to(tl.uint32)
                    raw = b0 | (b1 << 8)
                    v_idx_i = (raw >> shift) & ((1 << VALUE_BITS) - 1)
                    v_idx = v_idx_i.to(tl.float32)

                v_val = v_idx * v_scale[None, :] + v_min[None, :]

                if HAS_RESIDUAL:
                    sign_bit_pos = flat
                    sign_byte_pos = sign_bit_pos // 8
                    sign_shift = sign_bit_pos - sign_byte_pos * 8
                    vsb = tl.load(
                        value_residual_packed_ptr + p * residual_bytes_per_page + sign_byte_pos,
                        mask=mask,
                        other=0,
                    ).to(tl.uint32)
                    v_sign = ((vsb >> sign_shift) & 1).to(tl.float32) * 2.0 - 1.0
                    v_val = v_val + v_sign * v_res_scale[None, :]

                acc = acc * alpha + tl.sum(probs[:, None] * v_val, axis=0)
                l = l * alpha + tl.sum(probs, axis=0)
                m = m_new

        out = acc / l
        tl.store(out_ptr + h * D + offs_d, out, mask=mask_d)

    @triton.jit
    def _ttq_split_k_stage1_kernel(
        q_ptr,
        key_packed_ptr,
        value_packed_ptr,
        key_min_ptr,
        key_scale_ptr,
        value_min_ptr,
        value_scale_ptr,
        key_residual_packed_ptr,
        value_residual_packed_ptr,
        key_residual_scale_ptr,
        value_residual_scale_ptr,
        partial_m_ptr,
        partial_l_ptr,
        partial_acc_ptr,
        H: tl.constexpr,
        S: tl.constexpr,
        D: tl.constexpr,
        PAGE: tl.constexpr,
        PAGES: tl.constexpr,
        SPLIT_K_SLABS: tl.constexpr,
        KEY_BITS: tl.constexpr,
        VALUE_BITS: tl.constexpr,
        HAS_RESIDUAL: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
        SCALE: tl.constexpr,
    ):
        # v0.10.7 measured split-K Stage-1 partial kernel.  Each program
        # computes slab-local online-softmax statistics for one head and one
        # sequence slab.  Stage 2 reduction remains a Torch reference path in
        # this release.
        h = tl.program_id(0)
        slab = tl.program_id(1)
        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        q = tl.load(q_ptr + h * D + offs_d, mask=mask_d, other=0.0).to(tl.float32)

        m = tl.full((), -3.4028234663852886e38, tl.float32)
        l = tl.full((), 0.0, tl.float32)
        acc = tl.zeros((BLOCK_D,), tl.float32)
        offs_m = tl.arange(0, BLOCK_M)

        values_per_page = H * PAGE * D
        key_bytes_per_page = (values_per_page * KEY_BITS + 7) // 8
        value_bytes_per_page = (values_per_page * VALUE_BITS + 7) // 8
        residual_bytes_per_page = (values_per_page + 7) // 8
        pages_per_slab = (PAGES + SPLIT_K_SLABS - 1) // SPLIT_K_SLABS
        page_start = slab * pages_per_slab

        for local_p in range(0, pages_per_slab):
            p = page_start + local_p
            page_active = p < PAGES
            k_min = tl.load(key_min_ptr + (p * H + h) * D + offs_d, mask=page_active & mask_d, other=0.0).to(tl.float32)
            k_scale = tl.load(key_scale_ptr + (p * H + h) * D + offs_d, mask=page_active & mask_d, other=1.0).to(tl.float32)
            v_min = tl.load(value_min_ptr + (p * H + h) * D + offs_d, mask=page_active & mask_d, other=0.0).to(tl.float32)
            v_scale = tl.load(value_scale_ptr + (p * H + h) * D + offs_d, mask=page_active & mask_d, other=1.0).to(tl.float32)

            if HAS_RESIDUAL:
                k_res_scale = tl.load(key_residual_scale_ptr + (p * H + h) * D + offs_d, mask=page_active & mask_d, other=0.0).to(tl.float32)
                v_res_scale = tl.load(value_residual_scale_ptr + (p * H + h) * D + offs_d, mask=page_active & mask_d, other=0.0).to(tl.float32)

            for start in range(0, PAGE, BLOCK_M):
                t = start + offs_m
                token_mask = (t < PAGE) & page_active
                flat = h * PAGE * D + t[:, None] * D + offs_d[None, :]
                mask = token_mask[:, None] & mask_d[None, :]

                if KEY_BITS == 8:
                    k_idx = tl.load(
                        key_packed_ptr + p * key_bytes_per_page + flat,
                        mask=mask,
                        other=0,
                    ).to(tl.float32)
                else:
                    k_bit_pos = flat * KEY_BITS
                    k_byte_pos = k_bit_pos // 8
                    k_shift = k_bit_pos - k_byte_pos * 8
                    kb0 = tl.load(
                        key_packed_ptr + p * key_bytes_per_page + k_byte_pos,
                        mask=mask,
                        other=0,
                    ).to(tl.uint32)
                    kb1 = tl.load(
                        key_packed_ptr + p * key_bytes_per_page + k_byte_pos + 1,
                        mask=mask & ((k_byte_pos + 1) < key_bytes_per_page),
                        other=0,
                    ).to(tl.uint32)
                    k_raw = kb0 | (kb1 << 8)
                    k_idx_i = (k_raw >> k_shift) & ((1 << KEY_BITS) - 1)
                    k_idx = k_idx_i.to(tl.float32)

                k_val = k_idx * k_scale[None, :] + k_min[None, :]

                if HAS_RESIDUAL:
                    sign_bit_pos = flat
                    sign_byte_pos = sign_bit_pos // 8
                    sign_shift = sign_bit_pos - sign_byte_pos * 8
                    ksb = tl.load(
                        key_residual_packed_ptr + p * residual_bytes_per_page + sign_byte_pos,
                        mask=mask,
                        other=0,
                    ).to(tl.uint32)
                    k_sign = ((ksb >> sign_shift) & 1).to(tl.float32) * 2.0 - 1.0
                    k_val = k_val + k_sign * k_res_scale[None, :]

                scores = tl.sum(k_val * q[None, :], axis=1) * SCALE
                scores = tl.where(token_mask, scores, -3.4028234663852886e38)
                block_m = tl.max(scores, axis=0)
                m_new = tl.maximum(m, block_m)
                alpha = tl.exp(m - m_new)
                probs = tl.exp(scores - m_new)

                if VALUE_BITS == 8:
                    v_idx = tl.load(
                        value_packed_ptr + p * value_bytes_per_page + flat,
                        mask=mask,
                        other=0,
                    ).to(tl.float32)
                else:
                    bit_pos = flat * VALUE_BITS
                    byte_pos = bit_pos // 8
                    shift = bit_pos - byte_pos * 8
                    b0 = tl.load(
                        value_packed_ptr + p * value_bytes_per_page + byte_pos,
                        mask=mask,
                        other=0,
                    ).to(tl.uint32)
                    b1 = tl.load(
                        value_packed_ptr + p * value_bytes_per_page + byte_pos + 1,
                        mask=mask & ((byte_pos + 1) < value_bytes_per_page),
                        other=0,
                    ).to(tl.uint32)
                    raw = b0 | (b1 << 8)
                    v_idx_i = (raw >> shift) & ((1 << VALUE_BITS) - 1)
                    v_idx = v_idx_i.to(tl.float32)

                v_val = v_idx * v_scale[None, :] + v_min[None, :]

                if HAS_RESIDUAL:
                    sign_bit_pos = flat
                    sign_byte_pos = sign_bit_pos // 8
                    sign_shift = sign_bit_pos - sign_byte_pos * 8
                    vsb = tl.load(
                        value_residual_packed_ptr + p * residual_bytes_per_page + sign_byte_pos,
                        mask=mask,
                        other=0,
                    ).to(tl.uint32)
                    v_sign = ((vsb >> sign_shift) & 1).to(tl.float32) * 2.0 - 1.0
                    v_val = v_val + v_sign * v_res_scale[None, :]

                acc = acc * alpha + tl.sum(probs[:, None] * v_val, axis=0)
                l = l * alpha + tl.sum(probs, axis=0)
                m = m_new

        idx = slab * H + h
        tl.store(partial_m_ptr + idx, m)
        tl.store(partial_l_ptr + idx, l)
        tl.store(partial_acc_ptr + idx * D + offs_d, acc, mask=mask_d)

    @triton.jit
    def _ttq_split_k_stage2_reduce_kernel(
        partial_m_ptr,
        partial_l_ptr,
        partial_acc_ptr,
        out_ptr,
        H: tl.constexpr,
        D: tl.constexpr,
        SLABS: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        # v0.10.7 Triton Stage-2 reducer.  One program reduces all split-K
        # slab partials for one head.  The Stage-1 contract is:
        # partial_acc = sum(exp(score - local_max) * V), i.e. unnormalized.
        h = tl.program_id(0)
        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < D

        global_m = tl.full((), -3.4028234663852886e38, tl.float32)
        for slab in range(0, SLABS):
            m = tl.load(partial_m_ptr + slab * H + h).to(tl.float32)
            global_m = tl.maximum(global_m, m)

        denom = tl.full((), 0.0, tl.float32)
        acc = tl.zeros((BLOCK_D,), tl.float32)
        for slab in range(0, SLABS):
            m = tl.load(partial_m_ptr + slab * H + h).to(tl.float32)
            l = tl.load(partial_l_ptr + slab * H + h).to(tl.float32)
            w = tl.exp(m - global_m)
            vals = tl.load(
                partial_acc_ptr + (slab * H + h) * D + offs_d,
                mask=mask_d,
                other=0.0,
            ).to(tl.float32)
            acc += w * vals
            denom += w * l

        out = acc / denom
        tl.store(out_ptr + h * D + offs_d, out, mask=mask_d)
else:  # pragma: no cover
    _ttq_affine_decode_kernel = None
    _ttq_split_k_stage1_kernel = None
    _ttq_split_k_stage2_reduce_kernel = None


def _is_supported_triton_config(query: torch.Tensor, page_table: CompressedKVPageTable) -> Tuple[bool, str]:
    if triton is None or tl is None or _ttq_affine_decode_kernel is None:
        return False, "triton package is not importable"
    if not query.is_cuda:
        return False, "query is not on CUDA"
    if query.ndim != 4 or query.shape[0] != 1 or query.shape[2] != 1:
        return False, "v0.10.7 Triton prototype supports batch=1 and query_len=1 only"
    if page_table.quantization_mode not in {"affine", "residual-affine"}:
        return False, "v0.10.7 Triton prototype supports affine and residual-affine layouts only"
    if page_table.page_count <= 0:
        return False, "page table is empty"
    if page_table.dense_shape[0] != 1:
        return False, "v0.10.7 Triton prototype supports batch=1 page tables only"
    if page_table.head_dim not in (32, 64, 128):
        return False, "v0.10.7 Triton prototype supports head_dim in {32,64,128}"
    if any(p.key_bits not in (4, 6, 8) for p in page_table.pages):
        return False, "v0.10.7 Triton prototype supports key_bits in {4,6,8}"
    if any(p.value_bits not in (4, 6, 8) for p in page_table.pages):
        return False, "v0.10.7 Triton prototype supports value_bits in {4,6,8}"
    if len({p.key_bits for p in page_table.pages}) != 1:
        return False, "all pages must use the same key_bits"
    if len({p.value_bits for p in page_table.pages}) != 1:
        return False, "all pages must use the same value_bits"
    if len({p.page_length for p in page_table.pages}) != 1:
        return False, "all pages must have the same page length"
    if page_table.pages[0].page_length != page_table.page_size:
        return False, "v0.10.7 Triton prototype requires full pages; pad or use divisible seq_len"
    if page_table.quantization_mode == "residual-affine":
        if any(p.key_residual_packed is None or p.value_residual_packed is None for p in page_table.pages):
            return False, "residual-affine pages must include residual sign payloads"
    return True, "supported"


def _concat_page_payloads(page_table: CompressedKVPageTable) -> Dict[str, torch.Tensor]:
    pages = page_table.pages
    key_packed = torch.cat([p.key_packed.reshape(-1) for p in pages]).contiguous()
    value_packed = torch.cat([p.value_packed.reshape(-1) for p in pages]).contiguous()
    key_min = torch.cat([p.key_min.squeeze(0).squeeze(1) for p in pages], dim=0).contiguous()
    key_scale = torch.cat([p.key_scale.squeeze(0).squeeze(1) for p in pages], dim=0).contiguous()
    value_min = torch.cat([p.value_min.squeeze(0).squeeze(1) for p in pages], dim=0).contiguous()
    value_scale = torch.cat([p.value_scale.squeeze(0).squeeze(1) for p in pages], dim=0).contiguous()

    device = key_packed.device
    if page_table.quantization_mode == "residual-affine":
        key_residual_packed = torch.cat([p.key_residual_packed.reshape(-1) for p in pages]).contiguous()
        value_residual_packed = torch.cat([p.value_residual_packed.reshape(-1) for p in pages]).contiguous()
        key_residual_scale = torch.cat([p.key_residual_scale.squeeze(0).squeeze(1) for p in pages], dim=0).contiguous()
        value_residual_scale = torch.cat([p.value_residual_scale.squeeze(0).squeeze(1) for p in pages], dim=0).contiguous()
    else:
        key_residual_packed = torch.empty(1, device=device, dtype=torch.uint8)
        value_residual_packed = torch.empty(1, device=device, dtype=torch.uint8)
        key_residual_scale = torch.empty_like(key_min)
        value_residual_scale = torch.empty_like(value_min)

    return {
        "key_packed": key_packed,
        "value_packed": value_packed,
        "key_min": key_min,
        "key_scale": key_scale,
        "value_min": value_min,
        "value_scale": value_scale,
        "key_residual_packed": key_residual_packed,
        "value_residual_packed": value_residual_packed,
        "key_residual_scale": key_residual_scale,
        "value_residual_scale": value_residual_scale,
    }


def _triton_fused_affine_decode_attention(
    query: torch.Tensor,
    page_table: CompressedKVPageTable,
    *,
    kernel_block_m: int = 64,
    kernel_num_warps: int = 4,
) -> torch.Tensor:
    ok, reason = _is_supported_triton_config(query, page_table)
    if not ok:
        raise RuntimeError(reason)
    if int(kernel_block_m) not in (8, 16, 32, 64, 128):
        raise RuntimeError("v0.10.7 Triton prototype supports kernel_block_m in {8,16,32,64,128}")
    if int(kernel_num_warps) not in (1, 2, 4, 8):
        raise RuntimeError("v0.10.7 Triton prototype supports kernel_num_warps in {1,2,4,8}")
    q_rot = page_table.rotate_query(query).contiguous()
    payloads = _concat_page_payloads(page_table)
    b, h, s, d = page_table.dense_shape
    page_size = int(page_table.page_size)
    pages = int(page_table.page_count)
    key_bits = int(page_table.pages[0].key_bits)
    value_bits = int(page_table.pages[0].value_bits)
    has_residual = page_table.quantization_mode == "residual-affine"
    out = torch.empty((1, h, 1, d), device=query.device, dtype=query.dtype)
    # v0.10.7 exposes BLOCK_M so users can tune page-loop granularity.
    block_d = 1 << ((d - 1).bit_length())
    block_m = int(kernel_block_m)
    _ttq_affine_decode_kernel[(h,)](
        q_rot,
        payloads["key_packed"],
        payloads["value_packed"],
        payloads["key_min"],
        payloads["key_scale"],
        payloads["value_min"],
        payloads["value_scale"],
        payloads["key_residual_packed"],
        payloads["value_residual_packed"],
        payloads["key_residual_scale"],
        payloads["value_residual_scale"],
        out,
        H=h,
        S=s,
        D=d,
        PAGE=page_size,
        PAGES=pages,
        KEY_BITS=key_bits,
        VALUE_BITS=value_bits,
        HAS_RESIDUAL=has_residual,
        BLOCK_M=block_m,
        BLOCK_D=block_d,
        SCALE=float(d ** -0.5),
        num_warps=int(kernel_num_warps),
    )
    return out

def triton_split_k_stage1_partials(
    query: torch.Tensor,
    page_table: CompressedKVPageTable,
    *,
    split_k_slabs: int = 4,
    kernel_block_m: int = 64,
    kernel_num_warps: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Run the v0.10.7 Triton Stage-1 split-K partial-statistics kernel.

    Returns ``(partial_m, partial_l, partial_acc, metadata)`` where:
    - partial_m is shaped ``(slabs, heads)``
    - partial_l is shaped ``(slabs, heads)``
    - partial_acc is shaped ``(slabs, heads, head_dim)``

    This is only Stage 1. Stage 2 reduction is intentionally a separate
    reference step in v0.10.7.
    """
    ok, reason = _is_supported_triton_config(query, page_table)
    if not ok:
        raise RuntimeError(reason)
    if triton is None or tl is None or _ttq_split_k_stage1_kernel is None:
        raise RuntimeError("v0.10.7 Triton split-K Stage-1 kernel is unavailable")
    if int(kernel_block_m) not in (8, 16, 32, 64, 128):
        raise RuntimeError("v0.10.7 Triton split-K Stage-1 supports kernel_block_m in {8,16,32,64,128}")
    if int(kernel_num_warps) not in (1, 2, 4, 8):
        raise RuntimeError("v0.10.7 Triton split-K Stage-1 supports kernel_num_warps in {1,2,4,8}")
    slabs = max(1, int(split_k_slabs))
    q_rot = page_table.rotate_query(query).contiguous()
    payloads = _concat_page_payloads(page_table)
    b, h, s, d = page_table.dense_shape
    page_size = int(page_table.page_size)
    pages = int(page_table.page_count)
    key_bits = int(page_table.pages[0].key_bits)
    value_bits = int(page_table.pages[0].value_bits)
    has_residual = page_table.quantization_mode == "residual-affine"
    partial_m = torch.empty((slabs, h), device=query.device, dtype=torch.float32)
    partial_l = torch.empty((slabs, h), device=query.device, dtype=torch.float32)
    partial_acc = torch.empty((slabs, h, d), device=query.device, dtype=torch.float32)
    block_d = 1 << ((d - 1).bit_length())
    _ttq_split_k_stage1_kernel[(h, slabs)](
        q_rot,
        payloads["key_packed"],
        payloads["value_packed"],
        payloads["key_min"],
        payloads["key_scale"],
        payloads["value_min"],
        payloads["value_scale"],
        payloads["key_residual_packed"],
        payloads["value_residual_packed"],
        payloads["key_residual_scale"],
        payloads["value_residual_scale"],
        partial_m,
        partial_l,
        partial_acc,
        H=h,
        S=s,
        D=d,
        PAGE=page_size,
        PAGES=pages,
        SPLIT_K_SLABS=slabs,
        KEY_BITS=key_bits,
        VALUE_BITS=value_bits,
        HAS_RESIDUAL=has_residual,
        BLOCK_M=int(kernel_block_m),
        BLOCK_D=block_d,
        SCALE=float(d ** -0.5),
        num_warps=int(kernel_num_warps),
    )
    meta = {
        "mode": "triton-split-k-stage1-partials",
        "split_k_slabs": slabs,
        "kernel_block_m": int(kernel_block_m),
        "kernel_num_warps": int(kernel_num_warps),
        "quantization_mode": page_table.quantization_mode,
        "uses_compressed_pages_directly": True,
        "constructs_full_dense_kv": False,
        "boundary": "Stage-1 partial statistics only; Stage-2 Triton reducer is future work",
    }
    return partial_m, partial_l, partial_acc, meta


def triton_split_k_stage2_reduce(
    partial_m: torch.Tensor,
    partial_l: torch.Tensor,
    partial_acc: torch.Tensor,
    *,
    output_dtype: Optional[torch.dtype] = None,
    kernel_num_warps: int = 1,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Run the v0.10.7 Triton Stage-2 split-K reducer.

    Stage 1 must provide unnormalized accumulators:
    ``partial_acc = sum(exp(score - local_max) * V)``.  The reducer combines
    the slab-local statistics with log-sum-exp correction entirely in Triton.
    """
    if triton is None or tl is None or _ttq_split_k_stage2_reduce_kernel is None:
        raise RuntimeError("v0.10.7 Triton split-K Stage-2 reducer is unavailable")
    if not partial_m.is_cuda or not partial_l.is_cuda or not partial_acc.is_cuda:
        raise RuntimeError("v0.10.7 Triton split-K Stage-2 reducer requires CUDA tensors")
    if partial_m.ndim != 2 or partial_l.ndim != 2 or partial_acc.ndim != 3:
        raise ValueError("partial_m/partial_l/partial_acc must be shaped (slabs, heads), (slabs, heads), (slabs, heads, head_dim)")
    slabs, heads = partial_m.shape
    if partial_l.shape != (slabs, heads) or partial_acc.shape[:2] != (slabs, heads):
        raise ValueError("split-K partial tensor shapes do not agree")
    d = int(partial_acc.shape[-1])
    if d not in (32, 64, 128):
        raise RuntimeError("v0.10.7 Triton Stage-2 reducer supports head_dim in {32,64,128}")
    if int(kernel_num_warps) not in (1, 2, 4, 8):
        raise RuntimeError("v0.10.7 Triton Stage-2 reducer supports kernel_num_warps in {1,2,4,8}")
    dtype = output_dtype or partial_acc.dtype
    out = torch.empty((1, heads, 1, d), device=partial_acc.device, dtype=dtype)
    block_d = 1 << ((d - 1).bit_length())
    _ttq_split_k_stage2_reduce_kernel[(heads,)](
        partial_m.contiguous(),
        partial_l.contiguous(),
        partial_acc.contiguous(),
        out,
        H=int(heads),
        D=int(d),
        SLABS=int(slabs),
        BLOCK_D=int(block_d),
        num_warps=int(kernel_num_warps),
    )
    meta = {
        "mode": "triton-split-k-stage2-reducer",
        "split_k_slabs": int(slabs),
        "kernel_num_warps": int(kernel_num_warps),
        "output_dtype": str(dtype).replace("torch.", ""),
        "stage1_contract": "partial_acc is unnormalized sum(exp(score-local_max) * V)",
        "boundary": "Stage-2 reducer only; full production FlashDecoding remains future work",
    }
    return out, meta


def triton_split_k_full_attention_reference(
    query: torch.Tensor,
    page_table: CompressedKVPageTable,
    *,
    split_k_slabs: int = 4,
    kernel_block_m: int = 64,
    kernel_num_warps: int = 4,
    reducer_num_warps: int = 1,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Run real Triton Stage 1 plus real Triton Stage 2 reducer.

    This is the v0.10.7 measured full split-K research path.  It is still a
    microbenchmark prototype, not a production serving kernel.
    """
    pm, pl, pa, meta1 = triton_split_k_stage1_partials(
        query,
        page_table,
        split_k_slabs=split_k_slabs,
        kernel_block_m=kernel_block_m,
        kernel_num_warps=kernel_num_warps,
    )
    out, meta2 = triton_split_k_stage2_reduce(
        pm,
        pl,
        pa,
        output_dtype=query.dtype,
        kernel_num_warps=reducer_num_warps,
    )
    meta = {
        "mode": "triton-two-stage-split-k",
        "stage1": meta1,
        "stage2": meta2,
        "uses_compressed_pages_directly": True,
        "constructs_full_dense_kv": False,
        "boundary": "measured research split-K path; not a production serving kernel",
    }
    return out, meta


def reduce_split_k_partials(partial_m: torch.Tensor, partial_l: torch.Tensor, partial_acc: torch.Tensor, *, output_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Reference Stage-2 log-sum-exp reduction for Stage-1 split-K partials.

    Contract fixed in v0.10.7:
    ``partial_acc`` must be the **unnormalized** slab accumulator
    ``sum(exp(score - local_max) * V)``.  Therefore the global reducer must
    scale ``partial_acc`` only by ``exp(local_max - global_max)``.  It must not
    multiply ``partial_acc`` by ``partial_l`` again.  Multiplying by
    ``partial_l`` is only correct when Stage 1 stores a locally normalized
    attention output, which this package does not do.
    """
    m_global = torch.max(partial_m, dim=0).values  # (H,)
    slab_scale = torch.exp(partial_m - m_global.unsqueeze(0))
    denom = torch.sum(slab_scale * partial_l, dim=0).clamp_min(1e-30)
    acc = torch.sum(slab_scale.unsqueeze(-1) * partial_acc, dim=0) / denom.unsqueeze(-1)
    if output_dtype is not None:
        acc = acc.to(output_dtype)
    return acc.unsqueeze(0).unsqueeze(2)


def triton_split_k_stage1_attention_reference(
    query: torch.Tensor,
    page_table: CompressedKVPageTable,
    *,
    split_k_slabs: int = 4,
    kernel_block_m: int = 64,
    kernel_num_warps: int = 4,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Run Triton Stage 1 plus Torch Stage 2 reduction for correctness checks."""
    pm, pl, pa, meta = triton_split_k_stage1_partials(
        query,
        page_table,
        split_k_slabs=split_k_slabs,
        kernel_block_m=kernel_block_m,
        kernel_num_warps=kernel_num_warps,
    )
    out = reduce_split_k_partials(pm, pl, pa, output_dtype=query.dtype)
    meta = dict(meta)
    meta["mode"] = "triton-stage1-plus-torch-reduce-reference"
    meta["stage2"] = "torch-logsumexp-reference"
    return out, meta


def experimental_fused_compressed_decode_attention(
    query: torch.Tensor,
    page_table: CompressedKVPageTable,
    *,
    prefer_triton: bool = True,
    kernel_block_m: int = 64,
    kernel_num_warps: int = 4,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Run the v0.10.7 experimental compressed decode-attention path.

    The function consumes the compressed page table directly and avoids full
    dense K/V reconstruction.  When the narrow Triton prototype supports the
    supplied configuration, it runs that kernel.  Otherwise it falls back to the
    verified PyTorch compressed-page reference and reports the fallback clearly.
    """

    if query.ndim != 4:
        raise ValueError("query must be shaped (B, H, 1, D)")
    if query.shape[2] != 1:
        raise ValueError("experimental fused decode attention supports query_len=1 only")

    runtime = _triton_runtime_status()
    supported, support_reason = _is_supported_triton_config(query, page_table)
    can_attempt_triton = bool(prefer_triton) and runtime["available"] and runtime["cuda_available"] and supported

    triton_error: Optional[str] = None
    if can_attempt_triton:
        try:
            out = _triton_fused_affine_decode_attention(query, page_table, kernel_block_m=kernel_block_m, kernel_num_warps=kernel_num_warps)
            meta = {
                "mode": ("triton-fused-residual-affine-decode" if page_table.quantization_mode == "residual-affine" else "triton-fused-affine-decode"),
                "triton_attempted": True,
                "triton_runtime": runtime,
                "reason": ("minimal v0.10.7 Triton fused residual-affine decode kernel executed" if page_table.quantization_mode == "residual-affine" else "minimal v0.10.7 Triton fused affine decode kernel executed"),
                "support_check": support_reason,
                "query_len": int(query.shape[2]),
                "kernel_block_m": int(kernel_block_m),
                "kernel_num_warps": int(kernel_num_warps),
                "quantization_mode": page_table.quantization_mode,
                "uses_compressed_pages_directly": True,
                "constructs_full_dense_kv": False,
                "boundary": "experimental Triton prototype; not a production serving kernel",
            }
            return out, meta
        except Exception as exc:  # pragma: no cover - depends on Triton runtime/GPU
            triton_error = repr(exc)

    reason = support_reason
    if not prefer_triton:
        reason = "prefer_triton=False; using PyTorch compressed-page fallback"
    elif not runtime["available"] or not runtime["cuda_available"]:
        reason = "triton runtime unavailable"
    elif triton_error:
        reason = f"triton kernel failed; using fallback: {triton_error}"

    out = compressed_page_attention_reference(query, page_table, rotate_query=True)
    meta = {
        "mode": "torch-compressed-page-fallback",
        "triton_attempted": bool(can_attempt_triton),
        "triton_runtime": runtime,
        "reason": reason,
        "support_check": support_reason,
        "triton_error": triton_error,
        "query_len": int(query.shape[2]),
        "kernel_block_m": int(kernel_block_m),
        "kernel_num_warps": int(kernel_num_warps),
        "uses_compressed_pages_directly": True,
        "constructs_full_dense_kv": False,
        "boundary": "experimental research path; fallback is not a production fused CUDA/Triton kernel",
    }
    return out, meta

def _build_random_qkv(config: FusedDecodeBenchConfig, device: str, dtype: torch.dtype):
    torch.manual_seed(int(config.seed))
    q = torch.randn(config.batch_size, config.heads, 1, config.head_dim, device=device, dtype=dtype)
    k = torch.randn(config.batch_size, config.heads, config.seq_len, config.head_dim, device=device, dtype=dtype)
    v = torch.randn_like(k)
    return q, k, v


def _quality_warning(metrics: Dict[str, float]) -> str:
    rel = float(metrics.get("relative_error", 999.0))
    cos = float(metrics.get("cosine_similarity", 0.0))
    if cos >= 0.99 and rel <= 0.05:
        return "ok: fused prototype output is close to dense in this benchmark."
    if cos >= 0.95:
        return "caution: fused prototype is directionally close, but error is still material."
    return "weak: fused prototype quality is not safe for kernel work with this layout."




def _safe_ratio(numerator: float, denominator: float) -> Optional[float]:
    if denominator is None or denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _candidate_block_values(config: FusedDecodeBenchConfig) -> List[int]:
    if config.tune_kernel:
        vals = [int(v) for v in config.tune_block_m_values]
    else:
        vals = [int(config.kernel_block_m)]
    out: List[int] = []
    for v in vals:
        if v in (8, 16, 32, 64, 128) and v not in out:
            out.append(v)
    if not out:
        out = [64]
    return out


def _candidate_warp_values(config: FusedDecodeBenchConfig) -> List[int]:
    if config.tune_num_warps:
        vals = [int(v) for v in config.tune_num_warps_values]
    else:
        vals = [int(config.kernel_num_warps)]
    out: List[int] = []
    for v in vals:
        if v in (1, 2, 4, 8) and v not in out:
            out.append(v)
    if not out:
        out = [4]
    return out

def _run_fused_decode_benchmark_core(config: FusedDecodeBenchConfig) -> Dict[str, Any]:
    """Benchmark dense, SDPA, compressed-page reference, and v0.9 fused path for one page size."""

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
    layout_build_seconds = time.perf_counter() - t0

    dense_out, dense_seconds = _time_call(
        lambda: dense_attention(q, k, v),
        device=device,
        warmup=config.warmup,
        repeats=config.repeats,
    )

    try:
        sdpa_out, sdpa_seconds = _time_call(
            lambda: sdpa_attention(q, k, v),
            device=device,
            warmup=config.warmup,
            repeats=config.repeats,
        )
        sdpa_report: Optional[Dict[str, Any]] = {
            "seconds": sdpa_seconds,
            "similarity_to_dense": attention_similarity(dense_out, sdpa_out),
        }
    except Exception as exc:  # pragma: no cover depends on torch build/backend
        sdpa_report = {"error": repr(exc)}

    reference_out, reference_seconds = _time_call(
        lambda: compressed_page_attention_reference(q, table, rotate_query=True),
        device=device,
        warmup=config.warmup,
        repeats=config.repeats,
    )

    fused_meta_holder: Dict[str, Any] = {}
    tuning_rows: List[Dict[str, Any]] = []
    selected_block_m = int(config.kernel_block_m)
    fused_out: Optional[torch.Tensor] = None
    fused_seconds: float = 0.0

    selected_num_warps = int(config.kernel_num_warps)
    for block_m in _candidate_block_values(config):
        for num_warps in _candidate_warp_values(config):
            local_meta: Dict[str, Any] = {}

            def _run_fused(block_m=block_m, num_warps=num_warps):
                out, meta = experimental_fused_compressed_decode_attention(
                    q,
                    table,
                    prefer_triton=config.prefer_triton,
                    kernel_block_m=block_m,
                    kernel_num_warps=num_warps,
                )
                local_meta.clear()
                local_meta.update(meta)
                return out

            out_i, sec_i = _time_call(
                _run_fused,
                device=device,
                warmup=config.warmup,
                repeats=config.repeats,
            )
            q_i = attention_similarity(dense_out, out_i)
            row = {
                "kernel_block_m": int(block_m),
                "kernel_num_warps": int(num_warps),
                "seconds": float(sec_i),
                "mode": local_meta.get("mode"),
                "relative_error": q_i.get("relative_error"),
                "cosine_similarity": q_i.get("cosine_similarity"),
            }
            tuning_rows.append(row)

            if fused_out is None or sec_i < fused_seconds:
                fused_out = out_i
                fused_seconds = float(sec_i)
                selected_block_m = int(block_m)
                selected_num_warps = int(num_warps)
                fused_meta_holder.clear()
                fused_meta_holder.update(local_meta)

    assert fused_out is not None
    fused_meta_holder["selected_kernel_block_m"] = int(selected_block_m)
    fused_meta_holder["selected_kernel_num_warps"] = int(selected_num_warps)
    fused_meta_holder["tune_kernel"] = bool(config.tune_kernel)
    fused_meta_holder["tune_num_warps"] = bool(config.tune_num_warps)

    graph_report: Dict[str, Any] = {
        "enabled": bool(config.cuda_graph),
        "attempted": False,
        "captured": False,
        "seconds": None,
        "speedup_vs_normal_fused": None,
        "quality_vs_dense": None,
        "quality_vs_normal_fused": None,
        "reason": "CUDA Graph replay disabled",
    }
    if bool(config.cuda_graph):
        def _run_selected_fused():
            out, _meta = experimental_fused_compressed_decode_attention(
                q,
                table,
                prefer_triton=config.prefer_triton,
                kernel_block_m=selected_block_m,
                kernel_num_warps=selected_num_warps,
            )
            return out

        graph_out, graph_seconds, graph_meta = _time_cuda_graph_call(
            _run_selected_fused,
            device=device,
            warmup=config.warmup,
            graph_replays=config.graph_replays,
        )
        graph_report.update(graph_meta)
        if graph_seconds is not None and graph_out is not None:
            graph_report["seconds"] = float(graph_seconds)
            graph_report["speedup_vs_normal_fused"] = _safe_ratio(fused_seconds, graph_seconds)
            graph_report["quality_vs_dense"] = attention_similarity(dense_out, graph_out)
            graph_report["quality_vs_normal_fused"] = attention_similarity(fused_out, graph_out)

    fused_meta_holder["cuda_graph_enabled"] = bool(config.cuda_graph)
    fused_meta_holder["cuda_graph_captured"] = bool(graph_report.get("captured", False))

    mem = table.memory_report().to_dict()
    fused_quality = attention_similarity(dense_out, fused_out)
    reference_quality = attention_similarity(dense_out, reference_out)
    warning = _quality_warning(fused_quality)
    if warning.startswith("weak"):
        warnings = list(warnings) + [warning]

    return {
        "version": "0.10.7",
        "benchmark": "experimental-fused-compressed-decode-attention",
        "config": {
            **asdict(config),
            "device": device,
            "dtype": str(dtype).replace("torch.", ""),
            "requested_device": requested_device,
            "requested_dtype": requested_dtype,
            "iters": int(config.repeats),
        },
        "warnings": warnings,
        "execution": fused_meta_holder,
        "layout": table.to_dict(include_pages=False),
        "memory": mem,
        "timing": {
            "layout_build_seconds": float(layout_build_seconds),
            "dense_attention_seconds": float(dense_seconds),
            "sdpa": sdpa_report,
            "compressed_page_reference_seconds": float(reference_seconds),
            "experimental_fused_seconds": float(fused_seconds),
            "selected_kernel_block_m": int(selected_block_m),
            "selected_kernel_num_warps": int(selected_num_warps),
            "kernel_tuning": tuning_rows,
            "kernel_algorithm_note": "v0.10.7 tunes BLOCK_M and Triton num_warps for the safe-layout/residual microkernel; it does not add split-K or production serving integration.",
            "speedup_vs_compressed_page_reference": _safe_ratio(reference_seconds, fused_seconds),
            "speedup_vs_dense_manual": _safe_ratio(dense_seconds, fused_seconds),
            "speedup_vs_sdpa": (
                _safe_ratio(sdpa_report["seconds"], fused_seconds)
                if isinstance(sdpa_report, dict) and "seconds" in sdpa_report
                else None
            ),
            "cuda_graph": graph_report,
            "graph_fused_seconds": graph_report.get("seconds"),
            "graph_speedup_vs_normal_fused": graph_report.get("speedup_vs_normal_fused"),
            "note": (
                "Timing is diagnostic. v0.10.7 adds CUDA Graph replay diagnostics around the minimal fused-attention prototype; "
                "production acceleration is not claimed."
            ),
        },
        "quality": {
            "fused_vs_dense": fused_quality,
            "reference_vs_dense": reference_quality,
            "fused_vs_reference": attention_similarity(reference_out, fused_out),
            "quality_warning": warning,
        },
        "interpretation": (
            "v0.10.7 benchmarks an experimental compressed decode-attention path over the compressed page layout with optional BLOCK_M/page-size tuning and CUDA Graph replay diagnostics. "
            "It reads compressed pages without constructing full dense K/V. When a production Triton kernel is not "
            "available, the benchmark uses the verified PyTorch compressed-page fallback and reports that mode explicitly."
        ),
    }



def _candidate_page_sizes(config: FusedDecodeBenchConfig) -> List[int]:
    if config.tune_page_size:
        vals = [int(v) for v in config.tune_page_size_values]
    else:
        vals = [int(config.page_size)]
    out: List[int] = []
    for v in vals:
        if v > 0 and config.seq_len % v == 0 and v not in out:
            out.append(v)
    if not out:
        out = [int(config.page_size)]
    return out


def _add_competitiveness_summary(report: Dict[str, Any]) -> Dict[str, Any]:
    timing = report.get("timing", {})
    fused = float(timing.get("experimental_fused_seconds") or 0.0)
    graph = timing.get("cuda_graph") if isinstance(timing.get("cuda_graph"), dict) else {}
    graph_seconds = graph.get("seconds") if graph and graph.get("captured") else None
    effective_fused = float(graph_seconds) if graph_seconds is not None else fused
    effective_mode = "cuda-graph-replay" if graph_seconds is not None else "normal-fused-call"

    dense = float(timing.get("dense_attention_seconds") or 0.0)
    sdpa = timing.get("sdpa")
    sdpa_seconds = float(sdpa.get("seconds")) if isinstance(sdpa, dict) and "seconds" in sdpa else None
    ref = float(timing.get("compressed_page_reference_seconds") or 0.0)

    def ratio(a: Optional[float], b: Optional[float]) -> Optional[float]:
        if a is None or b is None or b <= 0:
            return None
        return float(a) / float(b)

    fused_over_dense = ratio(fused, dense)
    fused_over_sdpa = ratio(fused, sdpa_seconds)
    ref_over_fused = ratio(ref, fused)
    effective_over_dense = ratio(effective_fused, dense)
    effective_over_sdpa = ratio(effective_fused, sdpa_seconds)
    ref_over_effective = ratio(ref, effective_fused)

    competitive_dense = effective_over_dense is not None and effective_over_dense <= 1.0
    competitive_sdpa = effective_over_sdpa is not None and effective_over_sdpa <= 1.0

    summary = {
        "target": report.get("config", {}).get("competitiveness_target", "sdpa"),
        "effective_mode": effective_mode,
        "effective_fused_seconds": effective_fused,
        "fused_over_dense": fused_over_dense,
        "fused_over_sdpa": fused_over_sdpa,
        "reference_over_fused": ref_over_fused,
        "effective_over_dense": effective_over_dense,
        "effective_over_sdpa": effective_over_sdpa,
        "reference_over_effective": ref_over_effective,
        "cuda_graph_captured": bool(graph.get("captured", False)) if graph else False,
        "faster_than_dense_manual": competitive_dense,
        "faster_than_sdpa": competitive_sdpa,
        "faster_than_compressed_page_reference": ref_over_effective is not None and ref_over_effective > 1.0,
        "verdict": (
            "competitive-with-sdpa" if competitive_sdpa else
            "competitive-with-dense-manual" if competitive_dense else
            "faster-than-python-reference-only" if ref_over_effective is not None and ref_over_effective > 1.0 else
            "not-competitive-yet"
        ),
        "boundary": "This is a microbenchmark diagnostic, not production inference acceleration.",
    }
    report["competitiveness"] = summary
    report.setdefault("timing", {})["fused_over_dense_manual"] = fused_over_dense
    report.setdefault("timing", {})["fused_over_sdpa"] = fused_over_sdpa
    report.setdefault("timing", {})["effective_fused_seconds"] = effective_fused
    report.setdefault("timing", {})["effective_fused_mode"] = effective_mode
    report.setdefault("timing", {})["effective_over_dense_manual"] = effective_over_dense
    report.setdefault("timing", {})["effective_over_sdpa"] = effective_over_sdpa
    return report


def run_fused_decode_benchmark(config: FusedDecodeBenchConfig) -> Dict[str, Any]:
    """Benchmark dense, SDPA, compressed-page reference, and v0.9 fused path.

    v0.10.7 can optionally sweep page sizes in addition to BLOCK_M.  The returned
    report is always the fastest fused candidate, with all candidates recorded
    under ``page_size_tuning`` when page-size tuning is enabled.
    """
    page_sizes = _candidate_page_sizes(config)
    if len(page_sizes) <= 1:
        return _add_competitiveness_summary(_run_fused_decode_benchmark_core(config))

    reports: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    for ps in page_sizes:
        cfg = FusedDecodeBenchConfig(**{**asdict(config), "page_size": int(ps), "tune_page_size": False})
        rep = _add_competitiveness_summary(_run_fused_decode_benchmark_core(cfg))
        reports.append(rep)
        q = rep.get("quality", {}).get("fused_vs_dense", {})
        rows.append({
            "page_size": int(ps),
            "selected_kernel_block_m": rep.get("timing", {}).get("selected_kernel_block_m"),
            "selected_kernel_num_warps": rep.get("timing", {}).get("selected_kernel_num_warps"),
            "experimental_fused_seconds": rep.get("timing", {}).get("experimental_fused_seconds"),
            "graph_fused_seconds": rep.get("timing", {}).get("graph_fused_seconds"),
            "effective_fused_seconds": rep.get("timing", {}).get("effective_fused_seconds"),
            "effective_fused_mode": rep.get("timing", {}).get("effective_fused_mode"),
            "dense_attention_seconds": rep.get("timing", {}).get("dense_attention_seconds"),
            "sdpa_seconds": (rep.get("timing", {}).get("sdpa") or {}).get("seconds") if isinstance(rep.get("timing", {}).get("sdpa"), dict) else None,
            "effective_compression_ratio": rep.get("memory", {}).get("effective_compression_ratio"),
            "effective_memory_saved_pct": rep.get("memory", {}).get("effective_memory_saved_pct"),
            "relative_error": q.get("relative_error"),
            "cosine_similarity": q.get("cosine_similarity"),
            "competitiveness_verdict": rep.get("competitiveness", {}).get("verdict"),
        })

    best = min(reports, key=lambda r: float(r.get("timing", {}).get("effective_fused_seconds") or r.get("timing", {}).get("experimental_fused_seconds") or 1e30))
    best["version"] = "0.10.7"
    best["page_size_tuning"] = rows
    best["execution"]["selected_page_size"] = int(best.get("config", {}).get("page_size", config.page_size))
    best["execution"]["tune_page_size"] = True
    best["timing"]["selected_page_size"] = int(best.get("config", {}).get("page_size", config.page_size))
    best["timing"]["page_size_tuning"] = rows
    best["interpretation"] = (
        "v0.10.7 benchmarks an experimental compressed decode-attention path over the compressed page layout with optional "
        "BLOCK_M, page-size tuning, and CUDA Graph replay diagnostics. It reads compressed pages without constructing full dense K/V. The competitiveness "
        "verdict is a microbenchmark diagnostic and is not a production acceleration claim."
    )
    return _add_competitiveness_summary(best)

def fused_decode_markdown_report(report: Dict[str, Any]) -> str:
    mem = report["memory"]
    timing = report["timing"]
    quality = report["quality"]
    execution = report.get("execution", {})
    lines = [
        "# tiny-turboquant v0.10.7 fused decode-attention benchmark",
        "",
        "## Execution",
        f"- Mode: {execution.get('mode')}",
        f"- Triton attempted: {execution.get('triton_attempted')}",
        f"- Uses compressed pages directly: {execution.get('uses_compressed_pages_directly')}",
        f"- Constructs full dense K/V: {execution.get('constructs_full_dense_kv')}",
        f"- Reason: {execution.get('reason')}",
        "",
        "## Memory",
        f"- FP16 K/V bytes: {mem['fp16_kv_bytes']}",
        f"- Actual compressed-layout bytes: {mem['actual_total_bytes']}",
        f"- Effective compression: {mem['effective_compression_ratio']:.4f}x",
        f"- Effective memory saved: {mem['effective_memory_saved_pct']:.2f}%",
        "",
        "## Timing",
        f"- Dense attention seconds: {timing['dense_attention_seconds']:.6f}",
        f"- Compressed page reference seconds: {timing['compressed_page_reference_seconds']:.6f}",
        f"- Experimental fused seconds: {timing['experimental_fused_seconds']:.6f}",
        f"- CUDA Graph replay: {timing.get('cuda_graph')}",
        f"- Selected BLOCK_M: {timing.get('selected_kernel_block_m')}",
        f"- Selected num_warps: {timing.get('selected_kernel_num_warps')}",
        f"- Selected page size: {timing.get('selected_page_size')}",
        f"- SDPA: {timing.get('sdpa')}",
        "",
        "## Competitiveness",
        f"- Summary: {report.get('competitiveness')}",
        "",
        "## Quality",
        f"- Fused vs dense: {quality['fused_vs_dense']}",
        f"- Reference vs dense: {quality['reference_vs_dense']}",
        f"- Fused vs reference: {quality['fused_vs_reference']}",
        f"- Warning: {quality['quality_warning']}",
        "",
        "## Boundary",
        "This is an experimental research benchmark. It does not claim production inference acceleration.",
    ]
    return "\n".join(lines) + "\n"


def save_fused_decode_json(report: Dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")


def save_fused_decode_markdown(report: Dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(fused_decode_markdown_report(report), encoding="utf-8")



@dataclass
class LongContextCompareConfig:
    """Compare compressed decode-attention layouts across longer contexts.

    v0.10.7 is a diagnostic layer over ``run_fused_decode_benchmark``.  It is
    meant to answer one question: does a more compressed residual layout become
    worthwhile as context length grows?  The benchmark is still synthetic and
    microbenchmark-oriented; it is not a production serving benchmark.
    """

    batch_size: int = 1
    heads: int = 8
    query_len: int = 1
    seq_lens: Tuple[int, ...] = (2048, 4096, 8192, 16384)
    head_dim: int = 64
    page_size: int = 256
    presets: Tuple[str, ...] = ("safe-layout", "residual-balanced", "residual-aggressive")
    device: str = "auto"
    dtype: str = "auto"
    warmup: int = 1
    repeats: int = 3
    seed: int = 123
    prefer_triton: bool = True
    kernel_block_m: int = 64
    kernel_num_warps: int = 4
    tune_kernel: bool = True
    tune_block_m_values: Tuple[int, ...] = (8, 16, 32, 64, 128)
    tune_num_warps: bool = False
    tune_num_warps_values: Tuple[int, ...] = (1, 2, 4, 8)
    tune_page_size: bool = False
    tune_page_size_values: Tuple[int, ...] = (64, 128, 256)
    cuda_graph: bool = True
    graph_replays: int = 100
    competitiveness_target: str = "sdpa"


def _row_from_fused_report(seq_len: int, preset: str, report: Dict[str, Any]) -> Dict[str, Any]:
    timing = report.get("timing", {})
    quality = report.get("quality", {}).get("fused_vs_dense", {})
    mem = report.get("memory", {})
    execution = report.get("execution", {})
    comp = report.get("competitiveness", {})
    sdpa = timing.get("sdpa") if isinstance(timing.get("sdpa"), dict) else {}
    return {
        "seq_len": int(seq_len),
        "preset": preset,
        "mode": execution.get("mode"),
        "quantization_mode": execution.get("quantization_mode", report.get("config", {}).get("quantization_mode")),
        "key_bits": report.get("config", {}).get("key_bits"),
        "value_bits": report.get("config", {}).get("value_bits"),
        "page_size": report.get("config", {}).get("page_size"),
        "selected_kernel_block_m": timing.get("selected_kernel_block_m"),
        "selected_kernel_num_warps": timing.get("selected_kernel_num_warps"),
        "uses_compressed_pages_directly": execution.get("uses_compressed_pages_directly"),
        "constructs_full_dense_kv": execution.get("constructs_full_dense_kv"),
        "fp16_kv_bytes": mem.get("fp16_kv_bytes"),
        "actual_total_bytes": mem.get("actual_total_bytes"),
        "effective_compression_ratio": mem.get("effective_compression_ratio"),
        "effective_memory_saved_pct": mem.get("effective_memory_saved_pct"),
        "dense_attention_seconds": timing.get("dense_attention_seconds"),
        "sdpa_seconds": sdpa.get("seconds") if isinstance(sdpa, dict) else None,
        "compressed_page_reference_seconds": timing.get("compressed_page_reference_seconds"),
        "normal_fused_seconds": timing.get("experimental_fused_seconds"),
        "graph_fused_seconds": timing.get("graph_fused_seconds"),
        "effective_fused_seconds": timing.get("effective_fused_seconds"),
        "effective_fused_mode": timing.get("effective_fused_mode"),
        "effective_over_sdpa": comp.get("effective_over_sdpa"),
        "effective_over_dense": comp.get("effective_over_dense"),
        "reference_over_effective": comp.get("reference_over_effective"),
        "faster_than_sdpa": comp.get("faster_than_sdpa"),
        "faster_than_dense_manual": comp.get("faster_than_dense_manual"),
        "faster_than_compressed_page_reference": comp.get("faster_than_compressed_page_reference"),
        "verdict": comp.get("verdict"),
        "relative_error": quality.get("relative_error"),
        "cosine_similarity": quality.get("cosine_similarity"),
        "quality_warning": report.get("quality", {}).get("quality_warning"),
    }


def _is_quality_acceptable(row: Dict[str, Any], *, max_relative_error: float = 0.08, min_cosine: float = 0.995) -> bool:
    rel = row.get("relative_error")
    cos = row.get("cosine_similarity")
    if rel is None or cos is None:
        return False
    return float(rel) <= max_relative_error and float(cos) >= min_cosine


def _summarize_long_context_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_seq: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        by_seq.setdefault(int(row["seq_len"]), []).append(row)

    summary_rows: List[Dict[str, Any]] = []
    for seq_len in sorted(by_seq):
        group = by_seq[seq_len]
        valid_speed = [r for r in group if r.get("effective_fused_seconds") is not None]
        quality_ok = [r for r in valid_speed if _is_quality_acceptable(r)]
        fastest = min(valid_speed, key=lambda r: float(r["effective_fused_seconds"])) if valid_speed else None
        fastest_quality_ok = min(quality_ok, key=lambda r: float(r["effective_fused_seconds"])) if quality_ok else None
        best_memory = max(group, key=lambda r: float(r.get("effective_memory_saved_pct") or -1.0)) if group else None
        best_quality = max(group, key=lambda r: float(r.get("cosine_similarity") or -1.0)) if group else None
        recommended = fastest_quality_ok or best_quality or fastest
        summary_rows.append({
            "seq_len": seq_len,
            "fastest_preset": fastest.get("preset") if fastest else None,
            "fastest_effective_seconds": fastest.get("effective_fused_seconds") if fastest else None,
            "fastest_quality_ok_preset": fastest_quality_ok.get("preset") if fastest_quality_ok else None,
            "fastest_quality_ok_seconds": fastest_quality_ok.get("effective_fused_seconds") if fastest_quality_ok else None,
            "best_memory_preset": best_memory.get("preset") if best_memory else None,
            "best_memory_saved_pct": best_memory.get("effective_memory_saved_pct") if best_memory else None,
            "best_quality_preset": best_quality.get("preset") if best_quality else None,
            "best_quality_cosine": best_quality.get("cosine_similarity") if best_quality else None,
            "recommended_preset": recommended.get("preset") if recommended else None,
            "recommended_reason": (
                "fastest quality-acceptable layout" if fastest_quality_ok else
                "highest-quality fallback because no fast layout met the quality gate" if best_quality else
                "fastest available layout"
            ),
        })

    # Global verdict is deliberately conservative.
    any_sdpa = any(bool(r.get("faster_than_sdpa")) for r in rows)
    any_dense = any(bool(r.get("faster_than_dense_manual")) for r in rows)
    any_ref = any(bool(r.get("faster_than_compressed_page_reference")) for r in rows)
    return {
        "quality_gate": {
            "max_relative_error": 0.08,
            "min_cosine_similarity": 0.995,
            "note": "Used only to rank layouts for research; not a production quality guarantee.",
        },
        "by_seq_len": summary_rows,
        "global_verdict": (
            "competitive-with-sdpa-in-at-least-one-case" if any_sdpa else
            "competitive-with-dense-manual-in-at-least-one-case" if any_dense else
            "faster-than-python-reference-only" if any_ref else
            "not-competitive-yet"
        ),
        "boundary": "This is a synthetic long-context microbenchmark diagnostic, not production inference acceleration.",
    }


def run_long_context_comparison(config: LongContextCompareConfig) -> Dict[str, Any]:
    """Run safe vs residual compressed decode benchmarks across context lengths."""
    rows: List[Dict[str, Any]] = []
    reports: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for seq_len in [int(x) for x in config.seq_lens]:
        for preset in [str(p) for p in config.presets]:
            try:
                cfg = FusedDecodeBenchConfig(
                    batch_size=config.batch_size,
                    heads=config.heads,
                    query_len=config.query_len,
                    seq_len=seq_len,
                    head_dim=config.head_dim,
                    page_size=config.page_size,
                    preset=preset,
                    device=config.device,
                    dtype=config.dtype,
                    warmup=config.warmup,
                    repeats=config.repeats,
                    seed=config.seed,
                    prefer_triton=config.prefer_triton,
                    kernel_block_m=config.kernel_block_m,
                    kernel_num_warps=config.kernel_num_warps,
                    tune_kernel=config.tune_kernel,
                    tune_block_m_values=tuple(int(x) for x in config.tune_block_m_values),
                    tune_num_warps=config.tune_num_warps,
                    tune_num_warps_values=tuple(int(x) for x in config.tune_num_warps_values),
                    tune_page_size=config.tune_page_size,
                    tune_page_size_values=tuple(int(x) for x in config.tune_page_size_values),
                    cuda_graph=config.cuda_graph,
                    graph_replays=config.graph_replays,
                    competitiveness_target=config.competitiveness_target,
                )
                rep = run_fused_decode_benchmark(cfg)
                reports.append(rep)
                rows.append(_row_from_fused_report(seq_len, preset, rep))
            except Exception as exc:
                msg = f"seq_len={seq_len}, preset={preset} failed: {exc!r}"
                warnings.append(msg)
                rows.append({
                    "seq_len": seq_len,
                    "preset": preset,
                    "error": repr(exc),
                    "verdict": "failed",
                })

    return {
        "version": "long-context-comparison-v0.10.7",
        "config": asdict(config),
        "warnings": warnings,
        "rows": rows,
        "summary": _summarize_long_context_rows([r for r in rows if "error" not in r]),
        "interpretation": (
            "v0.10.7 compares safe-layout and residual layouts across longer context lengths to find where "
            "extra residual correction overhead becomes worthwhile. It reuses the experimental fused decode benchmark, "
            "including CUDA Graph replay when enabled. This is not a production serving benchmark and does not claim production inference acceleration."
        ),
    }


def long_context_markdown_report(report: Dict[str, Any]) -> str:
    lines = [
        "# tiny-turboquant v0.10.7 long-context comparison",
        "",
        "## Boundary",
        "This is a synthetic microbenchmark diagnostic. It does not claim production inference acceleration.",
        "",
        "## Summary",
        f"- Global verdict: {report.get('summary', {}).get('global_verdict')}",
        f"- Quality gate: {report.get('summary', {}).get('quality_gate')}",
        "",
        "## Per-context recommendation",
    ]
    for row in report.get("summary", {}).get("by_seq_len", []):
        lines.append(
            f"- seq_len={row.get('seq_len')}: recommended={row.get('recommended_preset')} "
            f"({row.get('recommended_reason')}); fastest={row.get('fastest_preset')}; best_memory={row.get('best_memory_preset')}"
        )
    lines += ["", "## Rows", ""]
    for row in report.get("rows", []):
        if "error" in row:
            lines.append(f"- seq_len={row.get('seq_len')}, preset={row.get('preset')}: ERROR {row.get('error')}")
            continue
        lines.append(
            f"- seq_len={row.get('seq_len')}, preset={row.get('preset')}, "
            f"effective={row.get('effective_fused_seconds')}, sdpa_gap={row.get('effective_over_sdpa')}, "
            f"memory_saved={row.get('effective_memory_saved_pct')}, rel_err={row.get('relative_error')}, "
            f"cos={row.get('cosine_similarity')}, verdict={row.get('verdict')}"
        )
    return "\n".join(lines) + "\n"


def save_long_context_json(report: Dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")


def save_long_context_markdown(report: Dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(long_context_markdown_report(report), encoding="utf-8")



@dataclass
class SplitKCompareConfig:
    """Diagnostic split-K / sequence-parallel planning benchmark.

    v0.10.7 does not replace the single-pass Triton kernel with a production
    FlashDecoding implementation.  Instead it measures the current best fused
    safe-layout path and estimates how much sequence-parallel split-K would need
    to improve to beat SDPA.  This keeps the report honest while giving concrete
    targets for the next kernel rewrite.
    """

    batch_size: int = 1
    heads: int = 8
    query_len: int = 1
    seq_lens: Tuple[int, ...] = (8192, 16384, 32768)
    head_dim: int = 64
    page_size: int = 256
    preset: str = "safe-layout"
    device: str = "auto"
    dtype: str = "auto"
    warmup: int = 1
    repeats: int = 3
    seed: int = 123
    prefer_triton: bool = True
    kernel_block_m: int = 64
    kernel_num_warps: int = 4
    tune_kernel: bool = True
    tune_block_m_values: Tuple[int, ...] = (8, 16, 32, 64, 128)
    tune_num_warps: bool = True
    tune_num_warps_values: Tuple[int, ...] = (1, 2, 4, 8)
    cuda_graph: bool = True
    graph_replays: int = 100
    split_k_slabs: Tuple[int, ...] = (2, 4, 8, 16)
    reduce_overhead_fraction: float = 0.12
    measure_split_k: bool = True
    competitiveness_target: str = "sdpa"


def _split_k_projection_rows(base_row: Dict[str, Any], slabs: Tuple[int, ...], reduce_overhead_fraction: float) -> List[Dict[str, Any]]:
    """Return honest split-K projection rows from a measured single-pass row.

    This is not a production kernel measurement.  It estimates the target time
    if the page loop were parallelized into N slabs plus a small reduction pass.
    The purpose is to determine how much algorithmic parallelism is needed
    before implementing a true two-kernel/one-kernel split-K path.
    """
    eff = base_row.get("effective_fused_seconds")
    sdpa = base_row.get("sdpa_seconds")
    dense = base_row.get("dense_attention_seconds")
    if eff is None:
        return []
    eff = float(eff)
    sdpa_f = float(sdpa) if sdpa is not None else None
    dense_f = float(dense) if dense is not None else None
    rows: List[Dict[str, Any]] = []
    for n in slabs:
        n = max(1, int(n))
        reduction = eff * max(0.0, float(reduce_overhead_fraction))
        projected = eff / n + reduction
        rows.append({
            "split_k_slabs": n,
            "projected_seconds": projected,
            "projected_speedup_vs_single_pass": eff / projected if projected > 0 else None,
            "projected_over_sdpa": projected / sdpa_f if sdpa_f and sdpa_f > 0 else None,
            "projected_over_dense": projected / dense_f if dense_f and dense_f > 0 else None,
            "projected_faster_than_sdpa": bool(sdpa_f and projected <= sdpa_f),
            "projection_note": "projection only: assumes page/slab parallelism plus reduction overhead; not a measured split-K Triton kernel",
        })
    return rows



def _split_page_ranges(page_count: int, slabs: int) -> List[Tuple[int, int]]:
    slabs = max(1, min(int(slabs), int(page_count)))
    base = page_count // slabs
    rem = page_count % slabs
    ranges: List[Tuple[int, int]] = []
    start = 0
    for i in range(slabs):
        width = base + (1 if i < rem else 0)
        end = start + width
        if end > start:
            ranges.append((start, end))
        start = end
    return ranges


def split_k_attention_reference(query: torch.Tensor, page_table: CompressedKVPageTable, *, split_k_slabs: int = 4) -> torch.Tensor:
    """Measured two-stage split-K reference over compressed pages.

    Stage 1 computes per-slab online-softmax statistics from dequantized compressed
    pages: local max, local denominator, and local weighted-V accumulator.
    Stage 2 combines those partial statistics exactly with the standard
    log-sum-exp correction.  This is a measured correctness prototype, not the
    production Triton split-K kernel.
    """
    if query.ndim != 4 or query.shape[2] != 1:
        raise ValueError("split_k_attention_reference supports decode query_len=1 only")
    q = page_table.rotate_query(query).float()
    scale = float(page_table.head_dim ** -0.5)
    partials: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for lo, hi in _split_page_ranges(page_table.page_count, int(split_k_slabs)):
        ks: List[torch.Tensor] = []
        vs: List[torch.Tensor] = []
        for page in page_table.pages[lo:hi]:
            ks.append(page_table.dequant_key_page(page, rotated=True).float())
            vs.append(page_table.dequant_value_page(page).float())
        k_slab = torch.cat(ks, dim=2)
        v_slab = torch.cat(vs, dim=2)
        scores = torch.matmul(q, k_slab.transpose(-1, -2)) * scale
        local_m = scores.max(dim=-1, keepdim=True).values
        probs = torch.exp(scores - local_m)
        local_l = probs.sum(dim=-1, keepdim=True).clamp_min(1e-20)
        local_acc = torch.matmul(probs, v_slab)
        partials.append((local_m, local_l, local_acc))
    global_m = torch.stack([p[0] for p in partials], dim=0).max(dim=0).values
    denom = None
    acc = None
    for local_m, local_l, local_acc in partials:
        weight = torch.exp(local_m - global_m)
        term_l = weight * local_l
        term_acc = weight * local_acc
        denom = term_l if denom is None else denom + term_l
        acc = term_acc if acc is None else acc + term_acc
    out = acc / denom.clamp_min(1e-20)
    return out.to(query.dtype)


def _measure_split_k_rows(
    config: FusedDecodeBenchConfig,
    dense_out: torch.Tensor,
    q: torch.Tensor,
    table: CompressedKVPageTable,
    base_row: Dict[str, Any],
    slabs: Tuple[int, ...],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    device = str(q.device).split(":")[0]
    sdpa = base_row.get("sdpa_seconds")
    dense = base_row.get("dense_attention_seconds")
    sdpa_f = float(sdpa) if sdpa is not None else None
    dense_f = float(dense) if dense is not None else None
    single = base_row.get("effective_fused_seconds")
    single_f = float(single) if single is not None else None
    for n in slabs:
        n = max(1, int(n))
        try:
            out, seconds = _time_call(
                lambda n=n: split_k_attention_reference(q, table, split_k_slabs=n),
                device=device,
                warmup=config.warmup,
                repeats=config.repeats,
            )
            quality = attention_similarity(dense_out, out)
            rows.append({
                "split_k_slabs": n,
                "measured_seconds": float(seconds),
                "measured_speedup_vs_single_pass": (single_f / float(seconds)) if single_f and seconds > 0 else None,
                "measured_over_sdpa": (float(seconds) / sdpa_f) if sdpa_f and sdpa_f > 0 else None,
                "measured_over_dense": (float(seconds) / dense_f) if dense_f and dense_f > 0 else None,
                "measured_faster_than_sdpa": bool(sdpa_f and float(seconds) <= sdpa_f),
                "relative_error": quality.get("relative_error"),
                "cosine_similarity": quality.get("cosine_similarity"),
                "mode": "torch-two-stage-split-k-reference",
                "measurement_note": "measured two-stage split-K reference using compressed pages; not a production Triton split-K kernel",
            })
        except Exception as exc:
            rows.append({
                "split_k_slabs": n,
                "error": repr(exc),
                "mode": "torch-two-stage-split-k-reference",
                "measurement_note": "measured split-K reference failed",
            })
    return rows



def _measure_triton_stage1_rows(
    config: FusedDecodeBenchConfig,
    dense_out: torch.Tensor,
    q: torch.Tensor,
    table: CompressedKVPageTable,
    base_row: Dict[str, Any],
    slabs: Tuple[int, ...],
) -> List[Dict[str, Any]]:
    """Measure the real Triton Stage-1 split-K partial kernel plus Torch reduction."""
    rows: List[Dict[str, Any]] = []
    device = str(q.device).split(":")[0]
    sdpa = base_row.get("sdpa_seconds")
    dense = base_row.get("dense_attention_seconds")
    sdpa_f = float(sdpa) if sdpa is not None else None
    dense_f = float(dense) if dense is not None else None
    single = base_row.get("effective_fused_seconds")
    single_f = float(single) if single is not None else None
    for n in slabs:
        n = max(1, int(n))
        try:
            partials, stage1_seconds = _time_call(
                lambda n=n: triton_split_k_stage1_partials(
                    q,
                    table,
                    split_k_slabs=n,
                    kernel_block_m=config.kernel_block_m,
                    kernel_num_warps=config.kernel_num_warps,
                )[:3],
                device=device,
                warmup=config.warmup,
                repeats=config.repeats,
            )
            partial_m, partial_l, partial_acc = partials
            reduced_out, reduce_seconds = _time_call(
                lambda: reduce_split_k_partials(partial_m, partial_l, partial_acc, output_dtype=q.dtype),
                device=device,
                warmup=config.warmup,
                repeats=config.repeats,
            )
            total_seconds = float(stage1_seconds) + float(reduce_seconds)
            quality = attention_similarity(dense_out, reduced_out)
            rows.append({
                "split_k_slabs": n,
                "stage1_seconds": float(stage1_seconds),
                "torch_reduce_seconds": float(reduce_seconds),
                "stage1_plus_torch_reduce_seconds": total_seconds,
                "stage1_speedup_vs_single_pass": (single_f / float(stage1_seconds)) if single_f and stage1_seconds > 0 else None,
                "stage1_over_sdpa": (float(stage1_seconds) / sdpa_f) if sdpa_f and sdpa_f > 0 else None,
                "stage1_plus_reduce_over_sdpa": (total_seconds / sdpa_f) if sdpa_f and sdpa_f > 0 else None,
                "stage1_over_dense": (float(stage1_seconds) / dense_f) if dense_f and dense_f > 0 else None,
                "stage1_faster_than_sdpa": bool(sdpa_f and float(stage1_seconds) <= sdpa_f),
                "stage1_plus_reduce_faster_than_sdpa": bool(sdpa_f and total_seconds <= sdpa_f),
                "relative_error": quality.get("relative_error"),
                "cosine_similarity": quality.get("cosine_similarity"),
                "mode": "triton-split-k-stage1-plus-torch-reduce-reference",
                "measurement_note": "Stage 1 is a real Triton partial-statistics kernel; Stage 2 reduction is still a Torch reference path.",
            })
        except Exception as exc:
            rows.append({
                "split_k_slabs": n,
                "error": repr(exc),
                "mode": "triton-split-k-stage1-partials",
                "measurement_note": "Triton Stage-1 split-K partial kernel failed or was unsupported",
            })
    return rows

def _measure_triton_full_split_k_rows(
    config: FusedDecodeBenchConfig,
    dense_out: torch.Tensor,
    q: torch.Tensor,
    table: CompressedKVPageTable,
    base_row: Dict[str, Any],
    slabs: Tuple[int, ...],
) -> List[Dict[str, Any]]:
    """Measure real Triton Stage-1 plus real Triton Stage-2 reduction."""
    rows: List[Dict[str, Any]] = []
    device = str(q.device).split(":")[0]
    sdpa = base_row.get("sdpa_seconds")
    dense = base_row.get("dense_attention_seconds")
    sdpa_f = float(sdpa) if sdpa is not None else None
    dense_f = float(dense) if dense is not None else None
    single = base_row.get("effective_fused_seconds")
    single_f = float(single) if single is not None else None
    for n in slabs:
        n = max(1, int(n))
        try:
            # Time Stage 1 and Stage 2 separately to expose where overhead lives.
            partials, stage1_seconds = _time_call(
                lambda n=n: triton_split_k_stage1_partials(
                    q,
                    table,
                    split_k_slabs=n,
                    kernel_block_m=config.kernel_block_m,
                    kernel_num_warps=config.kernel_num_warps,
                )[:3],
                device=device,
                warmup=config.warmup,
                repeats=config.repeats,
            )
            partial_m, partial_l, partial_acc = partials
            reduced_out, stage2_seconds = _time_call(
                lambda: triton_split_k_stage2_reduce(
                    partial_m,
                    partial_l,
                    partial_acc,
                    output_dtype=q.dtype,
                    kernel_num_warps=1,
                )[0],
                device=device,
                warmup=config.warmup,
                repeats=config.repeats,
            )
            total_seconds = float(stage1_seconds) + float(stage2_seconds)
            quality = attention_similarity(dense_out, reduced_out)
            rows.append({
                "split_k_slabs": n,
                "stage1_seconds": float(stage1_seconds),
                "stage2_seconds": float(stage2_seconds),
                "stage1_plus_stage2_seconds": total_seconds,
                "full_speedup_vs_single_pass": (single_f / total_seconds) if single_f and total_seconds > 0 else None,
                "full_over_sdpa": (total_seconds / sdpa_f) if sdpa_f and sdpa_f > 0 else None,
                "full_over_dense": (total_seconds / dense_f) if dense_f and dense_f > 0 else None,
                "full_faster_than_sdpa": bool(sdpa_f and total_seconds <= sdpa_f),
                "relative_error": quality.get("relative_error"),
                "cosine_similarity": quality.get("cosine_similarity"),
                "mode": "triton-two-stage-split-k",
                "measurement_note": "Stage 1 and Stage 2 are real Triton kernels; this is still a research microbenchmark, not production FlashDecoding.",
            })
        except Exception as exc:
            rows.append({
                "split_k_slabs": n,
                "error": repr(exc),
                "mode": "triton-two-stage-split-k",
                "measurement_note": "Full Triton two-stage split-K failed or was unsupported",
            })
    return rows


def _split_k_base_benchmark(config: FusedDecodeBenchConfig) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor, CompressedKVPageTable, torch.Tensor]:
    """Run the fused benchmark core and return internals for measured split-K.

    This intentionally duplicates the deterministic synthetic setup used by
    run_fused_decode_benchmark so the measured split-K reference is compared
    against the same dense baseline and compressed page table.
    """
    requested_device = config.device
    requested_dtype = config.dtype
    device = _resolve_device(requested_device)
    dtype = _resolve_dtype(requested_dtype, device)
    q, k, v = _build_random_qkv(config, device, dtype)
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
    dense_out, _ = _time_call(lambda: dense_attention(q, k, v), device=device, warmup=config.warmup, repeats=config.repeats)
    rep = run_fused_decode_benchmark(config)
    return rep, dense_out, q, table, k

def run_split_k_comparison(config: SplitKCompareConfig) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for seq_len in [int(x) for x in config.seq_lens]:
        try:
            cfg = FusedDecodeBenchConfig(
                batch_size=config.batch_size,
                heads=config.heads,
                query_len=config.query_len,
                seq_len=seq_len,
                head_dim=config.head_dim,
                page_size=config.page_size,
                preset=config.preset,
                device=config.device,
                dtype=config.dtype,
                warmup=config.warmup,
                repeats=config.repeats,
                seed=config.seed,
                prefer_triton=config.prefer_triton,
                kernel_block_m=config.kernel_block_m,
                kernel_num_warps=config.kernel_num_warps,
                tune_kernel=config.tune_kernel,
                tune_block_m_values=tuple(int(x) for x in config.tune_block_m_values),
                tune_num_warps=config.tune_num_warps,
                tune_num_warps_values=tuple(int(x) for x in config.tune_num_warps_values),
                cuda_graph=config.cuda_graph,
                graph_replays=config.graph_replays,
                competitiveness_target=config.competitiveness_target,
            )
            if config.measure_split_k:
                rep, dense_out, q, table, _k = _split_k_base_benchmark(cfg)
            else:
                rep = run_fused_decode_benchmark(cfg)
                dense_out = q = table = None  # type: ignore[assignment]
            base = _row_from_fused_report(seq_len, config.preset, rep)
            base["split_k_projection"] = _split_k_projection_rows(base, tuple(int(x) for x in config.split_k_slabs), config.reduce_overhead_fraction)
            best_proj = min(base["split_k_projection"], key=lambda r: float(r["projected_seconds"])) if base["split_k_projection"] else None
            base["best_projected_split_k"] = best_proj
            if config.measure_split_k and dense_out is not None and q is not None and table is not None:
                measured_rows = _measure_split_k_rows(cfg, dense_out, q, table, base, tuple(int(x) for x in config.split_k_slabs))
                base["measured_split_k"] = measured_rows
                valid_measured = [r for r in measured_rows if "measured_seconds" in r]
                base["best_measured_split_k"] = min(valid_measured, key=lambda r: float(r["measured_seconds"])) if valid_measured else None

                stage1_rows = _measure_triton_stage1_rows(cfg, dense_out, q, table, base, tuple(int(x) for x in config.split_k_slabs))
                base["triton_stage1_split_k"] = stage1_rows
                valid_stage1 = [r for r in stage1_rows if "stage1_seconds" in r]
                base["best_triton_stage1_split_k"] = min(valid_stage1, key=lambda r: float(r["stage1_seconds"])) if valid_stage1 else None

                full_rows = _measure_triton_full_split_k_rows(cfg, dense_out, q, table, base, tuple(int(x) for x in config.split_k_slabs))
                base["triton_full_split_k"] = full_rows
                valid_full = [r for r in full_rows if "stage1_plus_stage2_seconds" in r]
                base["best_triton_full_split_k"] = min(valid_full, key=lambda r: float(r["stage1_plus_stage2_seconds"])) if valid_full else None
            else:
                base["measured_split_k"] = []
                base["best_measured_split_k"] = None
                base["triton_stage1_split_k"] = []
                base["best_triton_stage1_split_k"] = None
                base["triton_full_split_k"] = []
                base["best_triton_full_split_k"] = None
            base["measured_single_pass_verdict"] = rep.get("competitiveness", {}).get("verdict")
            rows.append(base)
        except Exception as exc:
            warnings.append(f"seq_len={seq_len} failed: {exc!r}")
            rows.append({"seq_len": seq_len, "preset": config.preset, "error": repr(exc)})

    any_projected_sdpa = any(
        bool(p.get("projected_faster_than_sdpa"))
        for r in rows if "error" not in r
        for p in r.get("split_k_projection", [])
    )
    any_measured_sdpa = any(
        bool(p.get("measured_faster_than_sdpa"))
        for r in rows if "error" not in r
        for p in r.get("measured_split_k", [])
    )
    measured_rows = [
        p for r in rows if "error" not in r
        for p in r.get("measured_split_k", [])
        if "measured_seconds" in p
    ]
    any_stage1_sdpa = any(
        bool(p.get("stage1_faster_than_sdpa"))
        for r in rows if "error" not in r
        for p in r.get("triton_stage1_split_k", [])
    )
    stage1_rows = [
        p for r in rows if "error" not in r
        for p in r.get("triton_stage1_split_k", [])
        if "stage1_seconds" in p
    ]
    any_full_sdpa = any(
        bool(p.get("full_faster_than_sdpa"))
        for r in rows if "error" not in r
        for p in r.get("triton_full_split_k", [])
    )
    full_rows = [
        p for r in rows if "error" not in r
        for p in r.get("triton_full_split_k", [])
        if "stage1_plus_stage2_seconds" in p
    ]
    return {
        "version": "split-k-comparison-v0.10.7",
        "config": asdict(config),
        "warnings": warnings,
        "rows": rows,
        "summary": {
            "global_projection_verdict": "split-k-projection-can-beat-sdpa" if any_projected_sdpa else "split-k-projection-still-needs-more-work",
            "global_measured_verdict": "measured-split-k-beats-sdpa" if any_measured_sdpa else "measured-split-k-not-competitive-yet" if measured_rows else "measured-split-k-not-run",
            "global_triton_stage1_verdict": "triton-stage1-beats-sdpa-in-at-least-one-case" if any_stage1_sdpa else "triton-stage1-measured",
            "global_triton_full_split_k_verdict": "triton-full-split-k-beats-sdpa-in-at-least-one-case" if any_full_sdpa else "triton-full-split-k-not-competitive-yet" if full_rows else "triton-full-split-k-unavailable",
            "measured_kernel_verdict": "v0.10.7 adds a real Triton split-K Stage-2 reducer and reports full Stage-1 + Stage-2 measured split-K rows",
            "boundary": "This is a split-K measured research diagnostic, not production inference acceleration.",
        },
        "interpretation": (
            "v0.10.7 measures real Triton Stage-1 partial statistics plus a real Triton Stage-2 reducer over compressed pages. "
            "This completes the measured two-stage split-K research path, but it is still not a production FlashDecoding serving kernel."
        ),
    }


def split_k_markdown_report(report: Dict[str, Any]) -> str:
    lines = [
        "# tiny-turboquant v0.10.7 split-K planning diagnostic",
        "",
        "This report is a planning diagnostic. Split-K rows are projections, not measured production kernels.",
        "",
        "## Summary",
        f"- {report.get('summary')}",
        "",
        "## Rows",
    ]
    for row in report.get("rows", []):
        lines.append(f"- seq_len={row.get('seq_len')}, preset={row.get('preset')}, measured_effective={row.get('effective_fused_seconds')}, sdpa={row.get('sdpa_seconds')}, verdict={row.get('measured_single_pass_verdict')}")
        for p in row.get("split_k_projection", []):
            lines.append(f"  - projected slabs={p.get('split_k_slabs')}: seconds={p.get('projected_seconds')}, over_sdpa={p.get('projected_over_sdpa')}, faster_than_sdpa={p.get('projected_faster_than_sdpa')}")
        for p in row.get("measured_split_k", []):
            if "error" in p:
                lines.append(f"  - torch measured slabs={p.get('split_k_slabs')}: ERROR {p.get('error')}")
            else:
                lines.append(f"  - torch measured slabs={p.get('split_k_slabs')}: seconds={p.get('measured_seconds')}, over_sdpa={p.get('measured_over_sdpa')}, faster_than_sdpa={p.get('measured_faster_than_sdpa')}, cos={p.get('cosine_similarity')}")
        for p in row.get("triton_stage1_split_k", []):
            if "error" in p:
                lines.append(f"  - Triton Stage-1 slabs={p.get('split_k_slabs')}: ERROR {p.get('error')}")
            else:
                lines.append(f"  - Triton Stage-1 slabs={p.get('split_k_slabs')}: stage1={p.get('stage1_seconds')}, stage1+torch_reduce={p.get('stage1_plus_torch_reduce_seconds')}, over_sdpa={p.get('stage1_plus_reduce_over_sdpa')}, cos={p.get('cosine_similarity')}")
        for p in row.get("triton_full_split_k", []):
            if "error" in p:
                lines.append(f"  - Triton full split-K slabs={p.get('split_k_slabs')}: ERROR {p.get('error')}")
            else:
                lines.append(f"  - Triton full split-K slabs={p.get('split_k_slabs')}: total={p.get('stage1_plus_stage2_seconds')}, over_sdpa={p.get('full_over_sdpa')}, faster_than_sdpa={p.get('full_faster_than_sdpa')}, cos={p.get('cosine_similarity')}")
    lines.extend([
        "",
        "## Boundary",
        "This is not a production acceleration claim.",
    ])
    return "\n".join(lines) + "\n"


def save_split_k_json(report: Dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")


def save_split_k_markdown(report: Dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(split_k_markdown_report(report), encoding="utf-8")
