"""
Demo 4 — TurboQuant-style KV-cache compression using a NumPy reference path.

Story for the audience:
  "This demonstrates the memory/quality trade-off of a TurboQuant-style
   compressed KV cache. The NumPy reference path is intentionally not optimized
   for latency. Production speed requires a torch/Triton/CUDA fused attention
   path that consumes packed cache pages directly."

This demo runs without torch / transformers — it simulates a multi-layer
attention decode loop using realistic K/V distributions (low-rank trend +
Gaussian noise, the same shape real models produce). The TurboQuantKVCache class below is a NumPy reference cache for demo purposes.
Use tiny_turboquant.TurboQuantKVCache for HuggingFace experiments.

Run:
    python -m demos.demo4_kv_cache
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from tiny_turboquant.numpy_reference import TurboQuantMSE


# ---------------------------------------------------------------------------
# 1. TurboQuant KV cache (numpy reference; mirrors HF DynamicCache contract)
# ---------------------------------------------------------------------------

@dataclass
class TurboQuantKVCache:
    """Drop-in replacement for transformers.DynamicCache.

    Stores K and V quantized to `bits` per coordinate per head. The
    quantizer is shared across layers (all heads have the same head_dim,
    so a single random rotation + codebook works), making setup O(1).
    """

    bits: int = 4
    head_dim: int | None = None   # set on first update()
    _key_idx:   list[np.ndarray] = field(default_factory=list)   # per layer
    _value_idx: list[np.ndarray] = field(default_factory=list)
    _quantizer: TurboQuantMSE | None = None

    def _ensure_quantizer(self, head_dim: int) -> None:
        if self._quantizer is None:
            self.head_dim = head_dim
            self._quantizer = TurboQuantMSE(d=head_dim, bits=self.bits, seed=0)

    def update(
        self,
        key_states:   np.ndarray,   # (B, H, S_new, D)
        value_states: np.ndarray,
        layer_idx:    int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Append new K/V slices and return the *full* (dequantized) cache."""
        B, H, S_new, D = key_states.shape
        self._ensure_quantizer(D)

        # Flatten (B, H, S_new, D) -> (N, D) for the quantizer.
        k_flat = key_states.reshape(-1, D)
        v_flat = value_states.reshape(-1, D)

        # TurboQuant assumes unit norm. Per-vector scaling is standard.
        k_norm = np.linalg.norm(k_flat, axis=1, keepdims=True) + 1e-12
        v_norm = np.linalg.norm(v_flat, axis=1, keepdims=True) + 1e-12
        k_idx = self._quantizer.quant(k_flat / k_norm)
        v_idx = self._quantizer.quant(v_flat / v_norm)

        # We pack the float norm alongside the indices. In production this
        # would be fp16 (2 bytes). Here we keep fp32 for clarity.
        k_pack = (k_idx, k_norm.reshape(B, H, S_new, 1))
        v_pack = (v_idx.reshape(B, H, S_new, -1), v_norm.reshape(B, H, S_new, 1))
        k_pack = (k_idx.reshape(B, H, S_new, -1), k_pack[1])

        while len(self._key_idx) <= layer_idx:
            self._key_idx.append(None)
            self._value_idx.append(None)

        if self._key_idx[layer_idx] is None:
            self._key_idx[layer_idx]   = k_pack
            self._value_idx[layer_idx] = v_pack
        else:
            old_k_idx, old_k_norm = self._key_idx[layer_idx]
            old_v_idx, old_v_norm = self._value_idx[layer_idx]
            self._key_idx[layer_idx] = (
                np.concatenate([old_k_idx,  k_pack[0]], axis=2),
                np.concatenate([old_k_norm, k_pack[1]], axis=2),
            )
            self._value_idx[layer_idx] = (
                np.concatenate([old_v_idx,  v_pack[0]], axis=2),
                np.concatenate([old_v_norm, v_pack[1]], axis=2),
            )

        return self._dequantize(layer_idx)

    def _dequantize(self, layer_idx: int) -> tuple[np.ndarray, np.ndarray]:
        k_idx, k_norm = self._key_idx[layer_idx]
        v_idx, v_norm = self._value_idx[layer_idx]
        B, H, S, _ = k_idx.shape
        D = self.head_dim
        K = self._quantizer.dequant(k_idx.reshape(-1, D)).reshape(B, H, S, D) * k_norm
        V = self._quantizer.dequant(v_idx.reshape(-1, D)).reshape(B, H, S, D) * v_norm
        return K, V

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self._key_idx) or self._key_idx[layer_idx] is None:
            return 0
        return int(self._key_idx[layer_idx][0].shape[2])

    def memory_bytes(self) -> int:
        """Approximate stored bytes (packed indices + per-vector fp16 norm)."""
        total = 0
        for k_pack, v_pack in zip(self._key_idx, self._value_idx):
            for idx, norm in (k_pack, v_pack):
                # bits/coord * S*D coords / 8
                total += (self.bits * idx.size) // 8
                total += norm.size * 2     # fp16
        return total


@dataclass
class FullPrecisionKVCache:
    """Reference fp16 cache (the realistic LLM baseline), same interface."""
    _keys:   list[np.ndarray] = field(default_factory=list)
    _values: list[np.ndarray] = field(default_factory=list)

    def update(self, k, v, layer_idx):
        k = k.astype(np.float16); v = v.astype(np.float16)
        while len(self._keys) <= layer_idx:
            self._keys.append(None)
            self._values.append(None)
        if self._keys[layer_idx] is None:
            self._keys[layer_idx]   = k.copy()
            self._values[layer_idx] = v.copy()
        else:
            self._keys[layer_idx]   = np.concatenate([self._keys[layer_idx],   k], axis=2)
            self._values[layer_idx] = np.concatenate([self._values[layer_idx], v], axis=2)
        return self._keys[layer_idx], self._values[layer_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return 0 if not self._keys or self._keys[layer_idx] is None \
               else int(self._keys[layer_idx].shape[2])

    def memory_bytes(self) -> int:
        return sum(k.nbytes + v.nbytes for k, v in zip(self._keys, self._values))


# ---------------------------------------------------------------------------
# 2. Synthetic K/V generator that looks like a real transformer
# ---------------------------------------------------------------------------

def make_realistic_kv(
    n_layers: int,
    n_heads: int,
    head_dim: int,
    seq_len: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (K, V) of shape (n_layers, 1, n_heads, seq_len, head_dim).

    Real attention K/V have:
      - smooth low-rank temporal structure (a few "topics" the head tracks)
      - per-channel scale variation (some channels are 'outliers')
      - bounded but heavy-tailed magnitudes
    We synthesise this so the demo is meaningful without a real model.
    """
    L, H, S, D = n_layers, n_heads, seq_len, head_dim
    rank = 8
    basis = rng.standard_normal((L, H, rank, D)) / np.sqrt(rank)
    coeffs = np.cumsum(rng.standard_normal((L, H, S, rank)) * 0.3, axis=2)
    smooth = np.einsum("lhsr,lhrd->lhsd", coeffs, basis)
    noise = 0.6 * rng.standard_normal((L, H, S, D))
    # Outlier channels — a few channels with 5x amplitude (typical in LLMs)
    outlier_mask = rng.random(D) < 0.05
    scale = np.where(outlier_mask, 5.0, 1.0)
    K = (smooth + noise) * scale
    V = (smooth + 0.5 * rng.standard_normal((L, H, S, D))) * scale
    return K[:, None], V[:, None]   # add batch dim -> (L, B=1, H, S, D)


def attention(Q, K, V):
    """Standard scaled dot-product attention. All shapes (B, H, S, D)."""
    d = Q.shape[-1]
    scores = np.einsum("bhsd,bhtd->bhst", Q, K) / np.sqrt(d)
    scores -= scores.max(-1, keepdims=True)
    p = np.exp(scores); p /= p.sum(-1, keepdims=True)
    return np.einsum("bhst,bhtd->bhsd", p, V)


# ---------------------------------------------------------------------------
# 3. Decode loop comparing full-precision vs TurboQuant cache
# ---------------------------------------------------------------------------

def run(bits: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    n_layers, n_heads, head_dim = 4, 8, 128
    prefill_len, decode_len = 512, 32                 # realistic prompt + decode

    seq_len = prefill_len + decode_len
    K_all, V_all = make_realistic_kv(n_layers, n_heads, head_dim, seq_len, rng)
    Q_query = rng.standard_normal((1, n_heads, 1, head_dim))   # last-token query

    full_cache = FullPrecisionKVCache()
    tq_cache   = TurboQuantKVCache(bits=bits)

    # Server-startup cost: build quantizer once. Excluded from timing because
    # in production this happens at model load, not per request.
    tq_cache._ensure_quantizer(head_dim)

    # --- Prefill: process all prompt tokens at once (one quantize call/layer) ---
    t_prefill_full = 0.0
    t_prefill_tq   = 0.0
    for layer in range(n_layers):
        k_pre = K_all[layer, :, :, :prefill_len, :]
        v_pre = V_all[layer, :, :, :prefill_len, :]
        t0 = time.perf_counter(); full_cache.update(k_pre, v_pre, layer); t_prefill_full += time.perf_counter() - t0
        t0 = time.perf_counter(); tq_cache.update(k_pre, v_pre, layer);   t_prefill_tq   += time.perf_counter() - t0

    # --- Decode: append one token at a time (the streaming workload) ---
    t_decode_full = 0.0
    t_decode_tq   = 0.0
    for s in range(prefill_len, seq_len):
        for layer in range(n_layers):
            k_new = K_all[layer, :, :, s : s + 1, :]
            v_new = V_all[layer, :, :, s : s + 1, :]
            t0 = time.perf_counter(); full_cache.update(k_new, v_new, layer); t_decode_full += time.perf_counter() - t0
            t0 = time.perf_counter(); tq_cache.update(k_new, v_new, layer);   t_decode_tq   += time.perf_counter() - t0

    # Last-token attention output (per layer) — what the model actually consumes.
    layer_errs = []
    layer_cos  = []
    for layer in range(n_layers):
        K_full, V_full = full_cache._keys[layer], full_cache._values[layer]
        K_tq,   V_tq   = tq_cache._dequantize(layer)
        out_full = attention(Q_query, K_full, V_full).ravel()
        out_tq   = attention(Q_query, K_tq,   V_tq).ravel()
        layer_errs.append(float(np.linalg.norm(out_full - out_tq) / np.linalg.norm(out_full)))
        layer_cos.append(float(out_full @ out_tq /
                               (np.linalg.norm(out_full) * np.linalg.norm(out_tq))))

    return {
        "bits":                bits,
        "full_bytes":          full_cache.memory_bytes(),
        "tq_bytes":            tq_cache.memory_bytes(),
        "compression":         full_cache.memory_bytes() / tq_cache.memory_bytes(),
        "rel_err":             float(np.mean(layer_errs)),
        "cosine":              float(np.mean(layer_cos)),
        "t_prefill_fp16_ms":   t_prefill_full * 1000,
        "t_prefill_tq_ms":     t_prefill_tq   * 1000,
        "t_decode_fp16_ms":    t_decode_full  * 1000,
        "t_decode_tq_ms":      t_decode_tq    * 1000,
    }


def main() -> None:
    print("Synthetic transformer ── 4 layers × 8 heads × (512 prompt + 32 decode) × d=128")
    print("Baseline = fp16 (the realistic LLM serving precision)\n")
    print(f"{'bits':>4} | {'fp16 cache':>11} | {'TQ cache':>10} | {'compress':>9} | "
          f"{'rel-err':>9} | {'cos sim':>8} | "
          f"{'prefill fp16':>12} | {'prefill TQ':>10} | "
          f"{'decode fp16':>11} | {'decode TQ':>9}")
    print("-" * 130)
    for b in (4, 3, 2):
        r = run(bits=b, seed=0)
        print(f"{b:>4} | {r['full_bytes']/1e6:>8.2f} MB | "
              f"{r['tq_bytes']/1e6:>7.2f} MB | "
              f"{r['compression']:>8.2f}× | "
              f"{r['rel_err']:>8.4f} | "
              f"{r['cosine']:>8.4f} | "
              f"{r['t_prefill_fp16_ms']:>10.1f}ms | "
              f"{r['t_prefill_tq_ms']:>8.1f}ms | "
              f"{r['t_decode_fp16_ms']:>9.1f}ms | "
              f"{r['t_decode_tq_ms']:>7.1f}ms")
    print("\nrel-err  = ‖attn_full − attn_tq‖ / ‖attn_full‖   (lower = better)")
    print("cos sim  = cosine similarity of attention outputs (higher = better)")
    print("timing   = NumPy reference timing only; not a production latency result.\n")


# ---------------------------------------------------------------------------
# HF integration snippet — shown to the audience, not executed here.
# ---------------------------------------------------------------------------

HF_INTEGRATION_SNIPPET = r"""
# To use TurboQuant inside a real HuggingFace generation:
#
#   from transformers import AutoModelForCausalLM, AutoTokenizer
#   from tiny_turboquant import TurboQuantKVCache
#
#   tok   = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
#   model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")
#   cache = TurboQuantKVCache(bits=4)
#
#   inputs = tok("The capital of France is", return_tensors="pt")
#   out = model.generate(**inputs, max_new_tokens=64, past_key_values=cache)
#   print(tok.decode(out[0]))
#
# This uses compressed storage but still returns dense dequantized K/V tensors.
# Latency improvement requires fused packed-cache attention kernels.
"""


if __name__ == "__main__":
    main()
    print(HF_INTEGRATION_SNIPPET)
