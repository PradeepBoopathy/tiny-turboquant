"""Real LLM KV-cache benchmarking helpers.

This module is intentionally optional-dependency based. Importing it does not
require transformers; running the benchmark does.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from importlib.metadata import PackageNotFoundError, version as _pkg_version
from .kv_cache import HybridTurboQuantKVCache


def _tiny_turboquant_version() -> str:
    try:
        return _pkg_version("tiny-turboquant")
    except PackageNotFoundError:
        return "0.7.1"
from .kv_estimator import estimate_kv_cache_memory
from .kv_presets import resolve_kv_cache_preset


PROMPT_MODES = {
    "short": "The capital of France is Paris. The capital of Germany is Berlin. What is the capital of Italy?",
    "medium": (
        "You are answering questions from a short knowledge note. "
        "France has Paris as its capital. Germany has Berlin as its capital. "
        "Spain has Madrid as its capital. Italy has Rome as its capital. "
        "Portugal has Lisbon as its capital. Answer only using the note. " * 12
        + "Question: What is the capital of Italy?"
    ),
    "long": (
        "This is a long-context synthetic note for KV-cache evaluation. "
        "The important fact is: Italy has Rome as its capital. "
        "Other filler facts discuss geography, logistics, retrieval, and memory. " * 96
        + "Question: What is the capital of Italy?"
    ),
    "stress": (
        "This is a stress prompt for KV-cache memory evaluation. "
        "The key answer remains: Italy has Rome as its capital. "
        "The rest of this context is intentionally repetitive filler for cache growth. " * 256
        + "Question: What is the capital of Italy?"
    ),
}


def first_divergence(a: torch.Tensor, b: torch.Tensor) -> int:
    n = min(int(a.numel()), int(b.numel()))
    for i in range(n):
        if int(a[i]) != int(b[i]):
            return i
    return n


@dataclass
class KVBenchConfig:
    model: str
    cache_type: str = "hybrid"
    preset: str = "balanced"
    prompt_mode: str = "short"
    prompt: Optional[str] = None
    max_new_tokens: int = 16
    device: str = "auto"
    dtype: str = "auto"
    kl_tokens: int = 4
    skip_kl: bool = False
    key_bits: Optional[int] = None
    value_bits: Optional[int] = None
    key_recent_window: Optional[int] = None
    value_recent_window: Optional[int] = None
    key_outlier_bits: Optional[int] = None
    value_outlier_bits: Optional[int] = None
    n_key_outliers: Optional[str | int] = None
    n_value_outliers: Optional[str | int] = None
    per_head_calibration: Optional[bool] = None


def _torch_dtype(name: str, device: str) -> torch.dtype:
    if name == "auto":
        return torch.float16 if device == "cuda" else torch.float32
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"unsupported dtype {name!r}")
    resolved = mapping[name]
    if device == "cpu" and resolved in {torch.float16, torch.bfloat16}:
        # CPU fp16/bf16 support varies by machine and operation. Keep CLI demos stable.
        return torch.float32
    return resolved


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        # v0.7.1: fallback instead of crashing with
        # "Torch not compiled with CUDA enabled".
        return "cpu"
    return device


def _runtime_warnings(requested_device: str, resolved_device: str, requested_dtype: str, resolved_dtype: torch.dtype) -> List[str]:
    warnings: List[str] = []
    if requested_device == "cuda" and resolved_device == "cpu":
        warnings.append(
            "device='cuda' was requested but CUDA is not available; falling back to CPU. "
            "Install a CUDA-enabled PyTorch build or pass --device cpu."
        )
    if requested_dtype in {"float16", "fp16", "bfloat16", "bf16"} and resolved_device == "cpu" and resolved_dtype == torch.float32:
        warnings.append(
            f"dtype={requested_dtype!r} was requested on CPU; using float32 for stable CPU execution."
        )
    return warnings


def _prompt_from_mode(mode: str, prompt: Optional[str]) -> str:
    if prompt is not None:
        return prompt
    if mode not in PROMPT_MODES:
        valid = ", ".join(sorted(PROMPT_MODES))
        raise ValueError(f"unknown prompt mode {mode!r}; valid modes: {valid}")
    return PROMPT_MODES[mode]


def _load_model_and_tokenizer(model_name: str, device: str, dtype: torch.dtype):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "transformers is required for kv-bench. Install with: pip install tiny-turboquant[llm]"
        ) from exc

    tok = AutoTokenizer.from_pretrained(model_name)
    try:
        # Newer transformers prefers dtype= over torch_dtype=.
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    except TypeError:  # pragma: no cover - transformers-version dependent
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    try:
        model.to(device)
    except (AssertionError, RuntimeError) as exc:
        if device == "cuda":
            raise RuntimeError(
                "CUDA was selected but this PyTorch build cannot use CUDA. "
                "Run with --device cpu --dtype float32, or install CUDA-enabled PyTorch."
            ) from exc
        raise
    model.eval()
    return model, tok


def _make_cache(config: KVBenchConfig) -> HybridTurboQuantKVCache:
    overrides: Dict[str, Any] = {
        "key_bits": config.key_bits,
        "value_bits": config.value_bits,
        "key_recent_window": config.key_recent_window,
        "value_recent_window": config.value_recent_window,
        "key_outlier_bits": config.key_outlier_bits,
        "value_outlier_bits": config.value_outlier_bits,
        "n_key_outliers": config.n_key_outliers,
        "n_value_outliers": config.n_value_outliers,
        "per_head_calibration": config.per_head_calibration,
    }
    return HybridTurboQuantKVCache.from_preset(config.preset, **overrides)


@torch.no_grad()
def _generate(model, tok, inputs: Dict[str, torch.Tensor], *, max_new_tokens: int, cache=None):
    t0 = time.time()
    kwargs = dict(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    if cache is not None:
        kwargs["past_key_values"] = cache
    out = model.generate(**kwargs)
    elapsed = time.time() - t0
    text = tok.decode(out[0], skip_special_tokens=True)
    return out[0].detach().cpu(), text, elapsed


@torch.no_grad()
def _next_logit_kl(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, cache_factory, max_positions: int) -> Dict[str, float]:
    """Compare next-token logits for prefixes using dense vs compressed cache.

    This is a quality diagnostic. It is deliberately small because each prefix
    creates a fresh compressed cache.
    """
    kls = []
    n = min(max_positions, max(1, int(input_ids.shape[1]) - 1))
    for end in range(1, n + 1):
        prefix = input_ids[:, :end]
        mask = attention_mask[:, :end]
        dense = model(input_ids=prefix, attention_mask=mask, use_cache=True).logits[:, -1, :].float()
        cache = cache_factory()
        comp = model(input_ids=prefix, attention_mask=mask, use_cache=True, past_key_values=cache).logits[:, -1, :].float()
        kl = F.kl_div(
            F.log_softmax(comp, dim=-1),
            F.softmax(dense, dim=-1),
            reduction="batchmean",
        ).item()
        if math.isfinite(kl):
            kls.append(float(kl))
    if not kls:
        return {"mean_kl": float("nan"), "max_kl": float("nan"), "kl_positions": 0}
    return {"mean_kl": float(sum(kls) / len(kls)), "max_kl": float(max(kls)), "kl_positions": len(kls)}


def run_kv_benchmark(config: KVBenchConfig) -> Dict[str, Any]:
    requested_device = config.device
    requested_dtype = config.dtype
    device = _resolve_device(requested_device)
    dtype = _torch_dtype(requested_dtype, device)
    warnings = _runtime_warnings(requested_device, device, requested_dtype, dtype)
    prompt = _prompt_from_mode(config.prompt_mode, config.prompt)

    model, tok = _load_model_and_tokenizer(config.model, device, dtype)
    inputs = tok(prompt, return_tensors="pt").to(device)
    prompt_tokens = int(inputs["input_ids"].shape[1])

    cache = _make_cache(config)
    baseline_ids, baseline_text, baseline_seconds = _generate(
        model, tok, inputs, max_new_tokens=config.max_new_tokens, cache=None
    )
    compressed_ids, compressed_text, compressed_seconds = _generate(
        model, tok, inputs, max_new_tokens=config.max_new_tokens, cache=cache
    )

    continuation_base = baseline_ids[prompt_tokens:]
    continuation_comp = compressed_ids[prompt_tokens:]
    first_div = first_divergence(continuation_base, continuation_comp)
    identical = bool(torch.equal(continuation_base, continuation_comp))

    fp16_bytes = cache.fp16_baseline_bytes() if hasattr(cache, "fp16_baseline_bytes") else 0
    compressed_bytes = cache.actual_memory_bytes() if hasattr(cache, "actual_memory_bytes") else 0
    compression_ratio = fp16_bytes / max(compressed_bytes, 1) if fp16_bytes else 0.0
    memory_saved = (1.0 - compressed_bytes / fp16_bytes) * 100.0 if fp16_bytes else 0.0

    kl_report = {"mean_kl": None, "max_kl": None, "kl_positions": 0}
    if not config.skip_kl:
        def cf():
            return _make_cache(config)
        kl_report = _next_logit_kl(
            model,
            inputs["input_ids"],
            inputs.get("attention_mask", torch.ones_like(inputs["input_ids"])),
            cf,
            max_positions=max(1, int(config.kl_tokens)),
        )

    # Use model config for a context-size estimate as an explanatory metric.
    model_cfg = getattr(model, "config", None)
    layers = int(getattr(model_cfg, "num_hidden_layers", 0) or 0)
    kv_heads = int(getattr(model_cfg, "num_key_value_heads", getattr(model_cfg, "num_attention_heads", 0)) or 0)
    hidden = int(getattr(model_cfg, "hidden_size", 0) or 0)
    heads = int(getattr(model_cfg, "num_attention_heads", max(kv_heads, 1)) or 1)
    head_dim = int(getattr(model_cfg, "head_dim", hidden // max(heads, 1) if hidden else 0) or 0)
    estimate = None
    if layers and kv_heads and head_dim:
        preset_cfg = resolve_kv_cache_preset(config.preset)
        estimate = estimate_kv_cache_memory(
            layers=layers,
            kv_heads=kv_heads,
            head_dim=head_dim,
            seq_len=prompt_tokens + config.max_new_tokens,
            batch_size=1,
            dtype_bytes=2 if dtype in (torch.float16, torch.bfloat16) else 4,
            key_bits=int(config.key_bits or preset_cfg["key_bits"]),
            value_bits=int(config.value_bits or preset_cfg["value_bits"]),
            key_recent_window=int(config.key_recent_window or preset_cfg["key_recent_window"]),
            value_recent_window=int(config.value_recent_window or preset_cfg["value_recent_window"]),
            key_outlier_bits=int(config.key_outlier_bits or preset_cfg["key_outlier_bits"]),
            value_outlier_bits=int(config.value_outlier_bits or preset_cfg["value_outlier_bits"]),
            n_key_outliers=config.n_key_outliers if config.n_key_outliers is not None else preset_cfg["n_key_outliers"],
            n_value_outliers=config.n_value_outliers if config.n_value_outliers is not None else preset_cfg["n_value_outliers"],
        ).to_dict()

    return {
        "version": _tiny_turboquant_version(),
        "model": config.model,
        "cache_type": config.cache_type,
        "preset": config.preset,
        "prompt_mode": config.prompt_mode,
        "prompt_tokens": prompt_tokens,
        "max_new_tokens": int(config.max_new_tokens),
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "requested_device": requested_device,
        "requested_dtype": requested_dtype,
        "config": asdict(config),
        "warnings": warnings,
        "memory": {
            "fp16_bytes": int(fp16_bytes),
            "compressed_bytes": int(compressed_bytes),
            # Stable alias for callers that use estimator-style names.
            "compressed_estimated_bytes": int(compressed_bytes),
            "total_fp16_bytes": int(fp16_bytes),
            "total_compressed_bytes": int(compressed_bytes),
            "compression_ratio": float(compression_ratio),
            "memory_saved_pct": float(memory_saved),
            "estimate": estimate,
        },
        "quality": {
            **kl_report,
            "first_divergence": int(first_div),
            "continuation_tokens_compared": int(min(continuation_base.numel(), continuation_comp.numel())),
            "identical_generation": identical,
        },
        "timing": {
            "baseline_seconds": float(baseline_seconds),
            "compressed_seconds": float(compressed_seconds),
            "note": "Timing is diagnostic only; this package does not claim production inference acceleration.",
        },
        "text": {
            "prompt": prompt,
            "baseline_output": baseline_text,
            "compressed_output": compressed_text,
        },
        "interpretation": (
            "This benchmark measures KV-cache memory and generation-quality behavior. "
            "It does not prove faster production inference."
        ),
    }


def save_json_report(report: Dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(report, indent=2, sort_keys=True))


def markdown_report(report: Dict[str, Any]) -> str:
    mem = report["memory"]
    q = report["quality"]
    t = report["timing"]
    warnings = report.get("warnings") or []
    warnings_md = "\n".join(f"- {w}" for w in warnings) if warnings else "- None"
    return f"""# tiny-turboquant KV benchmark report

## Setup

- Package version: `{report['version']}`
- Model: `{report['model']}`
- Preset: `{report['preset']}`
- Prompt mode: `{report['prompt_mode']}`
- Prompt tokens: `{report['prompt_tokens']}`
- Max new tokens: `{report['max_new_tokens']}`
- Device: `{report['device']}`
- dtype: `{report['dtype']}`
- Requested device: `{report.get('requested_device', report['device'])}`
- Requested dtype: `{report.get('requested_dtype', report['dtype'])}`

## Warnings

{warnings_md}

## Memory

| Metric | Value |
|---|---:|
| fp16 KV bytes | {mem['fp16_bytes']} |
| compressed KV bytes | {mem['compressed_bytes']} |
| compression ratio | {mem['compression_ratio']:.4f}x |
| memory saved | {mem['memory_saved_pct']:.2f}% |

## Quality

| Metric | Value |
|---|---:|
| mean KL | {q.get('mean_kl')} |
| max KL | {q.get('max_kl')} |
| KL positions | {q.get('kl_positions')} |
| first divergence | {q['first_divergence']} / {q['continuation_tokens_compared']} |
| identical generation | {q['identical_generation']} |

## Timing

| Metric | Seconds |
|---|---:|
| baseline generation | {t['baseline_seconds']:.4f} |
| compressed generation | {t['compressed_seconds']:.4f} |

> Timing is diagnostic only. This package does not claim production inference acceleration.

## Baseline output

```text
{report['text']['baseline_output']}
```

## Compressed output

```text
{report['text']['compressed_output']}
```
"""


def save_markdown_report(report: Dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(markdown_report(report))


__all__ = [
    "KVBenchConfig",
    "PROMPT_MODES",
    "first_divergence",
    "run_kv_benchmark",
    "markdown_report",
    "save_json_report",
    "save_markdown_report",
]
