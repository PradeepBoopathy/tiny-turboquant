"""Safe presets for quality-aware KV-cache experiments.

These presets are intentionally conservative labels for research benchmarking.
They do not imply production acceleration.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict


KV_CACHE_PRESETS: Dict[str, Dict[str, Any]] = {
    "safe": {
        "key_bits": 8,
        "value_bits": 6,
        "key_outlier_bits": 8,
        "value_outlier_bits": 8,
        "n_key_outliers": "auto",
        "n_value_outliers": "auto",
        "key_recent_window": 256,
        "value_recent_window": 256,
        "per_layer_calibration": True,
        "per_head_calibration": False,
    },
    "balanced": {
        "key_bits": 6,
        "value_bits": 4,
        "key_outlier_bits": 8,
        "value_outlier_bits": 8,
        "n_key_outliers": "auto",
        "n_value_outliers": "auto",
        "key_recent_window": 128,
        "value_recent_window": 64,
        "per_layer_calibration": True,
        "per_head_calibration": False,
    },
    "aggressive": {
        "key_bits": 4,
        "value_bits": 4,
        "key_outlier_bits": 8,
        "value_outlier_bits": 6,
        "n_key_outliers": "auto",
        "n_value_outliers": "auto",
        "key_recent_window": 64,
        "value_recent_window": 32,
        "per_layer_calibration": True,
        "per_head_calibration": False,
    },
    "quality-headwise": {
        "key_bits": 8,
        "value_bits": 6,
        "key_outlier_bits": 8,
        "value_outlier_bits": 8,
        "n_key_outliers": "auto",
        "n_value_outliers": "auto",
        "key_recent_window": 256,
        "value_recent_window": 128,
        "per_layer_calibration": True,
        "per_head_calibration": True,
    },
}


def available_kv_presets() -> list[str]:
    return sorted(KV_CACHE_PRESETS)


def resolve_kv_cache_preset(preset: str = "balanced", **overrides: Any) -> Dict[str, Any]:
    """Resolve a preset name into constructor kwargs.

    ``None`` override values are ignored so CLI callers can pass optional values
    directly without clobbering preset defaults.
    """
    if preset not in KV_CACHE_PRESETS:
        valid = ", ".join(available_kv_presets())
        raise ValueError(f"unknown KV cache preset {preset!r}; valid presets: {valid}")
    cfg = deepcopy(KV_CACHE_PRESETS[preset])
    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value
    return cfg


__all__ = ["KV_CACHE_PRESETS", "available_kv_presets", "resolve_kv_cache_preset"]
