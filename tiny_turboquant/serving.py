"""Serving-engine integration scaffolding.

This module intentionally avoids importing vLLM or TensorRT-LLM. It provides
small spec/adapter objects that make integration experiments explicit without
claiming drop-in production support.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class PagedKVCacheSpec:
    key_bits: int = 6
    value_bits: int = 4
    key_outlier_bits: int = 8
    value_outlier_bits: int = 8
    n_key_outliers: int = 32
    n_value_outliers: int = 16
    key_recent_window: int = 128
    value_recent_window: int = 128
    per_layer_calibration: bool = True
    per_head_calibration: bool = False
    page_size: int = 128

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ServingEngineAdapter:
    """Base class for future serving-engine experiments.

    Subclasses should translate ``PagedKVCacheSpec`` into engine-specific cache
    layout, scheduling, and kernel choices. This class does not implement vLLM
    or TensorRT-LLM internals.
    """

    engine_name = "generic"

    def __init__(self, spec: Optional[PagedKVCacheSpec] = None):
        self.spec = spec or PagedKVCacheSpec()

    def integration_plan(self) -> Dict[str, Any]:
        return {
            "engine": self.engine_name,
            "spec": self.spec.to_dict(),
            "steps": [
                "map compressed pages to the engine's paged KV-cache blocks",
                "preserve recent-window pages in dense precision",
                "route older pages through compressed dequant/attention path",
                "measure memory, first-token drift, KL, and throughput separately",
            ],
            "status": "planning scaffold; not a drop-in serving backend",
        }


class VLLMExperimentAdapter(ServingEngineAdapter):
    engine_name = "vLLM-style paged cache"


class TensorRTLLMExperimentAdapter(ServingEngineAdapter):
    engine_name = "TensorRT-LLM-style paged cache"
