"""Serving-style KV-cache capacity simulation utilities.

These utilities model memory pressure for serving-style paged KV-cache layouts.
They are intentionally estimators: they help reason about capacity under a fixed
memory budget, but they do not implement a production serving backend.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from .kv_estimator import OutlierSpec, estimate_kv_cache_memory
from .kv_presets import resolve_kv_cache_preset

GB = 1024**3


def _ceil_div(a: int, b: int) -> int:
    return (int(a) + int(b) - 1) // int(b)


def _bytes_from_gb(x: float) -> int:
    return int(float(x) * GB)


def _safe_ratio(a: float, b: float) -> float:
    return float(a) / max(float(b), 1.0)


def _compression_config_from_preset(
    preset: str = "balanced",
    *,
    key_bits: Optional[int] = None,
    value_bits: Optional[int] = None,
    key_outlier_bits: Optional[int] = None,
    value_outlier_bits: Optional[int] = None,
    n_key_outliers: Optional[OutlierSpec] = None,
    n_value_outliers: Optional[OutlierSpec] = None,
    key_recent_window: Optional[int] = None,
    value_recent_window: Optional[int] = None,
) -> Dict[str, Any]:
    cfg = dict(resolve_kv_cache_preset(preset))
    overrides = {
        "key_bits": key_bits,
        "value_bits": value_bits,
        "key_outlier_bits": key_outlier_bits,
        "value_outlier_bits": value_outlier_bits,
        "n_key_outliers": n_key_outliers,
        "n_value_outliers": n_value_outliers,
        "key_recent_window": key_recent_window,
        "value_recent_window": value_recent_window,
    }
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg


@dataclass(frozen=True)
class ServingSequence:
    """One active request/sequence in a serving simulation."""

    seq_id: str
    prompt_tokens: int
    decode_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return int(self.prompt_tokens) + int(self.decode_tokens)


@dataclass(frozen=True)
class PagedServingConfig:
    """Configuration for serving-style KV-cache memory simulation."""

    layers: int
    kv_heads: int
    head_dim: int
    page_size: int = 128
    dtype_bytes: int = 2
    preset: str = "balanced"
    key_bits: Optional[int] = None
    value_bits: Optional[int] = None
    key_outlier_bits: Optional[int] = None
    value_outlier_bits: Optional[int] = None
    n_key_outliers: Optional[OutlierSpec] = None
    n_value_outliers: Optional[OutlierSpec] = None
    key_recent_window: Optional[int] = None
    value_recent_window: Optional[int] = None

    def compression_config(self) -> Dict[str, Any]:
        return _compression_config_from_preset(
            self.preset,
            key_bits=self.key_bits,
            value_bits=self.value_bits,
            key_outlier_bits=self.key_outlier_bits,
            value_outlier_bits=self.value_outlier_bits,
            n_key_outliers=self.n_key_outliers,
            n_value_outliers=self.n_value_outliers,
            key_recent_window=self.key_recent_window,
            value_recent_window=self.value_recent_window,
        )

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["resolved_compression"] = self.compression_config()
        return data


@dataclass(frozen=True)
class SequenceMemoryRecord:
    seq_id: str
    prompt_tokens: int
    decode_tokens: int
    total_tokens: int
    page_size: int
    pages_allocated: int
    allocated_tokens: int
    unused_page_tokens: int
    fp16_bytes: int
    compressed_bytes: int
    compression_ratio: float
    memory_saved_pct: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ServingMemoryReport:
    config: Dict[str, Any]
    active_sequences: int
    total_actual_tokens: int
    total_allocated_tokens: int
    total_pages_allocated: int
    fp16_bytes: int
    compressed_estimated_bytes: int
    compression_ratio: float
    memory_saved_pct: float
    gpu_memory_budget_bytes: Optional[int] = None
    model_weight_bytes: int = 0
    available_kv_bytes: Optional[int] = None
    fp16_fits_budget: Optional[bool] = None
    compressed_fits_budget: Optional[bool] = None
    remaining_budget_fp16_bytes: Optional[int] = None
    remaining_budget_compressed_bytes: Optional[int] = None
    per_sequence: List[Dict[str, Any]] = field(default_factory=list)
    note: str = (
        "Serving simulation estimates paged KV-cache memory and capacity. "
        "It does not implement a production serving backend or claim inference acceleration."
    )

    def to_dict(self, *, include_per_sequence: bool = True) -> Dict[str, Any]:
        data = asdict(self)
        # v0.7.1 backward-compatible aliases for earlier demo scripts.
        data["total_fp16_bytes"] = data["fp16_bytes"]
        data["total_compressed_bytes"] = data["compressed_estimated_bytes"]
        data["compressed_bytes"] = data["compressed_estimated_bytes"]
        data["per_sequence_count"] = len(data.get("per_sequence", []))
        if not include_per_sequence:
            data.pop("per_sequence", None)
        return data


class PagedKVServingSimulator:
    """Serving-style paged KV-cache memory simulator.

    The simulator models active sequences, page allocation, and fp16 vs hybrid
    compressed KV-cache memory under optional GPU-memory budgets.
    """

    def __init__(
        self,
        *,
        layers: int,
        kv_heads: int,
        head_dim: int,
        page_size: int = 128,
        dtype_bytes: int = 2,
        preset: str = "balanced",
        gpu_memory_budget_gb: Optional[float] = None,
        model_weight_gb: float = 0.0,
        key_bits: Optional[int] = None,
        value_bits: Optional[int] = None,
        key_outlier_bits: Optional[int] = None,
        value_outlier_bits: Optional[int] = None,
        n_key_outliers: Optional[OutlierSpec] = None,
        n_value_outliers: Optional[OutlierSpec] = None,
        key_recent_window: Optional[int] = None,
        value_recent_window: Optional[int] = None,
    ):
        self.config = PagedServingConfig(
            layers=int(layers),
            kv_heads=int(kv_heads),
            head_dim=int(head_dim),
            page_size=int(page_size),
            dtype_bytes=int(dtype_bytes),
            preset=preset,
            key_bits=key_bits,
            value_bits=value_bits,
            key_outlier_bits=key_outlier_bits,
            value_outlier_bits=value_outlier_bits,
            n_key_outliers=n_key_outliers,
            n_value_outliers=n_value_outliers,
            key_recent_window=key_recent_window,
            value_recent_window=value_recent_window,
        )
        self.gpu_memory_budget_bytes = (
            _bytes_from_gb(gpu_memory_budget_gb) if gpu_memory_budget_gb is not None else None
        )
        self.model_weight_bytes = _bytes_from_gb(model_weight_gb)
        self._sequences: List[ServingSequence] = []

    @property
    def sequences(self) -> Sequence[ServingSequence]:
        return tuple(self._sequences)

    def add_sequence(self, seq_id: Union[str, int], prompt_tokens: int, decode_tokens: int = 0) -> "PagedKVServingSimulator":
        self._sequences.append(
            ServingSequence(str(seq_id), int(prompt_tokens), int(decode_tokens))
        )
        return self

    def add_uniform_sequences(
        self,
        *,
        users: int,
        prompt_tokens: int,
        decode_tokens: int = 0,
        prefix: str = "user",
    ) -> "PagedKVServingSimulator":
        for i in range(int(users)):
            self.add_sequence(f"{prefix}-{i + 1}", prompt_tokens, decode_tokens)
        return self

    def _sequence_record(self, seq: ServingSequence) -> SequenceMemoryRecord:
        c = self.config
        pages = _ceil_div(max(seq.total_tokens, 1), c.page_size)
        allocated_tokens = pages * c.page_size
        unused = allocated_tokens - seq.total_tokens
        cfg = c.compression_config()
        est = estimate_kv_cache_memory(
            layers=c.layers,
            kv_heads=c.kv_heads,
            head_dim=c.head_dim,
            seq_len=allocated_tokens,
            batch_size=1,
            dtype_bytes=c.dtype_bytes,
            key_bits=int(cfg["key_bits"]),
            value_bits=int(cfg["value_bits"]),
            key_outlier_bits=cfg.get("key_outlier_bits"),
            value_outlier_bits=cfg.get("value_outlier_bits"),
            n_key_outliers=cfg.get("n_key_outliers", "auto"),
            n_value_outliers=cfg.get("n_value_outliers", "auto"),
            key_recent_window=int(cfg.get("key_recent_window", 128)),
            value_recent_window=int(cfg.get("value_recent_window", 64)),
        )
        fp16 = int(est.fp16_bytes)
        comp = int(est.compressed_estimated_bytes)
        return SequenceMemoryRecord(
            seq_id=seq.seq_id,
            prompt_tokens=int(seq.prompt_tokens),
            decode_tokens=int(seq.decode_tokens),
            total_tokens=int(seq.total_tokens),
            page_size=int(c.page_size),
            pages_allocated=int(pages),
            allocated_tokens=int(allocated_tokens),
            unused_page_tokens=int(unused),
            fp16_bytes=fp16,
            compressed_bytes=comp,
            compression_ratio=_safe_ratio(fp16, comp),
            memory_saved_pct=(1.0 - comp / max(fp16, 1)) * 100.0,
        )

    def memory_report(self) -> ServingMemoryReport:
        records = [self._sequence_record(seq) for seq in self._sequences]
        fp16 = sum(r.fp16_bytes for r in records)
        comp = sum(r.compressed_bytes for r in records)
        budget = self.gpu_memory_budget_bytes
        available = None if budget is None else max(0, budget - self.model_weight_bytes)
        return ServingMemoryReport(
            config=self.config.to_dict(),
            active_sequences=len(records),
            total_actual_tokens=sum(r.total_tokens for r in records),
            total_allocated_tokens=sum(r.allocated_tokens for r in records),
            total_pages_allocated=sum(r.pages_allocated for r in records),
            fp16_bytes=int(fp16),
            compressed_estimated_bytes=int(comp),
            compression_ratio=_safe_ratio(fp16, comp),
            memory_saved_pct=(1.0 - comp / max(fp16, 1)) * 100.0 if fp16 else 0.0,
            gpu_memory_budget_bytes=budget,
            model_weight_bytes=int(self.model_weight_bytes),
            available_kv_bytes=available,
            fp16_fits_budget=None if available is None else fp16 <= available,
            compressed_fits_budget=None if available is None else comp <= available,
            remaining_budget_fp16_bytes=None if available is None else int(available - fp16),
            remaining_budget_compressed_bytes=None if available is None else int(available - comp),
            per_sequence=[r.to_dict() for r in records],
        )


@dataclass(frozen=True)
class ServingCapacityEstimate:
    gpu_memory_gb: float
    model_weight_gb: float
    available_kv_gb: float
    layers: int
    kv_heads: int
    head_dim: int
    page_size: int
    avg_prompt_tokens: int
    avg_decode_tokens: int
    allocated_tokens_per_user: int
    fp16_bytes_per_user: int
    compressed_bytes_per_user: int
    max_users_fp16: int
    max_users_compressed: int
    capacity_gain: float
    compression_ratio: float
    memory_saved_pct: float
    preset: str
    resolved_compression: Dict[str, Any]
    note: str = (
        "Capacity estimate models KV-cache memory only after reserving model weights. "
        "It does not model scheduler overhead, activations, fragmentation, or production throughput."
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def estimate_serving_capacity(
    *,
    gpu_memory_gb: float,
    model_weight_gb: float,
    layers: int,
    kv_heads: int,
    head_dim: int,
    avg_prompt_tokens: int,
    avg_decode_tokens: int = 0,
    page_size: int = 128,
    dtype_bytes: int = 2,
    preset: str = "balanced",
    key_bits: Optional[int] = None,
    value_bits: Optional[int] = None,
    key_outlier_bits: Optional[int] = None,
    value_outlier_bits: Optional[int] = None,
    n_key_outliers: Optional[OutlierSpec] = None,
    n_value_outliers: Optional[OutlierSpec] = None,
    key_recent_window: Optional[int] = None,
    value_recent_window: Optional[int] = None,
) -> ServingCapacityEstimate:
    total_tokens = int(avg_prompt_tokens) + int(avg_decode_tokens)
    allocated_tokens = max(1, _ceil_div(total_tokens, page_size) * int(page_size))
    cfg = _compression_config_from_preset(
        preset,
        key_bits=key_bits,
        value_bits=value_bits,
        key_outlier_bits=key_outlier_bits,
        value_outlier_bits=value_outlier_bits,
        n_key_outliers=n_key_outliers,
        n_value_outliers=n_value_outliers,
        key_recent_window=key_recent_window,
        value_recent_window=value_recent_window,
    )
    est = estimate_kv_cache_memory(
        layers=layers,
        kv_heads=kv_heads,
        head_dim=head_dim,
        seq_len=allocated_tokens,
        batch_size=1,
        dtype_bytes=dtype_bytes,
        key_bits=int(cfg["key_bits"]),
        value_bits=int(cfg["value_bits"]),
        key_outlier_bits=cfg.get("key_outlier_bits"),
        value_outlier_bits=cfg.get("value_outlier_bits"),
        n_key_outliers=cfg.get("n_key_outliers", "auto"),
        n_value_outliers=cfg.get("n_value_outliers", "auto"),
        key_recent_window=int(cfg.get("key_recent_window", 128)),
        value_recent_window=int(cfg.get("value_recent_window", 64)),
    )
    available = max(0, _bytes_from_gb(gpu_memory_gb) - _bytes_from_gb(model_weight_gb))
    max_fp16 = int(available // max(est.fp16_bytes, 1))
    max_comp = int(available // max(est.compressed_estimated_bytes, 1))
    return ServingCapacityEstimate(
        gpu_memory_gb=float(gpu_memory_gb),
        model_weight_gb=float(model_weight_gb),
        available_kv_gb=float(available / GB),
        layers=int(layers),
        kv_heads=int(kv_heads),
        head_dim=int(head_dim),
        page_size=int(page_size),
        avg_prompt_tokens=int(avg_prompt_tokens),
        avg_decode_tokens=int(avg_decode_tokens),
        allocated_tokens_per_user=int(allocated_tokens),
        fp16_bytes_per_user=int(est.fp16_bytes),
        compressed_bytes_per_user=int(est.compressed_estimated_bytes),
        max_users_fp16=max_fp16,
        max_users_compressed=max_comp,
        capacity_gain=_safe_ratio(max_comp, max_fp16),
        compression_ratio=float(est.compression_ratio),
        memory_saved_pct=float(est.memory_saved_pct),
        preset=preset,
        resolved_compression=cfg,
    )


@dataclass(frozen=True)
class DecodeGrowthPoint:
    step: int
    users: int
    tokens_per_user: int
    allocated_tokens_per_user: int
    fp16_bytes: int
    compressed_estimated_bytes: int
    compression_ratio: float
    memory_saved_pct: float
    fp16_fits_budget: Optional[bool] = None
    compressed_fits_budget: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # v0.7.1 backward-compatible aliases for earlier demo scripts.
        data["decode_step"] = data["step"]
        data["compressed_bytes"] = data["compressed_estimated_bytes"]
        return data


def simulate_decode_growth(
    *,
    users: int,
    prompt_tokens: int,
    decode_tokens: int,
    layers: int,
    kv_heads: int,
    head_dim: int,
    page_size: int = 128,
    dtype_bytes: int = 2,
    preset: str = "balanced",
    step_interval: int = 64,
    gpu_memory_gb: Optional[float] = None,
    model_weight_gb: float = 0.0,
    key_bits: Optional[int] = None,
    value_bits: Optional[int] = None,
    key_outlier_bits: Optional[int] = None,
    value_outlier_bits: Optional[int] = None,
    n_key_outliers: Optional[OutlierSpec] = None,
    n_value_outliers: Optional[OutlierSpec] = None,
    key_recent_window: Optional[int] = None,
    value_recent_window: Optional[int] = None,
) -> List[DecodeGrowthPoint]:
    steps = list(range(0, int(decode_tokens) + 1, max(1, int(step_interval))))
    if steps[-1] != int(decode_tokens):
        steps.append(int(decode_tokens))
    budget = None
    if gpu_memory_gb is not None:
        budget = max(0, _bytes_from_gb(gpu_memory_gb) - _bytes_from_gb(model_weight_gb))
    points: List[DecodeGrowthPoint] = []
    cfg = _compression_config_from_preset(
        preset,
        key_bits=key_bits,
        value_bits=value_bits,
        key_outlier_bits=key_outlier_bits,
        value_outlier_bits=value_outlier_bits,
        n_key_outliers=n_key_outliers,
        n_value_outliers=n_value_outliers,
        key_recent_window=key_recent_window,
        value_recent_window=value_recent_window,
    )
    for step in steps:
        tokens = int(prompt_tokens) + int(step)
        allocated = max(1, _ceil_div(tokens, page_size) * int(page_size))
        est = estimate_kv_cache_memory(
            layers=layers,
            kv_heads=kv_heads,
            head_dim=head_dim,
            seq_len=allocated,
            batch_size=int(users),
            dtype_bytes=dtype_bytes,
            key_bits=int(cfg["key_bits"]),
            value_bits=int(cfg["value_bits"]),
            key_outlier_bits=cfg.get("key_outlier_bits"),
            value_outlier_bits=cfg.get("value_outlier_bits"),
            n_key_outliers=cfg.get("n_key_outliers", "auto"),
            n_value_outliers=cfg.get("n_value_outliers", "auto"),
            key_recent_window=int(cfg.get("key_recent_window", 128)),
            value_recent_window=int(cfg.get("value_recent_window", 64)),
        )
        points.append(
            DecodeGrowthPoint(
                step=int(step),
                users=int(users),
                tokens_per_user=tokens,
                allocated_tokens_per_user=allocated,
                fp16_bytes=int(est.fp16_bytes),
                compressed_estimated_bytes=int(est.compressed_estimated_bytes),
                compression_ratio=float(est.compression_ratio),
                memory_saved_pct=float(est.memory_saved_pct),
                fp16_fits_budget=None if budget is None else est.fp16_bytes <= budget,
                compressed_fits_budget=None if budget is None else est.compressed_estimated_bytes <= budget,
            )
        )
    return points


def serving_markdown_report(report: Union[ServingMemoryReport, ServingCapacityEstimate, Dict[str, Any]]) -> str:
    data = report.to_dict() if hasattr(report, "to_dict") else dict(report)
    lines = ["# tiny-turboquant serving simulation report", ""]
    if "active_sequences" in data:
        lines += [
            "## Summary",
            f"- Active sequences: {data['active_sequences']}",
            f"- Total actual tokens: {data['total_actual_tokens']}",
            f"- Total allocated tokens: {data['total_allocated_tokens']}",
            f"- Pages allocated: {data['total_pages_allocated']}",
            f"- FP16 KV bytes: {data['fp16_bytes']}",
            f"- Compressed estimated bytes: {data['compressed_estimated_bytes']}",
            f"- Compression ratio: {data['compression_ratio']:.4f}x",
            f"- Memory saved: {data['memory_saved_pct']:.2f}%",
            "",
        ]
        if data.get("available_kv_bytes") is not None:
            lines += [
                "## Budget fit",
                f"- Available KV bytes: {data['available_kv_bytes']}",
                f"- FP16 fits: {data['fp16_fits_budget']}",
                f"- Compressed fits: {data['compressed_fits_budget']}",
                "",
            ]
    elif "max_users_fp16" in data:
        lines += [
            "## Capacity estimate",
            f"- GPU memory: {data['gpu_memory_gb']} GB",
            f"- Model weights reserved: {data['model_weight_gb']} GB",
            f"- Available KV memory: {data['available_kv_gb']:.2f} GB",
            f"- Allocated tokens/user: {data['allocated_tokens_per_user']}",
            f"- FP16 bytes/user: {data['fp16_bytes_per_user']}",
            f"- Compressed bytes/user: {data['compressed_bytes_per_user']}",
            f"- Max users FP16: {data['max_users_fp16']}",
            f"- Max users compressed: {data['max_users_compressed']}",
            f"- Capacity gain: {data['capacity_gain']:.4f}x",
            f"- Memory saved/user: {data['memory_saved_pct']:.2f}%",
            "",
        ]
    lines += [
        "## Interpretation",
        "This report estimates serving-style KV-cache memory and capacity. ",
        "It does not claim production inference acceleration or a drop-in serving backend.",
        "",
    ]
    return "\n".join(lines)


def decode_growth_markdown_report(points: Sequence[DecodeGrowthPoint]) -> str:
    lines = [
        "# tiny-turboquant decode-growth memory report",
        "",
        "| step | tokens/user | allocated/user | fp16 bytes | compressed bytes | ratio | saved % |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for p in points:
        lines.append(
            f"| {p.step} | {p.tokens_per_user} | {p.allocated_tokens_per_user} | "
            f"{p.fp16_bytes} | {p.compressed_estimated_bytes} | "
            f"{p.compression_ratio:.4f} | {p.memory_saved_pct:.2f} |"
        )
    lines += [
        "",
        "This is a memory-growth simulation, not a production throughput benchmark.",
    ]
    return "\n".join(lines)


def save_json(data: Any, path: Union[str, Path]) -> None:
    if hasattr(data, "to_dict"):
        data = data.to_dict()
    elif isinstance(data, list):
        data = [x.to_dict() if hasattr(x, "to_dict") else x for x in data]
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_markdown(text: str, path: Union[str, Path]) -> None:
    Path(path).write_text(text, encoding="utf-8")


def save_decode_growth_csv(points: Sequence[DecodeGrowthPoint], path: Union[str, Path]) -> None:
    rows = [p.to_dict() for p in points]
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


__all__ = [
    "ServingSequence",
    "PagedServingConfig",
    "SequenceMemoryRecord",
    "ServingMemoryReport",
    "PagedKVServingSimulator",
    "ServingCapacityEstimate",
    "estimate_serving_capacity",
    "DecodeGrowthPoint",
    "simulate_decode_growth",
    "serving_markdown_report",
    "decode_growth_markdown_report",
    "save_decode_growth_csv",
    "save_json",
    "save_markdown",
]
