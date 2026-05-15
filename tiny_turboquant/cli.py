"""Command-line utilities for tiny-turboquant."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

import torch
import torch.nn.functional as F

from . import __version__
from .metrics import exact_overlap, label_match_ratio
from .rag import RAGCompressedIndex, load_texts_from_folder, load_texts_from_jsonl
from .vector_index import CompressedVectorIndex
from .kv_bench import KVBenchConfig, run_kv_benchmark, save_json_report, save_markdown_report
from .attention_perf import (
    PageAttentionBenchConfig,
    page_attention_markdown_report,
    run_page_attention_benchmark,
    save_page_attention_json,
    save_page_attention_markdown,
)
from .kv_estimator import estimate_kv_cache_memory
from .kv_presets import available_kv_presets
from .layout import (
    LayoutBenchConfig,
    LayoutSweepConfig,
    available_layout_presets,
    run_layout_benchmark,
    run_layout_quality_sweep,
    save_layout_json,
    save_layout_markdown,
    save_layout_sweep_markdown,
    run_rotate_q_check,
    ResidualSweepConfig,
    run_residual_correction_sweep,
    save_residual_sweep_markdown,
)
from .fused_attention import (
    FusedDecodeBenchConfig,
    run_fused_decode_benchmark,
    save_fused_decode_json,
    save_fused_decode_markdown,
    LongContextCompareConfig,
    run_long_context_comparison,
    save_long_context_json,
    save_long_context_markdown,
    SplitKCompareConfig,
    run_split_k_comparison,
    save_split_k_json,
    save_split_k_markdown,
    EndToEndDecodeBenchConfig,
    run_end_to_end_decode_benchmark,
    save_end_to_end_decode_json,
    save_end_to_end_decode_markdown,
)

from .serving_sim import (
    PagedKVServingSimulator,
    estimate_serving_capacity,
    simulate_decode_growth,
    serving_markdown_report,
    decode_growth_markdown_report,
    save_json as save_serving_json,
    save_markdown as save_serving_markdown,
    save_decode_growth_csv,
)


def _synthetic_rag_corpus(docs_per_topic: int = 100):
    topics = {
        "rag": [
            "Embedding compression can reduce vector-store memory in retrieval-heavy systems.",
            "Compressed vector indexes help store more document chunks in limited memory.",
            "RAG systems retrieve relevant chunks and pass them to an LLM as context.",
            "Reranking helps recover quality after compressed vector retrieval.",
        ],
        "kv_cache": [
            "KV cache stores Key and Value tensors during autoregressive LLM generation.",
            "KV cache memory grows with context length and active sequences.",
            "Hybrid KV cache keeps recent tokens dense and compresses older tokens.",
            "Fused attention kernels are needed for production-speed compressed cache.",
        ],
        "serving": [
            "LLM serving systems manage batching, throughput, latency, and memory.",
            "Paged cache layouts help serving engines manage many active users.",
            "Production inference requires stable latency and measured output quality.",
            "Serving benchmarks should report tokens per second and memory usage.",
        ],
    }
    docs = []
    labels = []
    for label, templates in topics.items():
        for i in range(docs_per_topic):
            docs.append(f"{templates[i % len(templates)]} Scenario {i:04d} topic={label}.")
            labels.append(label)
    return docs, labels


def _embed_with_sentence_transformers(texts, model_name: str, batch_size: int = 64):
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "sentence-transformers is required for rag-bench text embedding. "
            "Install with: pip install tiny-turboquant[demos]"
        ) from exc
    model = SentenceTransformer(model_name)
    return model.encode(
        list(texts),
        convert_to_tensor=True,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=False,
    )


def rag_bench(args: argparse.Namespace) -> int:
    if args.synthetic:
        docs, labels = _synthetic_rag_corpus(args.docs_per_topic)
        query = args.query or "How can we reduce memory usage in a RAG vector store?"
        expected_label = args.expected_label or "rag"
    elif args.input_jsonl:
        docs = load_texts_from_jsonl(args.input_jsonl, text_field=args.text_field)
        labels = [None] * len(docs)
        query = args.query
        expected_label = args.expected_label
    elif args.input_folder:
        docs = load_texts_from_folder(args.input_folder)
        labels = [None] * len(docs)
        query = args.query
        expected_label = args.expected_label
    else:
        raise SystemExit("rag-bench requires --synthetic, --input-jsonl, or --input-folder")

    if not query:
        raise SystemExit("rag-bench requires --query unless --synthetic is used")

    embeddings = _embed_with_sentence_transformers(docs, args.embedding_model, args.batch_size)
    query_embedding = _embed_with_sentence_transformers([query], args.embedding_model, args.batch_size)[0]

    # Baseline full-precision search.
    scores = embeddings @ query_embedding
    full_scores, full_idx = torch.topk(scores, k=min(args.top_k, embeddings.shape[0]))
    full_ids = [int(i) for i in full_idx]
    full_labels = [labels[i] for i in full_ids]

    # Compressed search + rerank.
    index = CompressedVectorIndex(bits=args.bits, store_original_for_rerank=True).add(embeddings)
    compressed = index.search(query_embedding, top_k=args.top_k, rerank_top_k=args.rerank_top_k)
    compressed_ids = [r.index for r in compressed]
    compressed_labels = [labels[i] for i in compressed_ids]

    fp32 = index.fp32_baseline_bytes()
    compressed_bytes = index.compressed_payload_bytes()

    result = {
        "version": __version__,
        "documents": len(docs),
        "embedding_dim": int(embeddings.shape[1]),
        "bits": args.bits,
        "fp32_bytes": fp32,
        "compressed_payload_bytes": compressed_bytes,
        "compression_ratio": fp32 / max(compressed_bytes, 1),
        "memory_saved_pct": (1.0 - compressed_bytes / fp32) * 100.0,
        "top_k": args.top_k,
        "rerank_top_k": args.rerank_top_k,
        "exact_overlap": exact_overlap(full_ids, compressed_ids, args.top_k),
    }
    if expected_label is not None:
        result["full_label_match"] = label_match_ratio(full_labels, expected_label, args.top_k)
        result["compressed_label_match"] = label_match_ratio(compressed_labels, expected_label, args.top_k)

    print(json.dumps(result, indent=2))
    return 0



def _coerce_outlier(value):
    if value is None:
        return None
    if isinstance(value, str) and value.lower() == "auto":
        return "auto"
    return int(value)


def kv_estimate(args: argparse.Namespace) -> int:
    estimate = estimate_kv_cache_memory(
        layers=args.layers,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        dtype_bytes=args.dtype_bytes,
        key_bits=args.key_bits,
        value_bits=args.value_bits,
        key_outlier_bits=args.key_outlier_bits,
        value_outlier_bits=args.value_outlier_bits,
        n_key_outliers=_coerce_outlier(args.n_key_outliers),
        n_value_outliers=_coerce_outlier(args.n_value_outliers),
        key_recent_window=args.key_recent_window,
        value_recent_window=args.value_recent_window,
    )
    print(json.dumps(estimate.to_dict(), indent=2))
    return 0


def kv_bench(args: argparse.Namespace) -> int:
    cfg = KVBenchConfig(
        model=args.model,
        cache_type="hybrid",
        preset=args.preset,
        prompt_mode=args.prompt_mode,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        dtype=args.dtype,
        kl_tokens=args.kl_tokens,
        skip_kl=args.skip_kl,
        key_bits=args.key_bits,
        value_bits=args.value_bits,
        key_recent_window=args.key_recent_window,
        value_recent_window=args.value_recent_window,
        key_outlier_bits=args.key_outlier_bits,
        value_outlier_bits=args.value_outlier_bits,
        n_key_outliers=_coerce_outlier(args.n_key_outliers),
        n_value_outliers=_coerce_outlier(args.n_value_outliers),
        per_head_calibration=args.per_head_calibration,
    )
    report = run_kv_benchmark(cfg)
    if args.report_json:
        save_json_report(report, args.report_json)
    if args.report_md:
        save_markdown_report(report, args.report_md)
    print(json.dumps(report, indent=2))
    return 0


def page_attn_bench(args: argparse.Namespace) -> int:
    cfg = PageAttentionBenchConfig(
        batch_size=args.batch_size,
        heads=args.heads,
        query_len=args.query_len,
        seq_len=args.seq_len,
        head_dim=args.head_dim,
        page_size=args.page_size,
        device=args.device,
        dtype=args.dtype,
        warmup=args.warmup,
        repeats=args.repeats,
        seed=args.seed,
    )
    report = run_page_attention_benchmark(cfg)
    if args.report_json:
        save_page_attention_json(report, args.report_json)
    if args.report_md:
        save_page_attention_markdown(report, args.report_md)
    print(json.dumps(report, indent=2))
    return 0


def _write_optional_serving_reports(data, markdown_text, json_path=None, md_path=None):
    if json_path:
        save_serving_json(data, json_path)
    if md_path:
        save_serving_markdown(markdown_text, md_path)


def serving_sim(args: argparse.Namespace) -> int:
    sim = PagedKVServingSimulator(
        layers=args.layers,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        page_size=args.page_size,
        dtype_bytes=args.dtype_bytes,
        preset=args.preset,
        gpu_memory_budget_gb=args.gpu_memory_gb,
        model_weight_gb=args.model_weight_gb,
        key_bits=args.key_bits,
        value_bits=args.value_bits,
        key_outlier_bits=args.key_outlier_bits,
        value_outlier_bits=args.value_outlier_bits,
        n_key_outliers=_coerce_outlier(args.n_key_outliers),
        n_value_outliers=_coerce_outlier(args.n_value_outliers),
        key_recent_window=args.key_recent_window,
        value_recent_window=args.value_recent_window,
    )
    if args.sequence:
        for item in args.sequence:
            parts = item.split(":")
            if len(parts) not in {2, 3}:
                raise SystemExit("--sequence must be seq_id:prompt_tokens[:decode_tokens]")
            seq_id = parts[0]
            prompt = int(parts[1])
            decode = int(parts[2]) if len(parts) == 3 else args.decode_tokens
            sim.add_sequence(seq_id, prompt, decode)
    else:
        sim.add_uniform_sequences(
            users=args.users,
            prompt_tokens=args.prompt_tokens,
            decode_tokens=args.decode_tokens,
        )
    report = sim.memory_report()
    data = report.to_dict(include_per_sequence=args.include_per_sequence)
    _write_optional_serving_reports(data, serving_markdown_report(report), args.report_json, args.report_md)
    print(json.dumps(data, indent=2))
    return 0


def serving_capacity(args: argparse.Namespace) -> int:
    cap = estimate_serving_capacity(
        gpu_memory_gb=args.gpu_memory_gb,
        model_weight_gb=args.model_weight_gb,
        layers=args.layers,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        avg_prompt_tokens=args.avg_prompt_tokens,
        avg_decode_tokens=args.avg_decode_tokens,
        page_size=args.page_size,
        dtype_bytes=args.dtype_bytes,
        preset=args.preset,
        key_bits=args.key_bits,
        value_bits=args.value_bits,
        key_outlier_bits=args.key_outlier_bits,
        value_outlier_bits=args.value_outlier_bits,
        n_key_outliers=_coerce_outlier(args.n_key_outliers),
        n_value_outliers=_coerce_outlier(args.n_value_outliers),
        key_recent_window=args.key_recent_window,
        value_recent_window=args.value_recent_window,
    )
    data = cap.to_dict()
    _write_optional_serving_reports(data, serving_markdown_report(cap), args.report_json, args.report_md)
    print(json.dumps(data, indent=2))
    return 0


def decode_growth(args: argparse.Namespace) -> int:
    points = simulate_decode_growth(
        users=args.users,
        prompt_tokens=args.prompt_tokens,
        decode_tokens=args.decode_tokens,
        layers=args.layers,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        page_size=args.page_size,
        dtype_bytes=args.dtype_bytes,
        preset=args.preset,
        step_interval=args.step_interval,
        gpu_memory_gb=args.gpu_memory_gb,
        model_weight_gb=args.model_weight_gb,
        key_bits=args.key_bits,
        value_bits=args.value_bits,
        key_outlier_bits=args.key_outlier_bits,
        value_outlier_bits=args.value_outlier_bits,
        n_key_outliers=_coerce_outlier(args.n_key_outliers),
        n_value_outliers=_coerce_outlier(args.n_value_outliers),
        key_recent_window=args.key_recent_window,
        value_recent_window=args.value_recent_window,
    )
    data = [p.to_dict() for p in points]
    if args.report_json:
        save_serving_json(data, args.report_json)
    if args.report_md:
        save_serving_markdown(decode_growth_markdown_report(points), args.report_md)
    if args.report_csv:
        save_decode_growth_csv(points, args.report_csv)
    print(json.dumps(data, indent=2))
    return 0


def layout_bench(args: argparse.Namespace) -> int:
    cfg = LayoutBenchConfig(
        batch_size=args.batch_size,
        heads=args.heads,
        query_len=args.query_len,
        seq_len=args.seq_len,
        head_dim=args.head_dim,
        page_size=args.page_size,
        key_bits=args.key_bits,
        value_bits=args.value_bits,
        preset=args.preset,
        quantization_mode=args.quantization_mode,
        device=args.device,
        dtype=args.dtype,
        warmup=args.warmup,
        repeats=args.repeats,
        seed=args.seed,
    )
    report = run_layout_benchmark(cfg)
    if args.report_json:
        save_layout_json(report, args.report_json)
    if args.report_md:
        save_layout_markdown(report, args.report_md)
    print(json.dumps(report, indent=2))
    return 0



def layout_sweep(args: argparse.Namespace) -> int:
    cfg = LayoutSweepConfig(
        batch_size=args.batch_size,
        heads=args.heads,
        query_len=args.query_len,
        seq_len=args.seq_len,
        head_dim=args.head_dim,
        page_size=args.page_size,
        bit_pairs=args.bit_pairs,
        quantization_mode=args.quantization_mode,
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
    )
    report = run_layout_quality_sweep(cfg)
    if args.report_json:
        save_layout_json(report, args.report_json)
    if args.report_md:
        save_layout_sweep_markdown(report, args.report_md)
    print(json.dumps(report, indent=2))
    return 0



def residual_sweep(args: argparse.Namespace) -> int:
    cfg = ResidualSweepConfig(
        batch_size=args.batch_size,
        heads=args.heads,
        query_len=args.query_len,
        seq_len=args.seq_len,
        head_dim=args.head_dim,
        page_size=args.page_size,
        bit_pairs=args.bit_pairs,
        modes=args.modes,
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
    )
    report = run_residual_correction_sweep(cfg)
    if args.report_json:
        save_layout_json(report, args.report_json)
    if args.report_md:
        save_residual_sweep_markdown(report, args.report_md)
    print(json.dumps(report, indent=2))
    return 0

def rotate_q_check(args: argparse.Namespace) -> int:
    report = run_rotate_q_check(
        batch_size=args.batch_size,
        heads=args.heads,
        query_len=args.query_len,
        seq_len=args.seq_len,
        head_dim=args.head_dim,
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2))
    return 0

def fused_decode_bench(args: argparse.Namespace) -> int:
    cfg = FusedDecodeBenchConfig(
        batch_size=args.batch_size,
        heads=args.heads,
        query_len=args.query_len,
        seq_len=args.seq_len,
        head_dim=args.head_dim,
        page_size=args.page_size,
        key_bits=args.key_bits,
        value_bits=args.value_bits,
        preset=args.preset,
        quantization_mode=args.quantization_mode,
        device=args.device,
        dtype=args.dtype,
        warmup=args.warmup,
        repeats=args.repeats,
        seed=args.seed,
        prefer_triton=not args.no_triton,
        kernel_block_m=args.kernel_block_m,
        kernel_num_warps=args.kernel_num_warps,
        tune_kernel=args.tune_kernel,
        tune_block_m_values=tuple(int(x) for x in args.tune_block_m_values),
        tune_num_warps=args.tune_num_warps,
        tune_num_warps_values=tuple(int(x) for x in args.tune_num_warps_values),
        tune_page_size=args.tune_page_size,
        tune_page_size_values=tuple(int(x) for x in args.tune_page_size_values),
        cuda_graph=args.cuda_graph,
        graph_replays=args.graph_replays,
        competitiveness_target=args.competitiveness_target,
    )
    report = run_fused_decode_benchmark(cfg)
    if args.report_json:
        save_fused_decode_json(report, args.report_json)
    if args.report_md:
        save_fused_decode_markdown(report, args.report_md)
    print(json.dumps(report, indent=2))
    return 0


def long_context_compare(args: argparse.Namespace) -> int:
    cfg = LongContextCompareConfig(
        batch_size=args.batch_size,
        heads=args.heads,
        query_len=args.query_len,
        seq_lens=tuple(int(x) for x in args.seq_lens),
        head_dim=args.head_dim,
        page_size=args.page_size,
        presets=tuple(str(x) for x in args.presets),
        device=args.device,
        dtype=args.dtype,
        warmup=args.warmup,
        repeats=args.repeats,
        seed=args.seed,
        prefer_triton=not args.no_triton,
        kernel_block_m=args.kernel_block_m,
        kernel_num_warps=args.kernel_num_warps,
        tune_kernel=args.tune_kernel,
        tune_block_m_values=tuple(int(x) for x in args.tune_block_m_values),
        tune_num_warps=args.tune_num_warps,
        tune_num_warps_values=tuple(int(x) for x in args.tune_num_warps_values),
        tune_page_size=args.tune_page_size,
        tune_page_size_values=tuple(int(x) for x in args.tune_page_size_values),
        cuda_graph=args.cuda_graph,
        graph_replays=args.graph_replays,
        competitiveness_target=args.competitiveness_target,
    )
    report = run_long_context_comparison(cfg)
    if args.report_json:
        save_long_context_json(report, args.report_json)
    if args.report_md:
        save_long_context_markdown(report, args.report_md)
    print(json.dumps(report, indent=2))
    return 0


def split_k_compare(args: argparse.Namespace) -> int:
    cfg = SplitKCompareConfig(
        batch_size=args.batch_size,
        heads=args.heads,
        query_len=args.query_len,
        seq_lens=tuple(int(x) for x in args.seq_lens),
        head_dim=args.head_dim,
        page_size=args.page_size,
        preset=args.preset,
        device=args.device,
        dtype=args.dtype,
        warmup=args.warmup,
        repeats=args.repeats,
        seed=args.seed,
        prefer_triton=not args.no_triton,
        kernel_block_m=args.kernel_block_m,
        kernel_num_warps=args.kernel_num_warps,
        tune_kernel=args.tune_kernel,
        tune_block_m_values=tuple(int(x) for x in args.tune_block_m_values),
        tune_num_warps=args.tune_num_warps,
        tune_num_warps_values=tuple(int(x) for x in args.tune_num_warps_values),
        cuda_graph=args.cuda_graph,
        graph_replays=args.graph_replays,
        split_k_slabs=tuple(int(x) for x in args.split_k_slabs),
        reduce_overhead_fraction=args.reduce_overhead_fraction,
        competitiveness_target=args.competitiveness_target,
    )
    report = run_split_k_comparison(cfg)
    if args.report_json:
        save_split_k_json(report, args.report_json)
    if args.report_md:
        save_split_k_markdown(report, args.report_md)
    print(json.dumps(report, indent=2))
    return 0


def end_to_end_decode(args: argparse.Namespace) -> int:
    cfg = EndToEndDecodeBenchConfig(
        batch_size=args.batch_size,
        heads=args.heads,
        query_len=args.query_len,
        prompt_lens=tuple(int(x) for x in args.prompt_lens),
        decode_steps=args.decode_steps,
        head_dim=args.head_dim,
        page_size=args.page_size,
        preset=args.preset,
        device=args.device,
        dtype=args.dtype,
        warmup=args.warmup,
        repeats=args.repeats,
        seed=args.seed,
        prefer_triton=not args.no_triton,
        kernel_block_m=args.kernel_block_m,
        kernel_num_warps=args.kernel_num_warps,
        tune_kernel=args.tune_kernel,
        tune_block_m_values=tuple(int(x) for x in args.tune_block_m_values),
        tune_num_warps=args.tune_num_warps,
        tune_num_warps_values=tuple(int(x) for x in args.tune_num_warps_values),
        split_k_slabs=tuple(int(x) for x in args.split_k_slabs),
        cuda_graph=args.cuda_graph,
        graph_replays=args.graph_replays,
        include_single_pass_loop=not args.no_single_pass_loop,
        competitiveness_target=args.competitiveness_target,
    )
    report = run_end_to_end_decode_benchmark(cfg)
    if args.report_json:
        save_end_to_end_decode_json(report, args.report_json)
    if args.report_md:
        save_end_to_end_decode_markdown(report, args.report_md)
    print(json.dumps(report, indent=2))
    return 0

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tiny-tq", description="tiny-turboquant utilities")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("version", help="print package version")

    rb = sub.add_parser("rag-bench", help="run a small compressed RAG benchmark")
    rb.add_argument("--synthetic", action="store_true", help="use built-in synthetic corpus")
    rb.add_argument("--docs-per-topic", type=int, default=100)
    rb.add_argument("--input-jsonl", type=str, default=None)
    rb.add_argument("--input-folder", type=str, default=None)
    rb.add_argument("--text-field", type=str, default="text")
    rb.add_argument("--query", type=str, default=None)
    rb.add_argument("--expected-label", type=str, default=None)
    rb.add_argument("--embedding-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    rb.add_argument("--bits", type=int, default=4)
    rb.add_argument("--top-k", type=int, default=10)
    rb.add_argument("--rerank-top-k", type=int, default=50)
    rb.add_argument("--batch-size", type=int, default=64)
    rb.set_defaults(func=rag_bench)

    ke = sub.add_parser("kv-estimate", help="estimate dense vs hybrid KV-cache memory")
    ke.add_argument("--layers", type=int, required=True)
    ke.add_argument("--kv-heads", type=int, required=True)
    ke.add_argument("--head-dim", type=int, required=True)
    ke.add_argument("--seq-len", type=int, required=True)
    ke.add_argument("--batch-size", type=int, default=1)
    ke.add_argument("--dtype-bytes", type=int, default=2)
    ke.add_argument("--key-bits", type=int, default=6)
    ke.add_argument("--value-bits", type=int, default=4)
    ke.add_argument("--key-outlier-bits", type=int, default=8)
    ke.add_argument("--value-outlier-bits", type=int, default=8)
    ke.add_argument("--n-key-outliers", default="auto")
    ke.add_argument("--n-value-outliers", default="auto")
    ke.add_argument("--key-recent-window", type=int, default=128)
    ke.add_argument("--value-recent-window", type=int, default=64)
    ke.set_defaults(func=kv_estimate)

    kb = sub.add_parser("kv-bench", help="compare DynamicCache vs HybridTurboQuantKVCache on a real LLM")
    kb.add_argument("--model", type=str, required=True)
    kb.add_argument("--preset", type=str, default="balanced", choices=available_kv_presets())
    kb.add_argument("--prompt-mode", type=str, default="short", choices=["short", "medium", "long", "stress"])
    kb.add_argument("--prompt", type=str, default=None)
    kb.add_argument("--max-new-tokens", type=int, default=16)
    kb.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    kb.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"])
    kb.add_argument("--kl-tokens", type=int, default=4)
    kb.add_argument("--skip-kl", action="store_true")
    kb.add_argument("--key-bits", type=int, default=None)
    kb.add_argument("--value-bits", type=int, default=None)
    kb.add_argument("--key-recent-window", type=int, default=None)
    kb.add_argument("--value-recent-window", type=int, default=None)
    kb.add_argument("--key-outlier-bits", type=int, default=None)
    kb.add_argument("--value-outlier-bits", type=int, default=None)
    kb.add_argument("--n-key-outliers", default=None)
    kb.add_argument("--n-value-outliers", default=None)
    kb.add_argument("--per-head-calibration", action="store_true", default=None)
    kb.add_argument("--report-json", type=str, default=None)
    kb.add_argument("--report-md", type=str, default=None)
    kb.set_defaults(func=kv_bench)


    pab = sub.add_parser("page-attn-bench", help="benchmark dense vs page-wise attention research paths")
    pab.add_argument("--batch-size", type=int, default=1)
    pab.add_argument("--heads", type=int, default=8)
    pab.add_argument("--query-len", type=int, default=1)
    pab.add_argument("--seq-len", type=int, default=512)
    pab.add_argument("--head-dim", type=int, default=64)
    pab.add_argument("--page-size", type=int, default=128)
    pab.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    pab.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"])
    pab.add_argument("--warmup", type=int, default=2)
    pab.add_argument("--repeats", "--iters", dest="repeats", type=int, default=5)
    pab.add_argument("--seed", type=int, default=123)
    pab.add_argument("--report-json", type=str, default=None)
    pab.add_argument("--report-md", type=str, default=None)
    pab.set_defaults(func=page_attn_bench)


    lb = sub.add_parser("layout-bench", help="benchmark v0.8 compressed KV page layout and rotate-Q reference path")
    lb.add_argument("--batch-size", type=int, default=1)
    lb.add_argument("--heads", type=int, default=8)
    lb.add_argument("--query-len", type=int, default=1)
    lb.add_argument("--seq-len", type=int, default=1024)
    lb.add_argument("--head-dim", type=int, default=64)
    lb.add_argument("--page-size", type=int, default=128)
    lb.add_argument("--key-bits", type=int, default=8)
    lb.add_argument("--value-bits", type=int, default=6)
    lb.add_argument("--preset", type=str, default="safe-layout", choices=available_layout_presets() + available_kv_presets())
    lb.add_argument("--quantization-mode", type=str, default="affine", choices=["affine", "residual-affine", "codebook"])
    lb.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    lb.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"])
    lb.add_argument("--warmup", type=int, default=1)
    lb.add_argument("--repeats", "--iters", dest="repeats", type=int, default=3)
    lb.add_argument("--seed", type=int, default=123)
    lb.add_argument("--report-json", type=str, default=None)
    lb.add_argument("--report-md", type=str, default=None)
    lb.set_defaults(func=layout_bench)


    ls = sub.add_parser("layout-sweep", help="sweep compressed KV page bit settings and quality diagnostics")
    ls.add_argument("--batch-size", type=int, default=1)
    ls.add_argument("--heads", type=int, default=8)
    ls.add_argument("--query-len", type=int, default=1)
    ls.add_argument("--seq-len", type=int, default=512)
    ls.add_argument("--head-dim", type=int, default=64)
    ls.add_argument("--page-size", type=int, default=128)
    ls.add_argument("--bit-pairs", nargs="+", default=["8,8", "8,6", "8,4", "6,4", "4,4"], help="K,V bit pairs such as 8,6 or 6/4")
    ls.add_argument("--quantization-mode", type=str, default="affine", choices=["affine", "residual-affine", "codebook"])
    ls.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ls.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"])
    ls.add_argument("--seed", type=int, default=123)
    ls.add_argument("--report-json", type=str, default=None)
    ls.add_argument("--report-md", type=str, default=None)
    ls.set_defaults(func=layout_sweep)




    rs = sub.add_parser("residual-sweep", help="compare affine KV pages against v0.10 residual-affine correction")
    rs.add_argument("--batch-size", type=int, default=1)
    rs.add_argument("--heads", type=int, default=8)
    rs.add_argument("--query-len", type=int, default=1)
    rs.add_argument("--seq-len", type=int, default=1024)
    rs.add_argument("--head-dim", type=int, default=64)
    rs.add_argument("--page-size", type=int, default=128)
    rs.add_argument("--bit-pairs", nargs="+", default=["8,6", "6,4", "4,4"], help="K,V bit pairs such as 8,6 or 6/4")
    rs.add_argument("--modes", nargs="+", default=["affine", "residual-affine"], choices=["affine", "residual-affine", "codebook"], help="quantization modes to compare")
    rs.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    rs.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"])
    rs.add_argument("--seed", type=int, default=123)
    rs.add_argument("--report-json", type=str, default=None)
    rs.add_argument("--report-md", type=str, default=None)
    rs.set_defaults(func=residual_sweep)

    rq = sub.add_parser("rotate-q-check", help="validate rotate-Q dense attention equivalence")
    rq.add_argument("--batch-size", type=int, default=1)
    rq.add_argument("--heads", type=int, default=8)
    rq.add_argument("--query-len", type=int, default=1)
    rq.add_argument("--seq-len", type=int, default=1024)
    rq.add_argument("--head-dim", type=int, default=64)
    rq.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    rq.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"])
    rq.add_argument("--seed", type=int, default=123)
    rq.set_defaults(func=rotate_q_check)


    fb = sub.add_parser("fused-decode-bench", help="benchmark v0.9 experimental compressed decode-attention path")
    fb.add_argument("--batch-size", type=int, default=1)
    fb.add_argument("--heads", type=int, default=8)
    fb.add_argument("--query-len", type=int, default=1)
    fb.add_argument("--seq-len", type=int, default=2048)
    fb.add_argument("--head-dim", type=int, default=64)
    fb.add_argument("--page-size", type=int, default=128)
    fb.add_argument("--key-bits", type=int, default=8)
    fb.add_argument("--value-bits", type=int, default=6)
    fb.add_argument("--preset", type=str, default="safe-layout", choices=available_layout_presets())
    fb.add_argument("--quantization-mode", type=str, default="affine", choices=["affine", "residual-affine", "codebook"])
    fb.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    fb.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"])
    fb.add_argument("--warmup", type=int, default=1)
    fb.add_argument("--repeats", "--iters", dest="repeats", type=int, default=3)
    fb.add_argument("--seed", type=int, default=123)
    fb.add_argument("--kernel-block-m", type=int, default=64, choices=[8, 16, 32, 64, 128], help="Triton BLOCK_M candidate used when tuning is disabled")
    fb.add_argument("--kernel-num-warps", type=int, default=4, choices=[1, 2, 4, 8], help="Triton num_warps candidate used when warp tuning is disabled")
    fb.add_argument("--tune-kernel", action="store_true", help="benchmark several Triton BLOCK_M candidates and choose the fastest supported one")
    fb.add_argument("--tune-block-m-values", nargs="+", type=int, default=[8, 16, 32, 64, 128], help="candidate BLOCK_M values for --tune-kernel")
    fb.add_argument("--tune-num-warps", action="store_true", help="benchmark several Triton num_warps candidates and choose the fastest supported one")
    fb.add_argument("--tune-num-warps-values", nargs="+", type=int, default=[1, 2, 4, 8], help="candidate num_warps values for --tune-num-warps")
    fb.add_argument("--tune-page-size", action="store_true", help="benchmark several page sizes and choose the fastest fused path")
    fb.add_argument("--tune-page-size-values", nargs="+", type=int, default=[64, 128, 256], help="candidate page sizes for --tune-page-size; values must divide seq-len")
    fb.add_argument("--cuda-graph", action="store_true", help="try CUDA Graph capture/replay for the selected fused decode path")
    fb.add_argument("--graph-replays", type=int, default=100, help="number of CUDA Graph replays used when --cuda-graph is enabled")
    fb.add_argument("--competitiveness-target", type=str, default="sdpa", choices=["dense", "sdpa"], help="target baseline used in competitiveness summary")
    fb.add_argument("--no-triton", action="store_true", help="force the verified PyTorch compressed-page fallback")
    fb.add_argument("--report-json", type=str, default=None)
    fb.add_argument("--report-md", type=str, default=None)
    fb.set_defaults(func=fused_decode_bench)


    lc = sub.add_parser("long-context-compare", help="compare safe vs residual compressed decode layouts across context lengths")
    lc.add_argument("--batch-size", type=int, default=1)
    lc.add_argument("--heads", type=int, default=8)
    lc.add_argument("--query-len", type=int, default=1)
    lc.add_argument("--seq-lens", nargs="+", type=int, default=[2048, 4096, 8192, 16384], help="context lengths to compare")
    lc.add_argument("--head-dim", type=int, default=64)
    lc.add_argument("--page-size", type=int, default=256)
    lc.add_argument("--presets", nargs="+", default=["safe-layout", "residual-balanced", "residual-aggressive"], choices=available_layout_presets())
    lc.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    lc.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"])
    lc.add_argument("--warmup", type=int, default=1)
    lc.add_argument("--repeats", "--iters", dest="repeats", type=int, default=3)
    lc.add_argument("--seed", type=int, default=123)
    lc.add_argument("--kernel-block-m", type=int, default=64, choices=[8, 16, 32, 64, 128])
    lc.add_argument("--kernel-num-warps", type=int, default=4, choices=[1, 2, 4, 8])
    lc.add_argument("--tune-kernel", action="store_true")
    lc.add_argument("--tune-block-m-values", nargs="+", type=int, default=[8, 16, 32, 64, 128])
    lc.add_argument("--tune-num-warps", action="store_true")
    lc.add_argument("--tune-num-warps-values", nargs="+", type=int, default=[1, 2, 4, 8])
    lc.add_argument("--tune-page-size", action="store_true")
    lc.add_argument("--tune-page-size-values", nargs="+", type=int, default=[64, 128, 256])
    lc.add_argument("--cuda-graph", action="store_true")
    lc.add_argument("--graph-replays", type=int, default=100)
    lc.add_argument("--competitiveness-target", type=str, default="sdpa", choices=["dense", "sdpa"])
    lc.add_argument("--no-triton", action="store_true")
    lc.add_argument("--report-json", type=str, default=None)
    lc.add_argument("--report-md", type=str, default=None)
    lc.set_defaults(func=long_context_compare)


    sk = sub.add_parser("split-k-compare", help="v0.10.8 split-K / sequence-parallel measured diagnostic")
    sk.add_argument("--batch-size", type=int, default=1)
    sk.add_argument("--heads", type=int, default=8)
    sk.add_argument("--query-len", type=int, default=1)
    sk.add_argument("--seq-lens", nargs="+", type=int, default=[8192, 16384, 32768])
    sk.add_argument("--head-dim", type=int, default=64)
    sk.add_argument("--page-size", type=int, default=256)
    sk.add_argument("--preset", type=str, default="safe-layout", choices=available_layout_presets())
    sk.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    sk.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"])
    sk.add_argument("--warmup", type=int, default=1)
    sk.add_argument("--repeats", "--iters", dest="repeats", type=int, default=3)
    sk.add_argument("--seed", type=int, default=123)
    sk.add_argument("--kernel-block-m", type=int, default=64, choices=[8, 16, 32, 64, 128])
    sk.add_argument("--kernel-num-warps", type=int, default=4, choices=[1, 2, 4, 8])
    sk.add_argument("--tune-kernel", action="store_true")
    sk.add_argument("--tune-block-m-values", nargs="+", type=int, default=[8, 16, 32, 64, 128])
    sk.add_argument("--tune-num-warps", action="store_true")
    sk.add_argument("--tune-num-warps-values", nargs="+", type=int, default=[1, 2, 4, 8])
    sk.add_argument("--cuda-graph", action="store_true")
    sk.add_argument("--graph-replays", type=int, default=100)
    sk.add_argument("--split-k-slabs", nargs="+", type=int, default=[2, 4, 8, 16])
    sk.add_argument("--reduce-overhead-fraction", type=float, default=0.12)
    sk.add_argument("--competitiveness-target", type=str, default="sdpa", choices=["dense", "sdpa"])
    sk.add_argument("--no-triton", action="store_true")
    sk.add_argument("--report-json", type=str, default=None)
    sk.add_argument("--report-md", type=str, default=None)
    sk.set_defaults(func=split_k_compare)


    ed = sub.add_parser("end-to-end-decode", help="v0.11.0 fixed-cache repeated decode-loop diagnostic")
    ed.add_argument("--batch-size", type=int, default=1)
    ed.add_argument("--heads", type=int, default=8)
    ed.add_argument("--query-len", type=int, default=1)
    ed.add_argument("--prompt-lens", nargs="+", type=int, default=[16384, 32768])
    ed.add_argument("--decode-steps", type=int, default=32)
    ed.add_argument("--head-dim", type=int, default=64)
    ed.add_argument("--page-size", type=int, default=256)
    ed.add_argument("--preset", type=str, default="safe-layout", choices=available_layout_presets())
    ed.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ed.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"])
    ed.add_argument("--warmup", type=int, default=1)
    ed.add_argument("--repeats", "--iters", dest="repeats", type=int, default=3)
    ed.add_argument("--seed", type=int, default=123)
    ed.add_argument("--kernel-block-m", type=int, default=64, choices=[8, 16, 32, 64, 128])
    ed.add_argument("--kernel-num-warps", type=int, default=4, choices=[1, 2, 4, 8])
    ed.add_argument("--tune-kernel", action="store_true")
    ed.add_argument("--tune-block-m-values", nargs="+", type=int, default=[8, 16, 32, 64, 128])
    ed.add_argument("--tune-num-warps", action="store_true")
    ed.add_argument("--tune-num-warps-values", nargs="+", type=int, default=[1, 2, 4, 8])
    ed.add_argument("--split-k-slabs", nargs="+", type=int, default=[2, 4, 8, 16])
    ed.add_argument("--cuda-graph", action="store_true")
    ed.add_argument("--graph-replays", type=int, default=100)
    ed.add_argument("--competitiveness-target", type=str, default="sdpa", choices=["dense", "sdpa"])
    ed.add_argument("--no-triton", action="store_true")
    ed.add_argument("--no-single-pass-loop", action="store_true")
    ed.add_argument("--report-json", type=str, default=None)
    ed.add_argument("--report-md", type=str, default=None)
    ed.set_defaults(func=end_to_end_decode)


    ss = sub.add_parser("serving-sim", help="simulate serving-style paged KV-cache memory for active users")
    ss.add_argument("--users", type=int, default=32)
    ss.add_argument("--prompt-tokens", type=int, default=2048)
    ss.add_argument("--decode-tokens", type=int, default=128)
    ss.add_argument("--sequence", action="append", default=None, help="seq_id:prompt_tokens[:decode_tokens]; may be repeated")
    ss.add_argument("--layers", type=int, required=True)
    ss.add_argument("--kv-heads", type=int, required=True)
    ss.add_argument("--head-dim", type=int, required=True)
    ss.add_argument("--page-size", type=int, default=128)
    ss.add_argument("--dtype-bytes", type=int, default=2)
    ss.add_argument("--preset", type=str, default="balanced", choices=available_kv_presets())
    ss.add_argument("--gpu-memory-gb", type=float, default=None)
    ss.add_argument("--model-weight-gb", type=float, default=0.0)
    ss.add_argument("--key-bits", type=int, default=None)
    ss.add_argument("--value-bits", type=int, default=None)
    ss.add_argument("--key-outlier-bits", type=int, default=None)
    ss.add_argument("--value-outlier-bits", type=int, default=None)
    ss.add_argument("--n-key-outliers", default=None)
    ss.add_argument("--n-value-outliers", default=None)
    ss.add_argument("--key-recent-window", type=int, default=None)
    ss.add_argument("--value-recent-window", type=int, default=None)
    ss.add_argument("--report-json", type=str, default=None)
    ss.add_argument("--report-md", type=str, default=None)
    ss.add_argument(
        "--include-per-sequence",
        action="store_true",
        help="include the full per-sequence records in stdout/JSON; default is compact summary output",
    )
    ss.set_defaults(func=serving_sim)

    sc = sub.add_parser("serving-capacity", help="estimate max concurrent users under a KV memory budget")
    sc.add_argument("--gpu-memory-gb", type=float, required=True)
    sc.add_argument("--model-weight-gb", type=float, required=True)
    sc.add_argument("--layers", type=int, required=True)
    sc.add_argument("--kv-heads", type=int, required=True)
    sc.add_argument("--head-dim", type=int, required=True)
    sc.add_argument("--avg-prompt-tokens", type=int, default=2048)
    sc.add_argument("--avg-decode-tokens", type=int, default=128)
    sc.add_argument("--page-size", type=int, default=128)
    sc.add_argument("--dtype-bytes", type=int, default=2)
    sc.add_argument("--preset", type=str, default="balanced", choices=available_kv_presets())
    sc.add_argument("--key-bits", type=int, default=None)
    sc.add_argument("--value-bits", type=int, default=None)
    sc.add_argument("--key-outlier-bits", type=int, default=None)
    sc.add_argument("--value-outlier-bits", type=int, default=None)
    sc.add_argument("--n-key-outliers", default=None)
    sc.add_argument("--n-value-outliers", default=None)
    sc.add_argument("--key-recent-window", type=int, default=None)
    sc.add_argument("--value-recent-window", type=int, default=None)
    sc.add_argument("--report-json", type=str, default=None)
    sc.add_argument("--report-md", type=str, default=None)
    sc.set_defaults(func=serving_capacity)

    dg = sub.add_parser("decode-growth", help="simulate KV-cache memory growth over decode steps")
    dg.add_argument("--users", type=int, default=16)
    dg.add_argument("--prompt-tokens", type=int, default=2048)
    dg.add_argument("--decode-tokens", type=int, default=256)
    dg.add_argument("--step-interval", type=int, default=64)
    dg.add_argument("--layers", type=int, required=True)
    dg.add_argument("--kv-heads", type=int, required=True)
    dg.add_argument("--head-dim", type=int, required=True)
    dg.add_argument("--page-size", type=int, default=128)
    dg.add_argument("--dtype-bytes", type=int, default=2)
    dg.add_argument("--preset", type=str, default="balanced", choices=available_kv_presets())
    dg.add_argument("--gpu-memory-gb", type=float, default=None)
    dg.add_argument("--model-weight-gb", type=float, default=0.0)
    dg.add_argument("--key-bits", type=int, default=None)
    dg.add_argument("--value-bits", type=int, default=None)
    dg.add_argument("--key-outlier-bits", type=int, default=None)
    dg.add_argument("--value-outlier-bits", type=int, default=None)
    dg.add_argument("--n-key-outliers", default=None)
    dg.add_argument("--n-value-outliers", default=None)
    dg.add_argument("--key-recent-window", type=int, default=None)
    dg.add_argument("--value-recent-window", type=int, default=None)
    dg.add_argument("--report-json", type=str, default=None)
    dg.add_argument("--report-md", type=str, default=None)
    dg.add_argument("--report-csv", type=str, default=None)
    dg.set_defaults(func=decode_growth)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "version":
        print(__version__)
        return 0
    if hasattr(args, "func"):
        return int(args.func(args))
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
