from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tiny_turboquant import TurboQuantKVCache, TurboQuantMSE, TurboQuantProd
from tiny_turboquant.bitpack import pack_indices, unpack_indices
from tiny_turboquant.fwht import fwht
from tiny_turboquant.outlier_split import OutlierSplitTurboQuant
from tiny_turboquant.rotation import RandomRotation


def _unit(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def test_bitpack_roundtrip_all_supported_bits():
    g = torch.Generator().manual_seed(0)
    for bits in range(1, 9):
        idx = torch.randint(0, 1 << bits, (3, 5, 17), generator=g, dtype=torch.uint8)
        packed, shape = pack_indices(idx, bits)
        got = unpack_indices(packed, bits, shape)
        assert packed.dtype == torch.uint8
        assert tuple(got.shape) == tuple(idx.shape)
        assert torch.equal(got, idx)


def test_fwht_is_orthonormal():
    x = torch.randn(4, 64)
    y = fwht(fwht(x), force_torch=True)
    assert torch.allclose(x, y, atol=1e-5, rtol=1e-5)


def test_random_rotation_inverse():
    x = torch.randn(16, 48)
    rot = RandomRotation.make(48, seed=123)
    y = rot.apply(x)
    got = rot.apply_T(y)
    assert got.shape == x.shape
    assert torch.allclose(x, got, atol=1e-5, rtol=1e-5)


def test_mse_quantizer_shape_and_reasonable_distortion():
    x = _unit(torch.randn(256, 64))
    q = TurboQuantMSE.build(64, bits=4, seed=0)
    idx = q.quant(x)
    xh = q.dequant(idx)
    mse = ((x - xh) ** 2).sum(-1).mean().item()
    assert idx.dtype == torch.uint8
    assert idx.shape[-1] == q.d_pad
    assert xh.shape == x.shape
    assert mse < 0.08


def test_prod_quantizer_shape():
    x = _unit(torch.randn(64, 32))
    q = TurboQuantProd.build(32, bits=3, seed=0)
    idx, signs, gamma = q.quant(x)
    xh = q.dequant(idx, signs, gamma)
    assert idx.dtype == torch.uint8
    assert signs.dtype == torch.int8
    assert gamma.shape == (64, 1)
    assert xh.shape == x.shape


def test_outlier_split_roundtrip_shape():
    x = _unit(torch.randn(128, 64))
    q = OutlierSplitTurboQuant.calibrate(x, n_out=16, bits_out=3, bits_reg=2, seed=0)
    idx_out, idx_reg, n_out, n_reg = q.quant(x)
    xh = q.dequant(idx_out, idx_reg, n_out, n_reg)
    assert idx_out.dtype == torch.uint8
    assert idx_reg.dtype == torch.uint8
    assert xh.shape == x.shape
    assert 2.0 < q.effective_bits < 3.0


def test_kv_cache_uniform_actual_memory_less_than_fp16():
    torch.manual_seed(0)
    layers, batch, heads, seq, dim = 2, 1, 4, 16, 32
    cache = TurboQuantKVCache(bits=4)
    for layer in range(layers):
        k = torch.randn(batch, heads, seq, dim)
        v = torch.randn(batch, heads, seq, dim)
        kr, vr = cache.update(k, v, layer)
        assert kr.shape == k.shape
        assert vr.shape == v.shape

    for layer in range(layers):
        k = torch.randn(batch, heads, 1, dim)
        v = torch.randn(batch, heads, 1, dim)
        kr, vr = cache.update(k, v, layer)
        assert kr.shape[-2] == seq + 1
        assert vr.shape[-2] == seq + 1

    assert cache.get_seq_length(0) == seq + 1
    assert cache.actual_memory_bytes() == cache.memory_bytes()
    assert cache.actual_memory_bytes() < cache.fp16_baseline_bytes()
    assert abs(cache.actual_memory_bytes() - cache.theoretical_memory_bytes()) < 64


def test_kv_cache_outlier_split_uses_real_packed_storage():
    torch.manual_seed(1)
    cache = TurboQuantKVCache(bits=2, bits_outlier=3, n_outlier=8)
    k = torch.randn(1, 2, 8, 32)
    v = torch.randn(1, 2, 8, 32) * 1.7
    kr, vr = cache.update(k, v, 0)
    assert kr.shape == k.shape
    assert vr.shape == v.shape
    assert cache.get_seq_length(0) == 8
    assert cache.actual_memory_bytes() < cache.fp16_baseline_bytes()


def test_hybrid_cache_keeps_recent_tokens_dense_and_compresses_old_tokens():
    torch.manual_seed(2)
    from tiny_turboquant import HybridTurboQuantKVCache

    cache = HybridTurboQuantKVCache(bits=4, recent_window=2)
    k = torch.randn(1, 2, 12, 32)
    v = torch.randn(1, 2, 12, 32)
    kr, vr = cache.update(k, v, 0)

    assert kr.shape == k.shape
    assert vr.shape == v.shape
    assert cache.get_seq_length(0) == 12
    assert cache.compressed_seq_length(0) == 10
    assert cache.recent_seq_length(0) == 2
    assert cache.actual_memory_bytes() < cache.fp16_baseline_bytes()

    k2 = torch.randn(1, 2, 1, 32)
    v2 = torch.randn(1, 2, 1, 32)
    kr2, vr2 = cache.update(k2, v2, 0)

    assert kr2.shape[-2] == 13
    assert vr2.shape[-2] == 13
    assert cache.compressed_seq_length(0) == 11
    assert cache.recent_seq_length(0) == 2


def test_hybrid_cache_separate_key_value_bits_and_outliers():
    torch.manual_seed(3)
    from tiny_turboquant import HybridTurboQuantKVCache

    cache = HybridTurboQuantKVCache(
        key_bits=6,
        value_bits=4,
        key_outlier_bits=5,
        value_outlier_bits=4,
        n_key_outliers=8,
        n_value_outliers=0,
        recent_window=1,
    )
    k = torch.randn(1, 2, 8, 32)
    v = torch.randn(1, 2, 8, 32)
    cache.update(k, v, 0)

    key_page = cache._key_pages[0][0]
    value_page = cache._value_pages[0][0]

    assert key_page["type"] == "outlier_split"
    assert key_page["bits_reg"] == 6
    assert key_page["bits_out"] == 5
    assert value_page["type"] == "uniform"
    assert value_page["bits"] == 4


def test_hybrid_cache_per_layer_calibration_builds_distinct_quantizers():
    torch.manual_seed(4)
    from tiny_turboquant import HybridTurboQuantKVCache

    cache = HybridTurboQuantKVCache(bits=4, recent_window=1, per_layer_calibration=True)
    for layer in range(2):
        k = torch.randn(1, 2, 6, 32) + layer
        v = torch.randn(1, 2, 6, 32) - layer
        cache.update(k, v, layer)

    assert cache._q_key_by_layer[0] is not None
    assert cache._q_key_by_layer[1] is not None
    assert cache._q_key_by_layer[0] is not cache._q_key_by_layer[1]
    assert cache.get_seq_length(0) == 6
    assert cache.get_seq_length(1) == 6


def test_outlier_count_is_clamped_for_head_dim():
    torch.manual_seed(5)
    from tiny_turboquant import HybridTurboQuantKVCache

    # head_dim=16, but n_key_outliers is intentionally too large.
    # v0.3 should clamp instead of raising.
    cache = HybridTurboQuantKVCache(
        key_bits=4,
        value_bits=4,
        key_outlier_bits=8,
        value_outlier_bits=8,
        n_key_outliers=64,
        n_value_outliers=64,
        recent_window=1,
    )
    k = torch.randn(1, 2, 8, 16)
    v = torch.randn(1, 2, 8, 16)
    kr, vr = cache.update(k, v, 0)
    assert kr.shape == k.shape
    assert vr.shape == v.shape
    assert cache.actual_memory_bytes() < cache.fp16_baseline_bytes()


def test_hybrid_cache_per_head_calibration_and_paged_attention():
    torch.manual_seed(6)
    from tiny_turboquant import HybridTurboQuantKVCache, dense_attention, attention_similarity

    cache = HybridTurboQuantKVCache(
        key_bits=6,
        value_bits=4,
        key_outlier_bits=8,
        value_outlier_bits=8,
        n_key_outliers=8,
        n_value_outliers=4,
        recent_window=2,
        per_layer_calibration=True,
        per_head_calibration=True,
    )
    k = torch.randn(1, 3, 12, 16)
    v = torch.randn(1, 3, 12, 16)
    kr, vr = cache.update(k, v, 0)

    assert kr.shape == k.shape
    assert vr.shape == v.shape
    assert isinstance(cache._q_key_by_layer[0], list)
    assert len(cache._q_key_by_layer[0]) == 3

    q = torch.randn(1, 3, 1, 16)
    paged = cache.paged_attention(q, 0)
    dense = dense_attention(q, kr, vr)
    metrics = attention_similarity(dense, paged)
    assert paged.shape == dense.shape
    assert metrics["relative_error"] < 1e-4
    assert metrics["cosine_similarity"] > 0.999


def test_streaming_paged_attention_matches_dense_attention_on_pages():
    torch.manual_seed(7)
    from tiny_turboquant import dense_attention, streaming_paged_attention

    q = torch.randn(2, 4, 3, 16)
    k1 = torch.randn(2, 4, 5, 16)
    v1 = torch.randn(2, 4, 5, 16)
    k2 = torch.randn(2, 4, 7, 16)
    v2 = torch.randn(2, 4, 7, 16)

    dense = dense_attention(q, torch.cat([k1, k2], dim=2), torch.cat([v1, v2], dim=2))
    stream = streaming_paged_attention(q, [(k1, v1), (k2, v2)])
    assert torch.allclose(dense, stream, atol=1e-5, rtol=1e-5)


def test_compressed_vector_index_search_and_rerank():
    torch.manual_seed(8)
    from tiny_turboquant import CompressedVectorIndex

    x = torch.randn(200, 32)
    ids = [f"doc-{i}" for i in range(200)]
    index = CompressedVectorIndex(bits=4, store_original_for_rerank=True).add(x, ids=ids)

    results = index.search(x[0], top_k=5)
    reranked = index.search(x[0], top_k=5, rerank_top_k=50)

    assert len(results) == 5
    assert len(reranked) == 5
    assert index.compressed_payload_bytes() < index.fp32_baseline_bytes()
    assert index.compression_ratio() > 5.0
    assert reranked[0].id == "doc-0"


def test_serving_adapter_plan_is_explicit_scaffold():
    from tiny_turboquant import PagedKVCacheSpec, VLLMExperimentAdapter

    spec = PagedKVCacheSpec(key_bits=6, value_bits=4, page_size=64)
    plan = VLLMExperimentAdapter(spec).integration_plan()
    assert plan["engine"] == "vLLM-style paged cache"
    assert plan["spec"]["page_size"] == 64
    assert "not a drop-in" in plan["status"]


def test_hybrid_cache_paged_attention_handles_different_key_value_recent_windows():
    torch.manual_seed(9)
    from tiny_turboquant import HybridTurboQuantKVCache, dense_attention, attention_similarity

    # Use uniform low-bit quantizers so the regression test stays fast.
    # The bug being tested is page alignment when K and V use different
    # recent windows, not outlier or per-head calibration quality.
    cache = HybridTurboQuantKVCache(
        key_bits=4,
        value_bits=4,
        n_key_outliers=0,
        n_value_outliers=0,
        key_recent_window=6,
        value_recent_window=3,
        per_layer_calibration=True,
        per_head_calibration=False,
    )
    k = torch.randn(1, 2, 20, 16)
    v = torch.randn(1, 2, 20, 16)
    kr, vr = cache.update(k, v, 0)

    assert kr.shape == k.shape
    assert vr.shape == v.shape
    assert cache.compressed_seq_length(0) == 14
    assert cache.recent_seq_length(0) == 6

    q = torch.randn(1, 2, 2, 16)
    paged = cache.paged_attention(q, 0, page_size=5)
    dense = dense_attention(q, kr, vr)
    metrics = attention_similarity(dense, paged)

    assert paged.shape == dense.shape
    assert metrics["relative_error"] < 1e-4
    assert metrics["cosine_similarity"] > 0.999


def test_retrieval_metrics_basic():
    from tiny_turboquant import recall_at_k, precision_at_k, mrr_at_k, ndcg_at_k, exact_overlap

    retrieved = ["a", "b", "c", "d"]
    relevant = {"b", "d", "x"}

    assert recall_at_k(retrieved, relevant, 4) == 2 / 3
    assert precision_at_k(retrieved, relevant, 2) == 0.5
    assert mrr_at_k(retrieved, relevant, 4) == 0.5
    assert 0.0 < ndcg_at_k(retrieved, relevant, 4) <= 1.0
    assert exact_overlap([1, 2, 3], [2, 3, 4], 3) == 2 / 3


def test_chunk_text_and_make_chunks():
    from tiny_turboquant import chunk_text, make_chunks

    chunks = chunk_text("abcdefghij", chunk_size=4, overlap=1)
    assert chunks == ["abcd", "defg", "ghij", "j"]

    docs = make_chunks(["hello world"], chunk_size=5, overlap=0, ids=["doc-a"])
    assert docs[0].id == "doc-a::chunk-0"
    assert docs[0].metadata["source_id"] == "doc-a"


def test_rag_compressed_index_from_embeddings_search_and_save_load(tmp_path):
    torch.manual_seed(10)
    from tiny_turboquant import RAGCompressedIndex, DocumentChunk

    chunks = [
        DocumentChunk(id=f"doc-{i}", text=f"document {i}", metadata={"topic": "x" if i < 5 else "y"})
        for i in range(20)
    ]
    x = torch.randn(20, 16)
    x[0] = torch.ones(16)
    q = torch.ones(16)

    rag = RAGCompressedIndex.from_embeddings(chunks, x, bits=4, store_original_for_rerank=True)
    results = rag.search(q, top_k=3, rerank_top_k=10)
    assert len(results) == 3
    assert results[0].id == "doc-0"
    report = rag.memory_report()
    assert report["payload_compression_ratio"] > 5.0

    path = tmp_path / "rag.ttq"
    rag.save(path)
    loaded = RAGCompressedIndex.load(path)
    loaded_results = loaded.search(q, top_k=3, rerank_top_k=10)
    assert loaded_results[0].id == "doc-0"


def test_compressed_vector_index_save_load(tmp_path):
    torch.manual_seed(11)
    from tiny_turboquant import CompressedVectorIndex

    x = torch.randn(50, 16)
    x[0] = torch.ones(16)
    idx = CompressedVectorIndex(bits=4, store_original_for_rerank=True).add(x, ids=[f"id-{i}" for i in range(50)])
    path = tmp_path / "index.ttq"
    idx.save(str(path))
    loaded = CompressedVectorIndex.load(str(path))
    res = loaded.search(torch.ones(16), top_k=3, rerank_top_k=10)
    assert res[0].id == "id-0"


def test_cli_version_invocation(capsys):
    from tiny_turboquant.cli import main
    from tiny_turboquant import __version__

    assert main(["version"]) == 0
    captured = capsys.readouterr()
    assert __version__ in captured.out


def test_kv_presets_and_auto_outliers():
    from tiny_turboquant import HybridTurboQuantKVCache, available_kv_presets

    assert "balanced" in available_kv_presets()
    cache = HybridTurboQuantKVCache.from_preset("balanced", key_recent_window=32)
    assert cache.key_bits == 6
    assert cache.value_bits == 4
    assert cache.key_recent_window == 32
    assert cache.n_key_outliers == "auto"

    k = torch.randn(1, 2, 8, 16)
    v = torch.randn(1, 2, 8, 16)
    cache.update(k, v, 0)
    counts = cache.resolved_outlier_counts(16)
    assert counts["key"] == 8
    assert counts["value"] == 4


def test_kv_memory_estimator_reduces_memory_for_long_context():
    from tiny_turboquant import estimate_kv_cache_memory

    est = estimate_kv_cache_memory(
        layers=24,
        kv_heads=8,
        head_dim=128,
        seq_len=4096,
        batch_size=4,
        key_bits=6,
        value_bits=4,
        key_recent_window=128,
        value_recent_window=64,
    )
    d = est.to_dict()
    assert d["fp16_bytes"] > d["compressed_estimated_bytes"]
    assert d["compression_ratio"] > 1.0
    assert d["memory_saved_pct"] > 0.0
    assert d["n_key_outliers"] == 32
    assert d["n_value_outliers"] == 16


def test_kv_bench_report_markdown_and_first_divergence():
    from tiny_turboquant import KVBenchConfig, first_divergence, markdown_report

    a = torch.tensor([1, 2, 3, 4])
    b = torch.tensor([1, 2, 9, 4])
    assert first_divergence(a, b) == 2

    report = {
        "version": "0.5.0",
        "model": "dummy",
        "preset": "balanced",
        "prompt_mode": "short",
        "prompt_tokens": 4,
        "max_new_tokens": 2,
        "device": "cpu",
        "dtype": "float32",
        "memory": {
            "fp16_bytes": 100,
            "compressed_bytes": 50,
            "compression_ratio": 2.0,
            "memory_saved_pct": 50.0,
        },
        "quality": {
            "mean_kl": 0.1,
            "max_kl": 0.2,
            "kl_positions": 2,
            "first_divergence": 1,
            "continuation_tokens_compared": 2,
            "identical_generation": False,
        },
        "timing": {"baseline_seconds": 1.0, "compressed_seconds": 1.2},
        "text": {"baseline_output": "abc", "compressed_output": "abd"},
    }
    md = markdown_report(report)
    assert "KV benchmark report" in md
    assert "dummy" in md
    assert "production inference acceleration" in md

    cfg = KVBenchConfig(model="dummy")
    assert cfg.preset == "balanced"


def test_cli_kv_estimate_invocation(capsys):
    from tiny_turboquant.cli import main

    rc = main([
        "kv-estimate",
        "--layers", "2",
        "--kv-heads", "2",
        "--head-dim", "16",
        "--seq-len", "64",
    ])
    captured = capsys.readouterr()
    assert rc == 0
    assert "compression_ratio" in captured.out


def test_sdpa_attention_matches_dense_attention():
    torch.manual_seed(12)
    from tiny_turboquant import dense_attention, sdpa_attention, attention_similarity

    q = torch.randn(1, 2, 3, 16)
    k = torch.randn(1, 2, 9, 16)
    v = torch.randn(1, 2, 9, 16)
    dense = dense_attention(q, k, v)
    sdpa = sdpa_attention(q, k, v)
    metrics = attention_similarity(dense, sdpa)
    assert metrics["relative_error"] < 1e-5
    assert metrics["cosine_similarity"] > 0.99999


def test_page_attention_benchmark_runs_cpu():
    from tiny_turboquant import PageAttentionBenchConfig, run_page_attention_benchmark

    cfg = PageAttentionBenchConfig(
        batch_size=1,
        heads=2,
        query_len=1,
        seq_len=32,
        head_dim=16,
        page_size=8,
        device="cpu",
        dtype="float32",
        warmup=0,
        repeats=1,
        seed=13,
    )
    report = run_page_attention_benchmark(cfg)
    assert report["page_count"] == 4
    assert report["memory"]["dense_kv_bytes"] > report["memory"]["largest_page_kv_bytes"]
    assert report["quality"]["streaming_vs_dense"]["relative_error"] < 1e-5
    assert "Production acceleration" in report["interpretation"]


def test_page_attention_config_accepts_iters_alias():
    from tiny_turboquant import PageAttentionBenchConfig, run_page_attention_benchmark

    cfg = PageAttentionBenchConfig(
        batch_size=1,
        heads=1,
        query_len=1,
        seq_len=16,
        head_dim=8,
        page_size=4,
        device="cpu",
        dtype="float32",
        warmup=0,
        iters=1,
    )
    assert cfg.repeats == 1
    report = run_page_attention_benchmark(cfg)
    assert report["config"]["iters"] == 1
    assert report["quality"]["streaming_vs_dense"]["relative_error"] < 1e-5


def test_page_attention_cuda_request_falls_back_on_cpu_only(monkeypatch):
    from tiny_turboquant import PageAttentionBenchConfig, run_page_attention_benchmark

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cfg = PageAttentionBenchConfig(
        batch_size=1,
        heads=1,
        query_len=1,
        seq_len=8,
        head_dim=4,
        page_size=4,
        device="cuda",
        dtype="float16",
        warmup=0,
        repeats=1,
    )
    report = run_page_attention_benchmark(cfg)
    assert report["config"]["requested_device"] == "cuda"
    assert report["config"]["device"] == "cpu"
    assert report["config"]["dtype"] == "float32"
    assert report["warnings"]


def test_page_attention_markdown_report_contains_warning():
    from tiny_turboquant import PageAttentionBenchConfig, page_attention_markdown_report, run_page_attention_benchmark

    report = run_page_attention_benchmark(
        PageAttentionBenchConfig(
            batch_size=1,
            heads=1,
            query_len=1,
            seq_len=16,
            head_dim=8,
            page_size=4,
            device="cpu",
            dtype="float32",
            warmup=0,
            repeats=1,
        )
    )
    md = page_attention_markdown_report(report)
    assert "page-attention benchmark" in md
    assert "does not claim production inference acceleration" in md


def test_cli_page_attn_bench_invocation(capsys):
    from tiny_turboquant.cli import main

    rc = main([
        "page-attn-bench",
        "--device", "cpu",
        "--dtype", "float32",
        "--batch-size", "1",
        "--heads", "1",
        "--query-len", "1",
        "--seq-len", "16",
        "--head-dim", "8",
        "--page-size", "4",
        "--warmup", "0",
        "--repeats", "1",
    ])
    captured = capsys.readouterr()
    assert rc == 0
    assert "streaming_paged_seconds" in captured.out

    rc = main([
        "page-attn-bench",
        "--device", "cpu",
        "--dtype", "float32",
        "--batch-size", "1",
        "--heads", "1",
        "--query-len", "1",
        "--seq-len", "16",
        "--head-dim", "8",
        "--page-size", "4",
        "--warmup", "0",
        "--iters", "1",
    ])
    captured = capsys.readouterr()
    assert rc == 0
    assert '"iters": 1' in captured.out


def test_paged_kv_serving_simulator_reports_budget_fit():
    from tiny_turboquant import PagedKVServingSimulator

    sim = PagedKVServingSimulator(
        layers=4,
        kv_heads=2,
        head_dim=16,
        page_size=16,
        preset="balanced",
        gpu_memory_budget_gb=1,
        model_weight_gb=0.1,
        key_recent_window=8,
        value_recent_window=4,
    )
    sim.add_sequence("a", prompt_tokens=33, decode_tokens=2)
    sim.add_sequence("b", prompt_tokens=17, decode_tokens=0)
    report = sim.memory_report()
    d = report.to_dict()

    assert d["active_sequences"] == 2
    assert d["total_pages_allocated"] == 5
    assert d["total_allocated_tokens"] == 80
    assert d["fp16_bytes"] > d["compressed_estimated_bytes"]
    assert d["compression_ratio"] > 1.0
    assert d["compressed_fits_budget"] is True
    assert len(d["per_sequence"]) == 2
    assert d["total_fp16_bytes"] == d["fp16_bytes"]
    assert d["total_compressed_bytes"] == d["compressed_estimated_bytes"]
    compact = report.to_dict(include_per_sequence=False)
    assert "per_sequence" not in compact
    assert compact["per_sequence_count"] == 2


def test_estimate_serving_capacity_increases_user_capacity():
    from tiny_turboquant import estimate_serving_capacity

    cap = estimate_serving_capacity(
        gpu_memory_gb=16,
        model_weight_gb=8,
        layers=24,
        kv_heads=8,
        head_dim=128,
        avg_prompt_tokens=2048,
        avg_decode_tokens=128,
        page_size=128,
        preset="balanced",
    )
    d = cap.to_dict()
    assert d["available_kv_gb"] == 8.0
    assert d["max_users_compressed"] >= d["max_users_fp16"]
    assert d["capacity_gain"] >= 1.0
    assert d["compression_ratio"] > 1.0


def test_decode_growth_points_are_monotonic():
    from tiny_turboquant import simulate_decode_growth

    points = simulate_decode_growth(
        users=4,
        prompt_tokens=128,
        decode_tokens=64,
        step_interval=32,
        layers=4,
        kv_heads=2,
        head_dim=16,
        page_size=32,
        preset="balanced",
    )
    assert [p.step for p in points] == [0, 32, 64]
    assert points[0].fp16_bytes <= points[-1].fp16_bytes
    assert points[0].compressed_estimated_bytes <= points[-1].compressed_estimated_bytes
    assert points[-1].compression_ratio > 1.0
    d = points[-1].to_dict()
    assert d["decode_step"] == d["step"]
    assert d["compressed_bytes"] == d["compressed_estimated_bytes"]


def test_serving_reports_markdown_and_csv(tmp_path):
    from tiny_turboquant import (
        PagedKVServingSimulator,
        decode_growth_markdown_report,
        save_decode_growth_csv,
        serving_markdown_report,
        simulate_decode_growth,
    )

    sim = PagedKVServingSimulator(layers=2, kv_heads=2, head_dim=16, page_size=16)
    sim.add_uniform_sequences(users=2, prompt_tokens=64, decode_tokens=8)
    md = serving_markdown_report(sim.memory_report())
    assert "serving simulation report" in md
    assert "does not claim production inference acceleration" in md

    points = simulate_decode_growth(
        users=2,
        prompt_tokens=64,
        decode_tokens=16,
        step_interval=8,
        layers=2,
        kv_heads=2,
        head_dim=16,
    )
    md2 = decode_growth_markdown_report(points)
    assert "decode-growth" in md2
    path = tmp_path / "growth.csv"
    save_decode_growth_csv(points, path)
    assert path.exists()
    assert "fp16_bytes" in path.read_text()


def test_cli_serving_commands(capsys):
    from tiny_turboquant.cli import main

    rc = main([
        "serving-sim",
        "--users", "2",
        "--prompt-tokens", "64",
        "--decode-tokens", "8",
        "--layers", "2",
        "--kv-heads", "2",
        "--head-dim", "16",
        "--page-size", "16",
    ])
    out = capsys.readouterr().out
    assert rc == 0
    assert "active_sequences" in out
    assert "compressed_estimated_bytes" in out
    assert '"per_sequence":' not in out
    assert "per_sequence_count" in out

    rc = main([
        "serving-capacity",
        "--gpu-memory-gb", "4",
        "--model-weight-gb", "1",
        "--layers", "2",
        "--kv-heads", "2",
        "--head-dim", "16",
        "--avg-prompt-tokens", "64",
    ])
    out = capsys.readouterr().out
    assert rc == 0
    assert "max_users_compressed" in out

    rc = main([
        "decode-growth",
        "--users", "2",
        "--prompt-tokens", "64",
        "--decode-tokens", "16",
        "--step-interval", "8",
        "--layers", "2",
        "--kv-heads", "2",
        "--head-dim", "16",
    ])
    out = capsys.readouterr().out
    assert rc == 0
    assert "tokens_per_user" in out


def test_v080_compressed_kv_page_table_memory_and_reference_attention():
    torch.manual_seed(14)
    from tiny_turboquant import CompressedKVPageTable, compressed_page_attention_reference, dense_attention

    k = torch.randn(1, 2, 32, 16)
    v = torch.randn(1, 2, 32, 16)
    q = torch.randn(1, 2, 1, 16)
    table = CompressedKVPageTable.from_dense(k, v, key_bits=6, value_bits=4, page_size=8)

    assert table.page_count == 4
    mem = table.memory_report().to_dict()
    assert mem["compressed_payload_bytes"] < mem["fp16_kv_bytes"]
    assert mem["actual_total_bytes"] < mem["fp16_kv_bytes"]
    out = compressed_page_attention_reference(q, table, rotate_query=True)
    dense = dense_attention(q, k, v)
    assert out.shape == dense.shape
    assert table.to_dict()["memory"]["page_count"] == 4


def test_v080_rotate_q_check_is_numerically_equivalent():
    from tiny_turboquant import run_rotate_q_check

    report = run_rotate_q_check(
        batch_size=1,
        heads=2,
        query_len=1,
        seq_len=32,
        head_dim=16,
        device="cpu",
        dtype="float32",
    )
    assert report["quality"]["relative_error"] < 1e-5
    assert report["quality"]["cosine_similarity"] > 0.99999


def test_v080_layout_bench_and_cli(capsys):
    from tiny_turboquant import LayoutBenchConfig, run_layout_benchmark
    from tiny_turboquant.cli import main

    cfg = LayoutBenchConfig(
        batch_size=1,
        heads=1,
        query_len=1,
        seq_len=16,
        head_dim=8,
        page_size=4,
        device="cpu",
        dtype="float32",
        warmup=0,
        iters=1,
    )
    assert cfg.repeats == 1
    report = run_layout_benchmark(cfg)
    assert report["layout"]["page_count"] == 4
    assert report["memory"]["payload_compression_ratio"] > 1.0
    assert "rotate_q_dense_equivalence" in report["quality"]

    rc = main([
        "layout-bench",
        "--device", "cpu",
        "--dtype", "float32",
        "--batch-size", "1",
        "--heads", "1",
        "--query-len", "1",
        "--seq-len", "16",
        "--head-dim", "8",
        "--page-size", "4",
        "--warmup", "0",
        "--iters", "1",
    ])
    out = capsys.readouterr().out
    assert rc == 0
    assert "effective_compression_ratio" in out

    rc = main([
        "rotate-q-check",
        "--device", "cpu",
        "--dtype", "float32",
        "--batch-size", "1",
        "--heads", "1",
        "--query-len", "1",
        "--seq-len", "16",
        "--head-dim", "8",
    ])
    out = capsys.readouterr().out
    assert rc == 0
    assert "rotate-Q" in out or "Rotate-Q" in out


def test_v081_affine_layout_quality_and_sweep(capsys):
    from tiny_turboquant import (
        LayoutBenchConfig,
        LayoutSweepConfig,
        available_layout_presets,
        run_layout_benchmark,
        run_layout_quality_sweep,
    )
    from tiny_turboquant.cli import main

    assert "safe-layout" in available_layout_presets()
    cfg = LayoutBenchConfig(
        batch_size=1,
        heads=2,
        query_len=1,
        seq_len=64,
        head_dim=16,
        page_size=16,
        preset="safe-layout",
        device="cpu",
        dtype="float32",
        warmup=0,
        repeats=1,
    )
    report = run_layout_benchmark(cfg)
    assert report["config"]["quantization_mode"] == "affine"
    assert report["memory"]["quant_metadata_bytes"] > 0
    assert report["quality"]["compressed_page_vs_dense"]["cosine_similarity"] > 0.99
    assert "key_reconstruction" in report["quality"]
    assert "attention_scores" in report["quality"]

    sweep = run_layout_quality_sweep(
        LayoutSweepConfig(
            batch_size=1,
            heads=1,
            query_len=1,
            seq_len=32,
            head_dim=8,
            page_size=8,
            bit_pairs=["8,8", "8,6", "6,4"],
            device="cpu",
            dtype="float32",
        )
    )
    assert len(sweep["rows"]) == 3
    assert sweep["rows"][0]["attention_cosine_similarity"] > 0.99

    rc = main([
        "layout-sweep",
        "--device", "cpu",
        "--dtype", "float32",
        "--batch-size", "1",
        "--heads", "1",
        "--query-len", "1",
        "--seq-len", "32",
        "--head-dim", "8",
        "--page-size", "8",
        "--bit-pairs", "8,8", "6,4",
    ])
    out = capsys.readouterr().out
    assert rc == 0
    assert "attention_cosine_similarity" in out


def test_v090_fused_decode_benchmark_and_cli(capsys):
    from tiny_turboquant import FusedDecodeBenchConfig, run_fused_decode_benchmark
    from tiny_turboquant.cli import main

    cfg = FusedDecodeBenchConfig(
        batch_size=1,
        heads=1,
        query_len=1,
        seq_len=32,
        head_dim=8,
        page_size=8,
        preset="safe-layout",
        device="cpu",
        dtype="float32",
        warmup=0,
        repeats=1,
        prefer_triton=False,
    )
    report = run_fused_decode_benchmark(cfg)
    assert report["version"] == "0.10.9"
    assert report["execution"]["uses_compressed_pages_directly"] is True
    assert report["execution"]["constructs_full_dense_kv"] is False
    assert report["timing"]["cuda_graph"]["enabled"] is False
    assert report["timing"]["effective_fused_mode"] == "normal-fused-call"
    assert "selected_kernel_num_warps" in report["timing"]
    assert report["quality"]["fused_vs_dense"]["cosine_similarity"] > 0.99
    assert report["memory"]["actual_total_bytes"] < report["memory"]["fp16_kv_bytes"]

    rc = main([
        "fused-decode-bench",
        "--device", "cpu",
        "--dtype", "float32",
        "--batch-size", "1",
        "--heads", "1",
        "--query-len", "1",
        "--seq-len", "32",
        "--head-dim", "8",
        "--page-size", "8",
        "--warmup", "0",
        "--iters", "1",
        "--tune-num-warps",
        "--tune-num-warps-values", "1", "2",
        "--cuda-graph",
        "--graph-replays", "2",
        "--no-triton",
    ])
    out = capsys.readouterr().out
    assert rc == 0
    assert "experimental-fused-compressed-decode-attention" in out
    assert "constructs_full_dense_kv" in out


def test_v010_residual_affine_layout_and_sweep():
    from tiny_turboquant import (
        LayoutBenchConfig,
        ResidualSweepConfig,
        run_layout_benchmark,
        run_residual_correction_sweep,
        available_layout_presets,
    )

    assert "residual-balanced" in available_layout_presets()
    report = run_layout_benchmark(
        LayoutBenchConfig(
            batch_size=1,
            heads=1,
            query_len=1,
            seq_len=16,
            head_dim=8,
            page_size=8,
            key_bits=4,
            value_bits=4,
            preset=None,
            quantization_mode="residual-affine",
            device="cpu",
            dtype="float32",
            warmup=0,
            repeats=1,
        )
    )
    assert report["config"]["quantization_mode"] == "residual-affine"
    assert report["memory"]["actual_total_bytes"] > 0
    assert "inner_product_bias_abs_mean" in report["quality"]["attention_scores"]

    sweep = run_residual_correction_sweep(
        ResidualSweepConfig(
            heads=1,
            seq_len=16,
            head_dim=8,
            page_size=8,
            bit_pairs=("4,4",),
            device="cpu",
            dtype="float32",
        )
    )
    assert sweep["version"] == "residual-correction-sweep-v0.10.3"
    assert len(sweep["rows"]) == 2
    assert sweep["comparisons"]



def test_v0102_long_context_comparison_cpu_and_cli(capsys):
    from tiny_turboquant import LongContextCompareConfig, run_long_context_comparison
    from tiny_turboquant.cli import main

    cfg = LongContextCompareConfig(
        heads=1,
        query_len=1,
        seq_lens=(16, 32),
        head_dim=8,
        page_size=8,
        presets=("safe-layout", "residual-balanced"),
        device="cpu",
        dtype="float32",
        warmup=0,
        repeats=1,
        prefer_triton=False,
        cuda_graph=False,
    )
    report = run_long_context_comparison(cfg)
    assert report["version"] == "long-context-comparison-v0.10.9"
    assert len(report["rows"]) == 4
    assert report["summary"]["by_seq_len"]
    assert "production inference acceleration" in report["interpretation"]

    rc = main([
        "long-context-compare",
        "--device", "cpu",
        "--dtype", "float32",
        "--heads", "1",
        "--query-len", "1",
        "--seq-lens", "16", "32",
        "--head-dim", "8",
        "--page-size", "8",
        "--presets", "safe-layout", "residual-balanced",
        "--warmup", "0",
        "--iters", "1",
        "--tune-num-warps",
        "--tune-num-warps-values", "1", "2",
        "--no-triton",
    ])
    out = capsys.readouterr().out
    assert rc == 0
    assert "long-context-comparison-v0.10.9" in out
    assert "recommended_preset" in out

def test_v0104_split_k_comparison_cpu_and_cli(capsys):
    from tiny_turboquant import SplitKCompareConfig, run_split_k_comparison
    from tiny_turboquant.cli import main

    cfg = SplitKCompareConfig(
        heads=1,
        query_len=1,
        seq_lens=(16, 32),
        head_dim=8,
        page_size=8,
        preset="safe-layout",
        device="cpu",
        dtype="float32",
        warmup=0,
        repeats=1,
        prefer_triton=False,
        cuda_graph=False,
        split_k_slabs=(2, 4),
    )
    report = run_split_k_comparison(cfg)
    assert report["version"] == "split-k-comparison-v0.10.9"
    assert len(report["rows"]) == 2
    assert "split_k_projection" in report["rows"][0]
    assert report["summary"]["boundary"].startswith("This is a split-K")

    rc = main([
        "split-k-compare",
        "--device", "cpu",
        "--dtype", "float32",
        "--heads", "1",
        "--head-dim", "8",
        "--seq-lens", "16",
        "--page-size", "8",
        "--preset", "safe-layout",
        "--warmup", "0",
        "--iters", "1",
        "--no-triton",
        "--split-k-slabs", "2", "4",
    ])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "split-k-comparison-v0.10.9" in captured


def test_split_k_attention_reference_matches_compressed_reference():
    torch.manual_seed(123)
    from tiny_turboquant import CompressedKVPageTable
    from tiny_turboquant.fused_attention import split_k_attention_reference
    from tiny_turboquant.layout import compressed_page_attention_reference

    q = torch.randn(1, 2, 1, 16)
    k = torch.randn(1, 2, 32, 16)
    v = torch.randn_like(k)
    table = CompressedKVPageTable.from_dense(
        k,
        v,
        key_bits=8,
        value_bits=6,
        page_size=8,
        seed=123,
        dtype_bytes=4,
        quantization_mode="affine",
    )
    ref = compressed_page_attention_reference(q, table, rotate_query=True)
    split = split_k_attention_reference(q, table, split_k_slabs=4)
    assert split.shape == ref.shape
    assert torch.allclose(split, ref, atol=1e-5, rtol=1e-5)


def test_v0106_stage1_exports():
    import tiny_turboquant as ttq

    assert hasattr(ttq, "triton_split_k_stage1_partials")
    assert hasattr(ttq, "reduce_split_k_partials")
    assert hasattr(ttq, "triton_split_k_stage1_attention_reference")


def test_v0106_1_reduce_split_k_partials_uses_unnormalized_accumulator_contract():
    from tiny_turboquant.fused_attention import reduce_split_k_partials

    # Two slabs, one head, one value dimension.  Both slabs have local max 0.
    # Stage 1 stores an unnormalized accumulator:
    #   slab0: l=2, acc=4
    #   slab1: l=3, acc=9
    # Correct global output = (4 + 9) / (2 + 3) = 2.6.
    # The old v0.10.6 reducer incorrectly multiplied acc by l again,
    # producing (2*4 + 3*9) / 5 = 7.0.
    partial_m = torch.tensor([[0.0], [0.0]], dtype=torch.float32)
    partial_l = torch.tensor([[2.0], [3.0]], dtype=torch.float32)
    partial_acc = torch.tensor([[[4.0]], [[9.0]]], dtype=torch.float32)

    out = reduce_split_k_partials(partial_m, partial_l, partial_acc)

    assert out.shape == (1, 1, 1, 1)
    assert torch.allclose(out.flatten(), torch.tensor([2.6]), atol=1e-6)


def test_v0106_1_reduce_split_k_partials_with_different_local_maxima():
    from tiny_turboquant.fused_attention import reduce_split_k_partials

    partial_m = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
    partial_l = torch.tensor([[2.0], [3.0]], dtype=torch.float32)
    partial_acc = torch.tensor([[[4.0]], [[9.0]]], dtype=torch.float32)

    scale0 = torch.exp(torch.tensor(0.0 - 1.0))
    scale1 = torch.exp(torch.tensor(1.0 - 1.0))
    expected = (scale0 * 4.0 + scale1 * 9.0) / (scale0 * 2.0 + scale1 * 3.0)

    out = reduce_split_k_partials(partial_m, partial_l, partial_acc)

    assert torch.allclose(out.flatten(), expected.reshape(1), atol=1e-6)


def test_v0110_end_to_end_decode_cpu_report_shape():
    from tiny_turboquant import EndToEndDecodeBenchConfig, run_end_to_end_decode_benchmark

    cfg = EndToEndDecodeBenchConfig(
        prompt_lens=(16,),
        decode_steps=2,
        heads=1,
        head_dim=8,
        page_size=8,
        device="cpu",
        dtype="float32",
        warmup=0,
        repeats=1,
        include_single_pass_loop=False,
        tune_kernel=False,
        tune_num_warps=False,
    )
    report = run_end_to_end_decode_benchmark(cfg)

    assert report["version"] == "end-to-end-decode-v0.11.1"
    assert report["rows"][0]["prompt_len"] == 16
    assert "setup" in report["rows"][0]
    assert "sdpa_decode" in report["rows"][0]
    assert report["rows"][0]["split_k_decode"] is None


def test_cli_end_to_end_decode_cpu_smoke(capsys):
    from tiny_turboquant.cli import main

    rc = main([
        "end-to-end-decode",
        "--device", "cpu",
        "--dtype", "float32",
        "--heads", "1",
        "--head-dim", "8",
        "--prompt-lens", "16",
        "--decode-steps", "1",
        "--page-size", "8",
        "--warmup", "0",
        "--iters", "1",
        "--no-triton",
        "--no-single-pass-loop",
    ])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "end-to-end-decode-v0.11.1" in captured
