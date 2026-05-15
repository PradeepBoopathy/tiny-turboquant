# Tiny TurboQuant

## v0.5.0: real LLM KV-cache benchmarking

This release adds repeatable KV-cache benchmark utilities for real Hugging Face causal language models.

New capabilities:

- `tiny-tq kv-bench` for DynamicCache vs `HybridTurboQuantKVCache` comparison
- prompt modes: `short`, `medium`, `long`, and `stress`
- safe / balanced / aggressive / quality-headwise KV-cache presets
- automatic outlier count selection with `"auto"`
- KV-cache memory estimator via `tiny-tq kv-estimate`
- JSON and Markdown benchmark reports
- baseline vs compressed output comparison
- generation-drift diagnostics such as first divergence and optional logit KL

This release continues to focus on memory compression and quality measurement. It does **not** claim production inference acceleration.

Example:

```bash
tiny-tq kv-estimate --layers 24 --kv-heads 8 --head-dim 128 --seq-len 4096 --batch-size 4

tiny-tq kv-bench \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --preset safe \
  --prompt-mode short \
  --max-new-tokens 8 \
  --report-json kv_report.json \
  --report-md kv_report.md
```


Tiny TurboQuant is a lightweight PyTorch research toolkit for low-bit vector compression, compressed RAG retrieval, and KV-cache compression experiments.

Version **0.4.0** focuses on the RAG/vector-index roadmap:

- `RAGCompressedIndex` for compressed retrieval experiments
- document chunking helpers
- JSONL / folder text ingestion utilities
- compressed vector index save/load
- RAG index save/load
- retrieval metrics: recall@k, precision@k, MRR, nDCG, overlap, label-match ratio
- CLI entry point: `tiny-tq rag-bench`
- existing hybrid KV-cache and paged-attention research utilities from v0.3.x

## Important limitation

This package demonstrates packed memory compression and memory-quality benchmarking. It is **not** a production compressed-attention engine. Hugging Face generation still receives dense K/V tensors. The paged attention utility dequantizes page-by-page and avoids one full dense cache tensor, but it is not a fused CUDA/Triton kernel. Real latency gains require fused kernels or serving-engine integration.

Do not use this package to claim training acceleration, fine-tuning memory reduction, production legal/medical QA readiness, drop-in vLLM replacement, exact nearest-neighbor search, or faster LLM inference.

## Install

```bash
pip install tiny-turboquant
```

Optional demo dependencies:

```bash
pip install "tiny-turboquant[demos]"
```

## Compressed RAG index from precomputed embeddings

```python
import torch
from tiny_turboquant import RAGCompressedIndex

texts = [
    "Embedding compression can reduce vector-store memory in RAG systems.",
    "KV cache stores Key and Value tensors during LLM generation.",
]
embeddings = torch.randn(len(texts), 384)

index = RAGCompressedIndex.from_embeddings(
    texts,
    embeddings,
    bits=4,
    store_original_for_rerank=True,
)

results = index.search(embeddings[0], top_k=1, rerank_top_k=2)
print(index.memory_report())
print(results[0].text)
```

## Compressed RAG index from documents

Requires `sentence-transformers`:

```python
from tiny_turboquant import RAGCompressedIndex

docs = [
    "RAG systems retrieve relevant document chunks and pass them to an LLM.",
    "Compressed vector indexes reduce embedding memory usage.",
]

index = RAGCompressedIndex.from_documents(
    docs,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    bits=4,
    chunk_size=500,
    overlap=50,
)

results = index.search("How do we reduce vector-store memory?", top_k=3)
```

## Save and load

```python
index.save("rag_index.ttq")
loaded = RAGCompressedIndex.load("rag_index.ttq")
```

## Retrieval metrics

```python
from tiny_turboquant import recall_at_k, mrr_at_k, ndcg_at_k

retrieved = ["doc-1", "doc-2", "doc-3"]
relevant = {"doc-2", "doc-5"}

print(recall_at_k(retrieved, relevant, k=3))
print(mrr_at_k(retrieved, relevant, k=3))
print(ndcg_at_k(retrieved, relevant, k=3))
```

## CLI

```bash
tiny-tq version
```

Synthetic RAG benchmark:

```bash
tiny-tq rag-bench --synthetic --bits 4 --top-k 10 --rerank-top-k 50
```

JSONL benchmark:

```bash
tiny-tq rag-bench \
  --input-jsonl docs.jsonl \
  --text-field text \
  --query "How can we reduce vector-store memory?" \
  --bits 4 \
  --top-k 10 \
  --rerank-top-k 50
```

## Hybrid KV-cache usage

```python
from tiny_turboquant import HybridTurboQuantKVCache

cache = HybridTurboQuantKVCache(
    key_bits=6,
    value_bits=4,
    key_outlier_bits=8,
    value_outlier_bits=8,
    n_key_outliers=32,
    n_value_outliers=16,
    key_recent_window=128,
    value_recent_window=64,
    per_layer_calibration=True,
    per_head_calibration=True,
)
```

## Compressed vector index usage

```python
import torch
from tiny_turboquant import CompressedVectorIndex

emb = torch.randn(10_000, 384)
index = CompressedVectorIndex(bits=4, store_original_for_rerank=True).add(emb)
results = index.search(emb[0], top_k=5, rerank_top_k=100)

print(index.compression_ratio())
print(results[0])
```

## Project position

Current focus:

- memory compression
- retrieval quality measurement
- RAG/vector-index experiments
- KV-cache compression research

Future direction:

- real workload RAG benchmarks
- FAISS/vector database integration
- long-context real-model KV-cache benchmarks
- fused dequant + attention kernels
- serving-engine integration experiments


## Packaging note

The PyPI wheel installs only the core `tiny_turboquant` package. Demo, benchmark, example, and test files are kept in the source distribution / repository for reference and are not installed as top-level Python packages.


## v0.6.0: Page-wise Attention Performance Research

This release adds a diagnostic benchmark for dense attention vs page-wise streaming attention.
It is useful for studying memory movement and attention quality before implementing fused kernels.

```bash
tiny-tq page-attn-bench --seq-len 1024 --page-size 128 --device cuda --dtype float16
```

The benchmark reports dense K/V bytes, largest-page bytes, timing for dense/manual attention, streaming paged attention, optional PyTorch SDPA, and output similarity.

Important: this is not a production fused CUDA/Triton kernel and does not claim production inference acceleration.

## v0.7.0 serving-capacity simulation

v0.7.0 adds serving-style KV-cache memory simulation and capacity planning.
These tools model paged KV-cache allocation, multi-user memory pressure,
decode-time cache growth, and fp16-vs-compressed capacity under a fixed GPU
memory budget.

Example:

```bash
tiny-tq serving-sim \
  --users 32 \
  --prompt-tokens 2048 \
  --decode-tokens 128 \
  --layers 24 \
  --kv-heads 8 \
  --head-dim 128 \
  --page-size 128 \
  --preset balanced \
  --gpu-memory-gb 16 \
  --model-weight-gb 8
```

This is a capacity-planning and research utility. It does not implement a
production serving backend and does not claim production inference acceleration.

## v0.7.1 hardening notes

v0.7.1 is a bugfix and demo-hardening release. It does not change the core compression algorithms.

Changes:

- `kv-bench` and `page-attn-bench` now fall back to CPU when `--device cuda` is requested on a CPU-only PyTorch build.
- CPU fallback also coerces fp16/bf16 requests to fp32 for stable execution.
- `PageAttentionBenchConfig` accepts `iters=` as a backward-compatible alias for `repeats=`.
- CLI `page-attn-bench` accepts both `--repeats` and `--iters`.
- Serving reports include backward-compatible aliases such as `total_fp16_bytes`, `total_compressed_bytes`, `decode_step`, and `compressed_bytes`.
- `serving-sim` defaults to compact output and hides the full per-sequence list unless `--include-per-sequence` is passed.
- Benchmark reports include runtime warnings for device/dtype fallback.

Boundary: this release is still about memory compression, retrieval-quality measurement, KV-cache capacity simulation, and diagnostics. It does not claim production inference acceleration.

## v0.8.1: Compressed KV Page Layout Preparation

This release adds a kernel-ready compressed KV page layout and rotate-Q reference path for future fused compressed attention work.

New capabilities:

- `CompressedKVPage` and `CompressedKVPageTable`
- `RotatedCompressedKVCache`
- `compressed_page_attention_reference()`
- `rotate_q_attention_reference()` and `tiny-tq rotate-q-check`
- `tiny-tq layout-bench`
- memory accounting for payload, codebooks, rotation metadata, page-table overhead, and actual total bytes

Example:

```bash
tiny-tq layout-bench \
  --seq-len 1024 \
  --page-size 128 \
  --heads 8 \
  --head-dim 64 \
  --preset balanced \
  --device cuda \
  --dtype float16
```

Rotate-Q equivalence check:

```bash
tiny-tq rotate-q-check \
  --seq-len 1024 \
  --heads 8 \
  --head-dim 64 \
  --device cuda \
  --dtype float16
```

Correct claim:

> tiny-turboquant v0.8.1 introduces a compressed KV page layout and rotate-Q reference path for future fused compressed attention.

Boundary:

> v0.8.1 does not claim faster inference or production acceleration. It prepares the layout and correctness path required before fused CUDA/Triton kernels.

## v0.9.0: Experimental fused compressed decode-attention benchmark

v0.9.0 adds an experimental compressed decode-attention benchmark API and CLI:

```bash
tiny-tq fused-decode-bench --preset safe-layout --seq-len 2048 --page-size 128 --device cuda --dtype float16
```

The benchmark compares dense attention, PyTorch SDPA, the compressed-page reference path, and the v0.9 experimental fused decode-attention interface. The v0.9 path reads compressed pages directly and avoids constructing a full dense K/V cache. If a production Triton kernel is unavailable, it falls back to the verified PyTorch compressed-page path and reports that mode explicitly.

This release still does not claim production inference acceleration. It establishes the benchmark contract and correctness boundary for future fused CUDA/Triton kernels.


## v0.9.4 Triton tuning diagnostics

`tiny-tq fused-decode-bench` now supports `--kernel-block-m` and `--tune-kernel` to compare small Triton BLOCK_M candidates for the experimental fused compressed decode-attention prototype. The benchmark reports the selected candidate, candidate timings, speedup versus the PyTorch compressed-page reference, and continues to avoid production acceleration claims.


## v0.10.3 kernel algorithm tuning

`tiny-tq fused-decode-bench` now supports `--kernel-num-warps`, `--tune-num-warps`, and `--tune-num-warps-values` so the experimental Triton compressed decode-attention prototype can tune both BLOCK_M and Triton num_warps. This is a microbenchmark diagnostic for narrowing the SDPA gap; it is not a production inference acceleration claim.

Example:

```bash
tiny-tq fused-decode-bench \
  --preset safe-layout \
  --seq-len 8192 \
  --page-size 256 \
  --heads 8 \
  --head-dim 64 \
  --device cuda \
  --dtype float16 \
  --tune-kernel \
  --tune-block-m-values 8 16 32 64 128 \
  --tune-num-warps \
  --tune-num-warps-values 1 2 4 8 \
  --cuda-graph
```


## v0.10.6.1.1

Adds a real Triton split-K Stage-1 partial-statistics kernel for compressed KV pages. Stage 2 reduction remains a Torch reference path; this is not a production inference acceleration claim.
