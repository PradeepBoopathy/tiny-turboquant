# tiny-turboquant

`tiny-turboquant` is a PyTorch/Triton research toolkit for studying low-bit vector compression, compressed RAG retrieval, KV-cache memory reduction, and experimental compressed decode-attention kernels.

The package is built for measurement, learning, and reproducible research. It is not a drop-in serving engine, not a vLLM replacement, and not a production LLM inference accelerator.

## What it supports

- Low-bit vector quantization and packed storage
- Compressed vector indexes with optional reranking
- Compressed RAG index experiments
- Retrieval metrics such as recall, precision, MRR, nDCG, overlap, and label match
- KV-cache memory estimation
- Hybrid KV-cache research objects for Hugging Face-style experimentation
- Page-wise attention diagnostics
- Compressed KV page layouts
- Rotate-Q / rotated-key compressed attention diagnostics
- Residual-corrected compressed KV layout experiments
- Triton compressed decode-attention research kernels
- Split-K / sequence-parallel compressed decode-attention diagnostics
- Fixed-cache repeated decode-loop benchmarks with setup amortization reporting
- Serving-capacity simulation for KV-memory planning

## What it does not claim

Do not use this package to claim:

- production LLM inference acceleration
- full model-serving acceleration
- training or fine-tuning acceleration
- exact nearest-neighbor search
- drop-in replacement for FAISS, vLLM, TensorRT-LLM, or FlashAttention
- legal, medical, or safety-critical retrieval correctness
- universal speedups across hardware or workloads

The fastest paths are research microbenchmarks. Results must be interpreted with their reported boundaries: setup cost, payload preparation, CUDA Graph use, synthetic data, fixed-cache assumptions, and hardware-specific behavior.

## Install

```bash
pip install tiny-turboquant
```

Optional demo dependencies:

```bash
pip install "tiny-turboquant[demos]"
```

Check the installed package:

```bash
tiny-tq version
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

texts = [
    "RAG systems retrieve relevant document chunks and pass them to an LLM.",
    "Compressed vector indexes reduce embedding memory usage.",
]

index = RAGCompressedIndex.from_documents(
    texts,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    bits=4,
    chunk_size=500,
    overlap=50,
)

results = index.search("How do we reduce vector-store memory?", top_k=3)
```

Save and load:

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

## Synthetic RAG benchmark

```bash
tiny-tq rag-bench \
  --synthetic \
  --bits 4 \
  --top-k 10 \
  --rerank-top-k 50
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

## Compressed vector index

```python
import torch
from tiny_turboquant import CompressedVectorIndex

embeddings = torch.randn(10_000, 384)
index = CompressedVectorIndex(bits=4, store_original_for_rerank=True).add(embeddings)
results = index.search(embeddings[0], top_k=5, rerank_top_k=100)

print(index.compression_ratio())
print(results[0])
```

## KV-cache memory estimation

```bash
tiny-tq kv-estimate \
  --layers 24 \
  --kv-heads 8 \
  --head-dim 128 \
  --seq-len 4096 \
  --batch-size 4
```

## Real-model KV-cache benchmark

```bash
tiny-tq kv-bench \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --preset safe \
  --prompt-mode short \
  --max-new-tokens 8 \
  --report-json kv_report.json \
  --report-md kv_report.md
```

This benchmark is for memory and quality diagnostics. It should not be treated as production serving performance.

## Hybrid KV-cache object

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

## Page-wise attention diagnostic

```bash
tiny-tq page-attn-bench \
  --seq-len 1024 \
  --page-size 128 \
  --device cuda \
  --dtype float16
```

This measures dense attention, SDPA, and page-wise compressed attention behavior. It is a diagnostic, not a serving benchmark.

## Layout quality sweep

```bash
tiny-tq layout-sweep \
  --seq-len 2048 \
  --page-size 256 \
  --heads 8 \
  --head-dim 64 \
  --device cuda \
  --dtype float16 \
  --bit-pairs 8,8 8,6 8,4 6,4 \
  --report-json layout_sweep.json \
  --report-md layout_sweep.md
```

Use this to compare memory savings and attention-quality impact across compressed layouts.

## Residual-correction sweep

```bash
tiny-tq residual-sweep \
  --device cuda \
  --dtype float16 \
  --seq-len 2048 \
  --heads 8 \
  --head-dim 64 \
  --page-size 256 \
  --bit-pairs 8,6 6,4 4,4 \
  --report-json residual_sweep.json \
  --report-md residual_sweep.md
```

This tests affine vs residual-affine compressed KV layouts and reports quality metrics such as attention relative error, cosine similarity, QK score error, softmax KL, and inner-product bias.

## Fused compressed decode benchmark

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
  --cuda-graph \
  --graph-replays 100 \
  --report-json fused_report.json \
  --report-md fused_report.md
```

This measures the experimental compressed decode-attention path and compares it with dense attention and SDPA.

## Split-K compressed decode benchmark

```bash
tiny-tq split-k-compare \
  --seq-lens 16384 32768 \
  --preset safe-layout \
  --page-size 256 \
  --heads 8 \
  --head-dim 64 \
  --device cuda \
  --dtype float16 \
  --tune-kernel \
  --tune-block-m-values 8 16 32 64 128 \
  --tune-num-warps \
  --tune-num-warps-values 1 2 4 8 \
  --cuda-graph \
  --graph-replays 100 \
  --split-k-slabs 2 4 8 16 \
  --report-json split_k_report.json \
  --report-md split_k_report.md
```

This compares single-pass compressed decode attention, projected split-K, Torch reference split-K, and measured Triton split-K paths. Treat this as a kernel research benchmark.

## End-to-end fixed-cache decode benchmark

```bash
tiny-tq end-to-end-decode \
  --prompt-lens 16384 32768 \
  --decode-steps 32 \
  --amortization-steps 32 128 256 512 1024 \
  --preset safe-layout \
  --page-size 256 \
  --heads 8 \
  --head-dim 64 \
  --device cuda \
  --dtype float16 \
  --tune-kernel \
  --tune-block-m-values 8 16 32 64 128 \
  --tune-num-warps \
  --tune-num-warps-values 1 2 4 8 \
  --split-k-slabs 2 4 8 16 \
  --cuda-graph \
  --graph-replays 100 \
  --report-json e2e_decode_report.json \
  --report-md e2e_decode_report.md
```

This benchmark reports:

- setup cost
- payload preparation cost
- repeated decode-loop timing
- per-token timing
- token throughput estimate
- SDPA comparison
- quality vs dense attention
- amortized setup cost
- break-even decode length
- cached-layout reuse scenarios

This is still a fixed-cache synthetic diagnostic, not full model serving.

## Serving memory simulation

```bash
tiny-tq serving-capacity \
  --gpu-memory-gb 24 \
  --model-weight-gb 8 \
  --layers 24 \
  --kv-heads 8 \
  --head-dim 128 \
  --avg-prompt-tokens 4096 \
  --avg-decode-tokens 256 \
  --preset balanced
```

Use this to estimate how compressed KV layouts may affect memory capacity for long-context serving scenarios.

## How to read benchmark output

Prefer these fields:

- `effective_memory_saved_pct`
- `cosine_similarity`
- `relative_error`
- `seconds_per_token`
- `tokens_per_second`
- `over_sdpa_per_token`
- `speedup_vs_sdpa_per_token`
- `total_setup_seconds`
- `break_even_decode_steps_full_setup`
- `break_even_decode_steps_payload_prepare_only`
- `boundary`

A kernel-only win is not the same as production acceleration. Check whether setup cost, allocation, cache preparation, and full model execution are included.

## Recommended project claim

Use this wording:

> tiny-turboquant is a PyTorch/Triton research toolkit for compressed RAG retrieval, KV-cache memory analysis, and experimental compressed decode-attention benchmarking.

Avoid this wording:

> tiny-turboquant accelerates production LLM inference.

## Packaging note

The wheel installs the core `tiny_turboquant` package. Demo, benchmark, example, and test files are kept in the source distribution or repository for reference and are not installed as top-level Python packages.
