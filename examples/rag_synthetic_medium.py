"""Medium RAG example: full precision vs compressed vector index."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tiny_turboquant import CompressedVectorIndex


def main():
    torch.manual_seed(0)
    n_topics, docs_per_topic, dim = 6, 500, 128
    centers = F.normalize(torch.randn(n_topics, dim), dim=-1)
    embeddings = []
    labels = []
    for topic in range(n_topics):
        x = F.normalize(centers[topic] + 0.08 * torch.randn(docs_per_topic, dim), dim=-1)
        embeddings.append(x)
        labels.extend([topic] * docs_per_topic)
    embeddings = torch.cat(embeddings, dim=0)

    query_topic = 2
    query = F.normalize(centers[query_topic] + 0.04 * torch.randn(dim), dim=-1)

    baseline_scores = embeddings @ query
    _, baseline_idx = torch.topk(baseline_scores, k=10)

    index = CompressedVectorIndex(bits=4, store_original_for_rerank=True).add(embeddings)
    compressed = index.search(query, top_k=10, rerank_top_k=100)

    compressed_ids = [r.index for r in compressed]
    topic_match = sum(1 for i in compressed_ids if labels[i] == query_topic) / 10

    print("fp32 MB:", round(index.fp32_baseline_bytes() / (1024 * 1024), 3))
    print("compressed payload MB:", round(index.compressed_payload_bytes() / (1024 * 1024), 3))
    print("compression ratio:", round(index.compression_ratio(), 2))
    print("compressed top-10 topic match:", topic_match)
    print("top-10 exact overlap:", len(set(baseline_idx.tolist()) & set(compressed_ids)), "/ 10")


if __name__ == "__main__":
    main()
