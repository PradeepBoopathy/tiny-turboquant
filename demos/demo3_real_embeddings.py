"""
Demo 3 — Real-world embeddings: semantic search before/after compression.

This demo uses a real sentence-transformer when available and falls back to a
synthetic corpus otherwise.

Important evaluation detail:
    The corpus intentionally contains many near-duplicate sentences. Exact
    index-level recall can be low even when both systems retrieve the same
    semantic item, because duplicate copies occupy different row IDs. Therefore
    the demo reports both:

      1. strict index recall@5
      2. semantic-label recall@5

Run:
    python -m demos.demo3_real_embeddings
"""

from __future__ import annotations

from collections import Counter
import time

import numpy as np

from tiny_turboquant.numpy_reference import TurboQuantProd


BASE_SENTENCES = [
    "The cat sat on the mat.",
    "A dog is running in the park.",
    "Quantum computing exploits superposition.",
    "The Eiffel Tower is located in Paris.",
    "Machine learning models require training data.",
    "Photosynthesis converts sunlight into chemical energy.",
    "Shakespeare wrote Romeo and Juliet.",
    "Neural networks are inspired by the brain.",
    "The Pacific Ocean is the largest ocean on Earth.",
    "Python is a popular programming language.",
]


def load_corpus_and_embeddings():
    """Return sentences, semantic labels, and normalized embeddings."""
    try:
        from sentence_transformers import SentenceTransformer

        sentences: list[str] = []
        labels: list[int] = []

        # 10 topics × 500 copies = 5,000 docs. Copies are given unique document
        # IDs so printed neighbours are distinguishable, but the semantic label
        # lets us measure equivalence under duplicate / near-duplicate rows.
        for copy_id in range(500):
            for label, text in enumerate(BASE_SENTENCES):
                sentences.append(f"{text} [doc {copy_id:03d}]")
                labels.append(label)

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        emb = model.encode(
            sentences,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return sentences, np.asarray(labels, dtype=np.int64), np.asarray(emb, dtype=np.float64)

    except Exception as exc:  # noqa: BLE001
        print(f"(sentence-transformers unavailable: {exc}) -> using synthetic corpus")

        rng = np.random.default_rng(1)
        n, d = 5000, 384
        n_topics = 10

        centers = rng.standard_normal((n_topics, d))
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)

        labels = rng.integers(0, n_topics, size=n)
        emb = centers[labels] + 0.15 * rng.standard_normal((n, d))
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)

        sentences = [f"topic-{labels[i]} synthetic document #{i}" for i in range(n)]
        return sentences, labels.astype(np.int64), emb.astype(np.float64)


def _label_overlap(a: np.ndarray, b: np.ndarray, labels: np.ndarray) -> int:
    ca = Counter(labels[a].tolist())
    cb = Counter(labels[b].tolist())
    return int(sum((ca & cb).values()))


def main() -> None:
    sentences, labels, emb = load_corpus_and_embeddings()
    n, d = emb.shape

    print(
        f"\nCorpus: {n} sentences, embedding dim={d}, "
        f"fp32 footprint={n * d * 4 / 1e6:.1f} MB\n"
    )

    bits = 4
    t0 = time.perf_counter()
    tq = TurboQuantProd(d, bits=bits, seed=0)
    idx, signs, gamma = tq.quant(emb)
    t_quant = time.perf_counter() - t0

    # The NumPy reference reports theoretical payload size. The torch KV-cache
    # implementation contains physical uint8 bit-packing for actual storage.
    bytes_per_vec = ((bits - 1) * d + d + 7) // 8 + 4

    print(
        f"compressed footprint = {n * bytes_per_vec / 1e6:.2f} MB  "
        f"(~{(n * d * 4) / (n * bytes_per_vec):.1f}× smaller than fp32 payload)"
    )
    print(f"indexing time        = {t_quant:.3f} s")

    rng = np.random.default_rng(7)
    q_ids = rng.choice(n, 3, replace=False)
    emb_hat = tq.dequant(idx, signs, gamma)

    strict_recalls = []
    label_recalls = []

    print("\nTop-5 neighbours: full-precision vs TurboQuant\n")
    for qi in q_ids:
        sims_fp = emb @ emb[qi]
        sims_q = emb_hat @ emb[qi]

        top_fp = np.argsort(-sims_fp)[:5]
        top_q = np.argsort(-sims_q)[:5]

        strict_overlap = len(set(top_fp.tolist()) & set(top_q.tolist()))
        semantic_overlap = _label_overlap(top_fp, top_q, labels)

        strict_recalls.append(strict_overlap / 5)
        label_recalls.append(semantic_overlap / 5)

        print(f"query: {sentences[qi][:90]!r}")
        print(f"   fp32 top5 : {[sentences[i][:70] for i in top_fp]}")
        print(f"   tq   top5 : {[sentences[i][:70] for i in top_q]}")
        print(f"   strict index recall@5   = {strict_overlap}/5")
        print(f"   semantic-label recall@5 = {semantic_overlap}/5\n")

    print("Summary:")
    print(f"   mean strict index recall@5   = {np.mean(strict_recalls):.3f}")
    print(f"   mean semantic-label recall@5 = {np.mean(label_recalls):.3f}")


if __name__ == "__main__":
    main()
