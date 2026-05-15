"""
ANN A/B benchmark: TurboQuant vs faiss PQ vs faiss IVFPQ vs RaBitQ (optional).

Reports recall@k, indexing time, and queries/second for each method on the
same dataset. The dataset can be loaded from a .npy file (most realistic)
or generated synthetically.

Usage:
    python -m benchmarks.bench_ann --data path/to/embeddings.npy --bits 4 --k 10
    python -m benchmarks.bench_ann --synthetic 50000 --d 768 --bits 4
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from tiny_turboquant import TurboQuantProd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",      type=str, default=None,
                   help="Path to .npy of shape (N, D), float32.")
    p.add_argument("--synthetic", type=int, default=50_000)
    p.add_argument("--d",         type=int, default=768)
    p.add_argument("--bits",      type=int, default=4)
    p.add_argument("--k",         type=int, default=10)
    p.add_argument("--n_queries", type=int, default=500)
    p.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_data(args) -> tuple[np.ndarray, np.ndarray]:
    if args.data is not None:
        X = np.load(args.data).astype(np.float32)
    else:
        rng = np.random.default_rng(0)
        # Cluster structure mimicking real embedding stores.
        n_clusters = max(50, args.synthetic // 500)
        centers = rng.standard_normal((n_clusters, args.d)).astype(np.float32)
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)
        cluster_id = rng.integers(0, n_clusters, size=args.synthetic)
        X = centers[cluster_id] + 0.25 * rng.standard_normal((args.synthetic, args.d)).astype(np.float32)

    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    rng = np.random.default_rng(1)
    qi = rng.choice(X.shape[0], args.n_queries, replace=False)
    return X, X[qi].copy()


def recall_at_k(true_top: np.ndarray, est_top: np.ndarray) -> float:
    hits = sum(len(set(t.tolist()) & set(e.tolist()))
               for t, e in zip(true_top, est_top))
    return hits / true_top.size


# ---- methods ----------------------------------------------------------

def turboquant_run(X, Q, k, bits, device):
    Xt = torch.from_numpy(X).to(device)
    Qt = torch.from_numpy(Q).to(device)
    d = X.shape[1]

    t0 = time.perf_counter()
    q  = TurboQuantProd.build(d, bits, device=device, seed=0,
                              dtype=torch.float32)
    idx, signs, gamma = q.quant(Xt)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t_index = time.perf_counter() - t0

    t0 = time.perf_counter()
    Xh = q.dequant(idx, signs, gamma)
    sims = Qt @ Xh.T
    top = sims.topk(k, dim=-1).indices.cpu().numpy()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t_query = time.perf_counter() - t0
    return top, t_index, t_query


def faiss_pq_run(X, Q, k, bits):
    try:
        import faiss
    except Exception as e:   # pragma: no cover
        print(f"  (faiss unavailable: {e}) — skipping PQ")
        return None, None, None
    d = X.shape[1]
    # Match bit-budget: PQ stores log2(ks) bits per sub-quantizer;
    # M sub-vectors * log2(ks) / d total bits per coord.
    # Use ks=2**bits and M=d so PQ's bit-budget = bits per coord.
    M  = d
    ks = 2 ** bits
    if ks > 256:                          # faiss IndexPQ stores 1 byte per code
        ks = 256
    index = faiss.IndexPQ(d, M, int(np.log2(ks)))
    index.metric_type = faiss.METRIC_INNER_PRODUCT
    t0 = time.perf_counter(); index.train(X); index.add(X); t_index = time.perf_counter() - t0
    t0 = time.perf_counter()
    _, top = index.search(Q, k)
    t_query = time.perf_counter() - t0
    return top, t_index, t_query


def faiss_ivfpq_run(X, Q, k, bits):
    try:
        import faiss
    except Exception:
        return None, None, None
    d = X.shape[1]
    nlist = 100
    M = d
    ks_bits = min(bits, 8)
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, M, ks_bits)
    index.metric_type = faiss.METRIC_INNER_PRODUCT
    t0 = time.perf_counter(); index.train(X); index.add(X); t_index = time.perf_counter() - t0
    index.nprobe = 16
    t0 = time.perf_counter()
    _, top = index.search(Q, k)
    t_query = time.perf_counter() - t0
    return top, t_index, t_query


def rabitq_run(X, Q, k, bits):                  # pragma: no cover
    """Optional RaBitQ baseline; skipped silently if package missing."""
    try:
        import rabitqlib                          # noqa: F401
    except Exception as e:
        print(f"  (rabitq unavailable: {e}) — skipping RaBitQ")
        return None, None, None
    print("  TODO: integrate rabitqlib (left as exercise — official API churns)")
    return None, None, None


# ---- driver ----------------------------------------------------------

def main():
    args = parse_args()
    X, Q = load_data(args)
    n, d = X.shape

    # Ground truth: brute-force top-k inner product
    print(f"\nDataset: {n} vectors of dim {d}, {len(Q)} queries, k={args.k}, "
          f"bits={args.bits}, device={args.device}\n")
    t0 = time.perf_counter()
    sims = Q @ X.T
    true_top = np.argpartition(-sims, args.k, axis=1)[:, :args.k]
    # exact top-k order
    true_top = np.array([row[np.argsort(-sims[i, row])] for i, row in enumerate(true_top)])
    t_brute = time.perf_counter() - t0
    print(f"brute force baseline: {t_brute:.3f}s\n")

    rows = []
    for name, fn in (
        ("TurboQuant",       lambda: turboquant_run(X, Q, args.k, args.bits, args.device)),
        ("faiss PQ",         lambda: faiss_pq_run(X, Q, args.k, args.bits)),
        ("faiss IVF-PQ",     lambda: faiss_ivfpq_run(X, Q, args.k, args.bits)),
        ("RaBitQ (opt.)",    lambda: rabitq_run(X, Q, args.k, args.bits)),
    ):
        top, t_index, t_query = fn()
        if top is None:
            continue
        rec = recall_at_k(true_top, top)
        qps = len(Q) / max(t_query, 1e-9)
        rows.append((name, t_index, t_query, qps, rec))

    print(f"{'method':16s} | {'index(s)':>9} | {'query(s)':>9} | {'qps':>10} | {'recall@k':>9}")
    print("-" * 72)
    for name, ti, tq, qps, rec in rows:
        print(f"{name:16s} | {ti:>9.3f} | {tq:>9.3f} | {qps:>10.0f} | {rec:>9.3f}")
    print()

    if rows:
        tq_row = next((r for r in rows if r[0] == "TurboQuant"), None)
        pq_row = next((r for r in rows if r[0] == "faiss PQ"),   None)
        if tq_row and pq_row:
            print(f"TurboQuant indexing speedup vs PQ: {pq_row[1] / tq_row[1]:.1f}×")
            print(f"TurboQuant recall delta vs PQ    : {tq_row[4] - pq_row[4]:+.3f}")


if __name__ == "__main__":
    main()
