"""RAG-focused helpers built on top of :mod:`tiny_turboquant.vector_index`.

The API is intentionally lightweight and research-first. It compresses embedding
payloads, supports candidate retrieval + optional full-precision reranking, and
provides small document/chunk helpers for demos and benchmarking.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

import torch
import torch.nn.functional as F

from .metrics import exact_overlap, mrr_at_k, ndcg_at_k, precision_at_k, recall_at_k
from .vector_index import CompressedVectorIndex, VectorSearchResult


@dataclass(frozen=True)
class DocumentChunk:
    """A text chunk that can be embedded and stored in a compressed RAG index."""

    id: object
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RAGSearchResult:
    """Search result containing text and retrieval metadata."""

    id: object
    text: str
    score: float
    index: int
    metadata: dict[str, Any]


def chunk_text(text: str, *, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Split text into simple character chunks with overlap.

    This is deliberately dependency-free. For production RAG, replace this with
    a tokenizer-aware splitter.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = str(text)
    if not text:
        return []

    chunks: list[str] = []
    step = chunk_size - overlap
    start = 0
    while start < len(text):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def load_texts_from_jsonl(path: str | os.PathLike[str], *, text_field: str = "text") -> list[str]:
    """Load a JSONL file where each record contains a text field."""
    texts: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if text_field not in row:
                raise KeyError(f"line {line_no}: missing field {text_field!r}")
            texts.append(str(row[text_field]))
    return texts


def load_texts_from_folder(
    folder: str | os.PathLike[str],
    *,
    suffixes: Sequence[str] = (".txt", ".md"),
    recursive: bool = True,
) -> list[str]:
    """Load text files from a folder."""
    base = Path(folder)
    pattern = "**/*" if recursive else "*"
    allowed = {s.lower() for s in suffixes}
    texts: list[str] = []
    for p in sorted(base.glob(pattern)):
        if p.is_file() and p.suffix.lower() in allowed:
            texts.append(p.read_text(encoding="utf-8", errors="ignore"))
    return texts


def make_chunks(
    texts: Sequence[str],
    *,
    chunk_size: int = 800,
    overlap: int = 100,
    ids: Optional[Sequence[object]] = None,
    metadatas: Optional[Sequence[dict[str, Any]]] = None,
) -> list[DocumentChunk]:
    """Create :class:`DocumentChunk` objects from raw documents."""
    if ids is not None and len(ids) != len(texts):
        raise ValueError("ids length must match texts length")
    if metadatas is not None and len(metadatas) != len(texts):
        raise ValueError("metadatas length must match texts length")

    out: list[DocumentChunk] = []
    for doc_idx, text in enumerate(texts):
        doc_id = ids[doc_idx] if ids is not None else doc_idx
        base_meta = dict(metadatas[doc_idx]) if metadatas is not None else {}
        for chunk_idx, chunk in enumerate(chunk_text(text, chunk_size=chunk_size, overlap=overlap)):
            cid = f"{doc_id}::chunk-{chunk_idx}"
            meta = dict(base_meta)
            meta.update({"source_id": doc_id, "chunk_index": chunk_idx})
            out.append(DocumentChunk(id=cid, text=chunk, metadata=meta))
    return out


class RAGCompressedIndex:
    """Compressed RAG index with optional embedding generation.

    The object stores document text/metadata and a :class:`CompressedVectorIndex`.
    It can be built from precomputed embeddings or, if ``sentence-transformers``
    is installed, directly from text documents.
    """

    def __init__(
        self,
        *,
        bits: int = 4,
        normalize: bool = True,
        store_original_for_rerank: bool = True,
        seed: int = 0,
        embedding_model: str | Any | None = None,
    ):
        self.bits = int(bits)
        self.normalize = bool(normalize)
        self.store_original_for_rerank = bool(store_original_for_rerank)
        self.seed = int(seed)
        self.embedding_model = embedding_model
        self.chunks: list[DocumentChunk] = []
        self.index: CompressedVectorIndex | None = None
        self.embedding_dim: int | None = None

    @classmethod
    def from_embeddings(
        cls,
        chunks: Sequence[str | DocumentChunk],
        embeddings: torch.Tensor,
        *,
        bits: int = 4,
        normalize: bool = True,
        store_original_for_rerank: bool = True,
        seed: int = 0,
    ) -> "RAGCompressedIndex":
        obj = cls(
            bits=bits,
            normalize=normalize,
            store_original_for_rerank=store_original_for_rerank,
            seed=seed,
        )
        obj.add_embeddings(chunks, embeddings)
        return obj

    @classmethod
    def from_documents(
        cls,
        documents: Sequence[str],
        *,
        embedding_model: str | Any = "sentence-transformers/all-MiniLM-L6-v2",
        bits: int = 4,
        chunk_size: int = 800,
        overlap: int = 100,
        batch_size: int = 64,
        normalize_embeddings: bool = True,
        store_original_for_rerank: bool = True,
        seed: int = 0,
    ) -> "RAGCompressedIndex":
        chunks = make_chunks(documents, chunk_size=chunk_size, overlap=overlap)
        obj = cls(
            bits=bits,
            normalize=True,
            store_original_for_rerank=store_original_for_rerank,
            seed=seed,
            embedding_model=embedding_model,
        )
        embeddings = obj.embed_texts(
            [c.text for c in chunks],
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
        )
        obj.add_embeddings(chunks, embeddings)
        return obj

    def _normalize_chunks(self, chunks: Sequence[str | DocumentChunk]) -> list[DocumentChunk]:
        out: list[DocumentChunk] = []
        for i, item in enumerate(chunks):
            if isinstance(item, DocumentChunk):
                out.append(item)
            else:
                out.append(DocumentChunk(id=i, text=str(item), metadata={}))
        return out

    def _get_embedder(self):
        if self.embedding_model is None:
            raise ValueError("embedding_model is required to embed text queries/documents")
        if isinstance(self.embedding_model, str):
            try:
                from sentence_transformers import SentenceTransformer
            except Exception as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "sentence-transformers is required for text embedding. "
                    "Install with: pip install tiny-turboquant[demos]"
                ) from exc
            self.embedding_model = SentenceTransformer(self.embedding_model)
        return self.embedding_model

    def embed_texts(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 64,
        normalize_embeddings: bool = True,
    ) -> torch.Tensor:
        embedder = self._get_embedder()
        if hasattr(embedder, "encode"):
            return embedder.encode(
                list(texts),
                convert_to_tensor=True,
                normalize_embeddings=normalize_embeddings,
                batch_size=batch_size,
                show_progress_bar=False,
            )
        raise TypeError("embedding_model must be a model name or object with encode(...)")

    def add_embeddings(self, chunks: Sequence[str | DocumentChunk], embeddings: torch.Tensor) -> "RAGCompressedIndex":
        normalized = self._normalize_chunks(chunks)
        if len(normalized) != int(embeddings.shape[0]):
            raise ValueError("number of chunks must match number of embeddings")
        self.chunks = normalized
        self.embedding_dim = int(embeddings.shape[1])
        ids = [c.id for c in self.chunks]
        self.index = CompressedVectorIndex(
            bits=self.bits,
            normalize=self.normalize,
            store_original_for_rerank=self.store_original_for_rerank,
            seed=self.seed,
        ).add(embeddings, ids=ids)
        return self

    @property
    def size(self) -> int:
        return len(self.chunks)

    def search(
        self,
        query: str | torch.Tensor,
        *,
        top_k: int = 5,
        rerank_top_k: int | None = 50,
    ) -> list[RAGSearchResult]:
        if self.index is None:
            raise RuntimeError("index is empty")
        if isinstance(query, str):
            q = self.embed_texts([query], normalize_embeddings=True)[0]
        else:
            q = query
        raw = self.index.search(q, top_k=top_k, rerank_top_k=rerank_top_k)
        by_id = {c.id: (i, c) for i, c in enumerate(self.chunks)}
        out: list[RAGSearchResult] = []
        for r in raw:
            if r.id in by_id:
                i, chunk = by_id[r.id]
            else:
                i = r.index
                chunk = self.chunks[i]
            out.append(
                RAGSearchResult(
                    id=chunk.id,
                    text=chunk.text,
                    score=r.score,
                    index=i,
                    metadata=dict(chunk.metadata),
                )
            )
        return out

    def memory_report(self) -> dict[str, float]:
        if self.index is None:
            raise RuntimeError("index is empty")
        fp32 = self.index.fp32_baseline_bytes()
        payload = self.index.compressed_payload_bytes()
        actual = self.index.actual_memory_bytes()
        return {
            "fp32_baseline_bytes": float(fp32),
            "compressed_payload_bytes": float(payload),
            "actual_index_bytes": float(actual),
            "payload_compression_ratio": float(fp32 / max(payload, 1)),
            "payload_memory_saved_pct": float((1.0 - payload / fp32) * 100.0),
            "actual_compression_ratio": float(fp32 / max(actual, 1)),
        }

    def evaluate_id_retrieval(
        self,
        queries: Sequence[str | torch.Tensor],
        relevant_ids: Sequence[Iterable[object]],
        *,
        top_k: int = 10,
        rerank_top_k: int | None = 50,
    ) -> dict[str, float]:
        recalls: list[float] = []
        precisions: list[float] = []
        mrrs: list[float] = []
        ndcgs: list[float] = []
        for query, rel in zip(queries, relevant_ids):
            results = self.search(query, top_k=top_k, rerank_top_k=rerank_top_k)
            ids = [r.id for r in results]
            recalls.append(recall_at_k(ids, rel, top_k))
            precisions.append(precision_at_k(ids, rel, top_k))
            mrrs.append(mrr_at_k(ids, rel, top_k))
            ndcgs.append(ndcg_at_k(ids, rel, top_k))
        return {
            "recall_at_k": float(sum(recalls) / max(len(recalls), 1)),
            "precision_at_k": float(sum(precisions) / max(len(precisions), 1)),
            "mrr_at_k": float(sum(mrrs) / max(len(mrrs), 1)),
            "ndcg_at_k": float(sum(ndcgs) / max(len(ndcgs), 1)),
        }

    def save(self, path: str | os.PathLike[str]) -> None:
        torch.save(
            {
                "version": 1,
                "bits": self.bits,
                "normalize": self.normalize,
                "store_original_for_rerank": self.store_original_for_rerank,
                "seed": self.seed,
                "embedding_dim": self.embedding_dim,
                "chunks": self.chunks,
                "index": self.index,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | os.PathLike[str], *, embedding_model: str | Any | None = None) -> "RAGCompressedIndex":
        data = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(
            bits=data["bits"],
            normalize=data["normalize"],
            store_original_for_rerank=data["store_original_for_rerank"],
            seed=data["seed"],
            embedding_model=embedding_model,
        )
        obj.embedding_dim = data.get("embedding_dim")
        obj.chunks = list(data["chunks"])
        obj.index = data["index"]
        return obj


def compare_retrieval_ids(
    baseline_ids: Sequence[object],
    compressed_ids: Sequence[object],
    relevant_ids: Iterable[object],
    *,
    k: int | None = None,
) -> dict[str, float]:
    """Convenience comparison between baseline and compressed retrieval lists."""
    return {
        "baseline_recall": recall_at_k(baseline_ids, relevant_ids, k),
        "compressed_recall": recall_at_k(compressed_ids, relevant_ids, k),
        "exact_overlap": exact_overlap(baseline_ids, compressed_ids, k),
    }


__all__ = [
    "DocumentChunk",
    "RAGSearchResult",
    "RAGCompressedIndex",
    "chunk_text",
    "make_chunks",
    "load_texts_from_jsonl",
    "load_texts_from_folder",
    "compare_retrieval_ids",
]
