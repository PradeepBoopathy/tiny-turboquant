"""Compressed vector-index API for RAG/retrieval experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import torch
import torch.nn.functional as F

from .bitpack import pack_indices, tensor_nbytes, unpack_indices
from .quantizer import TurboQuantMSE


@dataclass(frozen=True)
class VectorSearchResult:
    id: object
    score: float
    index: int


class CompressedVectorIndex:
    """Low-bit compressed vector index with optional reranking.

    This is a research-friendly in-memory index. It is useful for studying
    memory/recall tradeoffs before moving to a production vector database.
    """

    def __init__(
        self,
        *,
        bits: int = 4,
        normalize: bool = True,
        store_original_for_rerank: bool = False,
        seed: int = 0,
    ):
        if not 1 <= int(bits) <= 8:
            raise ValueError(f"bits must be in [1, 8], got {bits}")
        self.bits = int(bits)
        self.normalize = bool(normalize)
        self.store_original_for_rerank = bool(store_original_for_rerank)
        self.seed = int(seed)

        self.quantizer: Optional[TurboQuantMSE] = None
        self._packed: Optional[torch.Tensor] = None
        self._shape: Optional[tuple[int, ...]] = None
        self._ids: List[object] = []
        self._original: Optional[torch.Tensor] = None
        self._dtype: Optional[torch.dtype] = None
        self._device: Optional[torch.device] = None
        self._dim: Optional[int] = None

    def add(self, embeddings: torch.Tensor, ids: Optional[Sequence[object]] = None) -> "CompressedVectorIndex":
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2-D (N, D), got {tuple(embeddings.shape)}")
        if self._packed is not None:
            raise RuntimeError("This simple research index supports one add() call. Rebuild for new data.")

        x = embeddings.detach()
        self._dtype = x.dtype
        self._device = x.device
        n, d = int(x.shape[0]), int(x.shape[1])
        self._dim = d
        if ids is None:
            self._ids = list(range(n))
        else:
            if len(ids) != n:
                raise ValueError("ids length must match number of embeddings")
            self._ids = list(ids)

        x_work = x.float()
        if self.normalize:
            x_work = F.normalize(x_work, dim=-1)

        self.quantizer = TurboQuantMSE.build(d=d, bits=self.bits, device=x_work.device, seed=self.seed, dtype=x_work.dtype)
        idx = self.quantizer.quant(x_work)
        self._packed, self._shape = pack_indices(idx, self.bits)

        if self.store_original_for_rerank:
            self._original = x_work.contiguous()
        return self

    @property
    def size(self) -> int:
        return len(self._ids)

    @property
    def dim(self) -> int:
        if self._dim is None:
            raise RuntimeError("index is empty")
        return self._dim

    def _approx_embeddings(self) -> torch.Tensor:
        if self.quantizer is None or self._packed is None or self._shape is None:
            raise RuntimeError("index is empty; call add() first")
        idx = unpack_indices(self._packed, self.bits, self._shape)
        x = self.quantizer.dequant(idx)
        if self.normalize:
            x = F.normalize(x.float(), dim=-1)
        return x

    def search(
        self,
        query: torch.Tensor,
        *,
        top_k: int = 10,
        rerank_top_k: Optional[int] = None,
        rerank_embeddings: Optional[torch.Tensor] = None,
    ) -> List[VectorSearchResult]:
        if query.ndim == 1:
            q = query[None, :]
        elif query.ndim == 2 and query.shape[0] == 1:
            q = query
        else:
            raise ValueError("query must be shaped (D,) or (1, D)")
        if q.shape[-1] != self.dim:
            raise ValueError(f"query dim {q.shape[-1]} does not match index dim {self.dim}")

        q = q.float()
        if self.normalize:
            q = F.normalize(q, dim=-1)

        approx = self._approx_embeddings().to(q.device)
        candidate_count = int(rerank_top_k or top_k)
        candidate_count = max(int(top_k), min(candidate_count, self.size))
        scores = (q @ approx.T).squeeze(0)
        cand_scores, cand_idx = torch.topk(scores, k=candidate_count)

        if rerank_top_k is not None:
            if rerank_embeddings is not None:
                exact = rerank_embeddings.detach().float().to(q.device)
                if self.normalize:
                    exact = F.normalize(exact, dim=-1)
            elif self._original is not None:
                exact = self._original.to(q.device)
            else:
                raise ValueError("reranking requires store_original_for_rerank=True or rerank_embeddings=...")
            exact_candidates = exact.index_select(0, cand_idx)
            exact_scores = (q @ exact_candidates.T).squeeze(0)
            rerank_scores, order = torch.topk(exact_scores, k=min(top_k, candidate_count))
            final_idx = cand_idx.index_select(0, order)
            final_scores = rerank_scores
        else:
            final_scores = cand_scores[:top_k]
            final_idx = cand_idx[:top_k]

        return [
            VectorSearchResult(
                id=self._ids[int(i)],
                score=float(s),
                index=int(i),
            )
            for s, i in zip(final_scores.detach().cpu(), final_idx.detach().cpu())
        ]

    def actual_memory_bytes(self) -> int:
        total = 0
        if self._packed is not None:
            total += tensor_nbytes(self._packed)
        if self._original is not None:
            total += tensor_nbytes(self._original)
        return int(total)

    def compressed_payload_bytes(self) -> int:
        return 0 if self._packed is None else tensor_nbytes(self._packed)

    def fp32_baseline_bytes(self) -> int:
        return int(self.size * self.dim * 4)

    def compression_ratio(self, *, payload_only: bool = True) -> float:
        denom = self.compressed_payload_bytes() if payload_only else self.actual_memory_bytes()
        return self.fp32_baseline_bytes() / max(denom, 1)

    def save(self, path: str) -> None:
        """Persist the in-memory compressed index with torch.save."""
        import torch

        torch.save({
            "version": 1,
            "bits": self.bits,
            "normalize": self.normalize,
            "store_original_for_rerank": self.store_original_for_rerank,
            "seed": self.seed,
            "quantizer": self.quantizer,
            "packed": self._packed,
            "shape": self._shape,
            "ids": self._ids,
            "original": self._original,
            "dtype": self._dtype,
            "device": str(self._device) if self._device is not None else None,
            "dim": self._dim,
        }, path)

    @classmethod
    def load(cls, path: str) -> "CompressedVectorIndex":
        """Load an index previously saved with :meth:`save`."""
        import torch

        data = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(
            bits=data["bits"],
            normalize=data["normalize"],
            store_original_for_rerank=data["store_original_for_rerank"],
            seed=data["seed"],
        )
        obj.quantizer = data["quantizer"]
        obj._packed = data["packed"]
        obj._shape = data["shape"]
        obj._ids = list(data["ids"])
        obj._original = data["original"]
        obj._dtype = data.get("dtype")
        obj._device = torch.device("cpu")
        obj._dim = data["dim"]
        return obj
