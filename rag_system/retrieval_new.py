import os
import json
import pickle
from dataclasses import dataclass
from typing import Callable, List, Dict, Tuple, Optional

import numpy as np

import faiss
from rank_bm25 import BM25Okapi


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _normalize_l2(x: np.ndarray) -> np.ndarray:
    # safe L2 normalize for cosine similarity via inner product
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


def _to_numpy_float32(x) -> np.ndarray:
    # torch.Tensor or np.ndarray -> np.float32
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if x.dtype != np.float32:
        x = x.astype(np.float32)
    return x


@dataclass
class Retrieved:
    scores: List[List[float]]
    doc_ids: List[List[str]]
    doc_texts: Optional[List[List[str]]] = None


class DenseFaissIndex:
    """
    Builds/loads a FAISS index + metadata.
    Uses IndexFlatIP by default (cosine if normalized).
    """
    def __init__(
        self,
        index_path: str,
        meta_path: str,
        use_gpu: bool = False,
        normalize: bool = True,
        gpu_device: int = 0,
    ):
        self.index_path = index_path
        self.meta_path = meta_path
        self.use_gpu = use_gpu
        self.normalize = normalize
        self.gpu_device = gpu_device

        self._cpu_index = None
        self._index = None  # cpu or gpu
        self.doc_ids: List[str] = []
        self.doc_texts: List[str] = []

    def exists(self) -> bool:
        return os.path.exists(self.index_path) and os.path.exists(self.meta_path)

    def build(
        self,
        corpus_texts: List[str],
        doc_ids: List[str],
        embed_fn: Callable[[List[str]], "np.ndarray"],
        batch_size: int = 256,
    ) -> None:
        assert len(corpus_texts) == len(doc_ids), "doc_ids and corpus_texts must align"

        # Encode corpus
        embs = []
        for i in range(0, len(corpus_texts), batch_size):
            batch = corpus_texts[i : i + batch_size]
            batch_emb = _to_numpy_float32(embed_fn(batch))
            embs.append(batch_emb)
        embs = np.vstack(embs)  # [N, d]

        if self.normalize:
            embs = _normalize_l2(embs)

        d = embs.shape[1]
        cpu_index = faiss.IndexFlatIP(d)  # inner product; with L2-normalized => cosine
        cpu_index.add(embs)

        _ensure_dir(os.path.dirname(self.index_path) or ".")
        faiss.write_index(cpu_index, self.index_path)

        meta = {"doc_ids": doc_ids, "doc_texts": corpus_texts, "normalize": self.normalize}
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)

        self._cpu_index = cpu_index
        self.doc_ids = doc_ids
        self.doc_texts = corpus_texts

        self._index = self._maybe_to_gpu(cpu_index)

    def load(self) -> None:
        cpu_index = faiss.read_index(self.index_path)
        with open(self.meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.doc_ids = meta["doc_ids"]
        self.doc_texts = meta["doc_texts"]
        self.normalize = bool(meta.get("normalize", True))

        self._cpu_index = cpu_index
        self._index = self._maybe_to_gpu(cpu_index)

    def _maybe_to_gpu(self, cpu_index):
        if self.use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            return faiss.index_cpu_to_gpu(res, self.gpu_device, cpu_index)
        return cpu_index

    def ensure(
        self,
        corpus_texts: List[str],
        doc_ids: List[str],
        embed_fn: Callable[[List[str]], "np.ndarray"],
        batch_size: int = 256,
        rebuild: bool = False,
    ) -> None:
        if rebuild or not self.exists():
            self.build(corpus_texts, doc_ids, embed_fn, batch_size=batch_size)
        else:
            self.load()

    def search(self, query_embs, k: int = 10) -> Retrieved:
        assert self._index is not None, "Index not loaded. Call ensure() or load()."

        q = _to_numpy_float32(query_embs)  # [B, d] or [d]
        if q.ndim == 1:
            q = q[None, :]

        if self.normalize:
            q = _normalize_l2(q)

        scores, idx = self._index.search(q, k)  # [B,k]
        out_ids = [[self.doc_ids[j] for j in row] for row in idx]
        out_txt = [[self.doc_texts[j] for j in row] for row in idx]
        return Retrieved(scores=scores.tolist(), doc_ids=out_ids, doc_texts=out_txt)


class BM25Index:
    """
    Pure-Python BM25 index (rank_bm25).
    Stores tokenized corpus + doc_ids + doc_texts.
    """
    def __init__(self, bm25_path: str, tokenizer: Optional[Callable[[str], List[str]]] = None):
        self.bm25_path = bm25_path
        self.tokenizer = tokenizer or (lambda s: s.lower().split())

        self.bm25: Optional[BM25Okapi] = None
        self.doc_ids: List[str] = []
        self.doc_texts: List[str] = []
        self._tokenized: List[List[str]] = []

    def exists(self) -> bool:
        return os.path.exists(self.bm25_path)

    def build(self, corpus_texts: List[str], doc_ids: List[str]) -> None:
        assert len(corpus_texts) == len(doc_ids), "doc_ids and corpus_texts must align"
        tokenized = [self.tokenizer(t) for t in corpus_texts]
        bm25 = BM25Okapi(tokenized)

        _ensure_dir(os.path.dirname(self.bm25_path) or ".")
        with open(self.bm25_path, "wb") as f:
            pickle.dump(
                {
                    "doc_ids": doc_ids,
                    "doc_texts": corpus_texts,
                    "tokenized": tokenized,
                },
                f,
            )

        self.bm25 = bm25
        self.doc_ids = doc_ids
        self.doc_texts = corpus_texts
        self._tokenized = tokenized

    def load(self) -> None:
        with open(self.bm25_path, "rb") as f:
            data = pickle.load(f)
        self.doc_ids = data["doc_ids"]
        self.doc_texts = data["doc_texts"]
        self._tokenized = data["tokenized"]
        self.bm25 = BM25Okapi(self._tokenized)

    def ensure(self, corpus_texts: List[str], doc_ids: List[str], rebuild: bool = False) -> None:
        if rebuild or not self.exists():
            self.build(corpus_texts, doc_ids)
        else:
            self.load()

    def search(self, queries: List[str], k: int = 10) -> Retrieved:
        assert self.bm25 is not None, "BM25 not loaded. Call ensure() or load()."

        all_scores: List[List[float]] = []
        all_ids: List[List[str]] = []
        all_txt: List[List[str]] = []

        for q in queries:
            q_tok = self.tokenizer(q)
            scores = self.bm25.get_scores(q_tok)  # [N]
            # top-k indices
            top_idx = np.argsort(scores)[::-1][:k]
            all_scores.append([float(scores[i]) for i in top_idx])
            all_ids.append([self.doc_ids[i] for i in top_idx])
            all_txt.append([self.doc_texts[i] for i in top_idx])

        return Retrieved(scores=all_scores, doc_ids=all_ids, doc_texts=all_txt)


def rrf_fuse(one_query_rankings: List[List[str]], rrf_k: int = 60, top_k: int = 10) -> List[str]:
    from collections import defaultdict
    scores = defaultdict(float)
    for ranking in one_query_rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] += 1.0 / (rrf_k + rank + 1)
    return [d for d, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]]


class SimpleHybridRetriever:
    """
    One entry point to ensure indices + run dense/bm25/hybrid retrieval.
    """
    def __init__(self, dense: DenseFaissIndex, bm25: BM25Index, doc_id_to_text: Optional[Dict[str, str]] = None):
        self.dense = dense
        self.bm25 = bm25
        self.doc_id_to_text = doc_id_to_text  # optional convenience; can be built from meta

    def ensure_indices(
        self,
        corpus_texts: List[str],
        doc_ids: List[str],
        embed_fn: Callable[[List[str]], "np.ndarray"],
        dense_batch_size: int = 256,
        rebuild_dense: bool = False,
        rebuild_bm25: bool = False,
    ) -> None:
        self.dense.ensure(corpus_texts, doc_ids, embed_fn, batch_size=dense_batch_size, rebuild=rebuild_dense)
        self.bm25.ensure(corpus_texts, doc_ids, rebuild=rebuild_bm25)

        if self.doc_id_to_text is None:
            self.doc_id_to_text = dict(zip(doc_ids, corpus_texts))

    def dense_search(self, query_embs, k: int = 10) -> Retrieved:
        return self.dense.search(query_embs, k=k)

    def bm25_search(self, query_texts: List[str], k: int = 10) -> Retrieved:
        return self.bm25.search(query_texts, k=k)

    def hybrid_search(self, query_texts: List[str], query_embs, k: int = 10, rrf_k: int = 60) -> Retrieved:
        dense = self.dense.search(query_embs, k=k)
        bm25 = self.bm25.search(query_texts, k=k)

        fused_ids: List[List[str]] = []
        fused_txt: List[List[str]] = []
        fused_scores: List[List[float]] = []  # RRF scores (optional / not calibrated)

        for d_ids, b_ids in zip(dense.doc_ids, bm25.doc_ids):
            ids = rrf_fuse([d_ids, b_ids], rrf_k=rrf_k, top_k=k)
            fused_ids.append(ids)
            fused_txt.append([self.doc_id_to_text[i] for i in ids])
            # If you want numeric scores, you can also compute RRF scores similarly; omitted for simplicity.
            fused_scores.append([0.0] * len(ids))

        return Retrieved(scores=fused_scores, doc_ids=fused_ids, doc_texts=fused_txt)
