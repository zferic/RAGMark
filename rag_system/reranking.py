# reranking.py
import torch
from sentence_transformers import CrossEncoder


class Reranker:
    """
    Cross-encoder reranker.

    Takes a query and a list of retrieved docs, scores each (query, doc) pair,
    and returns the top_n docs sorted by descending score.

    Default model: cross-encoder/ms-marco-MiniLM-L-6-v2
    (fast, good quality for passage re-ranking)
    """

    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", device="cuda:0"):
        if isinstance(device, torch.device):
            device = str(device)
        self.device = device
        self.model = CrossEncoder(model_name, device=device)
        print(f"[Reranker] Loaded '{model_name}' on {device}")

    def rerank(self, query: str, docs: list[str], top_n: int) -> list[str]:
        """
        Score all (query, doc) pairs and return the top_n docs by score.
        If top_n >= len(docs), all docs are returned (still sorted by score).
        """
        if not docs:
            return docs

        top_n = min(top_n, len(docs))
        pairs = [(query, doc) for doc in docs]
        scores = self.model.predict(pairs, show_progress_bar=False)

        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[:top_n]]

    def rerank_batch(self, queries: list[str], docs_batch: list[list[str]], top_n: int) -> list[list[str]]:
        """
        Rerank a batch of (query, docs) pairs. Returns a list of reranked doc lists.
        """
        return [self.rerank(q, docs, top_n) for q, docs in zip(queries, docs_batch)]
