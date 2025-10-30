from typing import List, Dict, Any
from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str | None = None) -> None:
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        pairs = [[query, c.get("content", "")] for c in candidates]
        scores = self.model.predict(pairs)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        # Sort by rerank score desc, fallback to retriever score
        candidates.sort(key=lambda x: (x.get("rerank_score", 0.0), x.get("score", 0.0)), reverse=True)
        return candidates[:top_k]
