import os
import pickle
from typing import List, Dict, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
try:
    from rank_bm25 import BM25Okapi  # type: ignore
except Exception:  # pragma: no cover
    BM25Okapi = None  # fallback to dense-only if not available
from .utils import ensure_dir, compute_id


class ChunkIndex:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device
        self.model: SentenceTransformer | None = None
        self.index: NearestNeighbors | None = None
        self.embeddings: np.ndarray | None = None
        self.meta: List[Dict[str, Any]] = []
        self._bm25: Any | None = None
        self._bm25_corpus: List[List[str]] = []

    def _get_model(self) -> SentenceTransformer:
        if self.model is None:
            self.model = SentenceTransformer(self.model_name, device=self.device)
        return self.model

    def build(self, chunks: List[Dict[str, Any]], n_neighbors: int = 8) -> None:
        texts = [c["content"] for c in chunks]
        model = self._get_model()
        embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True, batch_size=64)
        self.embeddings = embs.astype("float32")
        self.meta = []
        for ch in chunks:
            ch_id = compute_id(ch.get("paper_id", ""), str(ch.get("page", "")), ch.get("type", ""), ch.get("content", ""))
            m = dict(ch)
            m["id"] = ch_id
            self.meta.append(m)
        self.index = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
        self.index.fit(self.embeddings)
        # Build BM25 if available
        if BM25Okapi is not None:
            self._bm25_corpus = [t.lower().split() for t in texts]
            self._bm25 = BM25Okapi(self._bm25_corpus)
        else:
            self._bm25 = None
            self._bm25_corpus = []

    def query_dense(self, text: str, top_k: int = 5) -> List[Tuple[int, float]]:
        if self.index is None or self.embeddings is None:
            raise RuntimeError("Index not built")
        q = self._get_model().encode([text], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype("float32")
        distances, indices = self.index.kneighbors(q, n_neighbors=min(top_k, len(self.meta)))
        sim = 1.0 - distances[0]
        return list(zip(indices[0].tolist(), sim.tolist()))

    def query_bm25(self, text: str, top_k: int = 5) -> List[Tuple[int, float]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(text.lower().split())
        idxs = np.argsort(scores)[::-1][: min(top_k, len(scores))]
        return [(int(i), float(scores[int(i)])) for i in idxs]

    def retrieve(self, text: str, top_k: int = 10, alpha: float = 0.6) -> List[Tuple[int, float]]:
        dense = self.query_dense(text, top_k=top_k)
        bm25 = self.query_bm25(text, top_k=top_k)
        if not bm25:
            # BM25 not available; return dense only
            return dense[:top_k]
        scores: Dict[int, float] = {}
        # Normalize bm25 to [0,1]
        bm_vals = np.array([s for _, s in bm25])
        if bm_vals.max() > 0:
            bm_norm = {i: float(s / bm_vals.max()) for i, s in bm25}
        else:
            bm_norm = {i: 0.0 for i, s in bm25}
        for i, s in dense:
            scores[i] = scores.get(i, 0.0) + alpha * s
        for i, s in bm_norm.items():
            scores[i] = scores.get(i, 0.0) + (1 - alpha) * s
        combined = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return combined

    def query(self, text: str, top_k: int = 5) -> List[Tuple[int, float]]:
        return self.retrieve(text, top_k=top_k)

    def save(self, out_dir: str) -> None:
        ensure_dir(out_dir)
        np.save(os.path.join(out_dir, "embeddings.npy"), self.embeddings)
        with open(os.path.join(out_dir, "meta.pkl"), "wb") as f:
            pickle.dump(self.meta, f)
        with open(os.path.join(out_dir, "model_name.txt"), "w", encoding="utf-8") as f:
            f.write(self.model_name)

    @staticmethod
    def load(in_dir: str) -> "ChunkIndex":
        with open(os.path.join(in_dir, "model_name.txt"), "r", encoding="utf-8") as f:
            model_name = f.read().strip()
        ci = ChunkIndex(model_name=model_name)
        with open(os.path.join(in_dir, "meta.pkl"), "rb") as f:
            ci.meta = pickle.load(f)
        ci.embeddings = np.load(os.path.join(in_dir, "embeddings.npy"))
        ci.index = NearestNeighbors(n_neighbors=8, metric="cosine")
        ci.index.fit(ci.embeddings)
        # Rebuild BM25 from meta if available
        if BM25Okapi is not None:
            texts = [m.get("content", "") for m in ci.meta]
            ci._bm25_corpus = [t.lower().split() for t in texts]
            ci._bm25 = BM25Okapi(ci._bm25_corpus)
        else:
            ci._bm25 = None
            ci._bm25_corpus = []
        return ci
