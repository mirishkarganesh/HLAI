from typing import List, Dict, Any
from .indexer import ChunkIndex


def retrieve(index: ChunkIndex, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    neighbors = index.query(query, top_k=top_k)
    results: List[Dict[str, Any]] = []
    for idx, score in neighbors:
        meta = index.meta[idx]
        item = dict(meta)
        item["score"] = float(score)
        results.append(item)
    return results
