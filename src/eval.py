from typing import List, Dict, Any
import time
import math
from transformers import pipeline


_nli = None

def _get_nli():
    global _nli
    if _nli is None:
        try:
            _nli = pipeline("text-classification", model="roberta-large-mnli")
        except Exception:
            _nli = None
    return _nli


def chunk_relevancy_precision_at_k(retrieved: List[Dict[str, Any]], query: str, k: int = 5) -> float:
    lower_q = query.lower()
    keywords = [w for w in lower_q.split() if len(w) >= 5]
    hits = 0
    for item in retrieved[:k]:
        text = (item.get("content") or "").lower()
        if any(kw in text for kw in keywords):
            hits += 1
            continue
        refs = item.get("refs") or {}
        if any(refs.get(kind) for kind in ["figure", "table", "equation"]):
            if any(tok in lower_q for tok in ["figure", "table", "eq", "equation"]):
                hits += 1
    return hits / max(1, min(k, len(retrieved)))


def coverage_ratio(index_meta: List[Dict[str, Any]], original_pages: int | None) -> float:
    if not original_pages:
        return 0.0
    pages_covered = {m.get("page") for m in index_meta}
    return len(pages_covered) / max(1, int(original_pages))


def answer_correctness_proxy(answer: str, query: str) -> float:
    if not answer or answer.strip().lower().startswith("insufficient"):
        return 0.0
    lower = answer.lower()
    tokens = [t for t in query.lower().split() if len(t) >= 5]
    if not tokens:
        return 0.5 if len(lower) > 10 else 0.0
    return sum(1 for t in tokens if t in lower) / len(tokens)


def latency_ms(start_time: float) -> float:
    return (time.perf_counter() - start_time) * 1000.0


def hallucination_rate_proxy(answer: str, contexts: List[str]) -> float:
    context_all = " \n ".join(contexts).lower()
    a = answer.lower()
    nums = set([n for n in a.split() if any(ch.isdigit() for ch in n)])
    if not nums:
        return 0.0
    missing = [n for n in nums if n not in context_all]
    return len(missing) / max(1, len(nums))


def nli_support_score(answer: str, contexts: List[str]) -> float | None:
    clf = _get_nli()
    if clf is None:
        return None
    # Check if answer is entailed by any context chunk
    scores = []
    for ctx in contexts[:5]:
        premise = ctx.strip()
        hypothesis = answer.strip()
        if not premise or not hypothesis:
            continue
        out = clf(f"{premise} </s> {hypothesis}")
        # Expect labels entailment/neutral/contradiction
        label = out[0]['label'].lower()
        score = out[0]['score']
        if 'entail' in label:
            scores.append(score)
        elif 'contradict' in label:
            scores.append(-score)
        else:
            scores.append(0.0)
    if not scores:
        return 0.0
    return sum(scores) / len(scores)
