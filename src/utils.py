import os
import re
import time
import json
import hashlib
from typing import List, Dict, Any


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: Any) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_whitespace(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def naive_sentence_split(text: str) -> List[str]:
    # Very simple sentence splitter without external deps
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(\[])", text)
    sentences = [p.strip() for p in parts if p and p.strip()]
    return sentences


def sliding_window_chunks(sentences: List[str], max_chars: int = 1200, overlap_sentences: int = 1) -> List[str]:
    chunks: List[str] = []
    start = 0
    while start < len(sentences):
        current: List[str] = []
        total = 0
        i = start
        while i < len(sentences) and total + len(sentences[i]) + 1 <= max_chars:
            current.append(sentences[i])
            total += len(sentences[i]) + 1
            i += 1
        if not current:
            # Fallback for very long single sentence
            current = [sentences[start][:max_chars]]
            i = start + 1
        chunks.append(" ".join(current))
        if i >= len(sentences):
            break
        start = max(i - overlap_sentences, start + 1)
    return chunks


def compute_id(*parts: str) -> str:
    m = hashlib.sha256()
    for p in parts:
        m.update(p.encode("utf-8"))
    return m.hexdigest()[:16]


class Stopwatch:
    def __init__(self) -> None:
        self._t0 = time.perf_counter()

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000.0


FIGURE_REF_RE = re.compile(r"(?:Fig(?:ure)?\s*)(\d+[a-z]?)", re.IGNORECASE)
TABLE_REF_RE = re.compile(r"(?:Table\s*)(\d+[a-z]?)", re.IGNORECASE)
EQUATION_REF_RE = re.compile(r"(?:Eq(?:uation)?\.?\s*)(\d+[a-z]?)", re.IGNORECASE)


def extract_cross_refs(text: str) -> Dict[str, List[str]]:
    refs: Dict[str, List[str]] = {"figure": [], "table": [], "equation": []}
    for kind, regex in [("figure", FIGURE_REF_RE), ("table", TABLE_REF_RE), ("equation", EQUATION_REF_RE)]:
        matches = regex.findall(text or "")
        if matches:
            refs[kind] = list(dict.fromkeys(matches))
    return refs


def compute_files_digest(file_paths: List[str]) -> str:
    m = hashlib.sha256()
    for p in sorted(file_paths):
        try:
            st = os.stat(p)
            m.update(p.encode("utf-8"))
            m.update(str(st.st_size).encode("utf-8"))
            m.update(str(int(st.st_mtime)).encode("utf-8"))
        except FileNotFoundError:
            continue
    return m.hexdigest()[:16]
