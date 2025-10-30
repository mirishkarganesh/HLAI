from typing import List, Dict, Any
import pdfplumber
import re
from .utils import normalize_whitespace, naive_sentence_split, extract_cross_refs

CAPTION_RE = re.compile(r"^(Figure|Fig\.|Table)\s+\d+[a-z]?[:.)\-]\s*(.+)$", re.IGNORECASE)


def parse_pdf(pdf_path: str, paper_id: str) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages):
            page_num = page_index + 1
            text = page.extract_text() or ""
            text = normalize_whitespace(text)

            # Extract tables
            tables_data: List[str] = []
            try:
                tables = page.extract_tables() or []
                for t in tables:
                    # Convert table to TSV-like string
                    rows = ["\t".join([c if c is not None else "" for c in row]) for row in t]
                    tables_data.append("\n".join(rows))
            except Exception:
                pass

            # Extract captions heuristically from text lines
            captions: List[str] = []
            for line in text.split("\n"):
                m = CAPTION_RE.match(line.strip())
                if m:
                    captions.append(line.strip())

            # Main text chunks
            if text:
                sentences = naive_sentence_split(text)
                if sentences:
                    chunks.append({
                        "type": "text",
                        "paper_id": paper_id,
                        "page": page_num,
                        "content": text,
                        "sentences": sentences,
                        "refs": extract_cross_refs(text),
                    })

            for cap in captions:
                chunks.append({
                    "type": "caption",
                    "paper_id": paper_id,
                    "page": page_num,
                    "content": cap,
                    "sentences": [cap],
                    "refs": extract_cross_refs(cap),
                })

            for tsv in tables_data:
                chunks.append({
                    "type": "table",
                    "paper_id": paper_id,
                    "page": page_num,
                    "content": tsv,
                    "sentences": [tsv],
                    "refs": extract_cross_refs(tsv),
                })
    return chunks
