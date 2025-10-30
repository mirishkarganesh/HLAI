from typing import List, Dict, Any
from .utils import sliding_window_chunks


def build_chunks(parsed_pages: List[Dict[str, Any]], max_chars: int = 1200, overlap_sentences: int = 1) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for page_item in parsed_pages:
        if page_item["type"] == "text":
            for chunk in sliding_window_chunks(page_item.get("sentences", []), max_chars=max_chars, overlap_sentences=overlap_sentences):
                output.append({
                    "type": "text",
                    "paper_id": page_item["paper_id"],
                    "page": page_item["page"],
                    "content": chunk,
                    "refs": page_item.get("refs", {}),
                })
        else:
            # For captions and tables, keep as-is
            output.append({
                "type": page_item["type"],
                "paper_id": page_item["paper_id"],
                "page": page_item["page"],
                "content": page_item["content"],
                "refs": page_item.get("refs", {}),
            })
    # Deduplicate identical chunks per paper/page/type
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for ch in output:
        key = (ch["paper_id"], ch["page"], ch["type"], ch["content"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ch)
    return deduped
