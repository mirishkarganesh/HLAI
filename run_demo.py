import os
import math
import argparse
from typing import List, Dict, Any
from rich import print

from src.downloader import download_latest_pdfs
from src.parser import parse_pdf
from src.chunker import build_chunks
from src.indexer import ChunkIndex
from src.retriever import retrieve
from src.qa import SimpleGenerator
from src.eval import chunk_relevancy_precision_at_k, coverage_ratio, answer_correctness_proxy, hallucination_rate_proxy, nli_support_score


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PDF_DIR = os.path.join(DATA_DIR, "pdfs")
INDEX_DIR = os.path.join(DATA_DIR, "index")


SAMPLE_QUERIES = [
    {"q": "What is SLA in the paper titled 'SLA: Beyond Sparsity in Diffusion Transformers'?", "k": 5},
    {"q": "In Table 1, what is the value of parameter α?", "k": 5},
    {"q": "In Figure 3, what phenomenon is illustrated?", "k": 5},
    {"q": "How does the proposed method compare versus baselines in terms of speed and accuracy?", "k": 7},
]


def main(limit: int = 3, force_rebuild: bool = False, style: str = "concise") -> None:
    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(INDEX_DIR, exist_ok=True)

    print("[bold]Downloading PDFs...[/bold]")
    papers = download_latest_pdfs(PDF_DIR, limit=limit)
    if not papers:
        print("[red]No papers downloaded from any source.[/red]")
        return

    print(f"Downloaded {len(papers)} papers")

    print("[bold]Parsing and chunking...[/bold]")
    all_chunks: List[Dict[str, Any]] = []
    total_pages = 0
    for p in papers:
        parsed = parse_pdf(p["pdf"], paper_id=p["title"])  # using title as id
        total_pages += len({item["page"] for item in parsed})
        per_paper_chunks = build_chunks(parsed)
        all_chunks.extend(per_paper_chunks)
    if not all_chunks:
        print("[red]No chunks produced from PDFs.[/red]")
        return

    print(f"Total chunks: {len(all_chunks)} across ~{total_pages} pages")

    print("[bold]Building index...[/bold]")
    index = ChunkIndex()
    index.build(all_chunks)

    print("[bold]Running demo queries...[/bold]")
    generator = SimpleGenerator(style=style)

    for i, q in enumerate(SAMPLE_QUERIES, 1):
        query = q["q"]
        k = q.get("k", 5)
        results = retrieve(index, query, top_k=k)
        contexts = [r["content"] for r in results]
        answer = generator.answer(query, contexts)

        # Metrics
        p_at_k = chunk_relevancy_precision_at_k(results, query, k=k)
        cov = coverage_ratio(index.meta, total_pages)
        halluc = hallucination_rate_proxy(answer, contexts)
        acc_proxy = answer_correctness_proxy(answer, query)
        nli_score = nli_support_score(answer, contexts)

        print(f"\n[cyan]Q{i}[/cyan]: {query}")
        print("Retrieved chunks (top-k with scores):")
        for r in results:
            meta = f"{r.get('paper_id')} p.{r.get('page')} [{r.get('type')}]"
            preview = (r.get("content") or "").replace("\n", " ")
            if len(preview) > 180:
                preview = preview[:180] + "..."
            print(f"  - score={r['score']:.3f} | {meta} | {preview}")
        print("Answer:")
        print(f"  {answer}")
        print("Manual check (short justification):")
        print("  Please verify the answer vs the retrieved previews above. This is a proxy demo.")
        print("Metrics:")
        print(f"  chunk_relevancy_precision@{k}: {p_at_k:.3f}")
        print(f"  coverage_ratio: {cov:.3f}")
        print(f"  answer_correctness_proxy: {acc_proxy:.3f}")
        print(f"  hallucination_rate_proxy: {halluc:.3f}")
        if nli_score is not None:
            print(f"  nli_support_score: {nli_score:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--style", type=str, default="concise", choices=["concise", "detailed", "bullet", "citation"]) 
    args = parser.parse_args()
    main(limit=args.limit, style=args.style)
