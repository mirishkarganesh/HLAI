import os
import argparse
from typing import List, Dict, Any
from rich import print
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.downloader import download_from_url
from src.parser import parse_pdf
from src.chunker import build_chunks
from src.indexer import ChunkIndex
from src.chat import ChatSession
from src.utils import compute_files_digest


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PDF_DIR = os.path.join(DATA_DIR, "pdfs")
INDEX_DIR = os.path.join(DATA_DIR, "index")


def parse_one(pdf_item: Dict[str, str]) -> List[Dict[str, Any]]:
    parsed = parse_pdf(pdf_item["pdf"], paper_id=pdf_item["title"])  # using title as id
    return build_chunks(parsed)


def build_index_from_pdfs(pdfs: List[Dict[str, str]], device: str | None, cache_dir: str) -> ChunkIndex:
    os.makedirs(cache_dir, exist_ok=True)
    digest = compute_files_digest([p["pdf"] for p in pdfs])
    cache_path = os.path.join(cache_dir, f"{digest}")

    # Try cache
    try:
        index = ChunkIndex.load(cache_path)
        return index
    except Exception:
        pass

    # Parallel parse
    all_chunks: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
        futures = [ex.submit(parse_one, p) for p in pdfs]
        for fut in as_completed(futures):
            all_chunks.extend(fut.result())

    index = ChunkIndex(device=device)
    index.build(all_chunks)
    index.save(cache_path)
    return index


def main(url: str, limit: int | None, style: str, best_only: bool, interactive: bool, question: str | None, device: str | None) -> None:
    os.makedirs(PDF_DIR, exist_ok=True)

    print(f"[bold]Downloading from[/bold] {url} ...")
    pdfs = download_from_url(PDF_DIR, url, limit=limit)
    if not pdfs:
        print("[red]No PDFs downloaded from the provided URL.[/red]")
        return
    print(f"Downloaded {len(pdfs)} PDFs")

    print("[bold]Indexing (with cache) ...[/bold]")
    index = build_index_from_pdfs(pdfs, device=device, cache_dir=os.path.join(INDEX_DIR, "cache"))

    session = ChatSession(index, style=style)

    if interactive:
        print("[green]Interactive chat. Type 'exit' to quit.[/green]")
        while True:
            try:
                q = input("You: ").strip()
            except EOFError:
                break
            if not q:
                continue
            if q.lower() in {"exit", "quit"}:
                break
            result = session.ask(q, top_k=7)
            print(f"Assistant: {result['answer']}")
            if not best_only:
                print("Retrieved chunks:")
                for r in result["chunks"]:
                    meta = f"{r.get('paper_id')} p.{r.get('page')} [{r.get('type')}]"
                    score = r.get('score', 0.0)
                    preview = (r.get("content") or "").replace("\n", " ")
                    if len(preview) > 160:
                        preview = preview[:160] + "..."
                    print(f"  - score={score:.3f} | {meta} | {preview}")
        return

    # Single-turn mode
    if not question:
        print("[red]No question provided. Use --interactive or --question.")
        return
    result = session.ask(question, top_k=7)

    if best_only:
        print(result["answer"])  # Only best answer
    else:
        print("Answer:")
        print(result["answer"])
        print("Retrieved chunks:")
        for r in result["chunks"]:
            meta = f"{r.get('paper_id')} p.{r.get('page')} [{r.get('type')}]"
            score = r.get('score', 0.0)
            preview = (r.get("content") or "").replace("\n", " ")
            if len(preview) > 160:
                preview = preview[:160] + "..."
            print(f"  - score={score:.3f} | {meta} | {preview}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, required=True, help="Page URL to crawl for papers/PDFs")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit; omit to fetch all")
    parser.add_argument("--style", type=str, default="concise", choices=["concise", "detailed", "bullet", "citation"]) 
    parser.add_argument("--best_only", action="store_true", help="If set, print only the final answer")
    parser.add_argument("--interactive", action="store_true", help="Start interactive chat mode")
    parser.add_argument("--question", type=str, required=False, help="The chat question to ask (single-turn)")
    parser.add_argument("--device", type=str, default=None, help="Device for encoders, e.g., 'cuda', 'cpu'")
    args = parser.parse_args()

    main(url=args.url, limit=args.limit, style=args.style, best_only=args.best_only, interactive=args.interactive, question=args.question, device=args.device)
