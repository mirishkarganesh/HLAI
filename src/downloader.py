import os
import re
from typing import List, Dict, Optional
import requests
from bs4 import BeautifulSoup
import feedparser
from .utils import ensure_dir

HF_DATE_URL = "https://huggingface.co/papers/date/2025-09-30"
ARXIV_RECENT_API = "http://export.arxiv.org/api/query?search_query=all&start=0&max_results={limit}&sortBy=submittedDate&sortOrder=descending"


def _is_valid_paper_href(href: str) -> bool:
    if not href or not href.startswith("/papers/"):
        return False
    if "/date/" in href or "/trending" in href:
        return False
    if "#" in href or "?" in href:
        return False
    return True


def fetch_paper_list(limit: int = 3) -> List[Dict[str, str]]:
    return fetch_paper_list_from_url(HF_DATE_URL, limit=limit)


def fetch_paper_list_from_url(url: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    items: List[Dict[str, str]] = []

    # Collect paper detail pages
    for a in soup.find_all("a"):
        href = a.get("href") or ""
        if _is_valid_paper_href(href):
            title = a.get_text(" ", strip=True) or href.rsplit("/", 1)[-1]
            full = f"https://huggingface.co{href}"
            items.append({"title": title, "url": full})

    # Also collect any direct PDF links on the page itself
    for a in soup.find_all("a"):
        href = a.get("href") or ""
        if href.endswith(".pdf"):
            title = a.get_text(" ", strip=True) or href.rsplit("/", 1)[-1]
            full = href if href.startswith("http") else f"https://huggingface.co{href}"
            items.append({"title": title, "pdf": full, "page": url})

    # Deduplicate
    dedup: List[Dict[str, str]] = []
    seen = set()
    for it in items:
        key = it.get("pdf") or it.get("url")
        if not key or key in seen:
            continue
        seen.add(key)
        dedup.append(it)

    if limit is not None:
        return dedup[:limit]
    return dedup


_ARXIV_ID_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})(?:v\d+)?", re.IGNORECASE)


def _find_arxiv_id(html: str) -> str | None:
    m = _ARXIV_ID_RE.search(html or "")
    if m:
        return m.group(1)
    return None


def resolve_pdf_url(paper_url: str) -> str:
    resp = requests.get(paper_url, timeout=60)
    resp.raise_for_status()
    html = resp.text
    soup = BeautifulSoup(html, "html.parser")
    arxiv_id = _find_arxiv_id(html)
    if arxiv_id:
        return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    for a in soup.find_all("a"):
        href = a.get("href") or ""
        if href.endswith(".pdf"):
            if href.startswith("http"):
                return href
            return f"https://huggingface.co{href}"
    raise RuntimeError("Could not find PDF link for paper page: " + paper_url)


def download_pdf(pdf_url: str, out_dir: str, filename: str) -> str:
    ensure_dir(out_dir)
    path = os.path.join(out_dir, filename)
    with requests.get(pdf_url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    return path


def _download_from_hf(out_dir: str, limit: Optional[int]) -> List[Dict[str, str]]:
    papers = fetch_paper_list(limit=limit or None)
    results: List[Dict[str, str]] = []
    for p in papers:
        try:
            if "pdf" in p:
                pdf_url = p["pdf"]
                title = p.get("title") or os.path.basename(pdf_url)
            else:
                pdf_url = resolve_pdf_url(p["url"]) 
                title = p["title"]
            safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", title)[:80]
            filename = f"{safe_name}.pdf"
            path = download_pdf(pdf_url, out_dir, filename)
            results.append({"title": title, "page": p.get("url") or p.get("page"), "pdf": path})
        except Exception as e:
            print(f"[warn] Failed to download {p.get('title','')} from HF: {e}")
    return results


def _download_from_arxiv(out_dir: str, limit: int) -> List[Dict[str, str]]:
    url = ARXIV_RECENT_API.format(limit=limit)
    feed = feedparser.parse(url)
    results: List[Dict[str, str]] = []
    for entry in feed.entries[:limit]:
        title = entry.title
        pdf_link = None
        for link in entry.links:
            if link.rel == 'related' and link.type == 'application/pdf':
                pdf_link = link.href
                break
        if not pdf_link:
            m = re.search(r"\d{4}\.\d{4,5}", entry.id)
            if m:
                pdf_link = f"https://arxiv.org/pdf/{m.group(0)}.pdf"
        if not pdf_link:
            continue
        safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", title)[:80]
        filename = f"{safe_name}.pdf"
        try:
            path = download_pdf(pdf_link, out_dir, filename)
            results.append({"title": title, "page": entry.link, "pdf": path})
        except Exception as e:
            print(f"[warn] Failed to download from arXiv: {title}: {e}")
    return results


def download_latest_pdfs(out_dir: str, limit: int = 3) -> List[Dict[str, str]]:
    results = _download_from_hf(out_dir, limit)
    if not results:
        print("[info] Falling back to arXiv recent feed.")
        results = _download_from_arxiv(out_dir, limit or 20)
    return results


def download_from_url(out_dir: str, url: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
    papers = fetch_paper_list_from_url(url, limit=limit)
    results: List[Dict[str, str]] = []
    for p in papers:
        try:
            if "pdf" in p:
                pdf_url = p["pdf"]
                title = p.get("title") or os.path.basename(pdf_url)
            else:
                pdf_url = resolve_pdf_url(p["url"]) 
                title = p["title"]
            safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", title)[:80]
            filename = f"{safe_name}.pdf"
            path = download_pdf(pdf_url, out_dir, filename)
            results.append({"title": title, "page": p.get("url") or p.get("page", url), "pdf": path})
        except Exception as e:
            print(f"[warn] Failed to download {p.get('title','')} from {url}: {e}")
    return results
