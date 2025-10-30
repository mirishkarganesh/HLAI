import os
import shutil
import math
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from .utils import ensure_dir, compute_id
from .parser import parse_pdf
from .chunker import build_chunks
from .indexer import ChunkIndex
from .downloader import download_from_url
from .qa import SimpleGenerator
from .rerank import Reranker


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
PDF_DIR = os.path.join(DATA_DIR, "pdfs")
INDEX_DIR = os.path.join(DATA_DIR, "index")
IDS_DIR = os.path.join(INDEX_DIR, "by_id")
META_PATH = os.path.join(INDEX_DIR, "metadata.json")


class CorpusManager:
    def __init__(self, device: Optional[str] = None) -> None:
        ensure_dir(PDF_DIR)
        ensure_dir(IDS_DIR)
        self.device = device
        self._indices: Dict[str, ChunkIndex] = {}
        self._file_map: Dict[str, Dict[str, str]] = {}
        self._generator = SimpleGenerator()
        self._reranker = Reranker()
        self._load_metadata()
        self._load_existing_indices()

    def _load_metadata(self) -> None:
        if os.path.exists(META_PATH):
            try:
                with open(META_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self._file_map = data
            except Exception:
                self._file_map = {}
        else:
            self._file_map = {}

    def _save_metadata(self) -> None:
        ensure_dir(os.path.dirname(META_PATH))
        tmp = META_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._file_map, f, ensure_ascii=False, indent=2)
        os.replace(tmp, META_PATH)

    def _generate_uid_for_file(self, pdf_path: str) -> str:
        st = os.stat(pdf_path)
        base = os.path.basename(pdf_path)
        uid = compute_id(base, str(st.st_size), str(int(st.st_mtime)), str(time.time()))
        return uid

    def _index_dir_for_uid(self, uid: str) -> str:
        return os.path.join(IDS_DIR, uid)

    def _build_index_for_pdf(self, pdf_path: str, uid: str) -> ChunkIndex:
        title = os.path.basename(pdf_path)
        parsed = parse_pdf(pdf_path, paper_id=title)
        chunks = build_chunks(parsed)
        index = ChunkIndex(device=self.device)
        index.build(chunks)
        idx_dir = self._index_dir_for_uid(uid)
        ensure_dir(idx_dir)
        index.save(idx_dir)
        return index

    def _ensure_index_for_uid(self, uid: str, pdf_path: str) -> ChunkIndex:
        if uid in self._indices:
            return self._indices[uid]
        idx_dir = self._index_dir_for_uid(uid)
        try:
            index = ChunkIndex.load(idx_dir)
        except Exception:
            index = self._build_index_for_pdf(pdf_path, uid)
        self._indices[uid] = index
        return index

    def _load_existing_indices(self) -> None:
        for filename, meta in list(self._file_map.items()):
            uid = meta.get("uid")
            pdf_path = meta.get("pdf_path")
            if not uid or not pdf_path or not os.path.exists(pdf_path):
                self._file_map.pop(filename, None)
                continue
            try:
                self._ensure_index_for_uid(uid, pdf_path)
            except Exception:
                # Keep metadata; index will be rebuilt lazily on demand
                pass
        self._save_metadata()

    def _aggregate_query(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        results: List[Tuple[Dict[str, Any], float]] = []
        num_files = max(1, len(self._indices))
        per_file_k = max(3, math.ceil(top_k * 3 / num_files))
        for uid, idx in self._indices.items():
            try:
                neighbors = idx.query(query, top_k=per_file_k)
            except Exception:
                continue
            for idx_i, score in neighbors:
                meta = dict(idx.meta[idx_i])
                meta["score"] = float(score)
                results.append((meta, float(score)))
        candidates = [m for m, _ in results]
        reranked = self._reranker.rerank(query, candidates, top_k)
        return reranked

    def add_pdfs(self, file_paths: List[str]) -> Dict[str, Any]:
        ensure_dir(PDF_DIR)
        added_items: List[Dict[str, str]] = []
        for src in file_paths:
            if not os.path.isfile(src) or not src.lower().endswith(".pdf"):
                continue
            filename = os.path.basename(src)
            dst = os.path.join(PDF_DIR, filename)
            if os.path.abspath(src) != os.path.abspath(dst):
                shutil.copy2(src, dst)
            else:
                dst = src
            uid = self._generate_uid_for_file(dst)
            self._file_map[filename] = {"uid": uid, "pdf_path": dst}
            self._ensure_index_for_uid(uid, dst)
            added_items.append({"filename": filename, "uid": uid})
        self._save_metadata()
        return {"added": len(added_items), "items": added_items}

    def add_from_url(self, url: str, limit: Optional[int] = None) -> Dict[str, Any]:
        items = download_from_url(PDF_DIR, url, limit=limit)
        added_items: List[Dict[str, str]] = []
        for it in items:
            filename = os.path.basename(it["pdf"])
            uid = self._generate_uid_for_file(it["pdf"])
            self._file_map[filename] = {"uid": uid, "pdf_path": it["pdf"]}
            self._ensure_index_for_uid(uid, it["pdf"])
            added_items.append({"filename": filename, "uid": uid})
        self._save_metadata()
        return {"added": len(added_items), "items": added_items}

    def ask(self, question: str, top_k: int = 7, style: str = "concise") -> Dict[str, Any]:
        if not self._indices:
            return {"answer": "No PDFs indexed yet.", "chunks": []}
        fused = self._aggregate_query(question, top_k=top_k)
        contexts = [r.get("content", "") for r in fused]
        ids = [r.get("id") or f"C{i+1}" for i, r in enumerate(fused)]
        self._generator.set_style(style)
        answer = self._generator.answer(question, contexts, ids=ids)
        return {"answer": answer, "chunks": fused}

    def delete_pdf(self, filename: str) -> Dict[str, Any]:
        meta = self._file_map.get(filename)
        if not meta:
            return {"deleted": False, "reason": "File not found"}
        uid = meta.get("uid")
        pdf_path = meta.get("pdf_path")
        if pdf_path and os.path.exists(pdf_path):
            try:
                os.remove(pdf_path)
            except Exception:
                return {"deleted": False, "reason": "Failed to remove file"}
        if uid in self._indices:
            del self._indices[uid]
        idx_dir = self._index_dir_for_uid(uid)
        if os.path.isdir(idx_dir):
            for root, dirs, files in os.walk(idx_dir, topdown=False):
                for fn in files:
                    try:
                        os.remove(os.path.join(root, fn))
                    except Exception:
                        pass
                for dn in dirs:
                    try:
                        os.rmdir(os.path.join(root, dn))
                    except Exception:
                        pass
            try:
                os.rmdir(idx_dir)
            except Exception:
                pass
        self._file_map.pop(filename, None)
        self._save_metadata()
        return {"deleted": True}

    def list_pdfs(self) -> List[str]:
        return sorted(list(self._file_map.keys()))

    def reset_indexes_and_metadata(self) -> Dict[str, Any]:
        # Clear in-memory
        self._indices.clear()
        self._file_map.clear()
        # Remove by_id folder contents
        if os.path.isdir(IDS_DIR):
            for root, dirs, files in os.walk(IDS_DIR, topdown=False):
                for fn in files:
                    try:
                        os.remove(os.path.join(root, fn))
                    except Exception:
                        pass
                for dn in dirs:
                    try:
                        os.rmdir(os.path.join(root, dn))
                    except Exception:
                        pass
        # Remove metadata file
        if os.path.exists(META_PATH):
            try:
                os.remove(META_PATH)
            except Exception:
                pass
        # Recreate base dirs
        ensure_dir(IDS_DIR)
        self._save_metadata()
        return {"status": "ok"}
