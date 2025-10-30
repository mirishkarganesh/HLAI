import os
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse

from src.corpus import CorpusManager


app = FastAPI(title="PDF RAG Chatbot", version="1.0.0")

# Single global corpus (can be extended to multi-tenant by namespace)
manager = CorpusManager()


@app.post("/upload_pdf")
async def upload_pdf(files: List[UploadFile] = File(...)):
    saved_paths: List[str] = []
    for uf in files:
        # Save to disk under data/pdfs
        contents = await uf.read()
        dst = os.path.join(os.path.dirname(__file__), "data", "pdfs", uf.filename)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as f:
            f.write(contents)
        saved_paths.append(dst)
    result = manager.add_pdfs(saved_paths)
    return JSONResponse({"status": "ok", "added": result.get("added", 0), "files": [os.path.basename(p) for p in saved_paths], "items": result.get("items", [])})


@app.post("/give_url")
async def give_url(url: str = Form(...), limit: Optional[int] = Form(None)):
    result = manager.add_from_url(url, limit=limit)
    return JSONResponse({"status": "ok", "added": result.get("added", 0), "items": result.get("items", [])})


@app.post("/ask")
async def ask(question: str = Form(...), top_k: int = Form(7), style: str = Form("concise"), best_only: bool = Form(True)):
    result = manager.ask(question, top_k=top_k, style=style)
    if best_only:
        return JSONResponse({"answer": result.get("answer", "")})
    return JSONResponse(result)


@app.post("/delete_pdf")
async def delete_pdf(filename: str = Form(...)):
    result = manager.delete_pdf(filename)
    status = "ok" if result.get("deleted") else "not_found"
    return JSONResponse({"status": status, **result})


@app.post("/reset")
async def reset_all():
    result = manager.reset_indexes_and_metadata()
    return JSONResponse(result)


@app.get("/")
async def root():
    return {"service": "pdf-rag-chatbot", "endpoints": ["/upload_pdf", "/give_url", "/ask", "/delete_pdf", "/reset"], "pdfs": manager.list_pdfs()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=4000, reload=False)
