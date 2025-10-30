from typing import List, Dict, Any
from .indexer import ChunkIndex
from .retriever import retrieve
from .qa import SimpleGenerator


class ChatSession:
    def __init__(self, index: ChunkIndex, style: str = "concise") -> None:
        self.index = index
        self.history: List[Dict[str, str]] = []
        self.generator = SimpleGenerator(style=style)

    def set_style(self, style: str) -> None:
        self.generator.set_style(style)

    def _history_context(self, max_turns: int = 4) -> str:
        turns = self.history[-max_turns:]
        lines: List[str] = []
        for t in turns:
            lines.append(f"User: {t['q']}")
            lines.append(f"Assistant: {t['a']}")
        return "\n".join(lines)

    def ask(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        # Retrieve with history-sensitive query
        hist = self._history_context()
        effective_query = f"{hist}\n\nCurrent question: {query}" if hist else query
        results = retrieve(self.index, effective_query, top_k=top_k)
        contexts = [r["content"] for r in results]
        answer = self.generator.answer(query, contexts)
        self.history.append({"q": query, "a": answer})
        return {"answer": answer, "chunks": results}
