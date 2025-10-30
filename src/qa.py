from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class SimpleGenerator:
    def __init__(self, model_name: str = "google/flan-t5-small", device: str | None = None, style: str = "concise") -> None:
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        if device:
            self.model.to(device)
        self.style = style

    def set_style(self, style: str) -> None:
        self.style = style

    def _style_prefix(self) -> str:
        if self.style == "detailed":
            return "Write a thorough technical answer with inline citations like [C1], [C2]."
        if self.style == "bullet":
            return "Answer in concise bullet points with inline citations like [C1]."
        if self.style == "citation":
            return "Answer concisely and include [CITATION] tags referencing chunk ids."
        return "Answer briefly and precisely with citations."

    def answer(self, query: str, contexts: List[str], ids: List[str] | None = None, max_new_tokens: int = 128) -> str:
        if not contexts:
            return "Insufficient evidence."
        # Limit to top contexts for generation
        contexts = contexts[:4]
        if ids is None:
            ids = [f"C{i+1}" for i in range(len(contexts))]
        else:
            ids = ids[:len(contexts)]
        joined = []
        for cid, ctx in zip(ids, contexts):
            joined.append(f"[{cid}] {ctx}")
        context_str = "\n\n".join(joined)
        style_prefix = self._style_prefix()
        prompt = (
            f"{style_prefix} Use only the provided context. If the context does not contain the answer, respond exactly: Insufficient evidence.\n\n"
            f"Context:\n{context_str}\n\nQuestion: {query}\nAnswer (with citations):"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
