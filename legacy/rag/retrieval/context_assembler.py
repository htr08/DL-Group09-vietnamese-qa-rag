"""Context assembly for generation."""

from typing import Any, Callable, Dict, List, Optional

from langchain_core.documents import Document

from .utils import extract_heading_text_from_metadata


class ContextAssembler:
    """Build context text and attach dynamic system prompt."""

    def __init__(
        self,
        default_top_k: int,
        system_prompt_provider: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.default_top_k = default_top_k
        self.system_prompt_provider = system_prompt_provider

    def assemble(
        self,
        question: str,
        documents: List[Document],
        top_k: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        limit = max(1, int(top_k or self.default_top_k))
        selected_docs = documents[:limit]

        prompt = (system_prompt or "").strip()
        if not prompt and self.system_prompt_provider is not None:
            try:
                prompt = (self.system_prompt_provider(question) or "").strip()
            except Exception:
                prompt = ""

        context_lines: List[str] = []
        for index, doc in enumerate(selected_docs, start=1):
            metadata = doc.metadata or {}
            heading_text = extract_heading_text_from_metadata(metadata)
            if heading_text:
                context_lines.append(f"[{index}] Heading: {heading_text}")
            context_lines.append(f"[{index}] {doc.page_content.strip()}")
        context = "\n\n".join(line for line in context_lines if line)

        return {
            "question": question,
            "system_prompt": prompt,
            "context": context,
            "documents": selected_docs,
        }

