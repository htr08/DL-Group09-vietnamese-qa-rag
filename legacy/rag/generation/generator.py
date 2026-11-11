from datetime import datetime, timezone
from typing import Iterator, List, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from ..config import RAGSettings, rag_settings
from ..models import RAGSource, RAGResponse
from .prompts import (
    RAG_USER_PROMPT_TEMPLATE,
    SALE_ASSISTANT_SYSTEM_PROMPT,
)


class GeminiSalesGenerator:
    """Generator chiu trach nhiem format context va goi Gemini."""

    def __init__(
        self,
        settings: RAGSettings = rag_settings,
        llm: Optional[ChatGoogleGenerativeAI] = None,
    ) -> None:
        self.settings = settings
        self.settings.validate_gemini_api_key()

        self.llm = llm or ChatGoogleGenerativeAI(
            model=settings.GEMINI_CHAT_MODEL,
            google_api_key=settings.GEMINI_API_KEY,
            temperature=settings.GEMINI_TEMPERATURE,
            max_output_tokens=settings.GEMINI_MAX_OUTPUT_TOKENS,
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SALE_ASSISTANT_SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="history", optional=True),
                ("human", RAG_USER_PROMPT_TEMPLATE),
            ]
        )

    def generate(
        self,
        question: str,
        documents: List[Document],
        sources: Optional[List[RAGSource]] = None,
        history: Optional[List[BaseMessage]] = None,
    ) -> RAGResponse:
        """Sinh cau tra loi cuoi cung va gan sources."""

        context = self._format_context(documents)
        chain = self.prompt | self.llm
        response = chain.invoke(
            {"question": question, "context": context, "history": history or []}
        )

        answer = getattr(response, "content", None) or str(response)
        return RAGResponse(answer=answer.strip(), sources=sources or [])

    def stream(
        self,
        question: str,
        documents: List[Document],
        history: Optional[List[BaseMessage]] = None,
        system_prompt: Optional[str] = None,
    ) -> Iterator[str]:
        """Yield tung text chunk tu Gemini (streaming)."""

        context = self._format_context(documents)
        current_date = datetime.now(timezone.utc).strftime("%d/%m/%Y")
        base_system = system_prompt or SALE_ASSISTANT_SYSTEM_PROMPT
        dated_system = f"{base_system}\n\nNgày hiện tại: {current_date}"

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", dated_system),
                MessagesPlaceholder(variable_name="history", optional=True),
                ("human", RAG_USER_PROMPT_TEMPLATE),
            ]
        )

        chain = prompt | self.llm
        for chunk in chain.stream(
            {"question": question, "context": context, "history": history or []}
        ):
            text = getattr(chunk, "content", None) or str(chunk)
            if text:
                yield text

    @staticmethod
    def _format_context(documents: List[Document]) -> str:
        """Format top-k chunks thanh context ngan gon cho prompt."""

        if not documents:
            return "(Không có tài liệu nội bộ liên quan đến câu hỏi này.)"

        context_blocks: List[str] = []
        for index, doc in enumerate(documents, start=1):
            metadata = doc.metadata or {}

            headings = metadata.get("headings", [])
            heading_str = (
                " > ".join(headings)
                if headings
                else metadata.get("filename", "unknown")
            )

            context_blocks.append(
                "\n".join(
                    [
                        f"[Nguon {index}: {heading_str}]",
                        doc.page_content.strip(),
                    ]
                )
            )

        return "\n\n---\n\n".join(context_blocks)
