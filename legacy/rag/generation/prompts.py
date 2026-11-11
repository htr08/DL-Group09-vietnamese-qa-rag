"""Re-export prompt cho generation package."""

from ..prompts import (
    NO_CONTEXT_ANSWER,
    RAG_USER_PROMPT as RAG_USER_PROMPT_TEMPLATE,
    SALE_ASSISTANT_SYSTEM_PROMPT,
)

__all__ = [
    "NO_CONTEXT_ANSWER",
    "RAG_USER_PROMPT_TEMPLATE",
    "SALE_ASSISTANT_SYSTEM_PROMPT",
]
