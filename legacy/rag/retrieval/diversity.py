"""Diversity-aware selection to reduce redundant chunks."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set

from langchain_core.documents import Document

from .query_encoder import tokenize
from .utils import extract_heading_text_from_metadata


def select_diverse_documents(
    documents: List[Document],
    limit: int,
    lambda_relevance: float = 0.75,
    section_penalty: float = 0.12,
) -> List[Document]:
    """
    Select top docs with an MMR-like objective.

    score = lambda * relevance - (1-lambda) * max_similarity - section_penalty(if same section)
    """

    if limit <= 0 or not documents:
        return []
    if len(documents) <= limit:
        return documents

    candidates = documents[:]
    selected: List[Document] = []
    selected_sections: Set[str] = set()

    while candidates and len(selected) < limit:
        best_idx = 0
        best_score = -(10**9)

        for idx, candidate in enumerate(candidates):
            relevance = _doc_relevance(candidate)
            redundancy = _max_similarity(candidate, selected)
            section_dup = _is_same_section(candidate, selected_sections)
            mmr_score = (
                lambda_relevance * relevance
                - (1.0 - lambda_relevance) * redundancy
                - (section_penalty if section_dup else 0.0)
            )
            if mmr_score > best_score:
                best_idx = idx
                best_score = mmr_score

        chosen = candidates.pop(best_idx)
        selected.append(chosen)

        metadata = chosen.metadata or {}
        section = _section_key(metadata)
        if section:
            selected_sections.add(section)

    return selected


def _doc_relevance(doc: Document) -> float:
    metadata = doc.metadata or {}
    value = metadata.get("score")
    if isinstance(value, (float, int)):
        return float(value)
    return 0.0


def _max_similarity(doc: Document, selected: List[Document]) -> float:
    if not selected:
        return 0.0
    return max(_text_similarity(doc, item) for item in selected)


def _text_similarity(left: Document, right: Document) -> float:
    left_tokens = _doc_tokens(left)
    right_tokens = _doc_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    return overlap / math.sqrt(len(left_tokens) * len(right_tokens))


def _doc_tokens(doc: Document) -> Set[str]:
    metadata = doc.metadata or {}
    heading = extract_heading_text_from_metadata(metadata)
    text = f"{heading} {doc.page_content[:600]}"
    return set(tokenize(text))


def _is_same_section(doc: Document, selected_sections: Set[str]) -> bool:
    section = _section_key(doc.metadata or {})
    if not section:
        return False
    return section in selected_sections


def _section_key(metadata: Dict) -> Optional[str]:
    headings = metadata.get("headings")
    if isinstance(headings, list) and headings:
        return " > ".join(headings).strip().lower()
    heading = extract_heading_text_from_metadata(metadata)
    if heading:
        return heading.lower()
    return None
