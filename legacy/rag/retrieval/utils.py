"""Utility helpers for retrieval pipeline."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from .types import HitType


def extract_hits(response: Any) -> List[HitType]:
    """Normalize qdrant response to a list of hits."""

    if response is None:
        return []
    if isinstance(response, list):
        return response
    points = getattr(response, "points", None)
    if points is not None:
        return list(points)
    result = getattr(response, "result", None)
    if isinstance(result, list):
        return result
    return []


def can_rank_hits(hits: List[HitType]) -> bool:
    """Check whether hits expose enough structure for fusion/rerank."""

    if not hits:
        return True
    sample = hits[0]
    if isinstance(sample, dict):
        return "payload" in sample
    return hasattr(sample, "payload")


def hit_payload(hit: HitType) -> Dict[str, Any]:
    """Extract payload from a qdrant hit."""

    if isinstance(hit, dict):
        payload = hit.get("payload")
        return payload if isinstance(payload, dict) else {}
    payload = getattr(hit, "payload", None)
    return payload if isinstance(payload, dict) else {}


def hit_score(hit: HitType) -> Optional[float]:
    """Extract score from a qdrant hit."""

    if isinstance(hit, dict):
        score = hit.get("score")
        return float(score) if isinstance(score, (float, int)) else None
    score = getattr(hit, "score", None)
    if isinstance(score, (float, int)):
        return float(score)
    return None


def hit_key(hit: HitType) -> str:
    """Build deterministic key for dedup/fusion."""

    payload = hit_payload(hit)
    metadata = payload.get("metadata") or {}
    chunk_id = payload.get("chunk_id") or metadata.get("chunk_id")
    if chunk_id:
        return str(chunk_id)

    hit_id = hit.get("id") if isinstance(hit, dict) else getattr(hit, "id", None)
    if hit_id is not None:
        return str(hit_id)

    return str(
        (
            payload.get("filename") or metadata.get("filename"),
            payload.get("page", metadata.get("page")),
            payload.get("chunk_index", metadata.get("chunk_index")),
            hash(payload.get("page_content", "")),
        )
    )


def extract_heading_text_from_metadata(metadata: Dict[str, Any]) -> str:
    """Extract heading-like text from chunk metadata."""

    heading = metadata.get("heading")
    if isinstance(heading, str) and heading.strip():
        return heading.strip()

    headings = metadata.get("headings")
    parts: List[str] = []
    if isinstance(headings, str) and headings.strip():
        parts.append(headings.strip())
    elif isinstance(headings, dict):
        text = headings.get("text")
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    elif isinstance(headings, Iterable):
        for item in headings:
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())

    section_path = metadata.get("section_path")
    if isinstance(section_path, str) and section_path.strip():
        parts.append(section_path.strip())

    return " | ".join(part for part in parts if part)


def extract_heading_text(hit: HitType) -> str:
    """Extract heading text from a hit payload."""

    payload = hit_payload(hit)
    metadata = payload.get("metadata") or {}
    return extract_heading_text_from_metadata(metadata)


def normalize_filters(filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize metadata filters for matching and cache keys."""

    if not filters:
        return {}
    out: Dict[str, Any] = {}
    for key, value in filters.items():
        if value is None:
            continue
        out[str(key)] = value
    return out


def hit_matches_filters(hit: HitType, filters: Dict[str, Any]) -> bool:
    """Check if hit metadata matches requested filters (exact/in-list)."""

    if not filters:
        return True
    payload = hit_payload(hit)
    metadata = payload.get("metadata") or {}
    merged = {**payload, **metadata}

    for key, expected in filters.items():
        actual = merged.get(key)
        if isinstance(expected, (list, tuple, set)):
            if actual not in expected:
                return False
        else:
            if actual != expected:
                return False
    return True
