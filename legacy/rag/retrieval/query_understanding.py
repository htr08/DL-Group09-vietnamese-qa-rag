"""Lightweight query understanding helpers."""

from __future__ import annotations

import re
from typing import List


class QueryUnderstanding:
    """
    Generate lightweight query rewrites for harder retrieval cases.

    This is heuristic (not LLM rewrite) to keep latency low.
    """

    def generate_variants(self, question: str, max_variants: int = 2) -> List[str]:
        clean = " ".join((question or "").strip().split())
        if not clean:
            return []

        variants: List[str] = [clean]
        lowered = clean.lower()

        # Variant 1: remove stop intent words, keep product terms.
        intent_words = {
            "mình",
            "toi",
            "tôi",
            "muốn",
            "can",
            "cần",
            "tim",
            "tìm",
            "cho",
            "xin",
            "hãy",
            "giúp",
            "giup",
            "về",
            "ve",
        }
        no_punct = re.sub(r"[^\w\s]", " ", lowered)
        terms = [w for w in no_punct.split() if w and w not in intent_words]
        compact = " ".join(terms).strip()
        if compact and compact != lowered:
            variants.append(compact)

        model_terms = []

        for word in clean.split():
            w_clean = re.sub(r"[^\w]", "", word.lower())
            if not w_clean:
                continue
            # Bắt các từ có chứa chữ số (ví dụ: 15, s24, 256gb, m3) - dấu hiệu tự nhiên của SKU
            if any(char.isdigit() for char in w_clean):
                model_terms.append(word.strip(".,!?()[]{}\"'"))

        if model_terms:
            model_query = " ".join(model_terms[:8]).strip()
            if model_query and model_query.lower() not in {v.lower() for v in variants}:
                variants.append(model_query)

        uniq: List[str] = []
        seen = set()
        for item in variants:
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            uniq.append(item)
            if len(uniq) >= max(1, max_variants):
                break
        return uniq
