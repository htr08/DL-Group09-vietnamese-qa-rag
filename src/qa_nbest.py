"""PhoBERT extractive-QA postprocessing. No model/training dependencies.

Supports n-best start/end spans in one pre-tokenized context window and
threshold selection on a *development* set (never tune using Test).
"""
from __future__ import annotations

import math


def decode_window(input_ids, start_logits, end_logits, tokenizer,
                  n_best_size=20, max_answer_length=30, sep_token_id=2):
    """Return (top candidates, CLS null score), with end index inclusive.

    Expected input format from A's train_qa.py: <s> q </s></s> context </s>.
    Only context positions are eligible; CLS / question / separators excluded.
    """
    if n_best_size <= 0 or max_answer_length <= 0:
        raise ValueError("n_best_size and max_answer_length must be positive")
    if len(input_ids) != len(start_logits) or len(input_ids) != len(end_logits):
        raise ValueError("logits/input_ids length mismatch")
    null_score = float(start_logits[0]) + float(end_logits[0])
    context_start = next(
        (i + 2 for i in range(len(input_ids) - 1)
         if input_ids[i] == sep_token_id and input_ids[i + 1] == sep_token_id),
        None,
    )
    if context_start is None:
        raise ValueError("Cannot find pair separator </s></s> in QA input")
    context_end = len(input_ids) - 1  # exclusive; final </s> not eligible
    if context_start >= context_end:
        return [], null_score

    positions = range(context_start, context_end)
    starts = sorted(positions, key=lambda i: float(start_logits[i]), reverse=True)[:n_best_size]
    ends = sorted(positions, key=lambda i: float(end_logits[i]), reverse=True)[:n_best_size]
    pairs = []
    for s in starts:
        for e in ends:
            if e < s or e - s + 1 > max_answer_length:
                continue
            pairs.append((float(start_logits[s]) + float(end_logits[e]), s, e))
    pairs.sort(reverse=True)
    # Decode only high-scoring candidates, not all O(n_best^2) pairs.
    unique = []
    seen = set()
    for score, s, e in pairs:
        raw_text = tokenizer.decode(
            input_ids[s:e + 1],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()
        if not raw_text:
            continue
        text = raw_text.replace("_", " ")
        if text in seen:
            continue
        seen.add(text)
        unique.append({"text": text, "raw_text": raw_text, "score": score,
                       "start_token": s, "end_token": e})
        if len(unique) >= n_best_size:
            break
    return unique, null_score


def scores_at_threshold(items, threshold):
    """Returns (mean EM, mean F1) from dev per-example precomputed scores."""
    if not items:
        raise ValueError("Cannot evaluate an empty dev set")
    em = f1 = 0.0
    for item in items:
        answer = item["score_diff"] <= threshold
        em += item["answer_em"] if answer else item["null_em"]
        f1 += item["answer_f1"] if answer else item["null_f1"]
    return em / len(items), f1 / len(items)


def best_dev_threshold(items):
    """Exact sweep of every decision boundary; maximize dev F1, then EM.

    For score_diff = null_score - best_span_score, choose a span if <= threshold.
    Returns (threshold, EM, F1). Ties prefer the threshold closest to zero.
    """
    if not items:
        raise ValueError("Cannot tune on an empty dev set")
    rows = sorted(items, key=lambda r: r["score_diff"])
    em = sum(r["null_em"] for r in rows)
    f1 = sum(r["null_f1"] for r in rows)
    initial = math.nextafter(float(rows[0]["score_diff"]), -math.inf)
    best = (initial, em / len(rows), f1 / len(rows))
    i = 0
    while i < len(rows):
        boundary = rows[i]["score_diff"]
        while i < len(rows) and rows[i]["score_diff"] == boundary:
            r = rows[i]
            em += r["answer_em"] - r["null_em"]
            f1 += r["answer_f1"] - r["null_f1"]
            i += 1
        cur = (float(boundary), em / len(rows), f1 / len(rows))
        key_cur = (round(cur[2], 12), round(cur[1], 12), -abs(cur[0]))
        key_best = (round(best[2], 12), round(best[1], 12), -abs(best[0]))
        if key_cur > key_best:
            best = cur
    return best
