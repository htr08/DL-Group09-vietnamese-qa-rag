from __future__ import annotations

import unicodedata
from collections import Counter
from typing import Sequence


def normalize_answer(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = "".join(
        " " if unicodedata.category(char).startswith("P") else char
        for char in text
    )
    return " ".join(text.split())


def exact_match_score(prediction: str, ground_truths: str | Sequence[str]) -> bool:
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]

    prediction = normalize_answer(prediction)

    return any(
        prediction == normalize_answer(answer)
        for answer in ground_truths
    )


def token_f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()

    if not prediction_tokens or not ground_truth_tokens:
        return float(prediction_tokens == ground_truth_tokens)

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(prediction_tokens)
    recall = num_common / len(ground_truth_tokens)

    return 2 * precision * recall / (precision + recall)


def f1_score(prediction: str, ground_truths: str | Sequence[str]) -> float:
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]

    return max(
        (token_f1_score(prediction, answer) for answer in ground_truths),
        default=0.0,
    )
