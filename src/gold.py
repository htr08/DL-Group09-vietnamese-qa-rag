"""Tập 100 câu miền Quy chế ĐHQGHN: so khớp nhãn, đồng thuận, kiểm tra và sinh gold.

Định dạng tệp và quy tắc theo docs/annotation_guide.md (mục 5-8). Mã đoạn nguồn
viết ngắn là D<điều>.K<khoản> (D15 cho Điều không chia Khoản).
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.qa_metrics import exact_match_score, f1_score

Label = Mapping[str, Any]

# Quota theo tác giả (mục 6 của hướng dẫn): trong phạm vi, ngoài phạm vi,
# dev (trong, ngoài), test (trong, ngoài).
AUTHOR_QUOTA = {
    "A": {"total": 33, "in": 30, "out": 3, "dev_in": 9, "dev_out": 1},
    "B": {"total": 33, "in": 30, "out": 3, "dev_in": 9, "dev_out": 1},
    "C": {"total": 34, "in": 30, "out": 4, "dev_in": 9, "dev_out": 1},
}
IN_SCOPE_TYPES = {"so_lieu", "dieu_kien", "thu_tuc", "dinh_nghia", "liet_ke", "bang", "co_khong"}
OOS_TYPES = {"oos_gan_mien", "oos_tien_de_sai", "oos_ngoai_mien"}
MAX_PER_ARTICLE = 3
MAX_TYPE = {"bang": 3, "co_khong": 10}

_FORBIDDEN = re.compile(
    r"\b(điều|khoản)\s*\d+|hiện hành|hiện nay|mới nhất|theo điều này|ở đây|như trên",
    re.IGNORECASE,
)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def source_code(label: Label) -> str | None:
    """Mã đoạn nguồn D<điều>.K<khoản>; None nếu ngoài phạm vi."""
    if not label.get("in_scope") or label.get("article") is None:
        return None
    if label.get("clause") is None:
        return f"D{label['article']}"
    return f"D{label['article']}.K{label['clause']}"


def _answers(label: Label, use_alt: bool = True) -> list[str]:
    alts = label.get("alt_answers", []) if use_alt else []
    return [a for a in [label.get("answer"), *alts] if a]


# --------------------------------------------------------------------------
# Đồng thuận (mục 7)
# --------------------------------------------------------------------------


def compare_labels(author: Label, annotator: Label, use_alt: bool = True) -> dict[str, Any]:
    """So khớp nhãn lượt đầu của một câu; trả về khóa `agreement` (mục 5.3).

    use_alt=False chỉ so `answer` với `answer`, bỏ qua `alt_answers` (dùng khi alt_answers
    của người gán nhãn có thể đã được bổ sung sau khi thấy nhãn của tác giả).
    """
    same_scope = bool(author["in_scope"]) == bool(annotator["in_scope"])
    result: dict[str, Any] = {
        "scope": "same" if same_scope else "different",
        "source": "n/a",
        "answer_f1": None,
        "answer_em": None,
    }
    if not (author["in_scope"] and annotator["in_scope"]):
        return result
    if author["article"] == annotator["article"]:
        same_clause = author.get("clause") == annotator.get("clause")
        result["source"] = "same_clause" if same_clause else "same_article"
    else:
        result["source"] = "different"
    # Câu có alt_answers: lấy cặp cho điểm cao nhất.
    result["answer_f1"] = max(
        (f1_score(a, [b]) for a in _answers(author, use_alt) for b in _answers(annotator, use_alt)),
        default=0.0,
    )
    result["answer_em"] = float(
        any(exact_match_score(a, _answers(annotator, use_alt)) for a in _answers(author, use_alt))
    )
    return result


def agreement_summary(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Tổng hợp ba chỉ số (a) đoạn nguồn, (b) đáp án, (c) phạm vi từ danh sách `agreement`."""
    both = [p for p in pairs if p["source"] != "n/a"]
    n_both = len(both)

    def rate(values: Sequence[float]) -> float | None:
        return sum(values) / len(values) if values else None

    return {
        "n_total": len(pairs),
        "scope_agree": rate([p["scope"] == "same" for p in pairs]),
        "scope_disagree_count": sum(p["scope"] == "different" for p in pairs),
        "n_both_in_scope": n_both,
        "same_clause": rate([p["source"] == "same_clause" for p in both]),
        "same_article": rate([p["source"] in ("same_clause", "same_article") for p in both]),
        "answer_em": rate([p["answer_em"] for p in both]),
        "answer_f1": rate([p["answer_f1"] for p in both]),
    }


# --------------------------------------------------------------------------
# Kiểm tra nhãn của tác giả (mục 9)
# --------------------------------------------------------------------------


def validate_authors(labels_by_author: Mapping[str, Sequence[Label]]) -> list[str]:
    """Trả về danh sách lỗi (rỗng nếu hợp lệ). Kiểm tra mọi mục không cần mở PDF."""
    errors: list[str] = []
    all_labels: list[tuple[str, Label]] = []
    for author, labels in labels_by_author.items():
        quota = AUTHOR_QUOTA.get(author)
        if quota is None:
            errors.append(f"tác giả lạ: {author}")
            continue
        n_in = sum(bool(x["in_scope"]) for x in labels)
        if len(labels) != quota["total"]:
            errors.append(f"{author}: {len(labels)} câu, cần {quota['total']}")
        if n_in != quota["in"]:
            errors.append(f"{author}: {n_in} câu trong phạm vi, cần {quota['in']}")
        for x in labels:
            if not x["id"].startswith(author):
                errors.append(f"{x['id']}: id không bắt đầu bằng {author}")
        all_labels += [(author, x) for x in labels]

    ids = Counter(x["id"] for _, x in all_labels)
    errors += [f"id trùng: {i}" for i, n in ids.items() if n > 1]
    qs = Counter(re.sub(r"\s+", " ", x["question"]).strip().lower() for _, x in all_labels)
    errors += [f"câu hỏi trùng: {q[:60]}" for q, n in qs.items() if n > 1]

    for _, x in all_labels:
        if _FORBIDDEN.search(x["question"]):
            errors.append(f"{x['id']}: câu hỏi nêu số Điều/Khoản hoặc dùng cụm bị cấm")
        if x["in_scope"] and (x.get("article") is None or not x.get("answer")):
            errors.append(f"{x['id']}: trong phạm vi nhưng thiếu article/answer")
        if not x["in_scope"] and (x.get("article") is not None or x.get("answer")):
            errors.append(f"{x['id']}: ngoài phạm vi nhưng có article/answer")

    per_article = Counter(x["article"] for _, x in all_labels if x["in_scope"])
    errors += [f"Điều {a}: {n} câu (tối đa {MAX_PER_ARTICLE})"
               for a, n in per_article.items() if n > MAX_PER_ARTICLE]

    types = Counter(x.get("question_type") for _, x in all_labels)
    missing = [x["id"] for _, x in all_labels if x.get("question_type") is None]
    if missing:
        errors.append(f"thiếu question_type: {len(missing)} câu ({', '.join(missing[:5])}...)")
    else:
        for x in (x for _, x in all_labels):
            valid = OOS_TYPES if not x["in_scope"] else IN_SCOPE_TYPES
            if x["question_type"] not in valid:
                errors.append(f"{x['id']}: question_type '{x['question_type']}' không hợp lệ")
        errors += [f"{t}: {types[t]} câu (tối đa {m})" for t, m in MAX_TYPE.items() if types[t] > m]
        errors += [f"nhóm {t}: {types[t]} câu (tối thiểu 2)" for t in sorted(OOS_TYPES) if types[t] < 2]
    return errors


# --------------------------------------------------------------------------
# Chia dev/test (mục 6) và sinh gold cuối (mục 5.3)
# --------------------------------------------------------------------------


def _draw(ids: Sequence[str], n_dev: int, seed: int) -> set[str]:
    return set(random.Random(seed).sample(sorted(ids), n_dev))


def draw_split(labels_by_author: Mapping[str, Sequence[Label]], seed: int) -> set[str]:
    """Tập id dev: bốc độc lập trong từng nhóm (trong/ngoài phạm vi) của từng tác giả."""
    dev: set[str] = set()
    for author, labels in labels_by_author.items():
        quota = AUTHOR_QUOTA[author]
        in_ids = [x["id"] for x in labels if x["in_scope"]]
        out_ids = [x["id"] for x in labels if not x["in_scope"]]
        dev |= _draw(in_ids, quota["dev_in"], seed)
        dev |= _draw(out_ids, quota["dev_out"], seed)
    return dev


def clause_leakage(labels: Sequence[Label], dev_ids: set[str]) -> int:
    """Số Khoản (mã nguồn) có câu ở cả dev lẫn test."""
    dev = {source_code(x) for x in labels if x["in_scope"] and x["id"] in dev_ids}
    test = {source_code(x) for x in labels if x["in_scope"] and x["id"] not in dev_ids}
    return len(dev & test)


def choose_split_seed(
    labels_by_author: Mapping[str, Sequence[Label]], max_seed: int = 2000
) -> tuple[int, set[str], int]:
    """Chọn seed (từ 0) cho ít Khoản bị chia sang cả dev lẫn test nhất; hòa thì seed nhỏ hơn.

    Vẫn giữ đúng quota của mục 6; chỉ chọn trong các cách bốc hợp lệ, không chọn tay câu nào.
    Trả về (seed, dev_ids, số Khoản bị rò).
    """
    flat = [x for labels in labels_by_author.values() for x in labels]
    best: tuple[int, set[str], int] | None = None
    for seed in range(max_seed):
        dev = draw_split(labels_by_author, seed)
        leak = clause_leakage(flat, dev)
        if best is None or leak < best[2]:
            best = (seed, dev, leak)
            if leak == 0:
                break
    assert best is not None
    return best


def build_gold(
    authors: Mapping[str, Sequence[Label]],
    annotations: Mapping[str, Sequence[Label]],
    final_labels: Mapping[str, Label] | None,
    dev_ids: set[str],
) -> list[dict[str, Any]]:
    """Ghép nhãn cuối theo mục 5.3.

    authors: nhãn lượt đầu của tác giả, khóa là chữ cái tác giả.
    annotations: nhãn lượt đầu của người gán nhãn, khóa là chữ cái TÁC GIẢ được gán.
    final_labels: nhãn sau họp (id -> {in_scope, article, clause, answer, answers, resolution,
        question?}); câu không có trong đây giữ nhãn của tác giả (resolution = none).
    """
    # Vòng tròn A→B→C→A: A gán câu của B, B gán câu của C, C gán câu của A.
    annotator_of = {"B": "A", "C": "B", "A": "C"}
    final_labels = final_labels or {}
    rows = []
    for author, labels in authors.items():
        ann = {x["id"]: x for x in annotations.get(author, [])}
        for x in labels:
            a = ann.get(x["id"])
            agreement = compare_labels(x, a) if a else None
            fin = final_labels.get(x["id"], {})
            in_scope = fin.get("in_scope", x["in_scope"])
            answers = fin.get("answers")
            if answers is None:
                answers = []
                if in_scope:
                    for t in _answers(x) + (_answers(a) if a else []):
                        if t not in answers:
                            answers.append(t)
            rows.append({
                "id": x["id"],
                "question": fin.get("question", x["question"]),
                "in_scope": in_scope,
                "article": fin.get("article", x["article"]) if in_scope else None,
                "clause": fin.get("clause", x["clause"]) if in_scope else None,
                "question_type": fin.get("question_type", x.get("question_type")),
                "answers": answers if in_scope else [],
                "author": author,
                "annotator": annotator_of[author],
                "split": "dev" if x["id"] in dev_ids else "test",
                "agreement": agreement and {
                    "source": agreement["source"],
                    "answer_f1": agreement["answer_f1"],
                    "scope": agreement["scope"],
                },
                "resolution": fin.get("resolution", "none"),
            })
    return rows
