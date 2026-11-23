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
    errors += [f"thiếu nhãn của tác giả {a} (cần đủ {', '.join(sorted(AUTHOR_QUOTA))})"
               for a in sorted(set(AUTHOR_QUOTA) - set(labels_by_author))]
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


def check_annotation_coverage(
    authors: Mapping[str, Sequence[Label]], annotations: Mapping[str, Sequence[Label]]
) -> list[str]:
    """Lỗi nếu nhãn của người gán nhãn thiếu tệp, thiếu id, thừa id hoặc trùng id so với tác giả."""
    errors: list[str] = []
    for author, labels in authors.items():
        ann = annotations.get(author)
        if ann is None:
            errors.append(f"thiếu nhãn của người gán nhãn cho tác giả {author}")
            continue
        want = [x["id"] for x in labels]
        got = [x["id"] for x in ann]
        missing, extra = sorted(set(want) - set(got)), sorted(set(got) - set(want))
        dup = sorted(i for i, n in Counter(got).items() if n > 1)
        for text, ids in (("thiếu id", missing), ("id thừa", extra), ("id trùng", dup)):
            if ids:
                errors.append(f"nhãn gán cho tác giả {author}: {text} {', '.join(ids[:5])}"
                              f"{'...' if len(ids) > 5 else ''} ({len(ids)})")
    return errors


RESOLUTION_VALUES = {"author_label", "annotator_label", "merged", "rewritten"}


def validate_resolutions(
    authors: Mapping[str, Sequence[Label]], final_labels: Mapping[str, Label]
) -> list[str]:
    """Lỗi nếu nhãn sau họp trỏ tới id không có hoặc ghi `resolution` không hợp lệ."""
    known = {x["id"] for labels in authors.values() for x in labels}
    errors = [f"resolution: id không có trong nhãn tác giả: {i}" for i in sorted(set(final_labels) - known)]
    for i, r in final_labels.items():
        if r.get("resolution") not in RESOLUTION_VALUES:
            errors.append(f"resolution {i}: giá trị '{r.get('resolution')}' không hợp lệ "
                          f"(một trong {', '.join(sorted(RESOLUTION_VALUES))})")
    return errors


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def validate_gold(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Kiểm tra gold cuối sau khi áp dụng resolutions: nhất quán trường và đúng quota (mục 6)."""
    errors: list[str] = []
    errors += [f"id trùng trong gold: {i}" for i, n in Counter(r["id"] for r in rows).items() if n > 1]
    for r in rows:
        i = r["id"]
        if r["split"] not in ("dev", "test"):
            errors.append(f"{i}: split '{r['split']}' không hợp lệ")
        if not isinstance(r["question"], str) or not r["question"].strip():
            errors.append(f"{i}: câu hỏi rỗng hoặc không phải chuỗi")
        elif _FORBIDDEN.search(r["question"]):
            errors.append(f"{i}: câu hỏi nêu số Điều/Khoản hoặc dùng cụm bị cấm")
        if not isinstance(r["in_scope"], bool):
            errors.append(f"{i}: in_scope phải là true/false ({r['in_scope']!r})")
            continue
        if not isinstance(r["answers"], list) or not all(isinstance(a, str) and a.strip() for a in r["answers"]):
            errors.append(f"{i}: answers phải là danh sách chuỗi không rỗng")
            continue
        if r["in_scope"]:
            if not _is_int(r["article"]):
                errors.append(f"{i}: trong phạm vi nhưng article không phải số nguyên ({r['article']!r})")
            if r["clause"] is not None and not _is_int(r["clause"]):
                errors.append(f"{i}: clause không phải số nguyên hoặc null ({r['clause']!r})")
            if not r["answers"]:
                errors.append(f"{i}: trong phạm vi nhưng không có đáp án")
            if r["question_type"] not in IN_SCOPE_TYPES:
                errors.append(f"{i}: question_type '{r['question_type']}' không hợp lệ cho câu trong phạm vi")
        else:
            if r["article"] is not None or r["clause"] is not None or r["answers"]:
                errors.append(f"{i}: ngoài phạm vi nhưng có article/clause/answers")
            if r["question_type"] not in OOS_TYPES:
                errors.append(f"{i}: question_type '{r['question_type']}' không hợp lệ cho câu ngoài phạm vi")
    for author, q in AUTHOR_QUOTA.items():
        mine = [r for r in rows if r["author"] == author]
        got = {"total": len(mine), "in": sum(r["in_scope"] is True for r in mine),
               "dev_in": sum(r["in_scope"] is True and r["split"] == "dev" for r in mine),
               "dev_out": sum(r["in_scope"] is False and r["split"] == "dev" for r in mine)}
        for key, n in got.items():
            if n != q[key]:
                errors.append(f"{author}: {key} = {n}, quota {q[key]}")
    return errors


def load_split(path: str | Path) -> tuple[int, set[str], dict[str, Any]]:
    """Đọc chia dev/test đã khóa (gold_split.json): (seed, dev_ids, toàn bộ nội dung tệp).

    Ném ValueError nếu tệp không đọc được hoặc thiếu/sai `seed`, `dev_ids`.
    """
    try:
        meta = json.loads(Path(path).read_text(encoding="utf-8"))
        ids = meta["dev_ids"]
        if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids) or len(set(ids)) != len(ids):
            raise ValueError("dev_ids phải là danh sách id (chuỗi) không trùng")
        return meta["seed"], set(ids), meta
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as e:
        raise ValueError(f"{Path(path).name} không hợp lệ: {e!r}") from e


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
