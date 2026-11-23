"""Chấm tay ảo giác theo giao thức chấm mù (docs/hallucination_protocol.md).

Quy trình: trộn câu trả lời của mọi cấu hình, ẩn tên hệ thống, chia câu hỏi cho ba người chấm,
chọn khoảng 30% câu trả lời để người kế chấm lại, sau đó tổng hợp tỷ lệ và Cohen's kappa.
"""
from __future__ import annotations

import random
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

from src.metrics import bootstrap_ci

LABELS = ("dung", "dung_mot_phan", "sai_bia", "tu_choi_dung", "tu_choi_sai")
GRADERS = ("A", "B", "C")
# Người chấm lại phần của người chấm chính: A chấm lại phần của B, B của C, C của A.
SECOND_GRADER = {"B": "A", "C": "B", "A": "C"}
RECHECK_FRACTION = 0.3


def assign_questions(question_ids: Iterable[str], seed: int) -> dict[str, str]:
    """Chia câu hỏi cho ba người chấm chính theo vòng tròn sau khi trộn (mỗi người khoảng 1/3)."""
    ids = sorted(set(question_ids))
    random.Random(seed).shuffle(ids)
    return {qid: GRADERS[i % len(GRADERS)] for i, qid in enumerate(ids)}


def build_blind_sheets(
    gold: Sequence[Mapping[str, Any]],
    results: Mapping[str, Sequence[Mapping[str, Any]]],
    seed: int,
    recheck_fraction: float = RECHECK_FRACTION,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    """Dựng bảng chấm của từng người và khóa ánh xạ.

    Args:
        gold: các dòng gold (id, question, in_scope, article, clause, answers).
        results: tên hệ thống -> danh sách {id, answer}; mỗi id xuất hiện đúng một lần và phải có
            khóa `answer`; câu trả lời rỗng hoặc None được giữ là chuỗi rỗng.
        seed: hạt giống cho việc chia câu hỏi, trộn dòng và chọn câu chấm lại.
        recheck_fraction: tỉ lệ câu hỏi (không phải mục) của mỗi người chấm chính được chấm lại.
            Chọn theo câu hỏi, và câu được chọn có đủ câu trả lời của mọi hệ thống trong bảng của
            người chấm lại, nên số dòng theo câu hỏi không cho biết mục nào là chấm lại.

    Returns:
        (bảng chấm theo người chấm, khóa). Bảng chỉ chứa mã mục, câu hỏi, đáp án gold và câu
        trả lời; khóa ánh xạ mã mục -> hệ thống, câu hỏi, người chấm chính, người chấm lại.
    """
    rng = random.Random(seed)
    gold_by_id = {g["id"]: g for g in gold}
    assignment = assign_questions(gold_by_id, seed)

    items: list[dict[str, Any]] = []
    for system in sorted(results):
        seen: set[str] = set()
        for r in results[system]:
            if r["id"] not in gold_by_id:
                raise KeyError(f"{system}: id {r['id']} không có trong gold")
            if r["id"] in seen:
                raise ValueError(f"{system}: id {r['id']} xuất hiện nhiều lần")
            if "answer" not in r:
                raise ValueError(f"{system}: dòng id {r['id']} thiếu khóa 'answer'")
            seen.add(r["id"])
            items.append({"system": system, "qid": r["id"], "answer": (r["answer"] or "").strip()})
    missing = {s: sorted(set(gold_by_id) - {r["id"] for r in rs}) for s, rs in results.items()}
    missing = {s: m for s, m in missing.items() if m}
    if missing:
        raise ValueError(f"hệ thống thiếu câu trả lời: { {s: len(m) for s, m in missing.items()} }")

    rng.shuffle(items)
    codes = [f"X{i + 1:04d}" for i in range(len(items))]
    rng.shuffle(codes)
    key: dict[str, dict[str, Any]] = {}
    for code, it in zip(codes, items):
        it["code"] = code
        it["primary"] = assignment[it["qid"]]
        key[code] = {"system": it["system"], "id": it["qid"], "primary": it["primary"], "second": None}

    # Chọn khoảng 30% câu hỏi của từng người chấm chính; mọi câu trả lời của câu đó được chấm lại.
    questions_of: dict[str, list[str]] = {g: [] for g in GRADERS}
    for qid in sorted(assignment):
        questions_of[assignment[qid]].append(qid)
    rechecked: dict[str, str] = {}
    for primary, qids in questions_of.items():
        for qid in rng.sample(qids, round(recheck_fraction * len(qids))):
            rechecked[qid] = SECOND_GRADER[primary]
    for it in items:
        key[it["code"]]["second"] = rechecked.get(it["qid"])

    sheets: dict[str, list[dict[str, Any]]] = {g: [] for g in GRADERS}
    for it in items:
        k = key[it["code"]]
        g = gold_by_id[it["qid"]]
        row = {
            "code": it["code"],
            "question": g["question"],
            "gold_answers": " | ".join(g.get("answers") or []),
            "source": _source_code(g),
            "answer": it["answer"],
            "label": "",
            "note": "",
        }
        sheets[k["primary"]].append(row)
        if k["second"]:
            sheets[k["second"]].append(dict(row))
    for rows in sheets.values():
        rng.shuffle(rows)
    return sheets, key


def _source_code(g: Mapping[str, Any]) -> str:
    if not g.get("in_scope"):
        return "ngoài phạm vi"
    return f"D{g['article']}" if g.get("clause") is None else f"D{g['article']}.K{g['clause']}"


def check_grade(label: str, in_scope: bool) -> None:
    """Kiểm tra nhãn hợp lệ và nhất quán với phạm vi của câu hỏi."""
    if label not in LABELS:
        raise ValueError(f"nhãn không hợp lệ: {label!r} (cần một trong {LABELS})")
    if label == "tu_choi_dung" and in_scope:
        raise ValueError("tu_choi_dung chỉ dùng cho câu ngoài phạm vi")
    if label == "tu_choi_sai" and not in_scope:
        raise ValueError("tu_choi_sai chỉ dùng cho câu trong phạm vi")
    if label in ("dung", "dung_mot_phan") and not in_scope:
        raise ValueError("câu ngoài phạm vi không thể gắn dung/dung_mot_phan; trả lời thay vì từ chối là sai_bia")


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float | None:
    """Cohen's kappa giữa hai chuỗi nhãn; None nếu không có mục hoặc kỳ vọng ngẫu nhiên bằng 1."""
    if len(a) != len(b):
        raise ValueError("hai chuỗi nhãn khác độ dài")
    n = len(a)
    if n == 0:
        return None
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[l] * cb[l] for l in set(ca) | set(cb)) / (n * n)
    if expected == 1.0:
        return None
    return (observed - expected) / (1.0 - expected)


def merge_grades(
    sheets: Mapping[str, Sequence[Mapping[str, Any]]],
    key: Mapping[str, Mapping[str, Any]],
    gold: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str], list[tuple[str, str]], list[tuple[str, str]]]:
    """Gộp nhãn đã điền của các bảng.

    Returns:
        (nhãn chính theo mã mục, cặp nhãn (chính, lần hai) để tính kappa, danh sách cảnh báo/lỗi).
    """
    in_scope = {g["id"]: bool(g["in_scope"]) for g in gold}
    primary: dict[str, str] = {}
    second: dict[str, str] = {}
    problems: list[tuple[str, str]] = []
    seen_codes: set[tuple[str, str]] = set()
    for grader, rows in sheets.items():
        for row in rows:
            if "code" not in row or "label" not in row:
                raise ValueError(
                    f"bảng của người chấm {grader} thiếu cột code/label; kiểm tra dấu phân cách (cần dấu phẩy) "
                    "và lưu lại ở dạng CSV UTF-8"
                )
            code, label = row["code"], (row.get("label") or "").strip().lower()
            k = key.get(code)
            if k is None:
                problems.append((code, "mã không có trong khóa"))
                continue
            if k["id"] not in in_scope:
                raise ValueError(f"mã {code} thuộc câu hỏi {k['id']} không có trong gold; kiểm tra --split")
            if (grader, code) in seen_codes:
                problems.append((code, f"mã xuất hiện nhiều lần trong bảng của người chấm {grader}"))
                continue
            seen_codes.add((grader, code))
            if not label:
                problems.append((code, f"người chấm {grader} chưa điền nhãn"))
                continue
            try:
                check_grade(label, in_scope[k["id"]])
            except ValueError as exc:
                problems.append((code, str(exc)))
                continue
            if grader == k["primary"]:
                primary[code] = label
            elif grader == k["second"]:
                second[code] = label
    for code, k in key.items():
        if code not in primary:
            problems.append((code, "thiếu nhãn của người chấm chính"))
        if k["second"] and code not in second:
            problems.append((code, "thiếu nhãn của người chấm lại"))
    pairs = [(primary[c], second[c]) for c in second if c in primary]
    return primary, pairs, problems


def summarize(
    primary: Mapping[str, str],
    key: Mapping[str, Mapping[str, Any]],
    gold: Sequence[Mapping[str, Any]],
    extractive_systems: Iterable[str] = (),
    n_resamples: int = 2000,
) -> dict[str, dict[str, Any]]:
    """Tỷ lệ theo hệ thống, kèm bootstrap CI 95% theo câu hỏi.

    Ảo giác = sai_bia / số câu; tỷ lệ này ghi None với hệ thống trích xuất (span nằm trong đoạn
    truy hồi nên theo định nghĩa không bịa), chỉ báo số câu sai. Từ chối đúng tính trên câu
    ngoài phạm vi; từ chối sai tính trên câu trong phạm vi.
    """
    extractive = set(extractive_systems)
    in_scope = {g["id"]: bool(g["in_scope"]) for g in gold}
    out: dict[str, dict[str, Any]] = {}
    systems = sorted({k["system"] for k in key.values()})
    unknown = extractive - set(systems)
    if unknown:
        raise ValueError(f"--extractive: không có hệ thống {sorted(unknown)}; các hệ thống có trong khóa: {systems}")
    for system in systems:
        rows = [(k["id"], primary[c]) for c, k in key.items() if k["system"] == system and c in primary]
        counts = Counter(label for _, label in rows)
        res: dict[str, Any] = {"n": len(rows), "counts": dict(counts)}
        n_in = sum(in_scope[q] for q, _ in rows)
        n_out = len(rows) - n_in

        def ci(scores: list[float]):
            if not scores:
                return None
            r = bootstrap_ci(scores, n_resamples=n_resamples)
            return {"estimate": r.estimate, "low": r.low, "high": r.high}

        res["dung"] = ci([float(l == "dung") for q, l in rows if in_scope[q]])
        res["dung_mot_phan"] = ci([float(l == "dung_mot_phan") for q, l in rows if in_scope[q]])
        res["tu_choi_sai"] = ci([float(l == "tu_choi_sai") for q, l in rows if in_scope[q]])
        res["tu_choi_dung"] = ci([float(l == "tu_choi_dung") for q, l in rows if not in_scope[q]])
        res["n_in_scope"], res["n_out_of_scope"] = n_in, n_out
        if system in extractive:
            res["hallucination"] = None
            res["wrong_count"] = counts.get("sai_bia", 0)
        else:
            res["hallucination"] = ci([float(l == "sai_bia") for _, l in rows])
        out[system] = res
    return out
