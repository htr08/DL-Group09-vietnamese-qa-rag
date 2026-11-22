"""Unit test cho src/gold.py (tập 100 câu miền Quy chế)."""
import pytest

from src.gold import (
    AUTHOR_QUOTA, agreement_summary, build_gold, choose_split_seed, clause_leakage,
    compare_labels, draw_split, source_code, validate_authors,
)


def lab(id_, art=21, clause=2, ans="2 tuần", in_scope=True, alt=(), q=None, qtype="so_lieu"):
    return {
        "id": id_, "question": q or f"Câu hỏi số {id_}?", "in_scope": in_scope,
        "article": art if in_scope else None, "clause": clause if in_scope else None,
        "answer": ans if in_scope else None, "alt_answers": list(alt),
        "question_type": qtype if in_scope else "oos_ngoai_mien", "note": "",
    }


def make_author(letter, n_in, n_out, start_article=1):
    rows = [lab(f"{letter}{i + 1:02d}", art=start_article + i // 3, clause=i % 3 + 1,
                q=f"Nội dung hỏi {letter} {i}?") for i in range(n_in)]
    rows += [lab(f"{letter}{n_in + j + 1:02d}", in_scope=False, q=f"Ngoài phạm vi {letter} {j}?")
             for j in range(n_out)]
    return rows


# --- source_code / compare_labels ----------------------------------------


def test_source_code():
    assert source_code(lab("A01", 21, 2)) == "D21.K2"
    assert source_code(lab("A02", 15, None)) == "D15"
    assert source_code(lab("A03", in_scope=False)) is None


def test_compare_same_clause_and_exact_answer():
    r = compare_labels(lab("A01", ans="Hai tuần."), lab("A01", ans="hai tuần"))
    assert r["source"] == "same_clause" and r["scope"] == "same"
    assert r["answer_em"] == 1.0 and r["answer_f1"] == 1.0


def test_compare_same_article_different_clause():
    r = compare_labels(lab("A01", 21, 2), lab("A01", 21, 3))
    assert r["source"] == "same_article"


def test_compare_different_article():
    assert compare_labels(lab("A01", 21, 2), lab("A01", 5, 2))["source"] == "different"


def test_compare_scope_disagreement_has_no_source_or_answer():
    r = compare_labels(lab("A01"), lab("A01", in_scope=False))
    assert r == {"scope": "different", "source": "n/a", "answer_f1": None, "answer_em": None}


def test_compare_both_out_of_scope_is_same_scope():
    r = compare_labels(lab("A01", in_scope=False), lab("A01", in_scope=False))
    assert r["scope"] == "same" and r["source"] == "n/a"


def test_compare_partial_answer_f1():
    r = compare_labels(lab("A01", ans="hai tuần kể từ đầu"), lab("A01", ans="hai tuần"))
    assert r["answer_em"] == 0.0
    assert r["answer_f1"] == pytest.approx(2 * 0.4 / 1.4)


def test_alt_answers_take_best_pair_and_can_be_ignored():
    author, annot = lab("A01", ans="hai tuần"), lab("A01", ans="2 tuần", alt=["hai tuần"])
    assert compare_labels(author, annot)["answer_em"] == 1.0
    assert compare_labels(author, annot, use_alt=False)["answer_em"] == 0.0


# --- agreement_summary ----------------------------------------------------


def test_agreement_summary_rates():
    pairs = [
        compare_labels(lab("A01"), lab("A01")),
        compare_labels(lab("A02", 5, 1), lab("A02", 5, 2)),
        compare_labels(lab("A03", 7, 1), lab("A03", 9, 1)),
        compare_labels(lab("A04"), lab("A04", in_scope=False)),
        compare_labels(lab("A05", in_scope=False), lab("A05", in_scope=False)),
    ]
    s = agreement_summary(pairs)
    assert s["n_total"] == 5 and s["n_both_in_scope"] == 3
    assert s["same_clause"] == pytest.approx(1 / 3)
    assert s["same_article"] == pytest.approx(2 / 3)
    assert s["scope_agree"] == pytest.approx(4 / 5)
    assert s["scope_disagree_count"] == 1


def test_agreement_summary_empty_denominator_is_none():
    s = agreement_summary([compare_labels(lab("A01", in_scope=False), lab("A01", in_scope=False))])
    assert s["same_clause"] is None and s["answer_f1"] is None


# --- validate_authors -----------------------------------------------------


def full_set():
    return {"A": make_author("A", 30, 3), "B": make_author("B", 30, 3, 15),
            "C": make_author("C", 30, 4, 30)}


def test_valid_set_has_no_errors_except_type_quotas():
    # 10 câu ngoài phạm vi đều gắn oos_ngoai_mien: thiếu hai nhóm còn lại.
    errors = validate_authors(full_set())
    assert errors == ["nhóm oos_gan_mien: 0 câu (tối thiểu 2)", "nhóm oos_tien_de_sai: 0 câu (tối thiểu 2)"]


def test_valid_set_after_fixing_oos_types():
    data = full_set()
    oos = [x for rows in data.values() for x in rows if not x["in_scope"]]
    for i, x in enumerate(oos):
        x["question_type"] = ["oos_gan_mien", "oos_tien_de_sai", "oos_ngoai_mien"][i % 3]
    assert validate_authors(data) == []


def test_validate_reports_wrong_counts_and_forbidden_phrases():
    data = full_set()
    data["A"] = data["A"][:-1]
    data["B"][0]["question"] = "Điều 21 quy định gì?"
    data["C"][1]["question"] = "Quy định hiện hành về học phí?"
    errors = "\n".join(validate_authors(data))
    assert "A: 32 câu, cần 33" in errors
    assert "B01: câu hỏi nêu số Điều/Khoản" in errors
    assert "C02: câu hỏi nêu số Điều/Khoản" in errors


def test_validate_flags_duplicates_article_limit_and_type_limits():
    data = full_set()
    data["B"][1]["id"] = "B01"
    data["B"][2]["question"] = data["B"][3]["question"]
    for x in data["C"][:4]:
        x["article"] = 40
    for x in data["A"][:11]:
        x["question_type"] = "co_khong"
    errors = "\n".join(validate_authors(data))
    assert "id trùng: B01" in errors
    assert "câu hỏi trùng" in errors
    assert "Điều 40: 4 câu" in errors
    assert "co_khong: 11 câu" in errors


def test_validate_flags_inconsistent_scope_fields():
    data = full_set()
    data["A"][0]["answer"] = None
    data["A"][-1]["answer"] = "không được có"
    errors = "\n".join(validate_authors(data))
    assert "A01: trong phạm vi nhưng thiếu" in errors
    assert "A33: ngoài phạm vi nhưng có" in errors


# --- chia dev/test --------------------------------------------------------


def test_draw_split_respects_quota_and_is_deterministic():
    data = full_set()
    dev = draw_split(data, seed=7)
    assert dev == draw_split(data, seed=7)
    for letter, q in AUTHOR_QUOTA.items():
        rows = data[letter]
        assert sum(x["id"] in dev and x["in_scope"] for x in rows) == q["dev_in"]
        assert sum(x["id"] in dev and not x["in_scope"] for x in rows) == q["dev_out"]
    assert len(dev) == 30


def test_choose_split_seed_minimises_clause_leakage():
    data = full_set()
    # B hỏi đúng các Khoản mà A đã hỏi -> bốc ngẫu nhiên dễ chia một Khoản sang cả dev lẫn test.
    for a_row, b_row in zip(data["A"], data["B"]):
        b_row["article"], b_row["clause"] = a_row["article"], a_row["clause"]
    flat = [x for rows in data.values() for x in rows]
    seed, dev, leak = choose_split_seed(data, max_seed=300)
    assert leak == clause_leakage(flat, dev)
    assert leak <= clause_leakage(flat, draw_split(data, 0))
    assert len(dev) == 30


# --- build_gold -----------------------------------------------------------


def test_build_gold_merges_alternatives_and_marks_split():
    data = {"A": [lab("A01", ans="2 tuần"), lab("A02", in_scope=False)]}
    ann = {"A": [lab("A01", ans="hai tuần", alt=["2 tuần"]), lab("A02", in_scope=False)]}
    rows = build_gold(data, ann, None, dev_ids={"A02"})
    r1, r2 = rows
    assert r1["answers"] == ["2 tuần", "hai tuần"] and r1["split"] == "test"
    assert r1["annotator"] == "C" and r1["resolution"] == "none"
    assert r1["agreement"]["source"] == "same_clause"
    assert r2["answers"] == [] and r2["split"] == "dev" and r2["article"] is None


def test_build_gold_applies_final_labels():
    data = {"B": [lab("B01", 21, 2, "2 tuần")]}
    ann = {"B": [lab("B01", 21, 3, "3 tuần")]}
    final = {"B01": {"article": 21, "clause": 2, "answers": ["2 tuần"], "resolution": "author_label"}}
    (row,) = build_gold(data, ann, final, dev_ids=set())
    assert row["clause"] == 2 and row["answers"] == ["2 tuần"]
    assert row["resolution"] == "author_label" and row["agreement"]["source"] == "same_article"
    assert row["annotator"] == "A"
