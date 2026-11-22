"""Unit test cho src/grading.py (chấm mù ảo giác)."""
import pytest

from src.grading import (
    GRADERS, SECOND_GRADER, assign_questions, build_blind_sheets, check_grade, cohen_kappa,
    merge_grades, summarize,
)


def make_gold(n_in=12, n_out=3):
    rows = [{"id": f"Q{i:02d}", "question": f"Câu hỏi {i}?", "in_scope": True, "article": i + 1,
             "clause": 1, "answers": [f"đáp án {i}"]} for i in range(n_in)]
    rows += [{"id": f"Q{n_in + j:02d}", "question": f"Ngoài phạm vi {j}?", "in_scope": False,
              "article": None, "clause": None, "answers": []} for j in range(n_out)]
    return rows


def make_results(gold, systems=("sys_a", "sys_b")):
    return {s: [{"id": g["id"], "answer": f"{s} trả lời {g['id']}"} for g in gold] for s in systems}


# --- chia câu hỏi ---------------------------------------------------------


def test_assign_questions_is_balanced_and_deterministic():
    ids = [f"Q{i:02d}" for i in range(15)]
    a = assign_questions(ids, seed=1)
    assert a == assign_questions(ids, seed=1)
    counts = [list(a.values()).count(g) for g in GRADERS]
    assert counts == [5, 5, 5]
    assert a != assign_questions(ids, seed=2)


# --- bảng chấm mù ---------------------------------------------------------


def test_sheets_hide_system_names_and_cover_every_item():
    gold = make_gold()
    sheets, key = build_blind_sheets(gold, make_results(gold), seed=7)
    assert len(key) == 2 * len(gold)
    for rows in sheets.values():
        for row in rows:
            assert set(row) == {"code", "question", "gold_answers", "source", "answer", "label", "note"}
            assert row["label"] == ""
    # Mã không để lộ tên hệ thống và không theo thứ tự câu hỏi.
    assert all(c.startswith("X") and "sys" not in c for c in key)


def test_all_systems_answers_of_a_question_go_to_same_primary_grader():
    gold = make_gold()
    _, key = build_blind_sheets(gold, make_results(gold, ("s1", "s2", "s3")), seed=3)
    by_question = {}
    for k in key.values():
        by_question.setdefault(k["id"], set()).add(k["primary"])
    assert all(len(v) == 1 for v in by_question.values())


def test_recheck_goes_to_next_grader_and_fraction_is_about_30_percent():
    gold = make_gold(n_in=27, n_out=3)
    _, key = build_blind_sheets(gold, make_results(gold), seed=5)
    rechecked = [k for k in key.values() if k["second"]]
    assert all(k["second"] == SECOND_GRADER[k["primary"]] for k in rechecked)
    assert 0.25 <= len(rechecked) / len(key) <= 0.35


def test_second_grader_sheet_contains_rechecked_items_and_not_others():
    gold = make_gold()
    sheets, key = build_blind_sheets(gold, make_results(gold), seed=11)
    for grader, rows in sheets.items():
        for row in rows:
            k = key[row["code"]]
            assert grader in (k["primary"], k["second"])
    codes_in_sheets = [r["code"] for rows in sheets.values() for r in rows]
    assert len(codes_in_sheets) == len(key) + sum(1 for k in key.values() if k["second"])


def test_build_rejects_missing_answers_and_unknown_ids():
    gold = make_gold()
    results = make_results(gold)
    results["sys_a"] = results["sys_a"][:-1]
    with pytest.raises(ValueError, match="thiếu"):
        build_blind_sheets(gold, results, seed=1)
    results = make_results(gold)
    results["sys_a"][0]["id"] = "ZZZ"
    with pytest.raises(KeyError):
        build_blind_sheets(gold, results, seed=1)


# --- nhãn -----------------------------------------------------------------


def test_check_grade_enforces_scope_consistency():
    check_grade("dung", True)
    check_grade("sai_bia", False)
    check_grade("tu_choi_dung", False)
    check_grade("tu_choi_sai", True)
    for label, scope in [("tu_choi_dung", True), ("tu_choi_sai", False), ("dung", False),
                         ("dung_mot_phan", False), ("khac", True)]:
        with pytest.raises(ValueError):
            check_grade(label, scope)


# --- kappa ----------------------------------------------------------------


def test_cohen_kappa_known_values():
    assert cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == pytest.approx(1.0)
    assert cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == pytest.approx(0.0)
    # 2x2: đồng ý 20/25 -> po = 0.8; pe = 0.5*0.6 + 0.5*0.4 = 0.5; kappa = 0.6
    a = ["x"] * 10 + ["y"] * 10 + ["x"] * 2 + ["y"] * 3
    b = ["x"] * 10 + ["y"] * 10 + ["y"] * 2 + ["x"] * 3
    assert cohen_kappa(a, b) == pytest.approx(0.6, abs=0.05)
    assert cohen_kappa([], []) is None
    assert cohen_kappa(["a", "a"], ["a", "a"]) is None
    with pytest.raises(ValueError):
        cohen_kappa(["a"], [])


# --- gộp và tổng hợp ------------------------------------------------------


def fill_sheets(sheets, key, gold, labeller):
    in_scope = {g["id"]: g["in_scope"] for g in gold}
    for rows in sheets.values():
        for row in rows:
            k = key[row["code"]]
            row["label"] = labeller(k["system"], k["id"], in_scope[k["id"]])


def test_merge_and_summarize_end_to_end():
    gold = make_gold(n_in=12, n_out=3)
    sheets, key = build_blind_sheets(gold, make_results(gold), seed=9)

    def labeller(system, qid, in_scope):
        if system == "sys_a":  # luôn trả lời đúng, từ chối đúng câu ngoài phạm vi
            return "dung" if in_scope else "tu_choi_dung"
        return "sai_bia" if not in_scope or int(qid[1:]) % 2 else "dung"  # sys_b bịa nhiều

    fill_sheets(sheets, key, gold, labeller)
    primary, pairs, problems = merge_grades(sheets, key, gold)
    assert problems == [] and len(primary) == len(key)
    assert pairs and all(a == b for a, b in pairs)  # cùng người gán nhãn máy móc -> kappa tối đa
    s = summarize(primary, key, gold, n_resamples=200)
    assert s["sys_a"]["hallucination"]["estimate"] == 0.0
    assert s["sys_a"]["tu_choi_dung"]["estimate"] == 1.0
    assert s["sys_b"]["hallucination"]["estimate"] > 0.5
    assert s["sys_a"]["n_in_scope"] == 12 and s["sys_a"]["n_out_of_scope"] == 3


def test_merge_reports_missing_and_invalid_labels():
    gold = make_gold(n_in=6, n_out=3)
    sheets, key = build_blind_sheets(gold, make_results(gold, ("s",)), seed=2)
    rows = sheets["A"]
    rows[0]["label"] = "dung"  # có thể sai phạm vi; các dòng còn lại để trống
    _, _, problems = merge_grades(sheets, key, gold)
    assert any("chưa điền" in msg for _, msg in problems)
    assert any("thiếu nhãn của người chấm chính" in msg for _, msg in problems)


def test_extractive_system_has_no_hallucination_rate():
    gold = make_gold(n_in=6, n_out=3)
    sheets, key = build_blind_sheets(gold, make_results(gold, ("gen", "ext")), seed=4)
    fill_sheets(sheets, key, gold, lambda s, q, ins: ("sai_bia" if ins else "tu_choi_dung") if s == "ext"
                else ("dung" if ins else "tu_choi_dung"))
    primary, _, problems = merge_grades(sheets, key, gold)
    assert problems == []
    s = summarize(primary, key, gold, extractive_systems=["ext"], n_resamples=100)
    assert s["ext"]["hallucination"] is None and s["ext"]["wrong_count"] == 6
    assert s["gen"]["hallucination"]["estimate"] == 0.0
