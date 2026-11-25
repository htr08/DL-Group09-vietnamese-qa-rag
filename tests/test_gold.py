"""Unit test cho src/gold.py (tập 100 câu miền Quy chế)."""
import json

import pytest

import scripts.gold_tool as gold_tool
from src.gold import (
    AUTHOR_QUOTA, agreement_summary, build_gold, check_annotation_coverage, check_author_counts,
    check_author_fields, check_disagreements_resolved, choose_split_seed, clause_leakage, compare_labels, draw_split, source_code, validate_authors, validate_gold,
    validate_resolutions,
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


# --- validate: thiếu tác giả ----------------------------------------------


def valid_set():
    data = full_set()
    oos = [x for rows in data.values() for x in rows if not x["in_scope"]]
    for i, x in enumerate(oos):
        x["question_type"] = ["oos_gan_mien", "oos_tien_de_sai", "oos_ngoai_mien"][i % 3]
    return data


def annotations_of(data):
    # khóa là chữ cái tác giả; nhãn người gán giữ nguyên nhãn tác giả (đồng thuận hoàn toàn)
    return {a: [dict(x) for x in rows] for a, rows in data.items()}


def test_validate_rejects_missing_author_even_if_present_ones_are_valid():
    data = valid_set()
    assert validate_authors(data) == []
    del data["C"]
    errors = validate_authors(data)
    assert any("thiếu nhãn của tác giả C" in e for e in errors)


# --- độ đầy đủ của nhãn gán -------------------------------------------------


def test_annotation_coverage_ok_and_flags_missing_extra_duplicate():
    data = valid_set()
    ann = annotations_of(data)
    assert check_annotation_coverage(data, ann) == []
    ann["A"] = ann["A"][1:] + [dict(ann["A"][1])]  # thiếu A01, trùng A02
    ann["B"].append({**ann["B"][0], "id": "B99"})  # thừa B99
    del ann["C"]
    errors = "\n".join(check_annotation_coverage(data, ann))
    assert "thiếu id A01" in errors and "id trùng A02" in errors
    assert "id thừa B99" in errors
    assert "thiếu nhãn của người gán nhãn cho tác giả C" in errors


# --- resolutions và gold cuối ----------------------------------------------


def _data_and_dev():
    data = valid_set()
    _, dev, _ = choose_split_seed(data, max_seed=50)
    return data, dev


def test_validate_resolutions_flags_unknown_id_and_bad_value():
    data = valid_set()
    final = {"A01": {"resolution": "rewritten"}, "ZZ9": {"resolution": "merged"},
             "A02": {"resolution": "vi_ly_do_khac"}, "A03": {}}
    errors = "\n".join(validate_resolutions(data, final))
    assert "không có trong nhãn tác giả: ZZ9" in errors
    assert "resolution A02" in errors and "resolution A03" in errors
    assert "resolution A01" not in errors


def test_validate_gold_accepts_consistent_set():
    data, dev = _data_and_dev()
    assert validate_gold(build_gold(data, annotations_of(data), {}, dev)) == []


def test_validate_gold_flags_in_scope_without_article_or_answers_after_resolution():
    data, dev = _data_and_dev()
    oos_id = next(x["id"] for x in data["A"] if not x["in_scope"])
    final = {oos_id: {"in_scope": True, "resolution": "rewritten"},  # chuyển sang trong phạm vi nhưng thiếu nhãn
             "A01": {"answers": [], "resolution": "merged"}}
    errors = "\n".join(validate_gold(build_gold(data, annotations_of(data), final, dev)))
    assert f"{oos_id}: trong phạm vi nhưng article không phải số nguyên" in errors
    assert f"{oos_id}: trong phạm vi nhưng không có đáp án" in errors
    assert "A01: trong phạm vi nhưng không có đáp án" in errors


def test_validate_gold_flags_quota_break_and_invalid_type():
    data, dev = _data_and_dev()
    final = {"A01": {"in_scope": False, "resolution": "rewritten"},
             "A02": {"question_type": "khong_ton_tai", "resolution": "rewritten"}}
    errors = "\n".join(validate_gold(build_gold(data, annotations_of(data), final, dev)))
    assert "A: in = 29, quota 30" in errors
    assert "A02: question_type 'khong_ton_tai' không hợp lệ" in errors


# --- lệnh gold_tool: dữ liệu thiếu và split đã khóa ------------------------


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")


@pytest.fixture()
def workspace(tmp_path, monkeypatch):
    drafts = tmp_path / "drafts"
    drafts.mkdir()
    monkeypatch.setattr(gold_tool, "DRAFTS", drafts)
    monkeypatch.setattr(gold_tool, "GOLD", tmp_path / "gold")
    monkeypatch.setattr(gold_tool, "SPLIT_FILE", tmp_path / "splits" / "gold_split.json")
    for a, rows in valid_set().items():
        _write_jsonl(drafts / f"author_{a}.jsonl", rows)
        _write_jsonl(drafts / f"annot_{gold_tool.ANNOTATOR_OF[a]}_of_{a}.jsonl", rows)
    return tmp_path


def test_cmd_validate_fails_when_any_author_file_is_missing(workspace):
    assert gold_tool.main(["validate"]) == 0
    (workspace / "drafts" / "author_C.jsonl").unlink()
    assert gold_tool.main(["validate"]) == 1


def test_cmd_agreement_fails_on_incomplete_annotations_unless_allowed(workspace, capsys):
    out = workspace / "agreement.json"
    assert gold_tool.main(["agreement", "--json", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["complete"] is True
    ann = workspace / "drafts" / "annot_A_of_B.jsonl"
    ann.write_text("\n".join(ann.read_text(encoding="utf-8").splitlines()[:10]), encoding="utf-8")
    assert gold_tool.main(["agreement"]) == 1
    assert gold_tool.main(["agreement", "--allow-partial", "--json", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["complete"] is False and report["coverage_errors"]
    assert "MỘT PHẦN" in capsys.readouterr().out


def test_cmd_build_writes_once_then_refuses_and_leaves_files_untouched(workspace):
    assert gold_tool.main(["build", "--write"]) == 0
    split = workspace / "splits" / "gold_split.json"
    before = {p: p.read_bytes() for p in [split, *(workspace / "gold").iterdir()]}
    assert len(json.loads(split.read_text(encoding="utf-8"))["dev_ids"]) == 30
    assert gold_tool.main(["build", "--write"]) == 1
    assert {p: p.read_bytes() for p in before} == before


def test_cmd_build_uses_dev_ids_from_locked_split(workspace, capsys):
    split = workspace / "splits" / "gold_split.json"
    split.parent.mkdir()
    _, dev = _data_and_dev()
    split.write_text(json.dumps({"seed": 999999, "clause_leakage": 7, "dev_ids": sorted(dev)}), encoding="utf-8")
    assert gold_tool.main(["build"]) == 0
    out = capsys.readouterr().out
    assert "split đã khóa: seed=999999" in out and "dev=30" in out


def test_cmd_build_rejects_invalid_resolutions_and_inconsistent_gold(workspace):
    res = workspace / "res.jsonl"
    _write_jsonl(res, [{"id": "KHONG_CO", "resolution": "merged"}])
    assert gold_tool.main(["build", "--resolutions", str(res), "--force"]) == 1
    _write_jsonl(res, [{"id": "A01", "in_scope": True, "article": None, "answers": [],
                        "resolution": "rewritten"}])
    assert gold_tool.main(["build", "--resolutions", str(res), "--force", "--write"]) == 1
    assert not (workspace / "gold").exists()


def test_validate_gold_rejects_malformed_resolution_values():
    data, dev = _data_and_dev()
    final = {"A01": {"in_scope": None, "resolution": "merged"},
             "A02": {"answers": "abc", "resolution": "merged"},
             "A03": {"answers": [""], "resolution": "merged"},
             "A04": {"question": "", "resolution": "merged"},
             "A05": {"question": "Theo Điều 5 thì sao?", "resolution": "merged"}}
    errors = "\n".join(validate_gold(build_gold(data, annotations_of(data), final, dev)))
    for i in ("A01: in_scope", "A02: answers", "A03: answers", "A04: câu hỏi rỗng", "A05: câu hỏi nêu số"):
        assert i in errors


def test_cmd_build_rejects_bad_locked_split_and_bad_resolution_rows(workspace):
    split = workspace / "splits" / "gold_split.json"
    split.parent.mkdir()
    _, dev = _data_and_dev()
    split.write_text("{khong phai json", encoding="utf-8")
    assert gold_tool.main(["build"]) == 1
    split.write_text(json.dumps({"seed": 1}), encoding="utf-8")  # thiếu dev_ids
    assert gold_tool.main(["build"]) == 1
    split.write_text(json.dumps({"seed": 1, "dev_ids": sorted(dev) + ["Z99"]}), encoding="utf-8")
    assert gold_tool.main(["build"]) == 1
    split.unlink()
    res = workspace / "res.jsonl"
    _write_jsonl(res, [{"resolution": "merged"}])  # thiếu id
    assert gold_tool.main(["build", "--resolutions", str(res)]) == 1
    _write_jsonl(res, [{"id": "A01", "resolution": "merged"}, {"id": "A01", "resolution": "merged"}])
    assert gold_tool.main(["build", "--resolutions", str(res)]) == 1
    assert gold_tool.main(["build", "--resolutions", str(workspace / "khong_co.jsonl")]) == 1


def test_cmd_build_recomputes_leakage_after_resolutions(workspace, capsys):
    data, dev = _data_and_dev()
    split = workspace / "splits" / "gold_split.json"
    split.parent.mkdir()
    split.write_text(json.dumps({"seed": 1, "clause_leakage": 0, "dev_ids": sorted(dev)}), encoding="utf-8")
    dev_row = next(x for x in data["A"] if x["id"] in dev and x["in_scope"])
    test_row = next(x for x in data["B"] if x["id"] not in dev and x["in_scope"])
    res = workspace / "res.jsonl"
    _write_jsonl(res, [{"id": test_row["id"], "article": dev_row["article"], "clause": dev_row["clause"],
                        "resolution": "merged"}])
    assert gold_tool.main(["build", "--resolutions", str(res)]) == 0
    assert "Khoản có câu ở cả dev lẫn test: 1" in capsys.readouterr().out





def _snapshot(root):
    return {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_build_and_agreement_refuse_missing_author_file_even_with_force(workspace):
    (workspace / "drafts" / "author_C.jsonl").unlink()
    assert gold_tool.main(["build"]) == 1
    assert gold_tool.main(["build", "--force", "--write"]) == 1
    assert not (workspace / "gold").exists() and not (workspace / "splits").exists()
    assert gold_tool.main(["agreement"]) == 1
    out = workspace / "agreement.json"
    assert gold_tool.main(["agreement", "--allow-partial", "--json", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["complete"] is False and any("tác giả C" in e for e in report["coverage_errors"])


def test_build_refuses_incomplete_annotations_even_with_force(workspace):
    ann = workspace / "drafts" / "annot_A_of_B.jsonl"
    ann.write_text("\n".join(ann.read_text(encoding="utf-8").splitlines()[:10]), encoding="utf-8")
    for flags in ([], ["--write"], ["--force"], ["--force", "--write"]):
        assert gold_tool.main(["build", *flags]) == 1, flags
    assert not (workspace / "gold").exists() and not (workspace / "splits").exists()


@pytest.mark.parametrize("breakage", ["missing_file", "extra_id", "duplicate_id"])
def test_force_never_bypasses_annotation_coverage_errors(workspace, breakage):
    ann = workspace / "drafts" / "annot_A_of_B.jsonl"
    rows = [json.loads(line) for line in ann.read_text(encoding="utf-8").splitlines()]
    if breakage == "missing_file":
        ann.unlink()
    elif breakage == "extra_id":
        _write_jsonl(ann, rows + [dict(rows[0], id="B99")])
    else:
        _write_jsonl(ann, rows + [dict(rows[0])])
    assert gold_tool.main(["build", "--force", "--write"]) == 1
    assert not (workspace / "gold").exists() and not (workspace / "splits").exists()


def test_validate_gold_rejects_missing_independent_label_and_unresolved_disagreement():
    data, dev = _data_and_dev()
    ann = annotations_of(data)
    ann["B"] = ann["B"][1:]  # thiếu nhãn độc lập của câu đầu
    errors = "\n".join(validate_gold(build_gold(data, ann, {}, dev)))
    assert f"{data['B'][0]['id']}: thiếu nhãn độc lập" in errors

    ann = annotations_of(data)
    ann["A"][0]["clause"] = 9  # cùng Điều, khác Khoản, chưa có resolution
    first = data["A"][0]["id"]
    errors = "\n".join(validate_gold(build_gold(data, ann, {}, dev)))
    assert f"{first}: hai nhãn bất đồng (scope=same, source=same_article)" in errors
    resolved = {first: {"id": first, "resolution": "author_label"}}
    assert not any(e.startswith(f"{first}:") for e in validate_gold(build_gold(data, ann, resolved, dev)))


@pytest.mark.parametrize("change,expected", [
    ({"in_scope": False, "article": None, "clause": None, "answer": None}, "scope=different"),
    ({"article": 44, "clause": 1}, "source=different"),
])
def test_validate_gold_flags_scope_and_article_disagreement(change, expected):
    data, dev = _data_and_dev()
    ann = annotations_of(data)
    ann["A"][0].update(change)
    first = data["A"][0]["id"]
    errors = "\n".join(validate_gold(build_gold(data, ann, {}, dev)))
    assert f"{first}: hai nhãn bất đồng" in errors and expected in errors


def test_disagreement_needs_resolution_with_explicit_final_label():
    data, _ = _data_and_dev()
    ann = annotations_of(data)
    ann["A"][0].update({"in_scope": False, "article": None, "clause": None, "answer": None})
    first = data["A"][0]["id"]
    assert check_disagreements_resolved(data, ann, {})[0].startswith(f"{first}: hai nhãn bất đồng")
    bare = {first: {"id": first, "resolution": "annotator_label"}}  # chỉ có tên, không có nhãn cuối
    assert "thiếu nhãn cuối (in_scope, article, clause, answers)" in check_disagreements_resolved(data, ann, bare)[0]
    full = {first: {"id": first, "resolution": "annotator_label", "in_scope": False,
                    "article": None, "clause": None, "answers": []}}
    assert check_disagreements_resolved(data, ann, full) == []
    null_answers = {first: {**full[first], "in_scope": True, "answers": None}}
    assert "thiếu nhãn cuối (answers)" in check_disagreements_resolved(data, ann, null_answers)[0]
    null_scope = {first: {**full[first], "in_scope": None}}
    assert "thiếu nhãn cuối (in_scope)" in check_disagreements_resolved(data, ann, null_scope)[0]
    assert check_disagreements_resolved(data, annotations_of(data), {}) == []  # đồng thuận: không cần


def test_cmd_build_rejects_bare_resolution_for_disagreement(workspace):
    ann = workspace / "drafts" / "annot_C_of_A.jsonl"
    rows = [json.loads(line) for line in ann.read_text(encoding="utf-8").splitlines()]
    rows[0].update({"in_scope": False, "article": None, "clause": None, "answer": None})
    _write_jsonl(ann, rows)
    first = rows[0]["id"]
    res = workspace / "res.jsonl"
    _write_jsonl(res, [{"id": first, "resolution": "annotator_label"}])
    assert gold_tool.main(["build", "--resolutions", str(res), "--write"]) == 1
    assert not (workspace / "gold").exists()


def test_build_refuses_to_overwrite_when_only_gold_test_exists(workspace):
    gold = workspace / "gold"
    gold.mkdir()
    (gold / "gold_test.jsonl").write_text('{"id": "X"}\n', encoding="utf-8")
    before = _snapshot(workspace)
    assert gold_tool.main(["build", "--write"]) == 1
    assert _snapshot(workspace) == before and not (workspace / "splits").exists()


def test_build_refuses_invalid_author_labels_unless_forced(workspace):
    for name in ("author_A.jsonl", "annot_C_of_A.jsonl"):  # đổi cả hai nhãn để không phát sinh bất đồng
        p = workspace / "drafts" / name
        rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]
        for r in rows[:4]:
            r["article"] = 40  # 4 câu cùng một Điều: vượt MAX_PER_ARTICLE, gold vẫn nhất quán
        _write_jsonl(p, rows)
    assert gold_tool.main(["build"]) == 1
    assert gold_tool.main(["build", "--force"]) == 0  # --force chỉ bỏ qua lỗi nhãn tác giả (dry run)
    assert gold_tool.main(["build", "--force", "--write"]) == 0


def test_agreement_is_incomplete_when_same_ids_are_missing_from_both_label_sets(workspace):
    # tệp tác giả và tệp gán nhãn cùng thiếu 5 id: hai tệp khớp nhau nhưng số câu lệch quota
    for name in ("author_A.jsonl", "annot_C_of_A.jsonl"):
        p = workspace / "drafts" / name
        _write_jsonl(p, [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()][5:])
    assert gold_tool.main(["agreement"]) == 1
    out = workspace / "agreement.json"
    assert gold_tool.main(["agreement", "--allow-partial", "--json", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["complete"] is False
    assert any("A: 28 câu, cần 33" in e for e in report["coverage_errors"])


def test_check_author_counts_flags_missing_author_and_wrong_quota():
    data = valid_set()
    assert check_author_counts(data) == []
    data["A"] = data["A"][:-1]
    del data["C"]
    errors = "\n".join(check_author_counts(data))
    assert "thiếu nhãn của tác giả C" in errors and "A: 32 câu, cần 33" in errors


def test_check_author_fields_flags_prefix_missing_answer_and_oos_answer():
    data = valid_set()
    assert check_author_fields(data) == []
    data["A"][0]["id"] = "Z01"
    data["B"][0]["answer"] = None
    data["C"][-1]["answer"] = "không được có"
    errors = "\n".join(check_author_fields(data))
    assert "Z01: id không bắt đầu bằng A" in errors
    assert "B01: trong phạm vi nhưng thiếu article/answer" in errors
    assert f"{data['C'][-1]['id']}: ngoài phạm vi nhưng có article/answer" in errors


@pytest.mark.parametrize("breakage", ["author_without_answer", "wrong_id_prefix"])
def test_force_never_bypasses_required_author_fields(workspace, breakage):
    drafts = workspace / "drafts"
    path = drafts / "author_A.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if breakage == "author_without_answer":
        rows[0]["answer"], rows[0]["alt_answers"] = None, []  # người gán nhãn vẫn giữ đáp án của họ
    else:
        rows[0]["id"] = "Z01"
        ann = drafts / "annot_C_of_A.jsonl"
        ann_rows = [json.loads(line) for line in ann.read_text(encoding="utf-8").splitlines()]
        ann_rows[0]["id"] = "Z01"
        _write_jsonl(ann, ann_rows)
    _write_jsonl(path, rows)
    for flags in ([], ["--force"], ["--force", "--write"]):
        assert gold_tool.main(["build", *flags]) == 1, flags
    assert not (workspace / "gold").exists() and not (workspace / "splits").exists()


def test_build_refuses_to_overwrite_existing_gold_without_split_lock(workspace):
    gold = workspace / "gold"
    gold.mkdir()
    (gold / "gold_dev.jsonl").write_text('{"id": "X"}\n', encoding="utf-8")
    before = _snapshot(workspace)
    assert gold_tool.main(["build", "--write"]) == 1
    assert gold_tool.main(["build", "--force", "--write"]) == 1
    assert _snapshot(workspace) == before and not (workspace / "splits").exists()


def test_build_may_write_over_empty_placeholder_gold_files(workspace):
    gold = workspace / "gold"
    gold.mkdir()
    for name in ("gold_dev.jsonl", "gold_test.jsonl"):
        (gold / name).write_text("", encoding="utf-8")
    assert gold_tool.main(["build", "--write"]) == 0
    assert len((gold / "gold_dev.jsonl").read_text(encoding="utf-8").splitlines()) == 30
    assert (workspace / "splits" / "gold_split.json").exists()


def test_locked_split_blocks_write_and_leaves_every_file_untouched(workspace):
    split = workspace / "splits" / "gold_split.json"
    split.parent.mkdir()
    _, dev = _data_and_dev()
    split.write_text(json.dumps({"seed": 999999, "clause_leakage": 7, "dev_ids": sorted(dev)}), encoding="utf-8")
    before = _snapshot(workspace)
    assert gold_tool.main(["build", "--write"]) == 1
    assert _snapshot(workspace) == before  # không tạo gold, không đổi gold_split.json


def test_locked_split_dev_ids_are_the_ones_passed_to_build_gold(workspace, monkeypatch):
    data, default_dev = _data_and_dev()
    other = draw_split(data, seed=12345)  # hợp lệ theo quota nhưng khác choose_split_seed
    assert other != default_dev and other != choose_split_seed(data)[1]
    split = workspace / "splits" / "gold_split.json"
    split.parent.mkdir()
    split.write_text(json.dumps({"seed": 1, "dev_ids": sorted(other)}), encoding="utf-8")
    seen = {}
    real = gold_tool.build_gold
    monkeypatch.setattr(gold_tool, "build_gold",
                        lambda a, n, f, d: seen.setdefault("dev", set(d)) and real(a, n, f, d))
    assert gold_tool.main(["build"]) == 0
    assert seen["dev"] == other


def _gold_rows():
    data, dev = _data_and_dev()
    return build_gold(data, annotations_of(data), {}, dev)


def test_validate_gold_flags_each_structural_inconsistency():
    rows = _gold_rows()
    rows[0]["clause"] = "2"                       # Khoản không phải số nguyên
    rows[1]["article"] = True                     # bool không phải số nguyên
    rows[2]["split"] = "val"                      # split lạ
    rows[3]["id"] = rows[4]["id"]                 # id trùng
    oos = next(r for r in rows if not r["in_scope"])
    oos["article"], oos["answers"] = 5, ["x"]     # ngoài phạm vi nhưng có article/answers
    oos2 = [r for r in rows if not r["in_scope"]][1]
    oos2["question_type"] = "so_lieu"             # loại câu sai cho câu ngoài phạm vi
    errors = "\n".join(validate_gold(rows))
    assert "clause không phải số nguyên" in errors
    assert "article không phải số nguyên (True)" in errors
    assert "split 'val' không hợp lệ" in errors
    assert "id trùng trong gold" in errors
    assert f"{oos['id']}: ngoài phạm vi nhưng có article/clause/answers" in errors
    assert f"{oos2['id']}: question_type 'so_lieu' không hợp lệ cho câu ngoài" in errors


def test_validate_gold_checks_dev_quota_and_ignores_nothing_when_dev_is_wrong():
    data, _ = _data_and_dev()
    errors = "\n".join(validate_gold(build_gold(data, annotations_of(data), {}, set())))
    assert "A: dev_in = 0, quota 9" in errors and "A: dev_out = 0, quota 1" in errors


def test_locked_split_with_duplicate_dev_ids_is_rejected(workspace):
    _, dev = _data_and_dev()
    ids = sorted(dev)
    split = workspace / "splits" / "gold_split.json"
    split.parent.mkdir()
    split.write_text(json.dumps({"seed": 1, "dev_ids": ids + [ids[0]]}), encoding="utf-8")
    assert gold_tool.main(["build"]) == 1
