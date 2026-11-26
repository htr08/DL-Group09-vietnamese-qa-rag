"""Công cụ cho tập 100 câu: kiểm tra nhãn, tính đồng thuận, sinh gold dev/test.

Sử dụng (từ thư mục gốc repo):
    python scripts/gold_tool.py validate
    python scripts/gold_tool.py agreement [--json docs/results/gold_agreement.json]
    python scripts/gold_tool.py build [--resolutions data/gold/drafts/resolutions.jsonl] [--write]

`build` mặc định chỉ in thống kê (dry run); thêm --write để ghi data/gold/gold_dev.jsonl,
data/gold/gold_test.jsonl và data/splits/gold_split.json. Việc chia dev/test chỉ được
ghi MỘT LẦN (hướng dẫn gán nhãn, mục 6): nếu gold_split.json đã có, `build` dùng đúng danh
sách dev trong tệp đó (không bốc lại) và từ chối --write. `build` cũng từ chối --write khi
gold_dev/gold_test.jsonl đã có nội dung mà chưa có gold_split.json.

--force bỏ qua các lỗi của `validate` trừ id sai tiền tố và lỗi trường bắt buộc (thiếu
article/answer ở câu trong phạm vi, có article/answer ở câu ngoài phạm vi). Trong số lỗi bị bỏ
qua, quota, cụm bị cấm và question_type vẫn được kiểm lại ở gold cuối; chỉ bốn nhóm sau không
còn ai kiểm: câu hỏi trùng, tối đa mỗi Điều, tối đa mỗi loại câu (bang, co_khong) và tối thiểu
mỗi nhóm ngoài phạm vi. Thiếu nhãn độc lập, resolutions không hợp lệ và gold cuối không nhất
quán luôn làm build thất bại. `agreement` thoát với mã 1 khi số câu của một tác giả lệch quota.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from src.gold import (  # noqa: E402
    agreement_summary, build_gold, check_annotation_coverage, check_author_counts,
    check_author_fields, check_disagreements_resolved, choose_split_seed, clause_leakage, compare_labels, load_split, read_jsonl, validate_authors, validate_gold,
    validate_resolutions, write_jsonl,
)
from src.metrics import bootstrap_ci  # noqa: E402

DRAFTS = ROOT / "data" / "gold" / "drafts"
GOLD = ROOT / "data" / "gold"
SPLIT_FILE = ROOT / "data" / "splits" / "gold_split.json"
# tác giả -> người gán nhãn độc lập (vòng tròn A→B→C→A: A gán câu của B, ...)
ANNOTATOR_OF = {"B": "A", "C": "B", "A": "C"}


def load_drafts(drafts: Path | None = None):
    drafts = drafts or DRAFTS
    authors, annotations = {}, {}
    for a, ann in ANNOTATOR_OF.items():  # a = tác giả, ann = người gán nhãn
        p = drafts / f"author_{a}.jsonl"
        if p.exists():
            authors[a] = read_jsonl(p)
        else:
            print(f"[cảnh báo] thiếu {p.name}", file=sys.stderr)
        q = drafts / f"annot_{ann}_of_{a}.jsonl"
        if q.exists():
            annotations[a] = read_jsonl(q)
        else:
            print(f"[cảnh báo] thiếu {q.name}", file=sys.stderr)
    return authors, annotations


def cmd_validate(_args) -> int:
    authors, _ = load_drafts()
    errors = validate_authors(authors)  # gồm lỗi thiếu tác giả A/B/C
    for e in errors:
        print("LỖI:", e)
    print(f"{len(errors)} lỗi trên {sum(len(v) for v in authors.values())} câu")
    return 1 if errors else 0


def _pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def cmd_agreement(args) -> int:
    authors, annotations = load_drafts()
    # số câu lệch quota (kể cả khi file tác giả và file gán nhãn cùng thiếu một id) cũng là dữ liệu chưa đủ
    coverage = check_author_counts(authors) + check_annotation_coverage(authors, annotations)
    for e in coverage:
        print("THIẾU:", e, file=sys.stderr)
    if coverage and not args.allow_partial:
        print("Dữ liệu đồng thuận chưa đầy đủ; không tính (dùng --allow-partial để xem kết quả một phần).",
              file=sys.stderr)
        return 1
    report, pooled = {}, []
    for a, labels in authors.items():
        if a not in annotations:
            continue
        ann = {x["id"]: x for x in annotations[a]}
        pairs = [compare_labels(x, ann[x["id"]]) for x in labels if x["id"] in ann]
        summary = agreement_summary(pairs)
        strict = agreement_summary([compare_labels(x, ann[x["id"]], use_alt=False)
                                    for x in labels if x["id"] in ann])
        summary["answer_em_no_alt"], summary["answer_f1_no_alt"] = strict["answer_em"], strict["answer_f1"]
        both = [p for p in pairs if p["source"] != "n/a"]
        if both:
            summary["same_clause_ci"] = bootstrap_ci([p["source"] == "same_clause" for p in both])[1:3]
            summary["answer_f1_ci"] = bootstrap_ci([p["answer_f1"] for p in both])[1:3]
        report[f"{ANNOTATOR_OF[a]}_of_{a}"] = summary
        pooled += pairs
    report["pooled"] = agreement_summary(pooled)
    report["complete"] = not coverage
    report["coverage_errors"] = coverage
    if coverage:
        print("\nCẢNH BÁO: kết quả MỘT PHẦN, không dùng làm kết quả đồng thuận đầy đủ.")
    for name, s in report.items():
        if name in ("complete", "coverage_errors"):
            continue
        print(f"\n== {name}: n={s['n_total']}, cả hai cùng trong phạm vi={s['n_both_in_scope']}")
        print(f"  (a) cùng Khoản {_pct(s['same_clause'])} | cùng Điều {_pct(s['same_article'])}")
        print(f"  (b) đáp án EM {_pct(s['answer_em'])} | F1 {_pct(s['answer_f1'])} (có alt_answers)")
        if "answer_em_no_alt" in s:
            print(f"      chỉ so `answer`: EM {_pct(s['answer_em_no_alt'])} | F1 {_pct(s['answer_f1_no_alt'])}")
        print(f"  (c) cùng phạm vi {_pct(s['scope_agree'])} (bất đồng {s['scope_disagree_count']} câu)")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0



def _fail(errors, message) -> int:
    print("\n".join(errors))
    print(message)
    return 1


def cmd_build(args) -> int:
    authors, annotations = load_drafts()
    if set(authors) != set(ANNOTATOR_OF):
        print("Thiếu nhãn tác giả; không thể sinh gold.", file=sys.stderr)
        return 1
    coverage = check_annotation_coverage(authors, annotations)
    if coverage:  # thiếu nhãn độc lập: không bỏ qua được bằng --force
        return _fail(coverage, "Dừng vì nhãn độc lập chưa đầy đủ (không bỏ qua được bằng --force).")
    errors = check_author_fields(authors)  # không bỏ qua được bằng --force
    if errors:
        return _fail(errors, "Dừng vì nhãn tác giả thiếu trường bắt buộc (không bỏ qua được bằng --force).")
    errors = validate_authors(authors)
    if errors and not args.force:
        return _fail(errors, "Dừng vì nhãn chưa hợp lệ (dùng --force để bỏ qua).")
    final = {}
    if args.resolutions:
        try:
            res_rows = read_jsonl(args.resolutions)
        except (OSError, ValueError) as e:
            return _fail([f"không đọc được {args.resolutions}: {e!r}"], "Dừng vì resolutions không hợp lệ.")
        ids = [r.get("id") for r in res_rows]
        errors = [f"resolution thiếu id ở dòng {n}" for n, i in enumerate(ids, 1) if not i]
        errors += [f"resolution id trùng: {i}" for i in sorted({i for i in ids if i and ids.count(i) > 1})]
        final = {r["id"]: r for r in res_rows if r.get("id")}
        errors += validate_resolutions(authors, final)  # không bỏ qua được bằng --force
        errors += check_disagreements_resolved(authors, annotations, final)
        if errors:
            return _fail(errors, "Dừng vì resolutions không hợp lệ.")
    locked = SPLIT_FILE.exists()
    if locked:  # chia dev/test đã khóa: dùng lại, không bốc lại
        try:
            seed, dev_ids, _ = load_split(SPLIT_FILE)
        except ValueError as e:
            return _fail([str(e)], "Dừng vì gold_split.json không hợp lệ.")
        ghosts = dev_ids - {x["id"] for labels in authors.values() for x in labels}
        if ghosts:
            return _fail([f"gold_split.json có id không tồn tại: {', '.join(sorted(ghosts))}"],
                         "Dừng vì gold_split.json không khớp nhãn tác giả.")
    else:
        seed, dev_ids, _ = choose_split_seed(authors)
    rows = build_gold(authors, annotations, final, dev_ids)
    leak = clause_leakage(rows, dev_ids)  # tính lại trên gold cuối (sau resolutions)
    errors = validate_gold(rows)  # không bỏ qua được bằng --force
    if errors:
        return _fail(errors, "Dừng vì gold cuối không hợp lệ.")
    dev = [r for r in rows if r["split"] == "dev"]
    test = [r for r in rows if r["split"] == "test"]
    print(f"{'split đã khóa' if locked else 'split mới'}: seed={seed}, dev={len(dev)} "
          f"({sum(r['in_scope'] for r in dev)} trong phạm vi), "
          f"test={len(test)} ({sum(r['in_scope'] for r in test)} trong phạm vi), "
          f"Khoản có câu ở cả dev lẫn test: {leak}")
    if not args.write:
        print("Dry run: chưa ghi gì (thêm --write).")
        return 0
    if locked:
        print(f"{SPLIT_FILE.name} đã tồn tại: chia dev/test chỉ làm một lần, không ghi đè.", file=sys.stderr)
        return 1
    existing = [p.name for p in (GOLD / "gold_dev.jsonl", GOLD / "gold_test.jsonl")
                if p.exists() and p.stat().st_size > 0]
    if existing:  # gold đã có nội dung nhưng chưa có khóa chia: không ghi đè
        print(f"{', '.join(existing)} đã có nội dung nhưng chưa có {SPLIT_FILE.name}; "
              "không ghi đè gold hiện có.", file=sys.stderr)
        return 1
    GOLD.mkdir(parents=True, exist_ok=True)
    write_jsonl(GOLD / "gold_dev.jsonl", dev)
    write_jsonl(GOLD / "gold_test.jsonl", test)
    SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_FILE.write_text(json.dumps({
        "seed": seed, "method": "choose_split_seed (ít Khoản bị chia nhất)",
        "clause_leakage": leak, "dev_ids": sorted(dev_ids),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Đã ghi gold_dev.jsonl, gold_test.jsonl, gold_split.json")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate").set_defaults(fn=cmd_validate)
    a = sub.add_parser("agreement")
    a.add_argument("--json")
    a.add_argument("--allow-partial", action="store_true",
                   help="vẫn tính khi thiếu tệp hoặc id; kết quả được đánh dấu complete=false")
    a.set_defaults(fn=cmd_agreement)
    b = sub.add_parser("build")
    b.add_argument("--resolutions")
    b.add_argument("--write", action="store_true")
    b.add_argument("--force", action="store_true")
    b.set_defaults(fn=cmd_build)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
