"""Xuất bảng chấm mù và tổng hợp kết quả chấm ảo giác.

Sử dụng (từ thư mục gốc repo):
    python scripts/grading_tool.py export --split test --system b2=kq_b2.jsonl --system qwen=kq_qwen.jsonl \\
        --out-dir data/grading
    python scripts/grading_tool.py summarize --split test --dir data/grading \\
        --extractive b1 --extractive extractive_mode [--json docs/results/grading_test.json]

`export` ghi `sheet_A.csv`, `sheet_B.csv`, `sheet_C.csv` (gửi cho từng người chấm) và `key.json`
(khóa ánh xạ mã mục -> hệ thống). `key.json` không được mở hoặc gửi cho người chấm trước khi
mọi bảng chấm được nộp. Mỗi tệp kết quả hệ thống là JSON Lines có khóa `id` và `answer`.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from src.grading import (  # noqa: E402
    GRADERS, build_blind_sheets, cohen_kappa, merge_grades, summarize,
)

GOLD_DIR = ROOT / "data" / "gold"
SEED = 2026
FIELDS = ["code", "question", "gold_answers", "source", "answer", "label", "note"]


def read_jsonl(path):
    with open(path, encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_gold(split: str, gold_dir: str | None = None):
    splits = ["dev", "test"] if split == "all" else [split]
    base = Path(gold_dir) if gold_dir else GOLD_DIR
    return [r for s in splits for r in read_jsonl(base / f"gold_{s}.jsonl")]


def cmd_export(args) -> int:
    gold = load_gold(args.split, args.gold_dir)
    results = {}
    for spec in args.system:
        name, _, path = spec.partition("=")
        if not path:
            print(f"--system cần dạng ten=duong_dan: {spec}", file=sys.stderr)
            return 1
        if name in results:
            print(f"tên hệ thống trùng: {name}", file=sys.stderr)
            return 1
        ids = {g["id"] for g in gold}
        results[name] = [r for r in read_jsonl(path) if r["id"] in ids]
        n_empty = sum(1 for r in results[name] if not (r.get("answer") or "").strip())
        if n_empty:
            print(f"[cảnh báo] {name}: {n_empty} câu trả lời rỗng hoặc thiếu", file=sys.stderr)
    sheets, key = build_blind_sheets(gold, results, seed=args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for grader, rows in sheets.items():
        with open(out / f"sheet_{grader}.csv", "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
    (out / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    n_second = sum(1 for k in key.values() if k["second"])
    print(f"{len(key)} mục chấm ({len(results)} hệ thống × {len(gold)} câu); "
          + ", ".join(f"{g}: {len(r)} dòng" for g, r in sheets.items())
          + f"; {n_second} mục được chấm lại. Đã ghi {out}")
    return 0


def load_sheets(directory: Path):
    sheets = {}
    for g in GRADERS:
        p = directory / f"sheet_{g}.csv"
        if p.exists():
            try:
                with open(p, encoding="utf-8-sig", newline="") as f:
                    sheets[g] = list(csv.DictReader(f))
            except UnicodeDecodeError as exc:
                raise SystemExit(f"{p.name} không phải UTF-8; lưu lại ở dạng CSV UTF-8 ({exc.reason})")
    return sheets


def cmd_summarize(args) -> int:
    gold = load_gold(args.split, args.gold_dir)
    directory = Path(args.dir)
    key = json.loads((directory / "key.json").read_text(encoding="utf-8"))
    primary, pairs, problems = merge_grades(load_sheets(directory), key, gold)
    for code, msg in problems[:30]:
        print(f"[vấn đề] {code}: {msg}")
    if problems:
        print(f"{len(problems)} vấn đề; bỏ qua các mục lỗi khi tổng hợp.")
    summary = summarize(primary, key, gold, extractive_systems=args.extractive)
    kappa = {
        "n_pairs": len(pairs),
        "kappa_5_labels": cohen_kappa([a for a, _ in pairs], [b for _, b in pairs]),
        "kappa_hallucination": cohen_kappa([str(a == "sai_bia") for a, _ in pairs],
                                           [str(b == "sai_bia") for _, b in pairs]),
    }
    for system, s in summary.items():
        def f(x):
            return "n/a" if x is None else f"{100 * x['estimate']:.1f}% [{100 * x['low']:.1f}, {100 * x['high']:.1f}]"
        print(f"\n== {system}: n={s['n']} (trong phạm vi {s['n_in_scope']}, ngoài {s['n_out_of_scope']})")
        print(f"  đúng {f(s['dung'])} | đúng một phần {f(s['dung_mot_phan'])}")
        print(f"  ảo giác {f(s['hallucination'])}" + (f" (số câu sai: {s['wrong_count']})" if 'wrong_count' in s else ""))
        print(f"  từ chối đúng {f(s['tu_choi_dung'])} | từ chối sai {f(s['tu_choi_sai'])}")
    print(f"\nkappa trên {kappa['n_pairs']} cặp: 5 nhãn = {kappa['kappa_5_labels']}, "
          f"ảo giác (nhị phân) = {kappa['kappa_hallucination']}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps({"systems": summary, "agreement": kappa, "problems": len(problems)},
                                              ensure_ascii=False, indent=2), encoding="utf-8")
    return 1 if problems else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--split", choices=["dev", "test", "all"], default="test")
    e.add_argument("--system", action="append", required=True, metavar="TEN=DUONG_DAN")
    e.add_argument("--out-dir", default=str(ROOT / "data" / "grading"))
    e.add_argument("--seed", type=int, default=SEED)
    e.add_argument("--gold-dir")
    e.set_defaults(fn=cmd_export)
    s = sub.add_parser("summarize")
    s.add_argument("--split", choices=["dev", "test", "all"], default="test")
    s.add_argument("--dir", default=str(ROOT / "data" / "grading"))
    s.add_argument("--extractive", action="append", default=[], help="hệ thống trích xuất (ảo giác: không áp dụng)")
    s.add_argument("--json")
    s.add_argument("--gold-dir")
    s.set_defaults(fn=cmd_summarize)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
