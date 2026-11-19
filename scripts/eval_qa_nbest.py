"""Evaluate existing PhoBERT QA checkpoint on A's preprocessed ViQuAD DEV.

NO training. No VnCoreNLP invocation: dev features were already segmented.
Test data must not be supplied to this script to tune null threshold.

Example (run at repo root):
  python scripts/eval_qa_nbest.py --checkpoint artifacts/phobert_qa/best_model \
    --features data/processed/qa_features/dev.jsonl \
    --gold data/processed/viquad/dev.jsonl \
    --output-dir outputs/week3_nbest --batch-size 8
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModel, AutoTokenizer
from src.qa_nbest import decode_window, scores_at_threshold, best_dev_threshold
from src.metrics import exact_match_score, f1_score


def load_jsonl(path):
    rows = []
    with path.open(encoding="utf-8-sig") as file:
        for lineno, line in enumerate(file, 1):
            if line.strip():
                obj = json.loads(line)
                if "id" not in obj:
                    raise ValueError(f"Missing id: {path}:{lineno}")
                rows.append(obj)
    if not rows:
        raise ValueError(f"Empty JSONL: {path}")
    return rows


def find_literal_in_context(candidate, context):
    """Best-effort substring recovery; NOT a replacement for saved char offsets."""
    if not context or not candidate:
        return candidate, False
    normalized = " ".join(candidate.split())
    pattern = r"\s+".join(re.escape(x) for x in normalized.split(" "))
    m = re.search(pattern, context, flags=re.IGNORECASE)
    return (m.group(0), True) if m else (candidate, False)


def model_batch(model, head, feats, pad_id, device):
    max_len = max(len(row["input_ids"]) for row in feats)
    inp = torch.full((len(feats), max_len), pad_id, dtype=torch.long)
    attn = torch.zeros((len(feats), max_len), dtype=torch.long)
    for i, row in enumerate(feats):
        ids = row["input_ids"]
        inp[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        attn[i, :len(ids)] = 1
    inp, attn = inp.to(device), attn.to(device)
    with torch.inference_mode():
        hidden = model(input_ids=inp, attention_mask=attn).last_hidden_state
        logits = head(hidden).float().cpu()
    return logits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=Path("artifacts/phobert_qa/best_model"))
    p.add_argument("--features", type=Path, default=Path("data/processed/qa_features/dev.jsonl"))
    p.add_argument("--gold", type=Path, default=Path("data/processed/viquad/dev.jsonl"))
    p.add_argument("--dev-titles", type=Path, default=Path("data/splits/viquad_dev_titles.txt"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/week3_nbest"))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--n-best", type=int, default=20)
    p.add_argument("--max-answer-length", type=int, default=None)
    p.add_argument("--limit-features", type=int, default=None, help="Smoke test only. No metrics/threshold on partial data.")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    a = p.parse_args()
    if a.batch_size < 1 or a.n_best < 1:
        p.error("--batch-size / --n-best must be positive")
    if a.limit_features is not None and a.limit_features < 1:
        p.error("--limit-features must be positive")

    # Normalize absolute paths BEFORE any potential cwd-changing imports.
    a.checkpoint, a.features, a.gold, a.dev_titles, a.output_dir = [
        v.resolve() for v in (a.checkpoint, a.features, a.gold, a.dev_titles, a.output_dir)
    ]
    for path in (a.features, a.gold, a.dev_titles, a.checkpoint / "config.json", a.checkpoint / "qa_head.pt", a.checkpoint / "qa_config.json"):
        if not path.exists():
            raise FileNotFoundError(path)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available() else "cpu" if a.device == "auto" else a.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    cfg = json.loads((a.checkpoint / "qa_config.json").read_text(encoding="utf-8"))
    max_answer_length = a.max_answer_length or int(cfg.get("max_answer_length", 30))
    features = load_jsonl(a.features)
    gold = load_jsonl(a.gold)
    expected_titles = {line.strip() for line in a.dev_titles.read_text(encoding="utf-8-sig").splitlines() if line.strip()}
    actual_titles = {str(row.get("title", "")).strip() for row in gold}
    if len(expected_titles) != 13 or actual_titles != expected_titles:
        raise ValueError(
            f"Expected EXACT 13-title Dev split. Saved={len(expected_titles)}, "
            f"gold={len(actual_titles)}, missing={sorted(expected_titles - actual_titles)[:3]}, "
            f"unexpected={sorted(actual_titles - expected_titles)[:3]}"
        )
    if a.limit_features is not None:
        features = features[:a.limit_features]
    else:
        feature_ids = {str(row["id"]) for row in features}
        gold_ids = {str(row["id"]) for row in gold}
        if len(gold_ids) != len(gold) or feature_ids != gold_ids:
            raise ValueError("Dev feature IDs and gold IDs differ (or duplicate gold IDs): check your 13-title split")

    print(f"Device={device}; features={len(features)}; gold_questions={len(gold)}", flush=True)
    print(f"Checkpoint={a.checkpoint}; n_best={a.n_best}; max_answer_length={max_answer_length}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(a.checkpoint), use_fast=False)
    backbone = AutoModel.from_pretrained(str(a.checkpoint)).to(device).eval()
    head = torch.nn.Linear(backbone.config.hidden_size, 2).to(device)
    try:
        state = torch.load(a.checkpoint / "qa_head.pt", map_location="cpu", weights_only=True)
    except TypeError:  # older PyTorch
        state = torch.load(a.checkpoint / "qa_head.pt", map_location="cpu")
    head.load_state_dict(state, strict=True)
    head.eval()

    predictions = defaultdict(lambda: {"null_score": float("inf"), "candidates": []})
    for offset in range(0, len(features), a.batch_size):
        batch = features[offset:offset + a.batch_size]
        logits = model_batch(backbone, head, batch, tokenizer.pad_token_id, device)
        for i, feat in enumerate(batch):
            valid_length = len(feat["input_ids"])
            candidates, null = decode_window(
                feat["input_ids"], logits[i, :valid_length, 0].tolist(),
                logits[i, :valid_length, 1].tolist(), tokenizer,
                n_best_size=a.n_best, max_answer_length=max_answer_length,
                sep_token_id=tokenizer.eos_token_id,
            )
            item = predictions[str(feat["id"])]
            item["null_score"] = min(null, item["null_score"])
            item["candidates"].extend(candidates)
        if (offset // a.batch_size + 1) % 50 == 0 or offset + a.batch_size >= len(features):
            print(f"Processed {min(offset + a.batch_size, len(features))}/{len(features)} features", flush=True)

    gold_by_id = {str(row["id"]): row for row in gold}
    output = []
    scored = []
    n_aligned = 0
    for qid, rec in predictions.items():
        candidates = sorted(rec["candidates"], key=lambda c: c["score"], reverse=True)
        unique = []
        seen = set()
        for c in candidates:
            if c["text"] in seen:
                continue
            unique.append(c)
            seen.add(c["text"])
            if len(unique) == a.n_best:
                break
        if not unique:
            raise RuntimeError(f"No valid answer candidates for id={qid}")
        best = unique[0]
        row = gold_by_id.get(qid, {})
        original_text, aligned = find_literal_in_context(best["text"], row.get("context", ""))
        n_aligned += int(aligned)
        score_diff = rec["null_score"] - best["score"]
        out = {
            "id": qid, "best_answer": original_text, "aligned_to_original": aligned,
            "null_score": rec["null_score"], "best_span_score": best["score"],
            "score_diff": score_diff,
            "nbest": [
                {"text": c["text"], "score": c["score"],
                 "start_token_in_window": c["start_token"], "end_token_in_window": c["end_token"]}
                for c in unique
            ],
        }
        output.append(out)
        if a.limit_features is None:
            if row["is_impossible"]:
                refs = [""]
                category = "impossible"
            else:
                refs = row["answers"]["text"]
                category = "answerable"
            scored.append({
                "id": qid, "category": category, "score_diff": score_diff,
                "answer_em": float(exact_match_score(original_text, refs)),
                "answer_f1": float(f1_score(original_text, refs)),
                "null_em": float(exact_match_score("", refs)),
                "null_f1": float(f1_score("", refs)),
            })

    pred_path = a.output_dir / "nbest_dev.jsonl"
    with pred_path.open("w", encoding="utf-8") as f:
        for item in output:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved nbest: {pred_path}")
    print(f"Original-substring alignment (best span only): {n_aligned}/{len(output)}")
    if a.limit_features is not None:
        print("SMOKE ONLY: no metrics or threshold tuning from partial features.")
        return

    baseline = scores_at_threshold(scored, 0.0)
    threshold, tuned_em, tuned_f1 = best_dev_threshold(scored)
    summary = {
        "eval_split": "ViQuAD title-held-out Dev (not public Test)",
        "dev_questions": len(gold), "dev_features": len(features),
        "dev_titles": sorted(expected_titles),
        "checkpoint": str(a.checkpoint), "n_best_size": a.n_best,
        "max_answer_length": max_answer_length,
        "best_span_exact_original_match": n_aligned,
        "baseline_threshold_0": {"em": baseline[0], "f1": baseline[1]},
        "tuned_on_dev": {"null_threshold": threshold, "em": tuned_em, "f1": tuned_f1},
        "per_category": {
            category: {
                "count": sum(x["category"] == category for x in scored),
                "threshold_0": dict(zip(("em", "f1"), scores_at_threshold(
                    [x for x in scored if x["category"] == category], 0.0))),
                "tuned": dict(zip(("em", "f1"), scores_at_threshold(
                    [x for x in scored if x["category"] == category], threshold))),
            }
            for category in ("answerable", "impossible")
            if any(x["category"] == category for x in scored)
        },
        "notes": [
            "N-best and threshold selection evaluated on ViQuAD Dev only.",
            "Exact original-text offsets were not stored in old QA feature JSONL; substring alignment is a best-effort fallback.",
            "Do not compare to older run as a controlled improvement until checkpoint, Dev titles, metric normalization and decoding are verified identical.",
        ],
    }
    (a.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (a.output_dir / "dev_predictions.jsonl").open("w", encoding="utf-8") as f:
        for obj in output:
            prediction = "" if obj["score_diff"] > threshold else obj["best_answer"]
            f.write(json.dumps({"id": obj["id"], "prediction": prediction}, ensure_ascii=False) + "\n")
    print(f"Dev threshold=0: EM={baseline[0]*100:.2f}%, F1={baseline[1]*100:.2f}%")
    print(f"Dev tuned threshold={threshold:.5f}: EM={tuned_em*100:.2f}%, F1={tuned_f1*100:.2f}%")
    print(f"Saved summary: {a.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
