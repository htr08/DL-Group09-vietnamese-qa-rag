"""B1 domain Dev: BM25 (top-k) -> PhoBERT extractive QA -> EM/F1.

Source assumptions: repository htr08/DL-Group09-vietnamese-qa-rag,
commit 0cf7a0e (2026-11-22) compatible with the imported modules.

Important: Gold answers are NEVER passed to preprocess_example; mode='predict'
creates features independent of gold. Gold is used only after inference to score
and tune a domain-Dev threshold. Do NOT run on domain Test.

Usage (from repo root):
  python scripts/run_b1_domain_dev.py --top-k 3
  python scripts/run_b1_domain_dev.py --top-k 3 --limit 2  # SMOKE ONLY
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing_qa import preprocess_example
from src.qa_nbest import (
    best_dev_threshold,
    decode_window,
    require_gold_independent_features,
    scores_at_threshold,
)
from src.qa_metrics import exact_match_score, f1_score
from scripts.eval_qa_nbest import model_batch
from scripts.run_eval import load_bm25


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"Empty gold dataset: {path}")
    ids = [str(row.get("id", "")) for row in rows]
    if not all(ids) or len(set(ids)) != len(ids):
        raise ValueError("Gold ID missing or duplicated")
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_gold(rows: list[dict], expected: int) -> None:
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} domain Dev questions, got {len(rows)}")
    for row in rows:
        if not isinstance(row.get("question"), str) or not row["question"].strip():
            raise ValueError(f"Invalid question for {row.get('id')}")
        if row.get("split") not in (None, "dev"):
            raise ValueError(f"Refusing non-Dev gold row: {row.get('id')}")
        if type(row.get("in_scope")) is not bool:
            raise ValueError(f"Missing boolean in_scope: {row.get('id')}")
        answers = row.get("answers")
        if not isinstance(answers, list) or not all(isinstance(a, str) for a in answers):
            raise ValueError(f"Expected answers as a list of strings: {row.get('id')}")
        if row["in_scope"] and not any(a.strip() for a in answers):
            raise ValueError(f"Missing in-scope answer for {row.get('id')}")
        if not row["in_scope"] and any(a.strip() for a in answers):
            raise ValueError(f"Out-of-scope row has an answer: {row.get('id')}")
    scopes = sum(bool(r["in_scope"]) for r in rows)
    if expected == 30 and scopes != 27:
        raise ValueError(f"Expected 27 in-scope and 3 out-of-scope Dev rows; got {scopes}/{30-scopes}")


def load_chunks(path: Path, retriever) -> dict[str, dict]:
    meta = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(meta, list) or not meta:
        raise ValueError(f"Invalid chunks metadata: {path}")
    chunks = {str(x["id"]): x for x in meta}
    if len(chunks) != len(meta):
        raise ValueError("Duplicate chunk IDs in metadata")
    if any(not isinstance(x.get("text"), str) or not x["text"].strip() for x in meta):
        raise ValueError("Empty chunk text")
    index_ids = list(map(str, retriever.doc_ids))
    if len(set(index_ids)) != len(index_ids) or set(index_ids) != set(chunks):
        raise ValueError("BM25 doc IDs != chunks_meta doc IDs. Request matching artifacts from B.")
    return chunks


def segment(vn, text: str) -> list[str]:
    tokens = " ".join(vn.word_segment(text)).split()
    if "𨮁" in text:
        tokens = ["𨮁" if t == "讁" else t for t in tokens]
    if not tokens:
        raise ValueError("VnCoreNLP returned no tokens")
    return tokens


def score_one(question: str, q_id: str, bm25, chunks: dict, vn, tokenizer,
              backbone, head, device, top_k: int, batch_size: int,
              n_best_size: int, max_answer_length: int) -> dict:
    import torch

    retrieved = bm25.retrieve(question, top_k=top_k)
    if len(retrieved) != top_k:
        raise ValueError(f"{q_id}: retrieved {len(retrieved)} instead of {top_k}")
    q_tokens = segment(vn, question)
    all_candidates = []
    best_null = float("inf")
    feature_count = 0

    for rank, (chunk_id, bm25_score) in enumerate(retrieved, start=1):
        context = chunks[chunk_id]["text"]
        # PASS ONLY question+context; never insert gold answers into model inputs.
        features = preprocess_example(
            {"id": q_id, "question": question, "context": context},
            q_tokens, segment(vn, context), tokenizer, mode="predict"
        )
        require_gold_independent_features(features, f"{q_id}/{chunk_id}")
        if any(f["label_status"] != "unlabeled" for f in features):
            raise AssertionError("Inference features must be unlabeled")
        feature_count += len(features)
        for offset in range(0, len(features), batch_size):
            batch = features[offset:offset + batch_size]
            logits = model_batch(backbone, head, batch, tokenizer.pad_token_id, device)
            for i, feature in enumerate(batch):
                length = len(feature["input_ids"])
                candidates, null = decode_window(
                    feature["input_ids"], logits[i, :length, 0].tolist(),
                    logits[i, :length, 1].tolist(), tokenizer,
                    n_best_size=n_best_size, max_answer_length=max_answer_length,
                    sep_token_id=tokenizer.sep_token_id,
                    word_spans=feature["word_spans"],
                    subword_spans=feature["subword_spans"],
                    context=context,
                )
                best_null = min(best_null, float(null))
                for item in candidates:
                    all_candidates.append({
                        "chunk_id": chunk_id,
                        "rank": rank,
                        "bm25_score": float(bm25_score),
                        "text": item["text"],
                        "score": float(item["score"]),
                        "char_start": item.get("char_start"),
                        "char_end": item.get("char_end"),
                    })
    all_candidates.sort(key=lambda c: c["score"], reverse=True)
    deduped, seen = [], set()
    for c in all_candidates:
        key = (c["chunk_id"], c["char_start"], c["char_end"])
        if key not in seen:
            seen.add(key)
            deduped.append(c)
        if len(deduped) >= n_best_size:
            break
    best = deduped[0] if deduped else None
    score_diff = best_null - best["score"] if best else 1e12
    if not (float("-inf") < score_diff < float("inf")):
        raise ValueError(f"Nonfinite score_diff for {q_id}")
    return {
        "id": q_id,
        "question": question,
        "retrieved": [{"id": did, "bm25_score": float(s)} for did, s in retrieved],
        "features": feature_count,
        "best_answer": best["text"] if best else "",
        "best_chunk_id": best["chunk_id"] if best else None,
        "score_diff": score_diff,
        "null_score": best_null,
        "best_span_score": best["score"] if best else None,
        "nbest": deduped,
    }


def metrics(rows: list[dict], threshold: float) -> dict:
    scored = []
    for r in rows:
        gold_answers = r["gold_answers"] if r["in_scope"] else [""]
        p = r["best_answer"] if r["score_diff"] <= threshold else ""
        scored.append({
            "id": r["id"], "answer": p, "in_scope": r["in_scope"],
            "em": float(exact_match_score(p, gold_answers)),
            "f1": float(f1_score(p, gold_answers)),
        })
    def group(subset):
        return {
            "n": len(subset),
            "em": round(100 * sum(s["em"] for s in subset) / len(subset), 2) if subset else None,
            "f1": round(100 * sum(s["f1"] for s in subset) / len(subset), 2) if subset else None,
        }
    return {
        "all": group(scored),
        "in_scope": group([s for s in scored if s["in_scope"]]),
        "out_of_scope": group([s for s in scored if not s["in_scope"]]),
    }


def main():
    parser = argparse.ArgumentParser(description="B1 = BM25 + PhoBERT, domain Dev only")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts/phobert_qa/best_model")
    parser.add_argument("--gold", type=Path, default=ROOT / "data/gold/gold_dev.jsonl")
    parser.add_argument("--chunks", type=Path, default=ROOT / "data/index/domain/chunks_meta.json")
    parser.add_argument("--bm25", type=Path, default=ROOT / "data/index/domain/bm25.pkl")
    parser.add_argument("--vncorenlp", type=Path, default=ROOT / "data/raw/vncorenlp")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/b1_domain_dev")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--n-best", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-answer-length", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--limit", type=int, default=None, help="Smoke test only; does not publish metrics")
    args = parser.parse_args()

    if args.top_k < 1 or args.batch_size < 1 or args.n_best < 1:
        parser.error("top-k, batch-size and n-best must be positive")
    if args.limit is not None and not (1 <= args.limit <= 30):
        parser.error("limit must be between 1 and 30")
    paths = [args.checkpoint / "config.json", args.checkpoint / "qa_head.pt",
             args.checkpoint / "qa_config.json", args.gold, args.chunks, args.bm25]
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(f"Missing required file: {p}")
    if not args.vncorenlp.is_dir():
        raise FileNotFoundError(f"VnCoreNLP model directory: {args.vncorenlp}")

    gold = read_jsonl(args.gold)
    inspect_gold(gold, expected=30)
    bm25 = load_bm25("domain") if args.bm25.resolve() == (ROOT / "data/index/domain/bm25.pkl").resolve() else None
    if bm25 is None:
        import pickle
        from src.retrieval import BM25Retriever
        with args.bm25.open("rb") as handle:
            obj = pickle.load(handle)  # Only load trusted index artifacts from B.
        bm25 = BM25Retriever.__new__(BM25Retriever)
        bm25.doc_ids, bm25.bm25, bm25.tokenized_corpus = obj["doc_ids"], obj["bm25"], []
    chunks = load_chunks(args.chunks, bm25)
    print(f"Loaded {len(chunks)} chunks; Gold Dev: {len(gold)} questions; top_k={args.top_k}")

    import torch
    import py_vncorenlp
    from transformers import AutoModel, AutoTokenizer

    device_name = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA selected but not available")
    device = torch.device(device_name)
    cfg = json.loads((args.checkpoint / "qa_config.json").read_text(encoding="utf-8"))
    max_answer_length = args.max_answer_length or int(cfg.get("max_answer_length", 30))
    if max_answer_length < 1:
        raise ValueError("Invalid max_answer_length")
    tokenizer = AutoTokenizer.from_pretrained(str(args.checkpoint), use_fast=False, local_files_only=True)
    backbone = AutoModel.from_pretrained(str(args.checkpoint), local_files_only=True).to(device).eval()
    head = torch.nn.Linear(backbone.config.hidden_size, 2).to(device)
    try:
        weights = torch.load(args.checkpoint / "qa_head.pt", map_location="cpu", weights_only=True)
    except TypeError:
        weights = torch.load(args.checkpoint / "qa_head.pt", map_location="cpu")
    head.load_state_dict(weights, strict=True)
    head.eval()
    vn = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=str(args.vncorenlp))

    results = []
    workset = gold[:args.limit] if args.limit is not None else gold
    for i, row in enumerate(workset, 1):
        q_id = str(row["id"])
        pred = score_one(row["question"], q_id, bm25, chunks, vn, tokenizer,
                         backbone, head, device, args.top_k, args.batch_size,
                         args.n_best, max_answer_length)
        pred["in_scope"] = bool(row["in_scope"])
        pred["gold_answers"] = row["answers"]
        pred["question_type"] = row.get("question_type")
        pred["gold_source"] = {"article": row.get("article"), "clause": row.get("clause")}
        results.append(pred)
        print(f"[{i}/{len(workset)}] {q_id} best={pred['best_answer'][:70]!r} diff={pred['score_diff']:.3f} windows={pred['features']}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / ("smoke_nbest.jsonl" if args.limit else "nbest_domain_dev.jsonl")
    with raw_path.open("w", encoding="utf-8") as out:
        for row in results:
            out.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    print("Saved n-best:", raw_path)
    if args.limit:
        print("SMOKE ONLY. No metrics, no threshold tuning.")
        return

    scored = []
    for row in results:
        refs = row["gold_answers"] if row["in_scope"] else [""]
        scored.append({
            "score_diff": row["score_diff"],
            "answer_em": float(exact_match_score(row["best_answer"], refs)),
            "answer_f1": float(f1_score(row["best_answer"], refs)),
            "null_em": float(exact_match_score("", refs)),
            "null_f1": float(f1_score("", refs)),
        })
    tuned_threshold, tuned_em, tuned_f1 = best_dev_threshold(scored)
    fixed_em, fixed_f1 = scores_at_threshold(scored, 0.0)
    fixed_details = metrics(results, 0.0)
    tuned_details = metrics(results, tuned_threshold)
    pred_path = args.output_dir / "predictions_domain_dev.jsonl"
    with pred_path.open("w", encoding="utf-8") as out:
        for row in results:
            pred_fixed = row["best_answer"] if row["score_diff"] <= 0.0 else ""
            pred_tuned = row["best_answer"] if row["score_diff"] <= tuned_threshold else ""
            out.write(json.dumps({
                "id": row["id"], "question": row["question"],
                "in_scope": row["in_scope"], "question_type": row["question_type"],
                "gold_answers": row["gold_answers"], "gold_source": row["gold_source"],
                "retrieved_ids": [r["id"] for r in row["retrieved"]],
                "best_chunk_id": row["best_chunk_id"],
                "score_diff": row["score_diff"],
                "prediction_threshold_0": pred_fixed,
                "prediction_tuned_dev": pred_tuned,
                "em_tuned": bool(exact_match_score(pred_tuned, row["gold_answers"] if row["in_scope"] else [""])),
                "f1_tuned": float(f1_score(pred_tuned, row["gold_answers"] if row["in_scope"] else [""])),
            }, ensure_ascii=False, allow_nan=False) + "\n")

    summary = {
        "eval_split": "domain_dev", "questions": len(results),
        "in_scope": sum(r["in_scope"] for r in results),
        "out_of_scope": sum(not r["in_scope"] for r in results),
        "top_k": args.top_k, "chunk_count": len(chunks),
        "checkpoint": str(args.checkpoint.resolve()), "device": str(device),
        "preprocessing_mode": "predict_gold_independent",
        "decoder": "word_char_offsets; global best span vs min null across retrieved windows",
        "max_answer_length": max_answer_length,
        "fixed_threshold_0": {"threshold": 0.0, "em": round(100*fixed_em,2), "f1": round(100*fixed_f1,2), "breakdown": fixed_details},
        "tuned_on_domain_dev": {"threshold": tuned_threshold, "em": round(100*tuned_em,2), "f1": round(100*tuned_f1,2), "breakdown": tuned_details},
        "provenance": {
            "gold_sha256": sha256(args.gold),
            "bm25_sha256": sha256(args.bm25),
            "chunks_sha256": sha256(args.chunks),
            "head_sha256": sha256(args.checkpoint / "qa_head.pt"),
        },
        "caveats": [
            "Top-k retrieval logits are not calibrated across documents; baseline only.",
            "Domain Dev contains 3 out-of-scope questions, threshold tuning is high-variance.",
            "Never tune on domain Test. Do not compare this directly with ViQuAD Dev EM/F1.",
        ]
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"B1 Dev 30 | fixed@0 EM={100*fixed_em:.2f}% F1={100*fixed_f1:.2f}%")
    print(f"B1 Dev 30 | tuned EM={100*tuned_em:.2f}% F1={100*tuned_f1:.2f}% threshold={tuned_threshold:.5f}")
    print("Saved predictions:", pred_path)
    print("Saved summary:", summary_path)


if __name__ == "__main__":
    main()
