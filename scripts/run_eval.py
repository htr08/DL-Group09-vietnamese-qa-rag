"""Chạy đánh giá retrieval và sinh bảng kết quả từ log.

Sử dụng:
    python scripts/run_eval.py --split viquad_dev  --method all --k 1 3 5 10 20
    python scripts/run_eval.py --split domain_dev  --method all --k 1 3 10
    python scripts/run_eval.py --split domain_test --method all --k 1 3 10
"""
import argparse
import json
import pickle
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

INDEX_DIR = ROOT / "data" / "index"
SPLITS_DIR = ROOT / "data" / "splits"
RAW_DIR = ROOT / "data" / "raw"
GOLD_DIR = ROOT / "data" / "gold"

from src.metrics import recall_at_k, mrr_at_k, mean_metric
from src.retrieval import BM25Retriever, DenseRetriever, reciprocal_rank_fusion, rerank_by_title


def normalize_text(text: str) -> str:
    """Chuẩn hóa Unicode NFC và loại bỏ ký tự điều khiển."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text).strip()


# ---------------------------------------------------------------------------
# Tải index đã build
# ---------------------------------------------------------------------------

def load_bm25(corpus: str) -> BM25Retriever:
    """Tải BM25 đã lưu từ pickle."""
    pkl_path = INDEX_DIR / corpus / "bm25.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"Chưa có BM25 index tại {pkl_path}. Chạy build_index.py --corpus {corpus} trước."
        )
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    retriever = BM25Retriever.__new__(BM25Retriever)
    retriever.doc_ids = obj["doc_ids"]
    retriever.bm25 = obj["bm25"]
    retriever.tokenized_corpus = []
    return retriever


def load_dense(corpus: str) -> DenseRetriever:
    """Tải DenseRetriever với numpy index đã lưu."""
    retriever = DenseRetriever()
    retriever.load_index(INDEX_DIR / corpus / "dense")
    return retriever


def load_doc_titles(corpus: str) -> Dict[str, str]:
    """Tải tiêu đề của các chunk/tài liệu để phục vụ rerank theo tiêu đề."""
    if corpus == "domain":
        meta_path = INDEX_DIR / "domain" / "chunks_meta.json"
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                data = json.load(f)
            return {item["id"]: item.get("title", "") for item in data}
    elif corpus == "viquad":
        # Với ViQuAD, tiêu đề có thể lấy từ metadata nếu có
        return {}
    return {}


# ---------------------------------------------------------------------------
# Tải tập câu hỏi đánh giá
# ---------------------------------------------------------------------------

def load_viquad_dev_queries() -> List[Dict[str, Any]]:
    """
    Tải câu hỏi từ ViQuAD train (chỉ câu thuộc dev titles và có đáp án).
    Chuẩn hóa NFC cả hai phía để không bị mất câu hỏi âm thầm.
    """
    dev_titles_path = SPLITS_DIR / "viquad_dev_titles.txt"
    if not dev_titles_path.exists():
        raise FileNotFoundError(
            f"{dev_titles_path} chưa tồn tại. Chạy download_data.py trước."
        )
    dev_titles = set(normalize_text(t) for t in dev_titles_path.read_text(encoding="utf-8").splitlines() if t.strip())

    mapping_path = INDEX_DIR / "viquad" / "docid_to_text.json"
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"{mapping_path} chưa tồn tại. Chạy build_index.py --corpus viquad trước."
        )
    with open(mapping_path, encoding="utf-8") as f:
        docid_to_text = json.load(f)
    text_to_docid = {normalize_text(v): k for k, v in docid_to_text.items()}

    queries = []
    jsonl_path = RAW_DIR / "viquad_train.jsonl"
    total_dev_rows = 0
    no_answer_count = 0
    missing_context_count = 0

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            title = normalize_text(row.get("title", ""))
            if title not in dev_titles:
                continue
            total_dev_rows += 1

            answers = row.get("answers", {})
            answer_texts = answers.get("text", [])
            answerable = bool(answer_texts and any(a.strip() for a in answer_texts))
            if not answerable:
                no_answer_count += 1
                continue

            ctx = normalize_text(row.get("context", ""))
            relevant_id = text_to_docid.get(ctx)
            if relevant_id is None:
                missing_context_count += 1
                continue

            queries.append({
                "question": normalize_text(row["question"]),
                "relevant_id": relevant_id,
                "answerable": True,
            })

    print(f"  ViQuAD dev stats: {total_dev_rows} dòng tổng | {no_answer_count} câu không đáp án | {missing_context_count} context lệch | {len(queries)} câu hợp lệ dùng đánh giá")
    return queries


def load_gold_queries(split: str = "dev") -> List[Dict[str, Any]]:
    """
    Tải tập câu hỏi gold của miền chính (dev hoặc test).
    Đọc bằng encoding='utf-8-sig' chống BOM.
    Dựng relevant_id linh hoạt từ source_article + source_clause hoặc chunk_id.
    """
    path = GOLD_DIR / f"gold_{split}.jsonl"
    # Nếu file rỗng hoặc chưa có, kiểm tra file câu hỏi của Hiển
    if not path.exists() or path.stat().st_size == 0:
        hien_path = GOLD_DIR / "hien_33_questions.jsonl"
        if hien_path.exists() and split == "dev":
            path = hien_path
        else:
            raise FileNotFoundError(f"{path} chưa có dữ liệu.")

    queries = []
    skipped_count = 0
    with open(path, encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not item.get("in_scope", True):
                continue

            # Dựng relevant_id
            relevant_id = item.get("relevant_id") or item.get("chunk_id")
            if not relevant_id and "source_article" in item:
                # Dựng từ schema của A
                art_match = re.search(r"\d+", str(item.get("source_article", "")))
                if art_match:
                    d_num = art_match.group()
                    clause_match = re.search(r"\d+", str(item.get("source_clause", "")))
                    if clause_match:
                        relevant_id = f"dieu{d_num}_khoan{clause_match.group()}"
                    else:
                        relevant_id = f"dieu{d_num}"

            if not relevant_id:
                print(f"  [Cảnh báo dòng {line_no}] Bỏ qua câu thiếu relevant_id/source_article: {item.get('question')}")
                skipped_count += 1
                continue

            queries.append({
                "question": normalize_text(item["question"]),
                "relevant_id": relevant_id,
                "in_scope": True
            })

    print(f"  Gold queries ({split}): {len(queries)} câu hợp lệ | {skipped_count} câu bị bỏ qua")
    return queries


# ---------------------------------------------------------------------------
# Chạy retrieval và tính chỉ số (kèm đo độ trễ)
# ---------------------------------------------------------------------------

def evaluate_retriever(retriever, queries: List[Dict], ks: List[int],
                       method_name: str, top_k_retrieve: int = 20):
    recalls = {k: [] for k in ks}
    mrrs = []
    latencies = []

    for q in queries:
        question = normalize_text(q["question"])
        relevant_id = q["relevant_id"]

        t0 = time.time()
        results = retriever.retrieve(question, top_k=top_k_retrieve)
        latencies.append((time.time() - t0) * 1000)

        retrieved_ids = [doc_id for doc_id, _ in results]
        for k in ks:
            recalls[k].append(recall_at_k(retrieved_ids, relevant_id, k))
        mrrs.append(mrr_at_k(retrieved_ids, relevant_id, k=10))

    metrics = {f"R@{k}": round(mean_metric(recalls[k]), 4) for k in ks}
    metrics["MRR@10"] = round(mean_metric(mrrs), 4)
    metrics["Latency (ms)"] = round(sum(latencies) / len(latencies), 1) if latencies else 0.0
    metrics["n_queries"] = len(mrrs)
    return metrics


def evaluate_hybrid(bm25_retriever, dense_retriever, queries: List[Dict],
                    ks: List[int], rrf_k: int = 60, top_k_retrieve: int = 20,
                    doc_titles: Optional[Dict[str, str]] = None, do_rerank: bool = False):
    recalls = {k: [] for k in ks}
    mrrs = []
    latencies = []

    for q in queries:
        question = normalize_text(q["question"])
        relevant_id = q["relevant_id"]

        t0 = time.time()
        bm25_results = bm25_retriever.retrieve(question, top_k=top_k_retrieve)
        dense_results = dense_retriever.retrieve(question, top_k=top_k_retrieve)

        bm25_ids = [did for did, _ in bm25_results]
        dense_ids = [did for did, _ in dense_results]
        fused = reciprocal_rank_fusion([bm25_ids, dense_ids], rrf_k=rrf_k)

        if do_rerank and doc_titles:
            fused = rerank_by_title(fused, doc_titles, question)

        latencies.append((time.time() - t0) * 1000)
        retrieved_ids = [did for did, _ in fused[:top_k_retrieve]]

        for k in ks:
            recalls[k].append(recall_at_k(retrieved_ids, relevant_id, k))
        mrrs.append(mrr_at_k(retrieved_ids, relevant_id, k=10))

    metrics = {f"R@{k}": round(mean_metric(recalls[k]), 4) for k in ks}
    metrics["MRR@10"] = round(mean_metric(mrrs), 4)
    metrics["Latency (ms)"] = round(sum(latencies) / len(latencies), 1) if latencies else 0.0
    metrics["n_queries"] = len(mrrs)
    return metrics


# ---------------------------------------------------------------------------
# Hiển thị bảng kết quả
# ---------------------------------------------------------------------------

def print_table(rows: List[Dict], ks: List[int]):
    headers = ["Method"] + [f"R@{k}" for k in ks] + ["MRR@10", "Latency (ms)", "N"]
    col_w = max(len(h) for h in headers) + 2

    sep = "+" + "+".join("-" * col_w for _ in headers) + "+"
    print(sep)
    print("|" + "|".join(h.center(col_w) for h in headers) + "|")
    print(sep)
    for row in rows:
        vals = [row["method"]] + [str(row.get(f"R@{k}", "")) for k in ks]
        vals += [str(row.get("MRR@10", "")), str(row.get("Latency (ms)", "")), str(row.get("n_queries", ""))]
        print("|" + "|".join(v.center(col_w) for v in vals) + "|")
    print(sep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Đánh giá retrieval.")
    parser.add_argument(
        "--split",
        choices=["viquad_dev", "domain_dev", "domain_test"],
        required=True,
        help="Tập dữ liệu đánh giá",
    )
    parser.add_argument(
        "--method",
        choices=["bm25", "dense", "hybrid", "all"],
        default="all",
    )
    parser.add_argument(
        "--k", nargs="+", type=int,
        default=[1, 3, 5, 10, 20],
        help="Các giá trị k cho Recall@k",
    )
    parser.add_argument(
        "--rrf_k", type=int, default=60,
        help="Hằng số k trong công thức RRF (thử 15 và 60)",
    )
    args = parser.parse_args()

    ks = sorted(args.k)

    if args.split == "viquad_dev":
        corpus = "viquad"
        queries = load_viquad_dev_queries()
    elif args.split == "domain_dev":
        corpus = "domain"
        queries = load_gold_queries("dev")
    else:
        corpus = "domain"
        queries = load_gold_queries("test")

    rows = []
    run_bm25 = args.method in ("bm25", "all")
    run_dense = args.method in ("dense", "all")
    run_hybrid = args.method in ("hybrid", "all")

    bm25_ret: Optional[BM25Retriever] = None
    dense_ret: Optional[DenseRetriever] = None
    doc_titles = load_doc_titles(corpus)

    if run_bm25 or run_hybrid:
        print("Tải BM25 index...")
        bm25_ret = load_bm25(corpus)
    if run_dense or run_hybrid:
        print("Tải Dense index (BGE-M3)...")
        dense_ret = load_dense(corpus)

    if run_bm25 and bm25_ret:
        print("Đánh giá BM25...")
        m = evaluate_retriever(bm25_ret, queries, ks, "BM25")
        rows.append({"method": "BM25", **m})

    if run_dense and dense_ret:
        print("Đánh giá Dense (BGE-M3)...")
        m = evaluate_retriever(dense_ret, queries, ks, "Dense")
        rows.append({"method": "Dense (BGE-M3)", **m})

    if run_hybrid and bm25_ret and dense_ret:
        # Thử nghiệm cả k=60 và k=15 theo đúng yêu cầu kế hoạch (§8.2)
        print("Đánh giá Hybrid RRF (k=60)...")
        m60 = evaluate_hybrid(bm25_ret, dense_ret, queries, ks, rrf_k=60)
        rows.append({"method": "Hybrid RRF (k=60)", **m60})

        print("Đánh giá Hybrid RRF (k=15)...")
        m15 = evaluate_hybrid(bm25_ret, dense_ret, queries, ks, rrf_k=15)
        rows.append({"method": "Hybrid RRF (k=15)", **m15})

        # Hàng Hybrid + Rerank theo tiêu đề (Ablation study)
        if doc_titles:
            print("Đánh giá Hybrid RRF (k=60) + Rerank Title...")
            m_rerank = evaluate_hybrid(bm25_ret, dense_ret, queries, ks, rrf_k=60,
                                       doc_titles=doc_titles, do_rerank=True)
            rows.append({"method": "Hybrid + Rerank Title", **m_rerank})

    print(f"\n--- Kết quả ({args.split}) ---")
    print_table(rows, ks)

    out_dir = ROOT / "docs" / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"retrieval_{args.split}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"split": args.split, "ks": ks, "results": rows}, f, ensure_ascii=False, indent=2)
    print(f"→ Lưu kết quả: {out_path}")


if __name__ == "__main__":
    main()
