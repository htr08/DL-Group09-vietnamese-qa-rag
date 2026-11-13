"""Chạy đánh giá retrieval và sinh bảng kết quả từ log.

Sử dụng:
    python scripts/run_eval.py --split viquad_dev  --method all --k 1 3 5 10 20
    python scripts/run_eval.py --split domain_dev  --method all --k 1 3 10
    python scripts/run_eval.py --split domain_test --method all --k 1 3 10
"""
import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

INDEX_DIR = ROOT / "data" / "index"
SPLITS_DIR = ROOT / "data" / "splits"
RAW_DIR = ROOT / "data" / "raw"
GOLD_DIR = ROOT / "data" / "gold"

from src.metrics import recall_at_k, mrr_at_k, mean_metric
from src.retrieval import BM25Retriever, DenseRetriever, reciprocal_rank_fusion


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
    # Khởi tạo lại BM25Retriever từ dữ liệu đã lưu
    retriever = BM25Retriever.__new__(BM25Retriever)
    retriever.doc_ids = obj["doc_ids"]
    retriever.bm25 = obj["bm25"]
    retriever.tokenized_corpus = []  # không cần lại
    return retriever


def load_dense(corpus: str) -> DenseRetriever:
    """Tải DenseRetriever với numpy index đã lưu."""
    retriever = DenseRetriever()
    retriever.load_index(INDEX_DIR / corpus / "dense")
    return retriever


# ---------------------------------------------------------------------------
# Tải tập câu hỏi đánh giá
# ---------------------------------------------------------------------------

def load_viquad_dev_queries() -> List[Dict[str, Any]]:
    """
    Tải câu hỏi từ ViQuAD train (chỉ câu thuộc dev titles và có đáp án).
    Mỗi phần tử: {question, relevant_id, answerable}.
    relevant_id là doc_id tương ứng với context trong index.
    """
    dev_titles_path = SPLITS_DIR / "viquad_dev_titles.txt"
    if not dev_titles_path.exists():
        raise FileNotFoundError(
            f"{dev_titles_path} chưa tồn tại. Chạy download_data.py trước."
        )
    dev_titles = set(dev_titles_path.read_text(encoding="utf-8").splitlines())

    # Tải ánh xạ context → doc_id
    mapping_path = INDEX_DIR / "viquad" / "docid_to_text.json"
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"{mapping_path} chưa tồn tại. Chạy build_index.py --corpus viquad trước."
        )
    with open(mapping_path, encoding="utf-8") as f:
        docid_to_text = json.load(f)
    text_to_docid = {v: k for k, v in docid_to_text.items()}

    queries = []
    jsonl_path = RAW_DIR / "viquad_train.jsonl"
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["title"] not in dev_titles:
                continue
            answers = row.get("answers", {})
            answer_texts = answers.get("text", [])
            answerable = bool(answer_texts and any(a.strip() for a in answer_texts))
            # Câu không có đáp án bị loại khỏi Recall@k
            if not answerable:
                continue
            ctx = row["context"]
            relevant_id = text_to_docid.get(ctx)
            if relevant_id is None:
                continue
            queries.append({
                "question": row["question"],
                "relevant_id": relevant_id,
                "answerable": True,
            })
    return queries


def load_gold_queries(split: str = "dev") -> List[Dict[str, Any]]:
    """Tải tập câu hỏi gold của miền chính (dev hoặc test)."""
    path = GOLD_DIR / f"gold_{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} chưa tồn tại.")
    queries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            # Chỉ lấy câu trong phạm vi cho Recall@k
            if item.get("in_scope", True):
                queries.append(item)
    return queries


# ---------------------------------------------------------------------------
# Chạy retrieval và tính chỉ số
# ---------------------------------------------------------------------------

def evaluate_retriever(retriever, queries: List[Dict], ks: List[int],
                       method_name: str, top_k_retrieve: int = 20):
    """Chạy retriever trên danh sách queries, trả về dict chỉ số."""
    recalls = {k: [] for k in ks}
    mrrs = []

    for q in queries:
        question = q["question"]
        relevant_id = q.get("relevant_id") or q.get("chunk_id")
        if relevant_id is None:
            continue

        results = retriever.retrieve(question, top_k=top_k_retrieve)
        retrieved_ids = [doc_id for doc_id, _ in results]

        for k in ks:
            recalls[k].append(recall_at_k(retrieved_ids, relevant_id, k))
        mrrs.append(mrr_at_k(retrieved_ids, relevant_id, k=10))

    metrics = {f"R@{k}": round(mean_metric(recalls[k]), 4) for k in ks}
    metrics["MRR@10"] = round(mean_metric(mrrs), 4)
    metrics["n_queries"] = len(mrrs)
    return metrics


def evaluate_hybrid(bm25_retriever, dense_retriever, queries: List[Dict],
                    ks: List[int], rrf_k: int = 60, top_k_retrieve: int = 20):
    """Chạy hybrid RRF và tính chỉ số."""
    recalls = {k: [] for k in ks}
    mrrs = []

    for q in queries:
        question = q["question"]
        relevant_id = q.get("relevant_id") or q.get("chunk_id")
        if relevant_id is None:
            continue

        bm25_results = bm25_retriever.retrieve(question, top_k=top_k_retrieve)
        dense_results = dense_retriever.retrieve(question, top_k=top_k_retrieve)

        bm25_ids = [did for did, _ in bm25_results]
        dense_ids = [did for did, _ in dense_results]
        fused = reciprocal_rank_fusion([bm25_ids, dense_ids], rrf_k=rrf_k)
        retrieved_ids = [did for did, _ in fused[:top_k_retrieve]]

        for k in ks:
            recalls[k].append(recall_at_k(retrieved_ids, relevant_id, k))
        mrrs.append(mrr_at_k(retrieved_ids, relevant_id, k=10))

    metrics = {f"R@{k}": round(mean_metric(recalls[k]), 4) for k in ks}
    metrics["MRR@10"] = round(mean_metric(mrrs), 4)
    metrics["n_queries"] = len(mrrs)
    return metrics


# ---------------------------------------------------------------------------
# Hiển thị bảng kết quả
# ---------------------------------------------------------------------------

def print_table(rows: List[Dict], ks: List[int]):
    """In bảng Recall@k và MRR@10 ra stdout."""
    headers = ["Method"] + [f"R@{k}" for k in ks] + ["MRR@10", "N"]
    col_w = max(len(h) for h in headers) + 2

    sep = "+" + "+".join("-" * col_w for _ in headers) + "+"
    print(sep)
    print("|" + "|".join(h.center(col_w) for h in headers) + "|")
    print(sep)
    for row in rows:
        vals = [row["method"]] + [str(row.get(f"R@{k}", "")) for k in ks]
        vals += [str(row.get("MRR@10", "")), str(row.get("n_queries", ""))]
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

    # Xác định corpus và queries
    if args.split == "viquad_dev":
        corpus = "viquad"
        queries = load_viquad_dev_queries()
        print(f"ViQuAD dev: {len(queries)} câu có đáp án")
    elif args.split == "domain_dev":
        corpus = "domain"
        queries = load_gold_queries("dev")
        print(f"Miền chính dev: {len(queries)} câu trong phạm vi")
    else:  # domain_test
        corpus = "domain"
        queries = load_gold_queries("test")
        print(f"Miền chính test: {len(queries)} câu trong phạm vi")

    rows = []
    run_bm25 = args.method in ("bm25", "all")
    run_dense = args.method in ("dense", "all")
    run_hybrid = args.method in ("hybrid", "all")

    bm25_ret: Optional[BM25Retriever] = None
    dense_ret: Optional[DenseRetriever] = None

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
        print(f"Đánh giá Hybrid RRF (k={args.rrf_k})...")
        m = evaluate_hybrid(bm25_ret, dense_ret, queries, ks, rrf_k=args.rrf_k)
        rows.append({"method": f"Hybrid RRF (k={args.rrf_k})", **m})

    print(f"\n--- Kết quả ({args.split}) ---")
    print_table(rows, ks)

    # Lưu JSON để tạo bảng báo cáo
    out_dir = ROOT / "docs" / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"retrieval_{args.split}_rrf{args.rrf_k}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"split": args.split, "rrf_k": args.rrf_k,
                   "ks": ks, "results": rows}, f, ensure_ascii=False, indent=2)
    print(f"→ Lưu kết quả: {out_path}")


if __name__ == "__main__":
    main()
