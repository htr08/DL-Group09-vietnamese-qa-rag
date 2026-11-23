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

import os
from src.metrics import recall_at_k, mrr_at_k, mean_metric, paired_bootstrap_diff_ci
from src.retrieval import DEFAULT_TITLE_BETA, BM25Retriever, DenseRetriever, hybrid_retrieve
from src.tracking import git_commit_hash, init_run

# Tham số bootstrap CI của chênh lệch chỉ số (ghi vào JSON kết quả).
CI_N_RESAMPLES = 10000
CI_SEED = 42


def paired_ci_info(comparison: str, scores_a: List[float], scores_b: List[float], metric: str = "Recall@5") -> Dict[str, Any]:
    """Bootstrap CI 95% (ghép cặp) của chênh lệch metric giữa hai hệ thống trên cùng danh sách câu hỏi."""
    ci = paired_bootstrap_diff_ci(scores_a, scores_b, n_resamples=CI_N_RESAMPLES, seed=CI_SEED)
    return {
        "comparison": comparison,
        "metric": metric,
        "estimate": round(ci.estimate, 4),
        "ci_low": round(ci.low, 4),
        "ci_high": round(ci.high, 4),
        "confidence": 0.95,
        "n_resamples": CI_N_RESAMPLES,
        "seed": CI_SEED,
        "is_significant": bool(ci.low > 0 or ci.high < 0),
    }


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
    """
    Tải tiêu đề bài viết / tiêu đề Điều của các chunk để phục vụ rerank theo tiêu đề.
    - domain: lấy từ chunks_meta.json (trường 'title' là tiêu đề Điều).
    - viquad: lấy từ docid_to_text.json + viquad_train.jsonl, dựng ánh xạ doc_id → title.
    """
    if corpus == "domain":
        meta_path = INDEX_DIR / "domain" / "chunks_meta.json"
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                data = json.load(f)
            return {item["id"]: item.get("title", "") for item in data}
    elif corpus == "viquad":
        mapping_path = INDEX_DIR / "viquad" / "docid_to_text.json"
        jsonl_path = RAW_DIR / "viquad_train.jsonl"
        if not mapping_path.exists() or not jsonl_path.exists():
            return {}
        with open(mapping_path, encoding="utf-8") as f:
            docid_to_text = json.load(f)
        # Dựng ánh xạ text → title từ viquad_train.jsonl
        text_to_title: Dict[str, str] = {}
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                ctx = normalize_text(row.get("context", ""))
                title = row.get("title", "")
                if ctx and title:
                    text_to_title[ctx] = title
        return {
            doc_id: text_to_title.get(normalize_text(text), "")
            for doc_id, text in docid_to_text.items()
        }
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
    Tách riêng câu answerable và báo lỗi rõ nếu câu có đáp án mà thiếu relevant_id.
    """
    path = GOLD_DIR / f"gold_{split}.jsonl"
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(
            f"Tệp gold '{path.name}' rỗng hoặc chưa tồn tại. Đánh giá miền chính sẽ chạy sau khi cả nhóm hoàn thành gán nhãn chéo và thống nhất nhãn (Tuần 3)."
        )

    queries = []
    unanswerable_count = 0
    with open(path, encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)

            # Hỗ trợ cả trường 'answerable' (của A) và 'in_scope' (chuẩn của nhóm)
            is_answerable = item.get("answerable")
            if is_answerable is None:
                is_answerable = item.get("in_scope", True)

            if not is_answerable:
                unanswerable_count += 1
                continue

            # Dựng relevant_id từ schema chuẩn (article, clause) hoặc schema của A (source_article, source_clause)
            relevant_id = item.get("relevant_id") or item.get("chunk_id")
            if not relevant_id:
                art = item.get("article") or item.get("source_article")
                clause = item.get("clause") or item.get("source_clause")
                if art is not None:
                    art_match = re.search(r"\d+", str(art))
                    if art_match:
                        d_num = art_match.group()
                        clause_match = re.search(r"\d+", str(clause)) if clause else None
                        if clause_match:
                            relevant_id = f"dieu{d_num}_khoan{clause_match.group()}"
                        else:
                            # Ghi chú quy ước: câu chỉ có Điều được đánh giá lỏng hơn (khớp Điều)
                            relevant_id = f"dieu{d_num}"

            if not relevant_id:
                raise ValueError(
                    f"Dòng {line_no} trong {path.name}: Câu có đáp án (answerable=True) nhưng thiếu relevant_id/article: {item.get('question')}"
                )

            queries.append({
                "question": normalize_text(item["question"]),
                "relevant_id": relevant_id,
                "answerable": True,
            })

    print(f"  Gold queries ({split}): {len(queries)} câu hợp lệ | {unanswerable_count} câu ngoài phạm vi (không đáp án)")
    return queries


# ---------------------------------------------------------------------------
# Chạy retrieval và tính chỉ số (kèm đo độ trễ)
# ---------------------------------------------------------------------------

def _add_article_scores(retrieved_ids, relevant_id, ks, recalls, mrrs):
    """Thêm điểm một câu hỏi ở mức Điều (chỉ dùng cho kho Quy chế, ID dạng dieuN_...)."""
    for k in ks:
        recalls[k].append(recall_at_k(retrieved_ids, relevant_id, k, match_level="article"))
    mrrs.append(mrr_at_k(retrieved_ids, relevant_id, k=10, match_level="article"))


def _article_metrics(recalls, mrrs):
    """Chỉ số mức Điều, thêm hậu tố _dieu để tách khỏi chỉ số mức Khoản."""
    metrics = {f"R@{k}_dieu": round(mean_metric(v), 4) for k, v in recalls.items()}
    metrics["MRR@10_dieu"] = round(mean_metric(mrrs), 4)
    return metrics


def evaluate_retriever(retriever, queries: List[Dict], ks: List[int],
                       method_name: str, top_k_retrieve: int = 20,
                       article_level: bool = False):
    top_k_retrieve = max(top_k_retrieve, *ks)  # truy hồi đủ sâu cho k lớn nhất, tránh R@k bị cắt âm thầm
    recalls = {k: [] for k in ks}
    mrrs = []
    article_recalls = {k: [] for k in ks}
    article_mrrs = []
    latencies = []

    # Warm-up 1-2 truy vấn trước khi đo để loại bỏ chi phí khởi động/load tensor
    if queries:
        for w_q in queries[:min(2, len(queries))]:
            retriever.retrieve(normalize_text(w_q["question"]), top_k=top_k_retrieve)

    for q in queries:
        question = normalize_text(q["question"])
        relevant_id = q["relevant_id"]

        t0 = time.perf_counter()
        results = retriever.retrieve(question, top_k=top_k_retrieve)
        latencies.append((time.perf_counter() - t0) * 1000)

        retrieved_ids = [doc_id for doc_id, _ in results]
        for k in ks:
            recalls[k].append(recall_at_k(retrieved_ids, relevant_id, k))
        mrrs.append(mrr_at_k(retrieved_ids, relevant_id, k=10))
        if article_level:
            _add_article_scores(retrieved_ids, relevant_id, ks, article_recalls, article_mrrs)

    metrics = {f"R@{k}": round(mean_metric(recalls[k]), 4) for k in ks}
    metrics["MRR@10"] = round(mean_metric(mrrs), 4)
    metrics["Latency (ms)"] = round(sum(latencies) / len(latencies), 1) if latencies else 0.0
    metrics["n_queries"] = len(mrrs)
    if article_level:
        metrics.update(_article_metrics(article_recalls, article_mrrs))
    recalls["mrr"] = mrrs
    return metrics, recalls


def evaluate_hybrid(bm25_retriever, dense_retriever, queries: List[Dict],
                    ks: List[int], rrf_k: int = 60, top_k_retrieve: int = 20,
                    doc_titles: Optional[Dict[str, str]] = None, do_rerank: bool = False,
                    article_level: bool = False):
    top_k_retrieve = max(top_k_retrieve, *ks)  # truy hồi đủ sâu cho k lớn nhất, tránh R@k bị cắt âm thầm
    recalls = {k: [] for k in ks}
    mrrs = []
    article_recalls = {k: [] for k in ks}
    article_mrrs = []
    latencies = []

    # Warm-up 1-2 truy vấn trước khi đo
    if queries:
        for w_q in queries[:min(2, len(queries))]:
            hybrid_retrieve(bm25_retriever, dense_retriever, normalize_text(w_q["question"]),
                            top_k=top_k_retrieve, rrf_k=rrf_k, doc_titles=doc_titles if do_rerank else None)

    for q in queries:
        question = normalize_text(q["question"])
        relevant_id = q["relevant_id"]

        t0 = time.perf_counter()
        fused = hybrid_retrieve(bm25_retriever, dense_retriever, question, top_k=top_k_retrieve,
                                rrf_k=rrf_k, doc_titles=doc_titles if do_rerank else None)
        latencies.append((time.perf_counter() - t0) * 1000)
        retrieved_ids = [did for did, _ in fused]

        for k in ks:
            recalls[k].append(recall_at_k(retrieved_ids, relevant_id, k))
        mrrs.append(mrr_at_k(retrieved_ids, relevant_id, k=10))
        if article_level:
            _add_article_scores(retrieved_ids, relevant_id, ks, article_recalls, article_mrrs)

    metrics = {f"R@{k}": round(mean_metric(recalls[k]), 4) for k in ks}
    metrics["MRR@10"] = round(mean_metric(mrrs), 4)
    metrics["Latency (ms)"] = round(sum(latencies) / len(latencies), 1) if latencies else 0.0
    metrics["n_queries"] = len(mrrs)
    if article_level:
        metrics.update(_article_metrics(article_recalls, article_mrrs))
    recalls["mrr"] = mrrs
    return metrics, recalls


# ---------------------------------------------------------------------------
# Hiển thị bảng kết quả
# ---------------------------------------------------------------------------

def print_table(rows: List[Dict], ks: List[int]):
    headers = ["Method"] + [f"R@{k}" for k in ks] + ["MRR@10", "Latency (ms)", "N"]
    article_headers = [f"R@{k}_dieu" for k in ks] + ["MRR@10_dieu"]
    has_article = any("MRR@10_dieu" in row for row in rows)
    if has_article:
        headers += article_headers
    col_w = max(len(h) for h in headers) + 2

    sep = "+" + "+".join("-" * col_w for _ in headers) + "+"
    print(sep)
    print("|" + "|".join(h.center(col_w) for h in headers) + "|")
    print(sep)
    for row in rows:
        vals = [row["method"]] + [str(row.get(f"R@{k}", "")) for k in ks]
        vals += [str(row.get("MRR@10", "")), str(row.get("Latency (ms)", "")), str(row.get("n_queries", ""))]
        if has_article:
            vals += [str(row.get(h, "")) for h in article_headers]
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
        "--rrf_k", type=int, default=None,
        help="Hằng số k trong công thức RRF. Nếu không chỉ định, thử cả k=15 và k=60 rồi chọn k tốt nhất theo Recall@5.",
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

    # Chỉ số mức Điều chỉ có nghĩa với ID dạng dieuN_...; ID của kho ViQuAD (viquad_NNNNN)
    # không có cấu trúc Điều nên mức Điều sẽ khớp mọi đoạn.
    article_level = corpus == "domain"

    rows = []
    run_bm25 = args.method in ("bm25", "all")
    run_dense = args.method in ("dense", "all")
    run_hybrid = args.method in ("hybrid", "all")

    bm25_ret: Optional[BM25Retriever] = None
    dense_ret: Optional[DenseRetriever] = None
    doc_titles = load_doc_titles(corpus)
    if doc_titles:
        print(f"  Tải {len(doc_titles)} tiêu đề chunk để rerank.")

    if run_bm25 or run_hybrid:
        print("Tải BM25 index...")
        bm25_ret = load_bm25(corpus)
    if run_dense or run_hybrid:
        print("Tải Dense index (BGE-M3)...")
        dense_ret = load_dense(corpus)

    # Điểm từng câu cho BM25 và Dense (dùng tính Bootstrap CI)
    bm25_r1_scores: List[float] = []
    bm25_r5_scores: List[float] = []
    bm25_mrr_scores: List[float] = []
    dense_r1_scores: List[float] = []
    dense_r5_scores: List[float] = []
    dense_mrr_scores: List[float] = []

    if run_bm25 and bm25_ret:
        print("Đánh giá BM25...")
        m, bm25_recalls = evaluate_retriever(bm25_ret, queries, ks, "BM25", article_level=article_level)
        rows.append({"method": "BM25", **m})
        bm25_r1_scores = bm25_recalls.get(1, [])
        bm25_r5_scores = bm25_recalls.get(5, [])
        bm25_mrr_scores = bm25_recalls.get("mrr", [])

    if run_dense and dense_ret:
        print("Đánh giá Dense (BGE-M3)...")
        m, dense_recalls = evaluate_retriever(dense_ret, queries, ks, "Dense", article_level=article_level)
        rows.append({"method": "Dense (BGE-M3)", **m})
        dense_r1_scores = dense_recalls.get(1, [])
        dense_r5_scores = dense_recalls.get(5, [])
        dense_mrr_scores = dense_recalls.get("mrr", [])

    best_rrf_k: Optional[int] = None
    ci_dict: Dict[str, Any] = {}
    ci_info: Optional[Dict[str, Any]] = None
    rerank_ci_info: Optional[Dict[str, Any]] = None

    if run_hybrid and bm25_ret and dense_ret:
        # Mục 1: Trên tập test, rrf_k phải được chọn từ dev và truyền qua --rrf_k (§3.5c)
        if "test" in args.split:
            if args.rrf_k is None:
                raise ValueError(
                    f"Khi đánh giá trên tập test ({args.split}), --rrf_k phải được chọn từ dev "
                    f"và truyền tường minh qua CLI (ví dụ: --rrf_k 60). Không tự chọn lại k trên tập test (§3.5c)."
                )
            rrf_ks_to_try = [args.rrf_k]
        else:
            if args.rrf_k is not None:
                rrf_ks_to_try = [args.rrf_k]
            else:
                rrf_ks_to_try = [60, 15]

        hybrid_r1_scores: List[float] = []
        hybrid_r5_scores: List[float] = []
        hybrid_mrr_scores: List[float] = []
        best_rrf_k = rrf_ks_to_try[0]
        best_eval_tuple = (-1.0, -1.0)  # (MRR@10, R@1)

        hybrid_metrics = {}
        for rrf_k in rrf_ks_to_try:
            label = f"Hybrid RRF (k={rrf_k})"
            print(f"Đánh giá {label}...")
            m, h_recalls = evaluate_hybrid(bm25_ret, dense_ret, queries, ks, rrf_k=rrf_k,
                                           article_level=article_level)
            rows.append({"method": label, **m})
            hybrid_metrics[rrf_k] = m
            # Chọn theo MRR@10 (hòa thì theo R@1)
            curr_tuple = (m.get("MRR@10", 0.0), m.get("R@1", 0.0))
            if curr_tuple > best_eval_tuple:
                best_eval_tuple = curr_tuple
                best_rrf_k = rrf_k
                hybrid_r1_scores = h_recalls.get(1, [])
                hybrid_r5_scores = h_recalls.get(5, [])
                hybrid_mrr_scores = h_recalls.get("mrr", [])

        if len(rrf_ks_to_try) > 1:
            all_tied = len(set((m.get("MRR@10"), m.get("R@1")) for m in hybrid_metrics.values())) == 1
            if all_tied:
                print(f"  → Cả k=15 và k=60 cho kết quả tương đương trên {args.split} (MRR@10={best_eval_tuple[0]:.4f}, R@1={best_eval_tuple[1]:.4f}). Chọn k={best_rrf_k} theo §3.5c (dev không phân biệt được k).")
            else:
                print(f"  → Chọn rrf_k={best_rrf_k} theo MRR@10={best_eval_tuple[0]:.4f}, R@1={best_eval_tuple[1]:.4f} trên {args.split}")

        # Mục 2: Bootstrap CI cho R@1, MRR@10 và R@5
        if hybrid_r5_scores and (bm25_r5_scores or dense_r5_scores):
            if bm25_r5_scores:
                ci_bm25_r1 = paired_ci_info(f"Hybrid (k={best_rrf_k}) vs BM25", hybrid_r1_scores, bm25_r1_scores, metric="R@1")
                ci_bm25_mrr = paired_ci_info(f"Hybrid (k={best_rrf_k}) vs BM25", hybrid_mrr_scores, bm25_mrr_scores, metric="MRR@10")
                ci_bm25_r5 = paired_ci_info(f"Hybrid (k={best_rrf_k}) vs BM25", hybrid_r5_scores, bm25_r5_scores, metric="R@5")
                ci_dict["hybrid_vs_bm25"] = {
                    "R@1": ci_bm25_r1,
                    "MRR@10": ci_bm25_mrr,
                    "R@5": ci_bm25_r5,
                }
                print(f"\n  Bootstrap CI (Hybrid vs BM25):")
                print(f"    R@1:    estimate={ci_bm25_r1['estimate']:+.4f}, 95% CI=[{ci_bm25_r1['ci_low']:+.4f}, {ci_bm25_r1['ci_high']:+.4f}]")
                print(f"    MRR@10: estimate={ci_bm25_mrr['estimate']:+.4f}, 95% CI=[{ci_bm25_mrr['ci_low']:+.4f}, {ci_bm25_mrr['ci_high']:+.4f}]")
                if all(s == 1.0 for s in hybrid_r5_scores) and all(s == 1.0 for s in bm25_r5_scores):
                    print(f"    R@5:    Cả hai phương pháp đều đạt Recall@5 = 1.0 (chạm trần hiệu năng / bão hòa trên dev)")
                else:
                    print(f"    R@5:    estimate={ci_bm25_r5['estimate']:+.4f}, 95% CI=[{ci_bm25_r5['ci_low']:+.4f}, {ci_bm25_r5['ci_high']:+.4f}]")

            if dense_r5_scores:
                ci_dense_r1 = paired_ci_info(f"Hybrid (k={best_rrf_k}) vs Dense", hybrid_r1_scores, dense_r1_scores, metric="R@1")
                ci_dense_mrr = paired_ci_info(f"Hybrid (k={best_rrf_k}) vs Dense", hybrid_mrr_scores, dense_mrr_scores, metric="MRR@10")
                ci_dense_r5 = paired_ci_info(f"Hybrid (k={best_rrf_k}) vs Dense", hybrid_r5_scores, dense_r5_scores, metric="R@5")
                ci_dict["hybrid_vs_dense"] = {
                    "R@1": ci_dense_r1,
                    "MRR@10": ci_dense_mrr,
                    "R@5": ci_dense_r5,
                }
                print(f"\n  Bootstrap CI (Hybrid vs Dense):")
                print(f"    R@1:    estimate={ci_dense_r1['estimate']:+.4f}, 95% CI=[{ci_dense_r1['ci_low']:+.4f}, {ci_dense_r1['ci_high']:+.4f}]")
                print(f"    MRR@10: estimate={ci_dense_mrr['estimate']:+.4f}, 95% CI=[{ci_dense_mrr['ci_low']:+.4f}, {ci_dense_mrr['ci_high']:+.4f}]")
                if all(s == 1.0 for s in hybrid_r5_scores) and all(s == 1.0 for s in dense_r5_scores):
                    print(f"    R@5:    Cả hai phương pháp đều đạt Recall@5 = 1.0 (chạm trần hiệu năng / bão hòa trên dev)")
                else:
                    print(f"    R@5:    estimate={ci_dense_r5['estimate']:+.4f}, 95% CI=[{ci_dense_r5['ci_low']:+.4f}, {ci_dense_r5['ci_high']:+.4f}]")

            if bm25_r5_scores and dense_r5_scores:
                single_r5 = bm25_r5_scores if mean_metric(bm25_r5_scores) >= mean_metric(dense_r5_scores) else dense_r5_scores
                best_single_name = "BM25" if mean_metric(bm25_r5_scores) >= mean_metric(dense_r5_scores) else "Dense"
            elif bm25_r5_scores:
                single_r5 = bm25_r5_scores
                best_single_name = "BM25"
            else:
                single_r5 = dense_r5_scores
                best_single_name = "Dense"
            ci_info = paired_ci_info(f"Hybrid (k={best_rrf_k}) vs {best_single_name}",
                                     hybrid_r5_scores, single_r5, metric="Recall@5")
            ci_info["detailed"] = ci_dict
        else:
            print("  [CI] Bỏ qua: cần chạy cùng BM25 hoặc Dense để so sánh với hybrid.")

        # Rerank theo tiêu đề (ablation) - chạy với rrf_k tốt nhất
        if doc_titles:
            print(f"Đánh giá Hybrid RRF (k={best_rrf_k}) + Rerank Title...")
            m_rerank, rerank_recalls = evaluate_hybrid(bm25_ret, dense_ret, queries, ks, rrf_k=best_rrf_k,
                                                       doc_titles=doc_titles, do_rerank=True,
                                                       article_level=article_level)
            rows.append({"method": f"Hybrid + Rerank Title (k={best_rrf_k})", **m_rerank})
            if hybrid_r5_scores:
                rerank_ci_info = paired_ci_info(
                    f"Hybrid + Rerank Title (k={best_rrf_k}) vs Hybrid (k={best_rrf_k})",
                    rerank_recalls.get(5, []), hybrid_r5_scores, metric="Recall@5"
                )
                rerank_ci_info["detailed"] = {
                    "R@1": paired_ci_info(f"Hybrid + Rerank Title vs Hybrid", rerank_recalls.get(1, []), hybrid_r1_scores, metric="R@1"),
                    "MRR@10": paired_ci_info(f"Hybrid + Rerank Title vs Hybrid", rerank_recalls.get("mrr", []), hybrid_mrr_scores, metric="MRR@10"),
                    "R@5": paired_ci_info(f"Hybrid + Rerank Title vs Hybrid", rerank_recalls.get(5, []), hybrid_r5_scores, metric="R@5"),
                }
                print(f"  Bootstrap CI (Rerank vs Hybrid):")
                print(f"    R@1:    estimate={rerank_ci_info['detailed']['R@1']['estimate']:+.4f}, 95% CI=[{rerank_ci_info['detailed']['R@1']['ci_low']:+.4f}, {rerank_ci_info['detailed']['R@1']['ci_high']:+.4f}]")
                print(f"    MRR@10: estimate={rerank_ci_info['detailed']['MRR@10']['estimate']:+.4f}, 95% CI=[{rerank_ci_info['detailed']['MRR@10']['ci_low']:+.4f}, {rerank_ci_info['detailed']['MRR@10']['ci_high']:+.4f}]")
        else:
            print("  [Rerank] Bỏ qua: không có tiêu đề chunk.")

    print(f"\n--- Kết quả ({args.split}) ---")
    print_table(rows, ks)

    # Mục 2: Lưu đầy đủ commit hash, n_chunks, device, model, rrf_k, rerank_beta, bootstrap CI vào JSON
    beta_used = DEFAULT_TITLE_BETA if (run_hybrid and doc_titles) else None
    out_dir = ROOT / "docs" / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"retrieval_{args.split}.json"

    # Lấy thông tin thiết bị và số chunk
    device_str = "cpu"
    if dense_ret and hasattr(dense_ret, "model"):
        device_str = str(dense_ret.model.device)
    elif run_dense:
        try:
            import torch
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device_str = "cpu"

    n_chunks = None
    if bm25_ret and hasattr(bm25_ret, "doc_ids"):
        n_chunks = len(bm25_ret.doc_ids)
    elif dense_ret and hasattr(dense_ret, "doc_ids"):
        n_chunks = len(dense_ret.doc_ids)

    output_data = {
        "split": args.split,
        "commit_hash": git_commit_hash(),
        "n_chunks": n_chunks,
        "device": device_str,
        "model": "BAAI/bge-m3" if (run_dense or run_hybrid) else None,
        "rrf_k": best_rrf_k,
        "rerank_beta": beta_used,
        "ks": ks,
        "bootstrap_ci": ci_info,
        "bootstrap_ci_detailed": ci_dict,
        "bootstrap_ci_rerank": rerank_ci_info,
        "results": rows,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"→ Lưu kết quả: {out_path}")

    # Mục 2: Tự động log lên W&B nếu đã đăng nhập và không tắt WANDB_MODE
    if os.environ.get("WANDB_MODE") != "disabled":
        try:
            import wandb
            # Kiểm tra xem đã login chưa
            if wandb.api.api_key or os.environ.get("WANDB_API_KEY"):
                run_k_suffix = f"_k{best_rrf_k}" if best_rrf_k else ""
                # Tên run theo mẫu <role>_<model>_<cấu hình>_seed<k> (project_plan §8.2);
                # seed là seed của bootstrap CI vì truy hồi không có thành phần ngẫu nhiên khác.
                run_name = f"B_bge-m3_eval-{args.split}{run_k_suffix}_seed{CI_SEED}"
                wb_run = init_run(
                    name=run_name,
                    job_type="eval_retrieval",
                    config={
                        "split": args.split,
                        "method": args.method,
                        "n_chunks": n_chunks,
                        "device": device_str,
                        "model": "BAAI/bge-m3",
                        "rrf_k": best_rrf_k,
                        "rerank_beta": beta_used,
                        "ks": ks,
                        "commit": git_commit_hash(),
                    },
                )
                for row in rows:
                    m_prefix = row["method"].replace(" ", "_").replace("(", "").replace(")", "")
                    for metric_k, metric_v in row.items():
                        if metric_k != "method":
                            wb_run.summary[f"{m_prefix}/{metric_k}"] = metric_v
                for comp_key, comp_dict in ci_dict.items():
                    for metric_key, c_info in comp_dict.items():
                        wb_run.summary[f"bootstrap_ci/{comp_key}/{metric_key}/estimate"] = c_info["estimate"]
                        wb_run.summary[f"bootstrap_ci/{comp_key}/{metric_key}/ci_low"] = c_info["ci_low"]
                        wb_run.summary[f"bootstrap_ci/{comp_key}/{metric_key}/ci_high"] = c_info["ci_high"]
                if rerank_ci_info:
                    wb_run.summary["bootstrap_ci/rerank_vs_hybrid/R@5/estimate"] = rerank_ci_info["estimate"]
                    wb_run.summary["bootstrap_ci/rerank_vs_hybrid/R@5/ci_low"] = rerank_ci_info["ci_low"]
                    wb_run.summary["bootstrap_ci/rerank_vs_hybrid/R@5/ci_high"] = rerank_ci_info["ci_high"]
                    if "detailed" in rerank_ci_info:
                        for m_k, sub_c in rerank_ci_info["detailed"].items():
                            wb_run.summary[f"bootstrap_ci/rerank_vs_hybrid/{m_k}/estimate"] = sub_c["estimate"]
                            wb_run.summary[f"bootstrap_ci/rerank_vs_hybrid/{m_k}/ci_low"] = sub_c["ci_low"]
                            wb_run.summary[f"bootstrap_ci/rerank_vs_hybrid/{m_k}/ci_high"] = sub_c["ci_high"]
                wb_run.finish()
                print(f"✓ Đã log W&B run '{run_name}' lên dự án nhom09-vi-qa-rag/vi-qa-rag")
            else:
                print("  [W&B Note] Chưa đăng nhập tài khoản W&B. Chạy 'wandb login' để tự động sync lên dashboard nhóm.")
        except Exception as exc:
            print(f"  [W&B Note] Bỏ qua log cloud do lỗi kết nối: {exc}")


if __name__ == "__main__":
    main()
