import argparse
import re
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
import csv
import math
import json
import numpy as np
from collections import Counter
from dotenv import load_dotenv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Đánh giá RAG bằng LLM (Gemini) thực tế và thuật toán BM25/Cosine Similarity"
    )
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Đường dẫn đến file tài liệu (.md, .txt)",
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Câu hỏi cụ thể để test (Ví dụ: 'Dung lượng pin là bao nhiêu?')",
    )
    return parser.parse_args()


# ==========================================
# 1. CHUNKING STRATEGIES
# ==========================================
def naive_chunking(text, chunk_size=300):
    """Mô phỏng Naive RAG: Cắt bừa bãi theo số lượng ký tự cố định."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_len = 0
    for word in words:
        if current_len + len(word) > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_len = len(word)
        else:
            current_chunk.append(word)
            current_len += len(word) + 1
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


def advanced_structural_chunking(text):
    """Sử dụng chính xác logic chia chunk từ file splitter.py của dự án thật."""
    try:
        from langchain_core.documents import Document
        from langchain_text_splitters import (
            MarkdownHeaderTextSplitter,
            RecursiveCharacterTextSplitter,
        )

        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on, strip_headers=False
        )
        fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=80,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        md_chunks = markdown_splitter.split_text(text)
        if not md_chunks:
            md_chunks = [Document(page_content=text)]

        final_chunks = fallback_splitter.split_documents(md_chunks)
        return [
            chunk.page_content.strip()
            for chunk in final_chunks
            if chunk.page_content.strip()
        ]
    except ImportError:
        print("[!] Không tìm thấy thư viện langchain. Đang dùng hàm fallback.")
        raw_chunks = re.split(r"(?=\n#{1,3} )", text)
        chunks = [c.strip() for c in raw_chunks if c.strip()]
        if len(chunks) <= 1:
            raw_chunks = text.split("\n\n")
            chunks = [c.strip() for c in raw_chunks if len(c.strip()) > 30]
        return chunks


def run_evaluation(file_path, query):
    file_path = file_path.strip()  # Tự động xóa khoảng trắng thừa ở hai đầu đường dẫn
    load_dotenv()
    if not os.environ.get("GEMINI_API_KEY"):
        print("[!] LỖI: Không tìm thấy GEMINI_API_KEY trong file .env.")
        print("[!] Vui lòng cấu hình biến môi trường trước khi chạy.")
        return

    if not os.path.exists(file_path):
        print(f"[!] Không tìm thấy file: {file_path}")
        return

    if file_path.lower().endswith((".md", ".txt")):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        try:
            from markitdown import MarkItDown

            print("Đang convert file sang Markdown bằng MarkItDown...")
            md_result = MarkItDown().convert(file_path)
            text = getattr(md_result, "text_content", None) or getattr(
                md_result, "markdown", None
            )
            if not text:
                print("[!] Không thể trích xuất nội dung từ file.")
                return
        except ImportError:
            print(
                "[!] Vui lòng cài đặt markitdown (pip install markitdown) để đọc file PDF/DOCX."
            )
            return
        except Exception as e:
            print(f"[!] Lỗi khi đọc file: {e}")
            return

    print(f"\nĐang xử lý tài liệu: {os.path.basename(file_path)}")
    print("-" * 60)

    # 1. CHUNKING
    adv_chunks = advanced_structural_chunking(text)
    if not adv_chunks:
        print("[!] Tài liệu rỗng hoặc không thể chia chunk.")
        return
    print(
        f"[+] Tài liệu đã được chia thành {len(adv_chunks)} chunks dựa theo Heading Markdown."
    )

    # 2. LLM GROUND TRUTH
    print(
        "\n[*] Đang gọi LLM (Gemini 2.5) để đọc file và xác định Ground Truth (thứ tự chunk chuẩn nhất)..."
    )
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm_model = os.environ.get("LLM_MODEL", "gemini-2.5-flash")
        llm = ChatGoogleGenerativeAI(model=llm_model, temperature=0.0)

        chunks_text = ""
        for i, c in enumerate(adv_chunks):
            preview = c[:500].replace("\n", " ")
            chunks_text += f"[Chunk {i}] {preview}...\n"

        prompt = f"""Bạn là một hệ thống đánh giá thông tin (RAG Assessor).
Người dùng có câu hỏi: "{query}"

Dưới đây là danh sách các đoạn tài liệu (được đánh số thứ tự từ 0):
{chunks_text}

Nhiệm vụ:
1. Chọn ra TỐI ĐA 5 đoạn tài liệu có chứa thông tin phù hợp nhất để trả lời câu hỏi trên.
2. Xếp hạng chúng từ quan trọng nhất (Rank 1) đến giảm dần.
3. CHỈ TRẢ VỀ kết quả là một mảng JSON các số nguyên tương ứng với chỉ số Chunk. Ví dụ: [3, 0, 5]
Tuyệt đối không giải thích thêm bất kỳ ký tự nào khác. Nếu không có đoạn nào liên quan, trả về []"""
        response = llm.invoke(prompt)
        match = re.search(r"\[.*?\]", response.content)
        gt_ranking = json.loads(match.group()) if match else []
    except Exception as e:
        print(f"[!] Lỗi LLM Ground Truth: {e}")
        gt_ranking = []

    # 3. GỌI TRỰC TIẾP LOGIC RETRIEVAL THỰC TẾ TỪ DỰ ÁN
    print(
        "\n[*] Đang kết nối với Vector DB (Qdrant) và Logic Hybrid Retrieval thực tế..."
    )
    top_dense = []
    top_sparse = []
    top_hybrid = []

    try:
        from backend.rag.retrieval.retriever import MarketingRetriever

        # Khởi tạo retriever thực tế (sẽ tự đọc biến môi trường và kết nối Qdrant)
        retriever = MarketingRetriever()

        print("\n[*] Đang kiểm tra cấu hình Sparse Encoder...")
        sparse_backend = getattr(
            retriever.query_encoder.sparse_encoder, "backend", "unknown"
        )
        print(f"    - Backend hiện tại: {sparse_backend}")
        if sparse_backend != "fastembed":
            print(
                "    ⚠️ CẢNH BÁO: Đang dùng thuật toán nội bộ 'bm25' vì FastEmbed không khả dụng."
            )
            print(
                "    ⚠️ Nếu dữ liệu trong Qdrant được Ingest bằng FastEmbed, Sparse Vector sẽ lệch và trả về 0 kết quả!"
            )

        # 3.1. Encode câu hỏi
        clean_question = query.strip()
        queries = retriever.query_understanding.generate_variants(clean_question)
        main_query = queries[0] if queries else clean_question

        dense_vector, sparse_vector = retriever.query_encoder.encode_parallel(
            main_query
        )

        # 3.2. Tìm kiếm song song Dense & Sparse từ Qdrant
        search_limit = max(retriever.settings.DEFAULT_TOP_K * 6, 30)
        dense_hits, sparse_hits = retriever.hybrid_searcher.search_parallel(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            dense_limit=search_limit,
            sparse_limit=search_limit,
        )

        # 3.3. Hợp nhất RRF và Rerank (Hybrid)
        fused_hits = retriever.fuser.fuse(
            dense_hits, sparse_hits, rrf_k=retriever.settings.RRF_K
        )
        reranked_hits = retriever.reranker.rerank(
            question=clean_question,
            query_dense_vector=dense_vector,
            merged_hits=fused_hits,
        )

        def extract_hit_data(hit_obj):
            payload = getattr(hit_obj, "payload", None) or (
                hit_obj if isinstance(hit_obj, dict) else {}
            )
            chunk_content = ""
            filename = "Unknown"
            if isinstance(payload, dict):
                metadata = payload.get("metadata", {})
                if not isinstance(metadata, dict):
                    metadata = {}
                chunk_content = (
                    payload.get("page_content")
                    or payload.get("text")
                    or metadata.get("page_content")
                    or metadata.get("text")
                    or ""
                )
                filename = (
                    payload.get("filename")
                    or metadata.get("filename")
                    or payload.get("source_filename")
                    or "Unknown"
                )
            score = getattr(hit_obj, "score", 0.0)
            return chunk_content, score, filename

        def match_chunk_to_adv(chunk_text):
            # Cải tiến: Dùng tập hợp từ khóa (Bag of Words) để so khớp mờ (Fuzzy Match)
            if not chunk_text:
                return -1

            def get_word_set(text):
                clean_text = re.sub(r"[^\w\s]", "", text.lower())
                return set(clean_text.split())

            chunk_words = get_word_set(chunk_text)
            if not chunk_words:
                return -1

            best_match_idx = -1
            best_overlap = 0.0

            for idx, adv in enumerate(adv_chunks):
                adv_words = get_word_set(adv)
                if not adv_words:
                    continue

                intersection = chunk_words.intersection(adv_words)
                overlap = len(intersection) / len(chunk_words) if chunk_words else 0

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match_idx = idx

            # Nếu mức độ trùng lặp từ vựng trên 40% thì coi là cùng một chunk
            if best_overlap > 0.4:
                return best_match_idx
            return -1

        for rank, hit in enumerate(dense_hits):
            chunk_text, score, filename = extract_hit_data(hit)
            top_dense.append(
                {
                    "rank": rank + 1,
                    "index": match_chunk_to_adv(chunk_text),
                    "chunk": chunk_text,
                    "score": score,
                    "filename": filename,
                }
            )

        for rank, hit in enumerate(sparse_hits):
            chunk_text, score, filename = extract_hit_data(hit)
            top_sparse.append(
                {
                    "rank": rank + 1,
                    "index": match_chunk_to_adv(chunk_text),
                    "chunk": chunk_text,
                    "score": score,
                    "filename": filename,
                }
            )

        for rank, mhit in enumerate(reranked_hits):
            chunk_text, _, filename = extract_hit_data(mhit.hit)
            top_hybrid.append(
                {
                    "rank": rank + 1,
                    "index": match_chunk_to_adv(chunk_text),
                    "chunk": chunk_text,
                    "score": mhit.final_score,
                    "filename": filename,
                }
            )

    except ImportError as e:
        print(f"[!] Lỗi Import Dự án: {e}")
        print("[!] Gợi ý: Hãy gửi file retriever.py cho tôi để tôi ráp đúng tên hàm.")
        top_hybrid = []
    except Exception as e:
        print(f"[!] Lỗi khi chạy logic dự án: {e}")
        top_hybrid = []

    # 4. IN KẾT QUẢ ĐÁNH GIÁ ĐỐI CHIẾU
    print("\n" + "=" * 80)
    print(f"KẾT QUẢ ĐÁNH GIÁ RAG THỰC TẾ (LLM GROUND TRUTH vs REAL QDRANT DB)")
    print("=" * 80)
    print(f"Câu hỏi: {query}")
    print("-" * 80)

    print("[1] LLM GROUND TRUTH (Thứ tự chuẩn do AI đọc toàn bộ file lúc này):")
    if gt_ranking:
        for rank, idx in enumerate(gt_ranking, start=1):
            if idx < len(adv_chunks):
                chunk_preview = adv_chunks[idx].replace("\n", " ")[:70] + "..."
                print(f"  🏆 Rank {rank}: [Chunk {idx}] {chunk_preview}")
    else:
        print("  Không tìm thấy đoạn nào liên quan.")

    print("\n[2] DENSE VECTOR RANKING (Semantic - Qdrant):")
    if not top_dense:
        print("  ❌ Không có dữ liệu.")
    for item in top_dense[:5]:
        chunk_preview = item["chunk"].replace("\n", " ")[:70] + "..."
        is_match = "✅" if item["index"] in gt_ranking else "❌"
        idx_str = item["index"] if item["index"] != -1 else "?"
        print(
            f"  Rank {item['rank']:<2}: {is_match} [Chunk {idx_str:<2}] (Score: {item['score']:.4f}) [File: {item['filename']}] {chunk_preview}"
        )

    print("\n[3] SPARSE VECTOR RANKING (Keyword - Qdrant/BM25):")
    if not top_sparse:
        print("  ❌ Không có dữ liệu.")
    for item in top_sparse[:5]:
        chunk_preview = item["chunk"].replace("\n", " ")[:70] + "..."
        is_match = "✅" if item["index"] in gt_ranking else "❌"
        idx_str = item["index"] if item["index"] != -1 else "?"
        print(
            f"  Rank {item['rank']:<2}: {is_match} [Chunk {idx_str:<2}] (Score: {item['score']:.4f}) [File: {item['filename']}] {chunk_preview}"
        )

    print("\n[4] HYBRID RRF + RERANKING (Qdrant + Tối ưu Heading):")
    if not top_hybrid:
        print("  ❌ Không có dữ liệu.")
    for item in top_hybrid[:5]:
        chunk_preview = item["chunk"].replace("\n", " ")[:70] + "..."
        is_match = "✅" if item["index"] in gt_ranking else "❌"
        idx_str = item["index"] if item["index"] != -1 else "?"
        print(
            f"  Rank {item['rank']:<2}: {is_match} [Chunk {idx_str:<2}] (Score: {item['score']:.4f}) [File: {item['filename']}] {chunk_preview}"
        )

    print("\n" + "=" * 80)
    if gt_ranking:

        def count_top_5_hits(sorted_full_list, gt_list):
            # Chỉ kiểm tra 5 kết quả đầu tiên, đếm số lượng chunk thuộc danh sách Ground Truth
            hits = sum(1 for item in sorted_full_list[:5] if item["index"] in gt_list)
            return hits

        d_hits = count_top_5_hits(top_dense, gt_ranking)
        s_hits = count_top_5_hits(top_sparse, gt_ranking)
        h_hits = count_top_5_hits(top_hybrid, gt_ranking)

        max_possible = min(5, len(gt_ranking))

        print("[ĐÁNH GIÁ TỔNG QUAN - SỐ LƯỢNG CHUNK ĐÚNG TRONG TOP 5]")
        print(f" - Dense Search : {d_hits}/{max_possible} chunk đúng")
        print(f" - Sparse Search: {s_hits}/{max_possible} chunk đúng")
        print(f" - Hybrid RRF   : {h_hits}/{max_possible} chunk đúng")


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args.file, args.query)
