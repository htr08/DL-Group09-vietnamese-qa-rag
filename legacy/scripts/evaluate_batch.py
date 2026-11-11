import argparse
import json
import os
import re
import sys
from dotenv import load_dotenv

# Thêm gốc của project vào sys.path để import được module backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Chạy đánh giá hàng loạt và so sánh hiệu năng Naive RAG vs Advanced RAG."
    )
    parser.add_argument(
        "--questions",
        type=str,
        default="backend/rag/doc/test_questions.json",
        help="Đường dẫn đến file JSON chứa danh sách các câu hỏi test.",
    )
    parser.add_argument(
        "--mock-llm",
        action="store_true",
        help="Bật chế độ giả lập (Mock) cho LLM để tránh bị trừ quota hoặc giới hạn API.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.txt",
        help="File lưu kết quả xuất ra dạng TXT để phục vụ vẽ biểu đồ.",
    )
    return parser.parse_args()


def get_all_chunks_from_qdrant():
    """Lấy toàn bộ chunk trực tiếp từ Qdrant."""
    try:
        from qdrant_client import QdrantClient
        from backend.rag.config import rag_settings

        # Sử dụng config từ dự án hoặc fallback
        qdrant_url = getattr(
            rag_settings,
            "QDRANT_URL",
            os.environ.get("QDRANT_URL", "http://localhost:6333"),
        )
        collection_name = getattr(
            rag_settings, "QDRANT_COLLECTION_NAME", "marketing_docs"
        )

        print(
            f"[*] Đang tải toàn bộ document chunks từ Qdrant (Collection: {collection_name})..."
        )
        client = QdrantClient(url=qdrant_url)

        chunks = []
        offset = None
        while True:
            records, offset = client.scroll(
                collection_name=collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for record in records:
                payload = record.payload or {}
                # Lấy text tuỳ thuộc vào cấu trúc schema payload đang dùng trong DB
                text = payload.get("text") or payload.get("page_content") or ""
                if not text and isinstance(payload.get("metadata"), dict):
                    text = (
                        payload["metadata"].get("text")
                        or payload["metadata"].get("page_content")
                        or ""
                    )

                if text:
                    chunks.append(text.strip())

            if offset is None:
                break
        return chunks
    except Exception as e:
        print(f"[!] Lỗi khi lấy dữ liệu từ Qdrant: {e}")
        return []


def get_llm_ground_truth(document_chunks, query, llm, use_mock=False):
    """Gọi LLM để xác định các chunk liên quan nhất."""
    if use_mock:
        # Chế độ Mock: Trả về giả lập chunk đầu tiên và chunk ngẫu nhiên để test luồng
        return [0, len(document_chunks) // 2] if document_chunks else []

    chunks_text = ""
    for i, c in enumerate(document_chunks):
        preview = c[:500].replace("\n", " ")
        chunks_text += f"[Chunk {i}] {preview}...\n"

    prompt = f"""Bạn là một hệ thống đánh giá RAG.
Truy vấn/Từ khóa của người dùng: "{query}"
Dưới đây là danh sách các đoạn tài liệu (đánh số từ 0):
{chunks_text}
Nhiệm vụ: Chọn ra TỐI ĐA 5 đoạn tài liệu có chứa thông tin phù hợp nhất để giải đáp hoặc liên quan trực tiếp đến truy vấn trên.
CHỈ TRẢ VỀ một mảng JSON các số nguyên là chỉ số Chunk. Ví dụ: [3, 0, 5].
Nếu không có đoạn nào liên quan, trả về []."""

    try:
        response = llm.invoke(prompt)
        # Regex lấy đúng định dạng mảng số nguyên, tránh lỗi parse JSON nếu LLM sinh chữ linh tinh
        match = re.search(r"\[[0-9,\s]*\]", response.content)
        if match:
            result = json.loads(match.group())
            if not result:
                print(
                    "    [!] CẢNH BÁO: LLM trả về mảng rỗng []. Nguyên nhân: Không có chunk nào trong DB chứa đáp án cho câu hỏi này."
                )
            return result
        else:
            print(f"    [!] LLM trả về sai định dạng Regex: {response.content}")

        return []
    except Exception as e:
        print(f"    [!] Lỗi khi lấy Ground Truth từ LLM: {e}")
        return []


def match_chunk_to_adv(chunk_text, adv_chunks):
    """So khớp mờ nội dung từ Qdrant về index của chunk local."""
    if not chunk_text:
        return -1

    def get_word_set(text):
        clean_text = re.sub(r"[^\w\s]", "", text.lower())
        return set(clean_text.split())

    chunk_words = get_word_set(chunk_text)
    if not chunk_words:
        return -1

    best_match_idx, best_overlap = -1, 0.0
    for idx, adv in enumerate(adv_chunks):
        adv_words = get_word_set(adv)
        if not adv_words:
            continue
        intersection = chunk_words.intersection(adv_words)
        overlap = len(intersection) / len(chunk_words) if chunk_words else 0
        if overlap > best_overlap:
            best_overlap, best_match_idx = overlap, idx

    return best_match_idx if best_overlap > 0.4 else -1


def run_single_evaluation(doc_chunks, query, retriever, llm, use_mock=False):
    """Chạy đánh giá cho một câu hỏi duy nhất và trả về số liệu."""
    gt_ranking = get_llm_ground_truth(doc_chunks, query, llm, use_mock)
    if not gt_ranking:
        print("    [-] Bỏ qua câu hỏi vì Ground Truth trống.")
        return None

    # 1. Chạy logic RAG thực tế
    dense_vector, sparse_vector = retriever.query_encoder.encode_parallel(query)
    search_limit = max(retriever.settings.DEFAULT_TOP_K * 6, 30)
    dense_hits, sparse_hits = retriever.hybrid_searcher.search_parallel(
        dense_vector=dense_vector,
        sparse_vector=sparse_vector,
        dense_limit=search_limit,
        sparse_limit=search_limit,
    )
    fused_hits = retriever.fuser.fuse(
        dense_hits, sparse_hits, rrf_k=retriever.settings.RRF_K
    )
    reranked_hits = retriever.reranker.rerank(query, dense_vector, fused_hits)

    # 2. Trích xuất kết quả
    def get_indices(hits, is_reranked=False):
        indices = []
        for hit in hits[:5]:
            hit_obj = hit.hit if is_reranked else hit
            payload = getattr(hit_obj, "payload", {})
            content = payload.get("text") or payload.get("page_content") or ""
            indices.append(match_chunk_to_adv(content, doc_chunks))
        return indices

    advanced_indices = get_indices(reranked_hits, is_reranked=True)
    naive_indices = get_indices(dense_hits)  # Naive RAG = Dense Search only
    sparse_indices = get_indices(sparse_hits)  # Keyword Search only

    # 3. Đếm số chunk đúng (Dùng set() để loại bỏ các chunk bị trùng lặp do nạp nhiều lần vào DB)
    advanced_correct = sum(1 for idx in set(advanced_indices) if idx in gt_ranking)
    naive_correct = sum(1 for idx in set(naive_indices) if idx in gt_ranking)
    sparse_correct = sum(1 for idx in set(sparse_indices) if idx in gt_ranking)

    return {
        "query": query,
        "advanced_hits": advanced_correct,
        "naive_hits": naive_correct,
        "sparse_hits": sparse_correct,
        "ground_truth_count": len(gt_ranking),
    }


def main():
    args = parse_args()
    load_dotenv()

    # 1. Tải toàn bộ tài liệu từ Qdrant
    doc_chunks = get_all_chunks_from_qdrant()
    if not doc_chunks:
        print(
            "[!] Không có dữ liệu trong Qdrant để đánh giá. Vui lòng ingest tài liệu trước."
        )
        return
    print(
        f"[+] Tài liệu đã được kéo về thành công, tổng cộng {len(doc_chunks)} chunks từ Qdrant."
    )

    # 2. Tải câu hỏi test
    if not os.path.exists(args.questions):
        print(f"[!] Không tìm thấy file câu hỏi: {args.questions}")
        return
    with open(args.questions, "r", encoding="utf-8") as f:
        questions = json.load(f)

    # Giới hạn chạy trong đúng 10 câu hỏi theo yêu cầu
    questions = questions[:10]
    print(f"[+] Đã tải {len(questions)} câu hỏi từ '{args.questions}'.")

    # 3. Khởi tạo các thành phần RAG
    try:
        from backend.rag.retrieval.retriever import MarketingRetriever
        from backend.rag.config import rag_settings
        from langchain_google_genai import ChatGoogleGenerativeAI

        retriever = MarketingRetriever()
        model_name = rag_settings.GEMINI_CHAT_MODEL
        print(f"[*] Đang khởi tạo LLM với model: {model_name}")
        llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.0)
    except Exception as e:
        print(f"[!] Lỗi khởi tạo RAG engine: {e}")
        return

    # 4. Chạy đánh giá hàng loạt
    print("\n" + "=" * 80)
    print("BẮT ĐẦU CHẠY ĐÁNH GIÁ HÀNG LOẠT (TỐI ĐA 10 CÂU)")
    print("=" * 80)

    (
        total_advanced_hits,
        total_naive_hits,
        total_sparse_hits,
        total_gt_chunks_in_top5,
    ) = (0, 0, 0, 0)
    processed_questions = 0

    with open(args.output, "w", encoding="utf-8") as f_out:
        # Ghi Header cho file TXT phân tách bằng Tab
        f_out.write(
            "Query_Index\tVector_Search_Acc(%)\tSparse_Search_Acc(%)\tAdvanced_Hybrid_Acc(%)\tGround_Truth_Count\n"
        )

        for i, query in enumerate(questions):
            print(f"[*] Đang xử lý câu hỏi {i + 1}/{len(questions)}...")
            result = run_single_evaluation(
                doc_chunks, query.strip(), retriever, llm, use_mock=args.mock_llm
            )
            if result:
                processed_questions += 1
                adv_hits = result["advanced_hits"]
                naive_hits = result["naive_hits"]
                sparse_hits = result["sparse_hits"]
                gt_count = min(5, result["ground_truth_count"])

                total_advanced_hits += adv_hits
                total_naive_hits += naive_hits
                total_sparse_hits += sparse_hits
                total_gt_chunks_in_top5 += gt_count

                naive_acc = (naive_hits / gt_count * 100) if gt_count > 0 else 0.0
                sparse_acc = (sparse_hits / gt_count * 100) if gt_count > 0 else 0.0
                adv_acc = (adv_hits / gt_count * 100) if gt_count > 0 else 0.0

                # Ghi kết quả mỗi lần hỏi ra file txt
                f_out.write(
                    f"{i + 1}\t{naive_acc:.2f}\t{sparse_acc:.2f}\t{adv_acc:.2f}\t{gt_count}\n"
                )
                print(
                    f"    -> Acc | Vector: {naive_acc:.2f}% | Sparse: {sparse_acc:.2f}% | Advanced: {adv_acc:.2f}%"
                )

    print(f"\n[+] Đã lưu kết quả chi tiết vào file '{args.output}' để vẽ biểu đồ.")

    # 5. In báo cáo tổng hợp
    print("\n" + "=" * 80)
    print("BẢNG TỔNG HỢP ACCURACY TRUNG BÌNH CÁC PHƯƠNG PHÁP SEARCH")
    print("=" * 80)

    adv_accuracy = (
        (total_advanced_hits / total_gt_chunks_in_top5) * 100
        if total_gt_chunks_in_top5 > 0
        else 0
    )
    naive_accuracy = (
        (total_naive_hits / total_gt_chunks_in_top5) * 100
        if total_gt_chunks_in_top5 > 0
        else 0
    )
    sparse_accuracy = (
        (total_sparse_hits / total_gt_chunks_in_top5) * 100
        if total_gt_chunks_in_top5 > 0
        else 0
    )

    print(f"Tổng số câu hỏi đã test: {processed_questions}")
    print(
        f"Tổng số chunks mục tiêu (Ground Truth Max Top 5): {total_gt_chunks_in_top5}"
    )
    print("-" * 90)
    print(
        "| {:<40} | {:<20} | {:<15} |".format(
            "Phương pháp", "Tổng Chunks Đúng", "Trung bình Acc"
        )
    )
    print("-" * 90)
    print(
        "| {:<40} | {:<20} | {:<15.2f}% |".format(
            "Chỉ Vector Search (Dense)",
            f"{total_naive_hits}/{total_gt_chunks_in_top5}",
            naive_accuracy,
        )
    )
    print(
        "| {:<40} | {:<20} | {:<15.2f}% |".format(
            "Chỉ Keyword Search (Sparse)",
            f"{total_sparse_hits}/{total_gt_chunks_in_top5}",
            sparse_accuracy,
        )
    )
    print(
        "| {:<40} | {:<20} | {:<15.2f}% |".format(
            "Advanced RAG (Hybrid + Reranking)",
            f"{total_advanced_hits}/{total_gt_chunks_in_top5}",
            adv_accuracy,
        )
    )
    print("-" * 90)


if __name__ == "__main__":
    main()
