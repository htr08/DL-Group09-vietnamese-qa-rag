# Reuse vs. new work

Ghi nhận phần dùng lại từ hệ thống nền (AI Marketing Assistant, dự án trước đó, dùng làm điểm xuất phát theo Proposal) và phần làm mới trong dự án này. File này là nguồn cho bảng "tự cài đặt so với dùng sẵn" trong báo cáo và cho phần khai báo khi kiểm tra trùng lặp mã.

- Mã gốc của hệ thống nền được lưu trong `legacy/` chỉ để tham khảo. Mã đó không chạy được trong repository này (đã bỏ `backend.config`) và không được `import` từ `src/`.
- Một thành phần chỉ được coi là "đã dùng lại" khi mã tương ứng đã nằm trong `src/` hoặc `scripts/`. Mỗi lần chuyển mã sang, cập nhật cột *Trạng thái* và ghi tên Pull Request.
- Việc dùng công cụ AI được ghi riêng ở `docs/ai_usage.md`.

Trạng thái: **Dự kiến** (đã quyết định hướng, chưa chuyển mã), **Đã chuyển** (mã đã vào `src/`), **Không dùng**.
Bảng dưới phản ánh tình trạng ngày 2026-11-08: chưa có thành phần nào được chuyển.

## 1. Dùng lại từ hệ thống nền

| Thành phần | Nguồn trong `legacy/rag/` | Dự định trong dự án này | Người phụ trách | Trạng thái |
|---|---|---|---|---|
| Lưu và tìm vector trong Qdrant | `vectorstore/qdrant_store.py` | Tách thành module độc lập; đổi kích thước vector theo BGE-M3; bỏ phụ thuộc cấu hình của backend | B | Dự kiến |
| Tìm kiếm thưa (từ khóa) | `retrieval/query_encoder.py` (`SparseTextEncoder`), `retrieval/searcher.py` | Đối chiếu với `rank_bm25` của Proposal; B chọn một cách và ghi lý do vào bảng này | B | Dự kiến, chưa quyết định |
| Hợp nhất RRF | `retrieval/fusion.py` | Dùng làm tham chiếu; bản trong `src/` có tham số `k` (thử 15 và 60) và unit test | B | Dự kiến |
| Xếp hạng lại theo tiêu đề | `retrieval/reranker.py` | Dùng lại làm thành phần ablation "có/không xếp hạng lại" | B | Dự kiến |
| Nạp, xử lý và cắt đoạn tài liệu | `ingestion/loader.py`, `processor.py`, `splitter.py` | Giữ khung; sửa cách cắt cho văn bản quy chế (theo Điều) và loại phần mục lục; đổi thư viện đọc PDF sang `pdfplumber` | B | Dự kiến |
| Kiểu dữ liệu nguồn và câu trả lời | `models.py` (`RAGSource`, `RAGResponse`), `retrieval/source_mapper.py` | Dùng cho phần hiển thị nguồn trong demo | C | Dự kiến |
| Ghép ngữ cảnh đưa vào LLM | `retrieval/context_assembler.py` | Xem xét; có thể viết lại gọn hơn | C | Dự kiến, chưa quyết định |
| Chạy baseline B2 | `generation/`, `embeddings/gemini.py`, `pipeline.py`, `legacy/scripts/evaluate_*.py` | Chỉ để chạy B2 (Gemini, `RRF_K = 15` giữ nguyên cấu hình gốc) trên tập 100 câu | C | Dự kiến |
| Dữ liệu miền phụ | Hai PDF Samsung Galaxy S24 | Giữ cho thí nghiệm lệch ngôn ngữ (`data/domain/`) | B | Đã chuyển |

## 2. Không dùng lại

| Thành phần | Lý do |
|---|---|
| Frontend (Next.js), FastAPI, đăng nhập, PostgreSQL/Alembic, Celery/Redis, MinIO, nginx | Không thuộc yêu cầu của Đề tài 10; demo dùng Gradio |
| `retrieval/cache.py`, `retrieval/diversity.py`, `retrieval/query_understanding.py` | Mặc định không dùng, để tránh yếu tố gây nhiễu trong các ablation đã nêu trong Proposal. Có thể xem lại nếu có lý do rõ ràng |
| `pipeline.py` (`MarketingRAGEngine`) | Gắn với Gemini và luồng xử lý của backend; viết lại pipeline mới, chỉ tham khảo luồng gọi |

## 3. Làm mới trong dự án này

| Thành phần | Vị trí dự kiến | Người phụ trách |
|---|---|---|
| Tiền xử lý QA: tách từ VnCoreNLP, tokenize, sliding window, ánh xạ offset | `src/preprocessing_qa.py` | A |
| Huấn luyện PhoBERT và XLM-R cho extractive QA | `src/train_qa.py` | A |
| Chỉ số: EM/F1, Recall@k, MRR@10, bootstrap CI | `src/metrics.py` | A, B, C (bootstrap do C) |
| Truy hồi dense với BGE-M3 | `src/retrieval.py` | B |
| Sinh câu trả lời bằng Qwen2.5-7B 4-bit: prompt, ràng buộc ngữ cảnh, từ chối, kiểm tra CJK | `src/generate.py` | C |
| Chế độ trích xuất (PhoBERT đọc các đoạn truy hồi) | `src/pipeline.py` | C |
| Tập 100 câu có nhãn, hướng dẫn gán nhãn, giao thức chấm ảo giác | `data/gold/`, `docs/annotation_guide.md` | Cả nhóm |
| Script tải dữ liệu, dựng index, chạy đánh giá | `scripts/` | A, B, C |
| Demo chatbot hai chế độ (Gradio) | `app/app.py` | C |
| Notebook thử Qwen 4-bit và dữ liệu thử | `notebooks/00_kaggle_qwen_smoke_test.ipynb`, `kaggle/smoke_test_dataset/` | C |
| CI, cấu trúc repository, Project Charter | `.github/`, `docs/charter.md` | C |

## 4. Thư viện và mô hình dùng sẵn (không tự cài đặt)

| Thành phần | Nguồn |
|---|---|
| PhoBERT-base, XLM-R-base | Hugging Face (`vinai/phobert-base`, `FacebookAI/xlm-roberta-base`) |
| BGE-M3, Qwen2.5-7B-Instruct | Hugging Face (`BAAI/bge-m3`, `Qwen/Qwen2.5-7B-Instruct`) |
| VnCoreNLP (tách từ) | `py_vncorenlp` |
| Huấn luyện, lượng tử hóa, truy hồi | `transformers`, `accelerate`, `bitsandbytes`, `sentence-transformers`, `qdrant-client`, `rank-bm25` |
| Dữ liệu | UIT-ViQuAD 2.0 (Hugging Face `taidng/UIT-ViQuAD2.0`); Quy chế đào tạo đại học ĐHQGHN 2014 |

Bản đầy đủ kèm trích dẫn được hoàn thiện ở bảng "tự cài đặt so với dùng sẵn" của báo cáo cuối.

## 5. Lịch sử cập nhật

| Ngày | Nội dung |
|---|---|
| 2026-11-08 | Lập bản đầu từ cấu trúc mã trong `legacy/rag/` và kế hoạch dự án; chưa chuyển mã nào. |
