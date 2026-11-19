# Reuse vs. new work

Ghi nhận phần dùng lại từ hệ thống nền (AI Marketing Assistant, dự án trước đó, dùng làm điểm xuất phát theo Proposal) và phần làm mới trong dự án này. File này là nguồn cho bảng "tự cài đặt so với dùng sẵn" trong báo cáo và cho phần khai báo khi kiểm tra trùng lặp mã.

- Mã gốc của hệ thống nền nằm ở repo https://github.com/htr08/2526II-KNHTTTNT-Project-AIT3004-4-10 và không được chép vào repository này, để công cụ kiểm tra trùng lặp (MOSS/JPlag) không khớp mã gốc với repo của nhóm. Bản sao tạm trước đây (thư mục `legacy/`, nhập ở commit `ec633c1`, đã gỡ) vẫn xem được qua lịch sử git, ví dụ `git show ec633c1:legacy/rag/retrieval/fusion.py`. Cột "Nguồn" bên dưới ghi đường dẫn tương đối theo bản sao đó. Commit cố định của repo gốc dùng để dẫn link: `54e912bb6793d034e77ce99494fa7f6d0b5c6bd7`.
- Một thành phần chỉ được coi là "đã dùng lại" khi mã tương ứng đã nằm trong `src/` hoặc `scripts/`. Mỗi lần chuyển mã sang, cập nhật cột *Trạng thái* và ghi tên Pull Request.
- Việc dùng công cụ AI được ghi riêng ở `docs/ai_usage.md`.

Trạng thái: **Dự kiến** (đã quyết định hướng, chưa chuyển mã), **Đã chuyển** (mã đã vào `src/`), **Không dùng**.
Bảng dưới phản ánh tình trạng ngày 2026-11-08: chưa có thành phần nào được chuyển.

## 1. Dùng lại từ hệ thống nền

| Thành phần | Nguồn (đường dẫn trong `legacy/rag/` ở commit `ec633c1`) | Dự định trong dự án này | Người phụ trách | Trạng thái |
|---|---|---|---|---|
| Lưu và tìm vector trong Qdrant | `vectorstore/qdrant_store.py` | Đã cấu hình sync vector vào Qdrant trong `build_index.py`; trong giai đoạn đánh giá (eval), pipeline dùng `numpy` brute-force để đảm bảo tính tái lập 100% độc lập với daemon nền. Qdrant dùng cho demo/deploy (PR #11). | B | Đã chuyển (dùng cho demo; eval dùng numpy) |
| Tìm kiếm thưa (từ khóa) | `retrieval/query_encoder.py` (`SparseTextEncoder`), `retrieval/searcher.py` | Dùng `rank_bm25` (BM25Okapi) theo Proposal và tách từ regex `\w+` trong `src/retrieval.py` | B | Đã chuyển (PR #11) |
| Hợp nhất RRF | `retrieval/fusion.py` | Cài đặt trong `src/retrieval.py` với tham số `rrf_k` linh hoạt ($k \in \{15, 60\}$), có unit test kiểm tra công thức | B | Đã chuyển (PR #11) |
| Xếp hạng lại theo tiêu đề | `retrieval/reranker.py` | Dùng làm thành phần ablation trong `src/retrieval.py`, nhưng là biến thể đơn giản hơn bản gốc. Bản gốc tính `alpha * rrf * (1 + beta * heading_score)`, với `heading_score` kết hợp độ tương đồng embedding Gemini và độ trùng từ. Bản mới cộng `boost` cho mỗi từ trùng giữa câu hỏi và tiêu đề (loại stopword tiếng Việt), không có thành phần embedding. `boost = 0,0003` là giá trị cố định, chưa được tune trên dev, nên kết quả ablation chỉ phản ánh biến thể này. | B | Đã chuyển, có thay đổi công thức (PR #11) |
| Nạp, xử lý và cắt đoạn tài liệu | `ingestion/loader.py`, `processor.py`, `splitter.py` | Cắt theo Điều → Khoản → Điểm, gán ID chính xác bằng regex, hỗ trợ sliding window giữ ngắt dòng; đọc PDF bằng `pdfplumber` | B | Đã chuyển (PR #11) |
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
| 2026-11-12 | Quyết định thay `legacy/` bằng link tới repo gốc; cắt kho Quy chế theo Điều → Khoản. |
