# 🚀 CellphoneS Advanced RAG Engine - Technical Documentation

## 1. Tổng quan Dự án
Module RAG tại `backend/app/rag` là "bộ não" của hệ thống **CellphoneS AI Sales Assistant**. Thay vì sử dụng kiến trúc RAG cơ bản (Naive RAG) vốn dễ gặp lỗi khi xử lý dữ liệu bán lẻ phức tạp, chúng tôi đã triển khai hệ thống **Advanced RAG** dựa trên cấu trúc dữ liệu Markdown và tìm kiếm lai (Hybrid Search).

## 2. So sánh Kiến trúc: Cũ (Naive) vs. Mới (Advanced)
Việc nâng cấp này giải quyết triệt để các vấn đề về mất ngữ cảnh và sai lệch thông số kỹ thuật.

| Đặc điểm | Kiến trúc Cơ bản (Naive) | Kiến trúc Nâng cấp (Advanced) |
| :--- | :--- | :--- |
| **Xử lý file** | Chuyển PDF/TXT thành văn bản thô (Raw Text). | **Markdown-First**: Sử dụng Microsoft MarkItDown bảo toàn cấu trúc bảng & tiêu đề. |
| **Chia nhỏ dữ liệu** | Cắt theo số lượng ký tự (Fixed-size chunking). | **Advanced Chunking**: Cắt theo tiêu đề (#, ##), fallback đệ quy và tạo *tiêu đề giả lập* cho tài liệu phẳng. |
| **Cơ chế tìm kiếm** | Chỉ dùng Vector (Semantic Search). | **Hybrid Search**: Kết hợp Vector (ý định) và Keyword (SKU, tên máy chính xác). |
| **Độ chính xác** | Dễ nhầm lẫn giữa các model điện thoại cùng dòng. | **Reranking**: Sắp xếp lại kết quả dựa trên độ tương đồng của tiêu đề (Heading). |

## 3. Các Thành phần Kỹ thuật Cốt lõi

### 3.1. Chuẩn hóa Markdown & Advanced Chunking
* **Markdown-First**: Sử dụng `MarkItDown` để chuyển đổi đa định dạng (PDF, DOCX, XLSX) về Markdown. Giúp LLM đọc hiểu bảng thông số (Specs) chính xác hơn 40%.
* **Pseudo-Heading Generation**: Thuật toán chia nhỏ không chỉ dùng `MarkdownHeaderTextSplitter` và `RecursiveCharacterTextSplitter` mà còn tự động nhận diện tài liệu "phẳng" (không có thẻ Heading chuẩn) bằng cách quét các dòng viết hoa, in đậm, và so sánh độ dài để tự động gán *Tiêu đề giả lập*.

### 3.2. Chiến lược Lưu trữ Kép (Dual-Storage)
1. **MinIO (Object Storage)**: Lưu trữ nguyên bản các file gốc (PDF, DOCX, HTML, v.v.). Đây là "nguồn sự thật" để truy xuất lại nội dung gốc hoặc phục vụ việc xử lý lại (re-ingest) khi cần thiết.
2. **Qdrant (Vector Database)**: Lưu trữ Dense/Sparse Vector cùng với **Rich Payload** (Metadata: `document_id`, `chunk_id`, `headings`, `minio_url`). ID của chunk được băm (UUID5) đảm bảo tính duy nhất.

### 3.3. Xử lý Truy vấn & Hybrid Search
* **Query Understanding**: Sử dụng Regex tự động lọc nhiễu và bắt chính xác các từ khóa SKU/Model chứa chữ số (vd: `15`, `s24`, `256gb`).
* **Hybrid Search**: Thực hiện tìm kiếm song song Dense Vector (ý định chung) và Sparse Vector (từ khóa kỹ thuật).
* **Fusion & Reranking**: Kết hợp kết quả bằng thuật toán **RRF (Reciprocal Rank Fusion)** với `k=60`, sau đó sắp xếp lại (Reranking) dựa trên độ tương đồng tiêu đề (Heading Similarity).

## 4. Luồng Dữ liệu Hệ thống Moi vs Hệ Thống Cũ
![alt text](<Screenshot 2026-05-07 001638.png>)
