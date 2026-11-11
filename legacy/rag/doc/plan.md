0) Nguyên tắc & chuẩn dữ liệu (chốt ngay từ đầu)
document_id: Integer (đồng bộ Postgres), 1 tài liệu = 1 document_id.
document_key: string ổn định để đặt tên object MinIO và truy vết (gợi ý: doc_{document_id}_{sha1(source_file_bytes)[:12]}).
Mỗi chunk phải có payload tối thiểu:
document_id (int)
chunk_id (string/uuid)
headings (list/json) ví dụ: [{level:1,text:"..."},{level:2,text:"..."}]
minio_url (string)
source_filename, file_type
section_path (string) ví dụ: "H1 > H2 > H3" (để debug và hiển thị)
text (content chunk) hoặc không lưu text trong payload nếu Qdrant dùng payload text để BM25 thì vẫn cần.
Qdrant collection: 1 collection cho “marketing docs” (tên theo config), hỗ trợ dense + sparse.
1) Nâng cấp Ingestion Pipeline (Markdown-first + chunking)
1.1. Công cụ
MarkItDown (Microsoft) để chuyển đổi đa định dạng → Markdown.
LangChain splitters:
MarkdownHeaderTextSplitter cho level #, ##, ###
RecursiveCharacterTextSplitter fallback
1.2. Thông số đề xuất (chốt)
Header levels: #, ##, ###
Fallback splitter:
chunk_size = 800
chunk_overlap = 100
“Header chunk max length” (để quyết định fallback): gợi ý max_chunk_chars = 1200 (nếu chunk sau header split > 1200 chars thì tiếp tục recursive split)
Chuẩn hóa Markdown trước khi split:
Normalize line endings \n
Loại bỏ khoảng trắng dư ở heading (# Title → # Title)
Giữ nguyên bảng (tables) và code block
1.3. MinIO lưu Markdown
Bucket: processed-docs
Object key gợi ý: processed/{document_id}/{document_key}.md
Metadata object (optional nhưng nên có): processed/{document_id}/{document_key}.json chứa:
original filename, mime, created_at, sha1, page_count (nếu có), converter_version
minio_url trong payload: dạng URL nội bộ hoặc “s3 path” chuẩn hoá (ví dụ s3://processed-docs/processed/...md) để không phụ thuộc domain.
1.4. Luồng ingestion chuẩn
Nhận file (PDF/DOCX/XLSX/PPTX/TXT)
Convert → Markdown string
Upload .md lên MinIO → lấy minio_url
Split Markdown → list chunks + headings context
Tạo embedding dense (+ sparse index nếu bạn build riêng, hoặc rely Qdrant sparse)
Upsert points vào Qdrant với payload schema ở trên
2) Nâng cấp Vector DB & Retrieval (Hybrid + RRF)
2.1. Điểm cần “chốt” trước khi code (1 câu hỏi quan trọng)
Bạn ghi “Dense: text-embedding-004”. Tên này không phải model OpenAI phổ biến, và dimension sẽ quyết định cấu hình Qdrant.
Bạn muốn dense embedding chạy bằng:

Google Gemini/Vertex (thường mới có text-embedding-004), hay
OpenAI embeddings (thường là text-embedding-3-small/large)?
Nếu chưa chốt, plan vẫn triển khai được bằng cách: config hóa model name + vector_size, và khi bạn xác nhận provider thì set chính xác.

2.2. Qdrant collection config (mục tiêu)
Dense vector:
distance = Cosine
size = <theo embedding dims>
Sparse vector:
bật sparse_vectors_config (BM25 hoặc SPLADE tùy Qdrant version & pipeline)
Payload indexes (tối ưu filter):
index cho document_id (int)
index cho source_filename (keyword)
(optional) index cho section_path (keyword)
2.3. Hybrid search
Thực hiện song song:
Dense search: semantic
Sparse search: keyword/SKU/model
Fusion: RRF với k = 60
RRF score: sum(1 / (k + rank_i))
Kết quả trả về: topN fused points, kèm lý do/rank debug (optional)
2.4. Metadata schema bổ sung (bắt buộc)
document_id (Integer)
headings (JSON/List)
minio_url (String)
3) Post-processing & tối ưu trả lời
3.1. Heading similarity reranking
Tạo embedding cho:
Query embedding (dense)
Heading embedding: có 2 cách:
embed “chuỗi headings gộp” (nhanh): "H1:... | H2:... | H3:..."
embed từng heading và lấy max/avg (chính xác hơn)
Cosine similarity → heading_sim
Weighting đề xuất (đơn giản, hiệu quả):
final_score = alpha * fused_rrf_score + beta * heading_sim
gợi ý: alpha=1.0, beta=0.25 (tune sau)
Chỉ rerank trong topK sau RRF (ví dụ top 30) để tiết kiệm.
3.2. Dynamic prompting từ PostgreSQL
DB table gợi ý: system_prompts
id, name, content, is_active, updated_at
Generator nhận system_prompt động:
Nếu có prompt active theo “intent”/“tenant” → dùng
Nếu không → fallback prompt file hiện tại (prompts.py)
Cần cache:
TTL cache 30–120s hoặc cache theo updated_at để giảm query DB
3.3. Quản lý vòng đời dữ liệu
Function: delete_points_by_metadata(document_id)
Xóa points Qdrant theo filter payload document_id
(optional) xóa object .md trong MinIO theo prefix processed/{document_id}/
Nên có “idempotent delete”: gọi nhiều lần không lỗi.

Dưới đây là bản plan triển khai tiếp theo theo sprint/milestone (giả định bạn giữ nguyên embedding hiện tại, chỉ nâng ingestion/retrieval/post-process/lifecycle đúng như kế hoạch).

Sprint 0 — Chốt chuẩn & wiring (nền tảng)
Mục tiêu: mọi module mới “cắm” được vào MarketingRAGEngine mà không phá API hiện có.

Chuẩn payload Qdrant (bắt buộc)
document_id:int, chunk_id:str, headings:list, section_path:str, minio_url:str, source_filename:str, file_type:str, text:str
Chốt config RAGSettings
Chunking: header # ## ###, fallback chunk_size=800, overlap=100, max_chunk_chars=1200
Retrieval: rrf_k=60, dense_top_k=50, sparse_top_k=50, final_top_k=10
Rerank: rerank_top_k=30, alpha=1.0, beta=0.25
MinIO: bucket processed-docs, object prefix processed/{document_id}/...
File dự kiến chỉnh/sửa
Sửa: backend/config.py, backend/pipeline.py
Tạo: backend/retrieval/rrf.py (pure function), backend/rag_types.py (optional: dataclasses/typing cho chunk/payload)
Done khi: process_document() và ask_bot() vẫn chạy (chưa cần hybrid), nhưng đã có config & schema thống nhất để các sprint sau plug-in.

Sprint 1 — Ingestion “Markdown-first” + MinIO lưu .md
Mục tiêu: mọi file đầu vào → Markdown chuẩn → upload MinIO → chunk theo header + fallback.

Tích hợp MarkItDown
Hỗ trợ: PDF, DOCX, XLSX, PPTX (và TXT nếu cần)
Wrapper converter trả: markdown_text, stats (page_count nếu có, warnings)
Chunking theo MarkdownHeaderTextSplitter
Lấy context headings (headings + section_path) cho từng chunk
Nếu chunk > max_chunk_chars → fallback Recursive splitter
MinIO upload
Bucket: processed-docs
Key: processed/{document_id}/{document_key}.md
minio_url: chuẩn hóa dạng s3://processed-docs/... (không phụ thuộc domain)
File dự kiến
Tạo: backend/ingestion/markdown_converter.py, backend/ingestion/chunking.py, backend/storage/minio_client.py
Sửa: backend/ingestion/loader.py (đổi từ “load text” → “convert markdown”), backend/pipeline.py (gọi converter → upload → split)
Done khi:

Upload xong có object .md trong MinIO (bucket processed-docs)
Chunk tạo ra có headings, section_path, minio_url đầy đủ.
Sprint 2 — Qdrant schema + upsert theo payload mới
Mục tiêu: lưu vector + metadata đúng schema để chuẩn bị hybrid.

Đảm bảo collection & indexes
Dense vector giữ nguyên theo embedding bạn đang dùng (size/distance như hiện tại)
Tạo payload index cho document_id (int) và các field filter hay dùng
Upsert points
Mỗi chunk → 1 point
point.id gợi ý: uuid hoặc {document_id}_{chunk_ordinal}
File dự kiến
Tạo: backend/retrieval/qdrant_collections.py
Sửa: module upsert hiện có (thường nằm ở ingestion/indexing hoặc trong pipeline.py)
Done khi: truy vấn filter document_id trả đúng các points; payload có đủ 3 trường bắt buộc: document_id/headings/minio_url.

Sprint 3 — Hybrid Retrieval + RRF (k=60)
Mục tiêu: triển khai truy vấn dense + sparse song song, fuse bằng RRF.

Hybrid search
Dense search: như hiện tại
Sparse search: bật sparse_vectors_config và query sparse (BM25/SPLADE tùy bạn cấu hình Qdrant)
Fusion
Implement rrf_fuse(dense_results, sparse_results, k=60) (deterministic)
Giữ debug fields (optional): dense_rank, sparse_rank, rrf_score
File dự kiến
Tạo: backend/retrieval/hybrid_retriever.py, backend/retrieval/rrf.py
Sửa: backend/pipeline.py (ask_bot dùng hybrid retriever)
Done khi: hỏi các query chứa SKU/model cho kết quả “trúng keyword” tốt hơn so với dense-only, nhưng query ngữ nghĩa vẫn ổn.

Sprint 4 — Heading Similarity Reranking (post-process)
Mục tiêu: giảm nhiễu và ưu tiên chunk có heading khớp câu hỏi.

Rerank topK sau RRF
Lấy top rerank_top_k (ví dụ 30) → tính heading_sim → sắp xếp lại
final_score = alpha*rrf_score + beta*heading_sim
File dự kiến
Tạo: backend/postprocessing/heading_reranker.py
Sửa: backend/pipeline.py (ask_bot: retrieve → rerank → build context)
Done khi: query hỏi đúng 1 mục/section cụ thể (theo tiêu đề) ưu tiên chunk trong section đó.

Sprint 5 — Dynamic Prompting từ PostgreSQL
Mục tiêu: generator nhận system prompt động thay vì cố định ở prompts.py.

Thiết kế table (nếu backend đã có DB layer thì chỉ cần query)
system_prompts(id, name, content, is_active, updated_at, intent(optional))
Load prompt + cache
TTL 30–120s hoặc cache theo updated_at
Fallback
Nếu DB lỗi/không có active prompt → dùng prompts.py hiện tại
File dự kiến
Tạo: backend/storage/postgres_prompts.py
Sửa: backend/generation/generator.py
Done khi: đổi prompt trong DB có hiệu lực (sau TTL) mà không restart service.

Sprint 6 — Data lifecycle: delete_points_by_metadata(document_id)
Mục tiêu: hỗ trợ admin xóa tài liệu sạch cả vector lẫn object storage.

Qdrant delete by filter
Filter payload document_id == <id>
MinIO delete theo prefix (optional nhưng nên)
Xóa processed/{document_id}/...
Idempotent
Không lỗi nếu không còn points/objects
File dự kiến
Tạo: backend/rag_lifecycle.py
Sửa: backend/pipeline.py (expose method hoặc route layer gọi vào)