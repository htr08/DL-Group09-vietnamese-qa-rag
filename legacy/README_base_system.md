<br>
<div align="center">


<img src="https://readme-typing-svg.herokuapp.com?font=Orbitron&weight=700&size=53&duration=3000&pause=1000&color=4DA6FF&center=true&vCenter=true&width=900&lines=AI+Marketing+Assistant;Next-Gen+Marketing+System"/>

<br>
<br>


![Python](https://img.shields.io/badge/Python_3.13-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js_14-000000?style=flat-square&logo=next.js&logoColor=white)

![Qdrant](https://img.shields.io/badge/Qdrant-DC244C?style=flat-square&logo=qdrant&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini_AI-8E75B2?style=flat-square&logo=google&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL_15-4169E1?style=flat-square&logo=postgresql&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-DC382D?style=flat-square&logo=redis&logoColor=white)

</div>

<br>
<br>

<h1 align="center">✨𝗔𝗜 𝗠𝗮𝗿𝗸𝗲𝘁𝗶𝗻𝗴 𝗔𝘀𝘀𝗶𝘀𝘁𝗮𝗻𝘁✨</h1>

<p align="center">
  <i>teaching silicon how to sell.</i>
</p>

<p align="center">
  <img src="https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExcGJ1YW1mZW1rNzVqMWluaW9xaGQ5bTM2dDRtZ2t3anN6aHIxeGhvZCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/vVegyymxA90fkY8jkE/giphy.gif" width="50"/>
</p>

<p align="center">Hệ thống Trợ lý Marketing nội bộ dựa trên kiến trúc <b>Advanced RAG</b>, cho phép nhân viên đặt câu hỏi và nhận tư vấn theo thời gian thực từ kho tài liệu nội bộ của doanh nghiệp.</p>




## 📚 Mục lục

- [📚 Mục lục](#-mục-lục)
- [👥 Vai trò \& Quyền hạn](#-vai-trò--quyền-hạn)
  - [🛠️ Admin](#️-admin)
  - [💬 User](#-user)
- [✨ Tính năng nổi bật](#-tính-năng-nổi-bật)
- [🏗️ Kiến trúc tổng quan](#️-kiến-trúc-tổng-quan)
  - [Luồng Ingestion (Upload → Vectorize)](#luồng-ingestion-upload--vectorize)
  - [Luồng Chat (Query → SSE Streaming)](#luồng-chat-query--sse-streaming)
- [🗂️ Tech Stack](#️-tech-stack)
- [📁 Cấu trúc thư mục](#-cấu-trúc-thư-mục)
- [🚀 Cài đặt \& Khởi chạy](#-cài-đặt--khởi-chạy)
  - [Yêu cầu](#yêu-cầu)
  - [1. Clone \& cấu hình môi trường](#1-clone--cấu-hình-môi-trường)
  - [2. Khởi chạy toàn bộ stack](#2-khởi-chạy-toàn-bộ-stack)
  - [3. Kiểm tra hệ thống](#3-kiểm-tra-hệ-thống)
  - [4. Tài khoản mặc định](#4-tài-khoản-mặc-định)
  - [5. Dừng & dọn dẹp](#5-dừng--dọn-dẹp)
  - [6. Migration thủ công (dành cho developer)](#6-migration-thủ-công-dành-cho-developer)
- [🌐 Demo \& Truy cập](#-demo--truy-cập)
  - [Truy cập local (sau khi khởi chạy Docker)](#truy-cập-local-sau-khi-khởi-chạy-docker)
  - [Demo online](#demo-online)
- [🌍 Triển khai lên internet với ngrok](#-triển-khai-lên-internet-với-ngrok)
  - [Luồng hoạt động](#luồng-hoạt-động)
  - [1. Lấy Auth Token & Static Domain](#1-lấy-auth-token--static-domain)
  - [2. Cấu hình trong docker-compose.yml](#2-cấu-hình-trong-docker-composeyml)
  - [3. Khởi chạy & Cập nhật](#3-khởi-chạy--cập-nhật)
  - [4. Truy cập demo online](#4-truy-cập-demo-online)
  - [5. Giới hạn bản Free](#5-giới-hạn-bản-free-free-tier)
  - [6. Cách tắt ngrok khi không dùng](#6-cách-tắt-ngrok-khi-không-dùng)
- [📡 API Reference](#-api-reference)
- [✨ Giao Diện Hệ Thống](#-giao-diện-hệ-thống)
- [🔐 Authentication Interface](#-authentication-interface)
- [👤 User Dashboard](#-user-dashboard)
- [🛠️ Admin Dashboard](#️-admin-dashboard)
- [🔒 Giới hạn \& Bảo vệ hệ thống](#-giới-hạn--bảo-vệ-hệ-thống)
- [📊 Logging \& Xử lý lỗi](#-logging--xử-lý-lỗi)
  - [Format Log](#format-log)
  - [Phân loại lỗi AI](#phân-loại-lỗi-ai)
- [🧪 Chạy Test](#-chạy-test)
- [🛠️ Dịch vụ quản trị](#️-dịch-vụ-quản-trị)
- [📐 So sánh kiến trúc RAG](#-so-sánh-kiến-trúc-rag)
- [Đọc thêm](#đọc-thêm)
- [👨‍💻 Thành viên nhóm 9](#-thành-viên-nhóm-9)

---

## 👥 Vai trò & Quyền hạn 
 
Hệ thống phân quyền theo 2 role: **Admin** và **User**.
 
### 🛠️ Admin
| Chức năng | Mô tả |
|---|---|
| Quản lý tài liệu | Upload / xóa file (PDF, DOCX, MD, TXT, HTML) vào kho tri thức |
| Theo dõi ingestion | Xem trạng thái xử lý tài liệu (pending / processing / completed / failed) |
| Quản lý người dùng | Thêm / xóa tài khoản user trong hệ thống |
| Cấu hình hệ thống | Chỉnh system prompt, giới hạn kích thước file, số tài liệu tối đa |
 
### 💬 User
| Chức năng | Mô tả |
|---|---|
| Đặt câu hỏi | Gửi câu hỏi về sản phẩm, chính sách, thông tin nội bộ, hỗ trợ viết nội dung marketing trên các nền tảng |
| Nhận trả lời realtime | Câu trả lời được stream trực tiếp qua SSE (không cần chờ load) |
| Xem nguồn trích dẫn | Mỗi câu trả lời kèm danh sách tài liệu nguồn liên quan |
| Lịch sử chat | Xem lại các cuộc hội thoại theo session |

---

## ✨ Tính năng nổi bật
 
- **Markdown-First Ingestion** — Sử dụng `Microsoft MarkItDown` để bảo toàn cấu trúc bảng & tiêu đề khi xử lý PDF/DOCX/MD/TXT/HTML, cải thiện độ chính xác đọc thông số kỹ thuật lên ~40% so với text thuần.
- **Advanced Hybrid Search** — Kết hợp `Dense Vector` (hiểu ý định) và `Sparse Vector / BM25` (bắt chính xác SKU/tên sản phẩm), hợp nhất kết quả qua **RRF Fusion (k=60)**.
- **Advanced Chunking** — Chia theo tiêu đề Markdown (H1/H2/H3), fallback đệ quy `RecursiveCharacterTextSplitter`, và tự động sinh **Pseudo-Heading** cho tài liệu phẳng (không có heading chuẩn) bằng cách nhận diện dòng viết hoa / in đậm.
- **Query Understanding** — Xử lý stop words, tạo query variants, Regex tự động extract SKU/model keyword chứa chữ số (vd: `15 Pro`, `s24`, `256gb`).
- **Diversity Selection** — Lọc redundancy bằng thuật toán **MMR-like** sau reranking, đảm bảo các chunk trả về đa dạng và không trùng lặp.
- **Heading Reranking** — Sắp xếp lại kết quả tìm kiếm dựa trên độ tương đồng tiêu đề, giảm nhầm lẫn giữa các model cùng dòng sản phẩm.
- **SSE Streaming** — Phản hồi AI được stream trực tiếp về client qua `Server-Sent Events`, mang lại trải nghiệm realtime mượt mà.
- **JWT Auth & RBAC** — Phân quyền rõ ràng giữa Admin (quản lý tài liệu, người dùng) và User (chat).

---




## 🏗️ Kiến trúc tổng quan
 
<div align="center">
  <img 
    src="./pics/dc67d8a1-140b-424a-a228-f755f3964d1a.jpg" 
    alt="Admin Interface"
    width="70%" 
  />
  
  <br/>

</div>

```
  Chú thích:
  • Truy cập local  : http://localhost hoặc http://127.0.0.1 → Nginx :80
  • Truy cập online : ngrok tunnel (HTTPS) → forward về Nginx :80 (local)
    ⚠ Yêu cầu Docker stack đang chạy; ngrok free có thể bị giới hạn kết nối.

```

### Luồng Ingestion (Upload → Vectorize)

```
Admin upload file (PDF / DOCX / PPTX / XLSX / MD / TXT / HTML)
    ↓
POST /api/admin/upload
    Lưu file gốc → MinIO
    Tạo record → PostgreSQL (status: pending)
    Đẩy task → Celery Queue (Redis)
    ↓
Celery Worker
    Tải file từ MinIO
    [Loader]    MarkItDown → PyMuPDF → pdfplumber (fallback chain → Markdown)
    [Splitter]  Split theo Markdown headers (H1/H2/H3) → Recursive split
                → Pseudo-Heading tự động cho tài liệu phẳng
    [Embedding] Gemini embedding-001 (3072 dims) → Dense vector + Sparse BM25 vector
    [Qdrant]    Upsert dense + sparse vector + metadata payload
                (doc_id, chunk_id, headings, minio_url)
    Cập nhật status → PostgreSQL (completed)
```

### Luồng Chat (Query → SSE Streaming)

```
User gửi câu hỏi
    ↓
POST /api/chat  ← JWT Auth
    Validate / tạo session, tải chat history từ PostgreSQL
    ↓
Advanced RAG Engine
    [Query Understanding]    Xử lý stop words, tạo query variants
                             Regex extract SKU/model keyword (vd: 15 Pro, s24, 256gb)
    [Query Encoding]         Gemini embedding-001 (Dense) + BM25 (Sparse) song song
    [Hybrid Search]          Tìm kiếm Dense + Sparse song song trên Qdrant
    [Fusion - RRF]           Reciprocal Rank Fusion (k=60) kết hợp 2 ranking
    [Reranking]              Heading-awareness reranking → ưu tiên chunk đúng ngữ cảnh
    [Diversity Selection]    MMR-like filtering → loại bỏ chunk trùng lặp / redundant
    [Generation]             Dựng prompt: context + chat history
                             → Gemini 2.5 Flash (streaming mode)
    ↓
SSE Response → Client
    yield {"chunk": "..."}  × N
    yield {"DONE": true, "sources": [...]}
    Lưu messages → PostgreSQL (async)
```

---

## 🗂️ Tech Stack

| Layer | Công nghệ | Mục đích |
|---|---|---|
| **Frontend** | Next.js 14 + TypeScript | UI, routing |
| **Styling** | Tailwind CSS | Component styling |
| **Streaming** | @microsoft/fetch-event-source | SSE client |
| **Markdown render** | react-markdown + remark-gfm | Hiển thị response |
| **Web Framework** | FastAPI + Uvicorn | REST API, SSE |
| **AI / LLM** | Google Gemini | Embedding + Generation |
| **RAG Pipeline** | LangChain | Orchestration, chat history |
| **Vector DB** | Qdrant | Dense + Sparse vector search |
| **Object Storage** | MinIO | Lưu file .md gốc |
| **Relational DB** | PostgreSQL 15 | Metadata, users, sessions |
| **Task Queue** | Celery + Redis | Xử lý ingestion nền |
| **Auth** | JWT (PyJWT + bcrypt) | Xác thực & phân quyền |
| **Container** | Docker + Docker Compose | Triển khai toàn stack |
| **Reverse Proxy** | Nginx | Route traffic, prefix /api |
| **Doc parsing** | Microsoft MarkItDown | Markdown-first conversion |
| **Tunnel / Expose** | ngrok | Public HTTPS access (demo online) |

---

## 📁 Cấu trúc thư mục

```
.
├── frontend/                       # Next.js 14 application
│   ├── app/                        # App Router pages
│   │   ├── admin/                  # Trang quản trị (Admin dashboard)
│   │   ├── chat/                   # Trang chat với AI
│   │   ├── login/                  # Trang đăng nhập
│   │   ├── globals.css             # Global styles
│   │   ├── layout.tsx              # Root layout
│   │   └── page.tsx                # Trang chủ (redirect)
│   ├── lib/
│   │   └── chatService.ts          # Service gọi API chat & SSE handling
│   ├── public/                     # Static assets
│   ├── .env.example                # Template biến môi trường frontend
│   ├── .gitignore
│   ├── BACKEND_INTEGRATION_GUID... # Hướng dẫn tích hợp backend
│   ├── Dockerfile                  # Docker build cho frontend
│   ├── HUONG_DAN_CAI_DAT.md        # Hướng dẫn cài đặt chi tiết
│   ├── next.config.js              # Next.js configuration
│   ├── package-lock.json
│   ├── package.json
│   ├── postcss.config.js
│   ├── tailwind.config.js          # Tailwind CSS configuration
│   └── tsconfig.json               # TypeScript configuration
│
├── backend/
│   ├── main.py                     # FastAPI app, tất cả route definitions
│   ├── config.py                   # Pydantic Settings (đọc từ .env)
│   ├── database.py                 # SQLAlchemy engine & session
│   ├── models.py                   # ORM models (User, Document, Chat...)
│   ├── schemas.py                  # Pydantic request/response schemas
│   ├── auth.py                     # JWT tạo/xác thực token, bcrypt
│   ├── storage.py                  # MinIO client wrapper
│   ├── worker.py                   # Celery app instance
│   ├── tasks.py                    # Celery task: process_document_task
│   ├── ai_service.py               # Facade gọi RAG engine
│   ├── seed.py                     # Seed dữ liệu mặc định
│   ├── logging_config.py           # Cấu hình logging toàn cục
│   ├── services/
│   │   └── chat_service.py         # Session management, SSE generator
│   └── rag/
│       ├── pipeline.py             # MarketingRAGEngine (orchestrator)
│       ├── config.py               # RAG-specific settings
│       ├── models.py               # RAG data models
│       ├── prompts.py              # Prompt templates
│       ├── ingestion/
│       │   ├── loader.py           # MarkItDown: đọc & parse → Markdown
│       │   ├── splitter.py         # Structural splitter theo heading
│       │   └── processor.py        # Orchestrate ingestion flow
│       ├── embeddings/
│       │   └── gemini.py           # Gemini dense + sparse embedding
│       ├── retrieval/
│       │   └── retriever.py        # Hybrid Search + RRF Fusion + Rerank
│       ├── generation/
│       │   ├── generator.py        # GeminiSalesGenerator (stream)
│       │   └── prompts.py          # System/user prompt builders
│       ├── vectorstore/
│       │   └── qdrant_store.py     # Qdrant collection operations
│       └── tests/                  # Unit & smoke tests
│
├── alembic/                        # DB migration scripts
├── nginx/
│   └── nginx.conf                  # Reverse proxy config
├── docker-compose.yml
├── alembic.ini
└── .env                            # Biến môi trường (không commit)
```

---

## 🚀 Cài đặt & Khởi chạy

### Yêu cầu

| Công cụ | Phiên bản tối thiểu |
|---|---|
| Docker | 24+ |
| Docker Compose | v2.20+ |
| Python | 3.10+ (nếu chạy local) |
| Node.js | 18+ (nếu chạy frontend local) |
| Google Gemini API Key | — |

### 1. Clone & cấu hình môi trường

```bash
git clone https://github.com/htr08/2526II-KNHTTTNT-Project-AIT3004-4-10
cd 2526II-KNHTTTNT-Project-AIT3004-4-10

# Tạo file .env từ template
cp .env.example .env
```

Chỉnh sửa các biến bắt buộc trong `.env` (copy từ `.env.example`):

```env
# ==============================================================================
# GENERAL CONFIG
# ==============================================================================
PROJECT_NAME=Marketing_RAG
 
# ==============================================================================
# AI & LLM
# ==============================================================================
GEMINI_API_KEY=your_gemini_api_key_here   # ← Bắt buộc: lấy tại https://aistudio.google.com
EMBEDDING_MODEL=models/gemini-embedding-001
LLM_MODEL=models/gemini-2.5-flash
 
# ==============================================================================
# DATABASE & STORAGE
# ==============================================================================
# PostgreSQL
POSTGRES_USER=admin
POSTGRES_PASSWORD=secret_pass
POSTGRES_DB=marketing_db
POSTGRES_PORT=5432
 
# Khi chạy local (không Docker): để localhost
# Khi chạy Docker: docker-compose.yml tự ghi đè, không cần sửa
POSTGRES_HOST=localhost
DATABASE_URL=postgresql://admin:secret_pass@localhost:5432/marketing_db
 
# MinIO (Object Storage)
MINIO_ROOT_USER=admin
MINIO_ROOT_PASSWORD=minio_secret_pass
MINIO_BUCKET_NAME=marketing-docs
 
# Tương tự: để localhost cho local dev, Docker tự ghi đè khi chạy container
MINIO_ENDPOINT=localhost:9000
 
# Qdrant (Vector DB)
QDRANT_URL=http://localhost:6333
 
# Redis
REDIS_URL=redis://localhost:6379/0
 
# ==============================================================================
# FRONTEND
# ==============================================================================
# Khi chạy local: localhost:8000 (trực tiếp vào backend)
# Khi chạy Docker: đổi thành http://localhost/api (qua Nginx)
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_MINIO_URL=http://localhost:9000
 
# ==============================================================================
# NGROK
# ==============================================================================
# Lấy token tại: https://dashboard.ngrok.com → Your Authtoken
# Không có token → URL thay đổi mỗi lần restart, static domain không hoạt động
NGROK_AUTHTOKEN=your_ngrok_authtoken_here
```


> ⚠️ **Không commit file `.env`** — đã được thêm vào `.gitignore`. Chỉ commit `.env.example`.

### 2. Khởi chạy toàn bộ stack

```bash
docker-compose up -d --build
```

> ⏱️ **Lần đầu build có thể mất 3–5 phút** do tải Docker image và cài dependency.


Lệnh này sẽ khởi động **10 container** theo thứ tự phụ thuộc:

| Container | Port(s) | Mô tả |
|---|---|---|
| `rag_postgres` | 5432 | PostgreSQL database |
| `rag_qdrant` | 6333, 6334 | Vector DB + gRPC |
| `rag_minio` | 9000, 9001 | Object storage + Console UI |
| `minio-init-1` | — | Khởi tạo bucket MinIO (chạy một lần rồi tự dừng) |
| `rag_redis` | 6379 | Message broker cho Celery |
| `worker-1` | — | Celery worker (ingestion pipeline) |
| `rag_backend_server` | 8000 (internal) | FastAPI application |
| `frontend-1` | — (internal) | Next.js 14 frontend |
| `rag_nginx` | **80** | Reverse proxy (entry point) |
| `ngrok` |   — | Tunnel HTTPS ra Internet (demo online) |


> Backend tự động chạy `alembic upgrade head` và seed dữ liệu mặc định trước khi khởi động Uvicorn.


### 3. Kiểm tra hệ thống
 
```bash
# Kiểm tra tất cả container đang chạy
docker-compose ps
 
# Kiểm tra backend
curl http://localhost/api/
# {"status": "online", "message": "Backend AI Marketing"}
 
# Kiểm tra Qdrant
curl http://localhost:6333/healthz
# {"title": "qdrant - healthy"}
 
# Xem log nếu có lỗi
docker-compose logs -f backend
docker-compose logs -f worker
```


### 4. Tài khoản mặc định

Sau khi khởi động, hệ thống tự động seed tài khoản qua `seed.py`:

| Role | Username | Password | Quyền |
|---|---|---|---|
| **Admin** | `admin` | `admin123` | Quản lý tài liệu, người dùng, cấu hình hệ thống |

> ⚠️ **Đổi mật khẩu ngay sau lần đăng nhập đầu tiên** trên môi trường production.

### 5. Dừng & dọn dẹp
 
```bash
# Dừng toàn bộ stack (giữ lại data)
docker-compose down
 
# Dừng và xóa toàn bộ volume (mất data)
docker-compose down -v
```
 
---

### 6. Migration thủ công (dành cho developer)
 
Dùng khi muốn chạy backend ngoài Docker để debug:

```bash
pip install -r backend/requirements.txt

# Áp dụng tất cả migration
alembic upgrade head

# Tạo migration mới sau khi thay đổi models.py
alembic revision --autogenerate -m "mo_ta_thay_doi"
```

> 📦 **Dependencies:** `backend/requirements.txt` (Python) và `frontend/package.json` (Node.js) đã liệt kê đầy đủ các thư viện cần thiết cho từng thành phần.

---

## 🌐 Demo & Truy cập

### Truy cập local (sau khi khởi chạy Docker)

| URL | Mô tả |
|---|---|
| http://localhost | Giao diện chính (qua Nginx) |
| http://127.0.0.1 | Giao diện chính (alias) |

### Demo online

> ⚠️ **Lưu ý:** Link demo yêu cầu Docker stack đang chạy ở máy host. Nếu server không hoạt động, trình duyệt sẽ báo lỗi kết nối — đây là giới hạn của môi trường demo miễn phí qua ngrok.

| URL | Mô tả |
|---|---|
| https://removable-overreact-leggings.ngrok-free.dev/login | Demo online (miễn phí, cần Docker đang chạy) |

## 🌍 Triển khai lên internet với ngrok

> **ngrok** tạo tunnel HTTPS công khai trỏ vào `Nginx:80` trên máy local — hữu ích để demo hoặc test trên thiết bị khác mà không cần deploy lên cloud.

---

### Luồng hoạt động

```
Internet ──(HTTPS)──> your-domain.ngrok-free.dev ──> Nginx (:80) ──> backend / frontend
```

---

### 1. Lấy Auth Token & Static Domain

1. Đăng nhập tại [https://dashboard.ngrok.com](https://dashboard.ngrok.com)
2. Vào **Your Authtoken** → Copy token → Điền vào file `.env`:

```env
NGROK_AUTHTOKEN=your_ngrok_authtoken_here   # ← Thay bằng token thật của bạn
```

3. Vào **Domains** → Bấm tạo và copy domain tĩnh của bạn (dạng `xxx-xxx-xxx.ngrok-free.dev`).

---

### 2. Cấu hình trong `docker-compose.yml`

Tìm đến dịch vụ `ngrok` và thay thế bằng đoạn cấu hình dưới đây:

```yaml
# 6. NGROK
ngrok:
  image: ngrok/ngrok:latest
  container_name: ngrok
  restart: unless-stopped
  # thay đoạn command này
  command:
    - "http"
    - "nginx:80"
    - "--url=https://your-domain.ngrok-free.dev" # ← Thay domain của bạn vào đây
  environment:
    - NGROK_AUTHTOKEN=${NGROK_AUTHTOKEN}
  depends_on:
    - nginx 
  networks:
    - rag_network
```



---

### 3. Khởi chạy & Cập nhật

Chạy lệnh này để Docker xóa container cũ và dựng lại sạch:

```bash
docker compose up -d --force-recreate ngrok
```

Kiểm tra xem tunnel đã hoạt động ổn định chưa:

```bash
docker compose logs ngrok
```

Nếu thành công sẽ thấy dòng chữ:

```
t=... lvl=info msg="started tunnel" url=https://your-domain.ngrok-free.dev
```

---

### 4. Truy cập demo online

| URL | Mô tả |
|-----|-------|
| `https://your-domain.ngrok-free.dev` | Giao diện Frontend Online |
| `https://your-domain.ngrok-free.dev/api/docs` | Tài liệu Swagger UI của Backend |

> ⚠️ **Lưu ý:** URL hoạt động khi và chỉ khi Docker stack đang chạy trên máy host. Nếu tắt máy hoặc dừng Docker, URL sẽ trả về lỗi kết nối.

---

### 5. Giới hạn bản Free (Free Tier)

- **1 Static Domain:** Chỉ được cấp 1 domain cố định miễn phí cho mỗi tài khoản.
- **Giới hạn kết nối:** Khoảng vài chục request mỗi phút, nếu dùng quá nhanh sẽ bị màn hình cảnh báo của ngrok.

> Tối ưu cho việc Dev/Test, không dùng làm trang web thật cho nhiều người dùng.

---

### 6. Cách tắt ngrok khi không dùng

Nếu bạn chỉ muốn code một mình ở localhost và không muốn treo ngrok chạy ngầm tốn tài nguyên, chỉ cần tắt riêng nó — các dịch vụ khác vẫn chạy bình thường:

```bash
docker compose stop ngrok
```
---

### 📡 API Reference

Đọc thêm ở `backend/Readme.md`

---

## ✨ Giao Diện Hệ Thống

## 🔐 Authentication Interface

<div align="center">
  <img 
    src="./pics/ac353f12-4e6e-4926-ba08-ece3823617a7.jpg" 
    alt="Login Interface"
    width="85%" 
  />
  
  <br/>
  <sub>Secure login experience for administrators and users</sub>
</div>

---

## 👤 User Dashboard

<div align="center">
  <img 
    src="./pics/32275c8d-8ae3-4e28-84d3-87086265ce2c.jpg" 
    alt="User Interface"
    width="85%" 
  />
  
  <br/>
  <sub>Interactive AI assistant dashboard for marketing workflows</sub>
</div>

---

## 🛠️ Admin Dashboard

<div align="center">
  <img 
    src="./pics/745b4cf6-33d9-4477-b392-cbb1d7450ad6.jpg" 
    alt="Admin Interface"
    width="85%" 
  />
  
  <br/>
  <sub>Management panel for monitoring and system administration</sub>
</div>

---

## 🔒 Giới hạn & Bảo vệ hệ thống

| Giới hạn | Mặc định | Cấu hình qua |
|---|---|---|
| Kích thước file tối đa | 100 MB | `PUT /api/admin/config` |
| Số tài liệu tối đa | 100 | `PUT /api/admin/config` |
| Định dạng file | PDF, DOCX, TXT, MD, HTML | Hardcoded |
| Độ dài câu hỏi | 2.000 ký tự | Hardcoded |
| JWT token TTL | 60 phút | `.env → ACCESS_TOKEN_EXPIRE_MINUTES` |

---

## 📊 Logging & Xử lý lỗi

### Format Log

```
[2026-05-14T10:30:00] [INFO]  [backend.main]         Document queued: report.pdf (id=abc-123)
[2026-05-14T10:30:05] [INFO]  [backend.tasks]        Ingestion completed abc-123: 42 chunks
[2026-05-14T10:30:10] [ERROR] [backend.rag.pipeline] Gemini timeout (DeadlineExceeded): ...
```

### Phân loại lỗi AI

| Loại lỗi | Từ khoá phát hiện | Phản hồi client |
|---|---|---|
| Rate Limit / Quota | `quota`, `429`, `resourceexhausted`, `overloaded` | "AI đang quá tải, vui lòng thử lại sau" |
| Timeout | `timeouterror`, `deadlineexceeded`, `timeout` | "AI đang quá tải, vui lòng thử lại sau" |
| Lỗi hệ thống | Tất cả exception còn lại | "Hệ thống gặp lỗi khi xử lý câu hỏi, vui lòng thử lại." |

> Stack trace đầy đủ chỉ xuất hiện phía server, client chỉ nhận thông báo an toàn.

---

## 🧪 Chạy Test

```bash
pip install -r backend/requirements.txt

# Chạy toàn bộ test suite cho RAG module
pytest backend/rag/tests/ -v
```

| File | Nội dung |
|---|---|
| `test_splitter.py` | Logic chia chunk theo heading |
| `test_retriever_contract.py` | Contract test Qdrant retriever |
| `test_vectorstore_mapping.py` | Mapping metadata khi lưu vector |
| `test_pipeline_smoke.py` | Smoke test toàn bộ RAG pipeline |

---

## 🛠️ Dịch vụ quản trị

| Dịch vụ | URL | Mô tả |
|---|---|---|
| Swagger UI | http://localhost/api/docs | Tài liệu & test API |
| MinIO Console | http://localhost:9001 | Quản lý file storage |
| Qdrant Dashboard | http://localhost:6333/dashboard | Quản lý vector collections |

---

## 📐 So sánh kiến trúc RAG

| Đặc điểm | Naive RAG (cũ) | Advanced RAG (hiện tại) |
|---|---|---|
| **Xử lý file** | Raw text | MarkItDown → PyMuPDF → pdfplumber (fallback chain) |
| **Chia chunk** | Fixed-size (số ký tự) | H1/H2/H3 header split → Recursive fallback → Pseudo-Heading |
| **Embedding** | Vector đơn giản | Gemini embedding-001, 3072 dims |
| **Xử lý query** | Không có | Stop words, query variants, Regex extract SKU/model |
| **Tìm kiếm** | Vector search đơn thuần | Hybrid: Dense (Gemini) + Sparse (BM25) song song |
| **Fusion** | Không có | Reciprocal Rank Fusion (RRF, k=60) |
| **Reranking** | Không có | Heading-awareness reranking |
| **Diversity** | Không có | MMR-like selection — loại chunk redundant |
| **Độ chính xác** | Dễ nhầm lẫn cùng dòng | Phân biệt chính xác từng SKU |

--- 

## Đọc thêm

Đọc thêm về backend ở `backend/Readme.md`

Đọc thêm về Rag ở `backend/rag/doc/readme.md`

## 👨‍💻 Thành viên nhóm 9

| Họ và tên | Vai trò |
|---|---|
| Nguyễn Huy Hoàng | Frontend |
| Nguyễn Thị Hiền Trang | Backend |
| Phạm Thế Hưng | AI Engineer |
| Trương Ái Linh | DevOps |

<br>

<p align="center">
  <img src="https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExZzI3bWJrZ3Q1ODAwZmJ4YjcwdnZjODhvOTU0YnJmbzJqYTJydm00NSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/DzS5TM0axorWWG7DvX/giphy.gif" width="50%"/>
</p>



<p align="center">
  <i>fewer tasks. smoother days.</i>
</p>