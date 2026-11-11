# Khai báo sử dụng công cụ AI

Đề bài cho phép dùng công cụ AI với điều kiện khai báo trong báo cáo mục đích sử dụng; phần vấn đáp có thể hỏi cả đoạn code do AI viết. File này là nguồn cho phần khai báo trong báo cáo cuối. Mỗi lần sử dụng được ghi tại thời điểm thực hiện.

## 1. Công cụ AI hỗ trợ phát triển

| Ngày | Người dùng | Công cụ | Mục đích | Phần bị ảnh hưởng | Cách kiểm chứng |
|---|---|---|---|---|---|
| 2026-11-11 | C | Claude Code | Rà soát kế hoạch dự án so với đề bài; chỉnh nội dung kế hoạch | `docs/project_plan.md`, `docs/ai_usage.md` | Người dùng đọc lại và đối chiếu với đề bài và `docs/charter.md` |

## 2. API và mô hình dùng trong hệ thống

| Thành phần | Mô hình / dịch vụ | Phạm vi sử dụng |
|---|---|---|
| Baseline B2 | Gemini (embedding và Gemini 2.5 Flash) qua API | Chỉ để chạy hệ thống nền trên tập 100 câu; không dùng trong hệ thống chính |
| Hệ thống chính | BGE-M3, Qwen2.5-7B-Instruct, PhoBERT, XLM-R (mã nguồn mở) | Chạy cục bộ hoặc trên Kaggle/Colab |

## 3. Quy ước ghi

- Mỗi dòng ở mục 1 ghi: ngày, thành viên, công cụ, việc được giao cho công cụ, file hoặc module bị ảnh hưởng, cách người dùng kiểm chứng kết quả.
- Đoạn code do AI viết đều được thành viên phụ trách đọc, hiểu và chịu trách nhiệm giải thích khi vấn đáp.
- Mỗi lần gọi API Gemini cho B2 được ghi lại cùng cấu hình (mô hình, `RRF_K`, ngày chạy).
