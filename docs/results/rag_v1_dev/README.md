# RAG v1 trên 30 câu dev (kiểm tra kỹ thuật)

Kết quả của `notebooks/07_kaggle_rag_v1_dev_run.ipynb`, chạy trên Kaggle (Tesla T4) với dataset dựng từ commit `a569936`.

| File | Nội dung |
|---|---|
| `rag_v1_dev_answers.jsonl` | `id`, `answer` của 30 câu dev; đầu vào của `scripts/grading_tool.py export` |
| `rag_v1_dev_results.json` | môi trường, cấu hình, mã băm mã nguồn và gold, độ trễ, VRAM, từng câu |
| `rag_v1_dev_items.csv` | từng câu, mở được bằng bảng tính |

Lưu ý khi dùng:

- Cấu hình truy hồi (top-5, `rrf_k = 60`, không xếp hạng lại theo tiêu đề) là mặc định chưa chốt; retrieval v1 do B chốt sau.
- Đây là dữ liệu để thử giao thức chấm ảo giác trên dev, chưa có nhãn chấm; không dùng làm kết quả đánh giá trong báo cáo.
- Mã băm trong `code_snapshot` và `gold_dev_sha256_12` tính trên file có xuống dòng CRLF (dataset được dựng bằng `git archive` trên Windows). Sau khi chuẩn hóa xuống dòng, nội dung khớp commit `a569936`.
- `auto_flag = refused` chỉ nghĩa là câu trả lời đúng bằng câu từ chối cố định; nhãn đúng/sai/ảo giác do người chấm gán.
