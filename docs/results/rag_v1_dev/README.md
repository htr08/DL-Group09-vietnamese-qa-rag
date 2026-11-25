# RAG v1 trên 30 câu dev (kiểm tra kỹ thuật)

Kết quả của `notebooks/07_kaggle_rag_v1_dev_run.ipynb`, chạy trên Kaggle (Tesla T4) với dataset dựng từ commit `a569936`.

| File | Nội dung |
|---|---|
| `rag_v1_dev_answers.jsonl` | `id`, `answer` của 30 câu dev; đầu vào của `scripts/grading_tool.py export` |
| `rag_v1_dev_results.json` | môi trường, cấu hình, mã băm mã nguồn và gold, độ trễ, VRAM, từng câu |
| `rag_v1_dev_items.csv` | từng câu, mở được bằng bảng tính |

Lưu ý khi dùng:

- Cấu hình truy hồi (`rrf_k = 60`, không xếp hạng lại theo tiêu đề) khớp retrieval v1 do B chốt trên `main` (`docs/results/retrieval_domain_dev.json`: k = 15 và k = 60 cho kết quả bằng nhau trên 30 câu dev, chọn k = 60; xếp hạng lại theo tiêu đề không đổi kết quả). Số đoạn đưa vào LLM (top-5) là giá trị tạm cho đến khi có ablation k ∈ {3, 5, 10}.
- Lần chạy dùng mã trước khi `main` đổi thứ tự RRF khi đồng điểm sang ưu tiên nhánh dense (PR #23). Đã kiểm bằng cách chạy lại truy hồi trên CPU với mã hiện tại của `main` (hybrid, `rrf_k = 60`, top-5, 216 chunk): mã cũ tái tạo đúng 30/30 danh sách nguồn đã lưu, còn mã mới chỉ giữ nguyên thứ tự top-5 ở 9/30 câu và giữ nguyên tập đoạn ở 15/30 câu; số câu có Khoản đúng ở hạng 1, 3 và 5 là 25, 27 và 27 trên 27 câu (lần chạy này: 24, 27 và 27). Vì vậy các câu trả lời trong thư mục này không tương ứng với truy hồi hiện hành của `main` ở khoảng hai phần ba số câu và cần chạy lại.
- Đây là dữ liệu để thử giao thức chấm ảo giác trên dev, chưa có nhãn chấm; không dùng làm kết quả đánh giá trong báo cáo.
- Mã băm trong `code_snapshot` và `gold_dev_sha256_12` tính trên file có xuống dòng CRLF (dataset được dựng bằng `git archive` trên Windows). Sau khi chuẩn hóa xuống dòng, nội dung khớp commit `a569936`.
- `auto_flag = refused` chỉ nghĩa là câu trả lời đúng bằng câu từ chối cố định; nhãn đúng/sai/ảo giác do người chấm gán.
