# RAG v1 trên 30 câu dev (kiểm tra kỹ thuật)

Kết quả của `notebooks/07_kaggle_rag_v1_dev_run.ipynb`, chạy trên Kaggle (Tesla T4) với dataset dựng từ commit `323599d`, tức truy hồi hiện hành của `main` (RRF xếp nhánh dense trước khi đồng điểm, PR #23).

| File | Nội dung |
|---|---|
| `rag_v1_dev_answers.jsonl` | `id`, `answer` của 30 câu dev; đầu vào của `scripts/grading_tool.py export` |
| `rag_v1_dev_results.json` | môi trường, cấu hình, mã băm mã nguồn và gold, độ trễ, VRAM, từng câu |
| `rag_v1_dev_items.csv` | từng câu, mở được bằng bảng tính |

Tổng hợp (30 câu: 27 trong phạm vi, 3 ngoài phạm vi):

| Chỉ số | Giá trị |
|---|---|
| Khoản đúng nằm trong top-1 / top-3 / top-5 truy hồi (27 câu trong phạm vi) | 25 / 27 / 27 |
| Câu trong phạm vi bị từ chối | 1 (A20) |
| Câu ngoài phạm vi bị từ chối | 3/3 |
| Câu lẫn ký tự CJK, câu phải sinh lại, câu trích dẫn sai | 0, 0, 0 |
| Độ trễ trung vị: truy hồi / sinh (tối đa khi sinh) | 0,022 s / 2,34 s (4,32 s) |
| VRAM đỉnh (Qwen 4-bit cùng BGE-M3 trên GPU); thời gian nạp Qwen | 8,36 GB; 109,4 s |

Lưu ý khi dùng:

- Cấu hình truy hồi (`rrf_k = 60`, không xếp hạng lại theo tiêu đề) khớp retrieval v1 do B chốt trên `main` (`docs/results/retrieval_domain_dev.json`: k = 15 và k = 60 cho kết quả bằng nhau trên 30 câu dev, chọn k = 60; xếp hạng lại theo tiêu đề không đổi kết quả). Số đoạn đưa vào LLM (top-5) là giá trị tạm cho đến khi có ablation k ∈ {3, 5, 10}.
- Lần chạy này thay lần chạy trước (commit `a569936`, RRF xếp BM25 trước khi đồng điểm). Chạy lại truy hồi trên CPU cho thấy đổi tie-break làm top-5 đổi thứ tự ở 21/30 câu và đổi cả tập đoạn ở 15/30 câu; Khoản đúng ở hạng 1 tăng từ 24 lên 25 trên 27 câu. Câu trả lời đổi ở 8/30 câu (cách diễn đạt và số nguồn `[n]`; riêng A05 thêm chi tiết các tháng họp); nhãn từ chối, số câu lẫn CJK và độ phủ top-3, top-5 không đổi.
- Đây là dữ liệu để thử giao thức chấm ảo giác trên dev, chưa có nhãn chấm; không dùng làm kết quả đánh giá cuối trong báo cáo. Các câu dev đã được dùng để thiết kế.
- Mã băm trong `code_snapshot` và `gold_dev_sha256_12` tính trên file có xuống dòng CRLF (dataset được dựng bằng `git archive` trên Windows). Sau khi chuẩn hóa xuống dòng, nội dung khớp commit `323599d`.
- `auto_flag = refused` chỉ nghĩa là câu trả lời đúng bằng câu từ chối cố định; nhãn đúng/sai/ảo giác do người chấm gán.
