# Chế độ trích xuất trên 30 câu dev (kiểm tra kỹ thuật)

Kết quả của `notebooks/08_kaggle_extractive_dev_run.ipynb`, chạy trên Kaggle (Tesla T4) với mã tương ứng commit `942ae48` (xem lưu ý cuối) và checkpoint PhoBERT LR 5e-5, seed 12 của A (`run_manifest.json`: fingerprint `3e04aedb82c6…`).

| File | Nội dung |
|---|---|
| `extractive_dev_answers.jsonl` | `id`, `answer` của 30 câu dev; câu từ chối là câu từ chối cố định |
| `extractive_dev_results.json` | môi trường, cấu hình, mã băm mã nguồn và gold, tổng hợp, từng câu (kèm `score_diff`, đoạn nguồn, độ trễ) |
| `extractive_dev_items.csv` | từng câu, mở được bằng bảng tính |

Cấu hình: truy hồi hybrid (BM25 + BGE-M3, RRF), top-5, `rrf_k = 60`, không xếp hạng lại theo tiêu đề, 216 chunk; ngưỡng từ chối `score_diff <= 0`.

Tổng hợp (30 câu: 27 trong phạm vi, 3 ngoài phạm vi):

| Ngưỡng | EM tất cả | F1 tất cả | EM trong phạm vi | F1 trong phạm vi | Số câu trả lời |
|---|---|---|---|---|---|
| 0,0 (cấu hình chạy) | 66,67 | 82,09 | 62,96 | 80,10 | 26 |
| -1,489326 (tinh chỉnh trên ViQuAD Dev, tính lại từ `score_diff`) | 60,00 | 75,43 | 55,56 | 72,70 | 24 |

- Khoản đúng nằm trong top-1 / top-3 / top-5 truy hồi: 24 / 27 / 27 (trên 27 câu trong phạm vi).
- Đoạn PhoBERT chọn thuộc Khoản đúng: 22/27. Năm câu còn lại (A21, A23, B21, B23, B24) có Khoản đúng ở hạng 1 nhưng khoảng được chọn nằm ở đoạn khác.
- Cả 3 câu ngoài phạm vi bị từ chối (`score_diff` 19,16 / 3,66 / 7,99); câu trong phạm vi có `score_diff` cao nhất là 0,19 (C23, bị từ chối).
- Độ trễ trung vị mỗi câu: truy hồi 0,023 giây, trích xuất 0,095 giây; VRAM đỉnh 2,83 GB (chưa có Qwen).

Lưu ý khi dùng:

- Đây là 30 câu dev của bộ Gold chính thức (nhãn đã chốt, `data/gold/gold_dev.jsonl`), đã dùng để thiết kế; không đưa vào báo cáo như kết quả đánh giá cuối.
- Ngưỡng 0 không được chọn trên 30 câu này. Dòng tính lại với ngưỡng ViQuAD chỉ để tham khảo: chênh hai câu (A26 và C19 chuyển từ đúng sang bị từ chối), chưa đủ để kết luận ngưỡng ViQuAD có chuyển được sang miền Quy chế hay không. Ở đây `score_diff` là điểm null nhỏ nhất trên 5 đoạn trừ điểm khoảng tốt nhất trên 5 đoạn, khác với phân bố trên từng đoạn đơn nơi ngưỡng ViQuAD được chọn.
- B1 hiện hành của A (`docs/results/b1_domain_dev.md` trên `main`: BM25 top-3, cùng 216 chunk, cùng checkpoint, ngưỡng 0) đạt EM 70,00 / F1 86,73 (trong phạm vi: EM 66,67 / F1 85,25), cao hơn kết quả ở đây (chênh một câu về EM). Hai bên chỉ khác cấu hình truy hồi (BM25 top-3 so với hybrid top-5); 30 câu chưa đủ để kết luận cấu hình nào tốt hơn.
- Năm câu chọn nhầm đoạn chưa phân biệt được giữa điểm logit chưa hiệu chỉnh giữa các đoạn và lỗi đọc trong đoạn: ở A21, B21 và B23, đoạn được chọn cùng Điều với đoạn đúng nhưng khác Khoản, với `score_diff` rất âm (A21 -8,22; B23 -10,15), tức không sát ngưỡng.
- Truy hồi chạy với mã `main` lúc đó (RRF đồng điểm xếp BM25 trước). `main` hiện đã đổi sang xếp dense trước (PR #23 của B), nên chạy lại bằng mã hiện tại có thể đổi thứ hạng, `source_rank` và kết quả từng câu.
- Mã băm trong `code_snapshot` và `gold_dev_sha256_12` tính trên file có xuống dòng CRLF. Mã chạy trên Kaggle là bản trước khi thêm chuẩn hóa NFC cho câu hỏi vào `src/extractive.py` (commit cục bộ `79aa301`, đã được viết lại thành `942ae48`; notebook ghi `79aa301`). Sau khi đổi sang CRLF, mọi file khớp `942ae48`, riêng `src/extractive.py` khác đúng dòng chuẩn hóa NFC đó; câu hỏi trong 30 câu dev đã ở dạng NFC nên kết quả không đổi.
- Trường `config.checkpoint` đã thay tên tài khoản Kaggle bằng `<kaggle-user>`.
