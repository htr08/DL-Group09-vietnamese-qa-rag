# Tổng hợp tiến độ Tuần 1–3 

| Tuần | Đầu ra dự kiến | Trạng thái / nơi lưu |
|---|---|---|
| 1 | Chuẩn hóa split, môi trường QA | Có pipeline và kiểm thử theo title; xem báo cáo Tuần 2 |
| 2 | Tải/EDA ViQuAD, preprocessing và EM/F1 | Xem `docs/qa_week2_A_evidence.md` và `src/` / `tests/` |
| 2 | QA head + training loop + PhoBERT baseline | Có code và training evidence |
| 2 | Viết 33 Gold, gán nhãn 33 câu của B | Người thực hiện xác nhận đã xong; bộ 100 câu nhóm đã chốt |
| 2 | W&B và B1 sơ bộ | Có evidence huấn luyện; B1 BM25 Top-3 đã đánh giá trên 30 câu Domain Dev với 213 chunks (EM 66,67%; F1 84,89%). |
| 3 | PhoBERT tuning 3 LR, n-best, no-answer | Đã có evidence và bảng số liệu trong `docs/results/qa_week3_A_training_summary.md` |
| 3 | Phân tích lỗi sơ bộ | Đã có script analyze_clean_dev_errors.py; phân tích chi tiết và xác nhận nhãn thủ công thực hiện ở Tuần 4. |

- PhoBERT LR=5e-5, seed=12, epoch 3: **Dev EM 55,78%; Dev F1 69,94%**.
- Clean Dev: **2.768 câu; 13 titles; 3.155 features; 29 unalignable; 0 skipped**.

