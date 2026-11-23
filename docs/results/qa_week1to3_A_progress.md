# Tổng hợp tiến độ Tuần 1–3 

| Tuần | Đầu ra dự kiến | Trạng thái / nơi lưu |
|---|---|---|

| 1 | Chuẩn hóa split, môi trường QA | Có pipeline và kiểm thử theo title; xem báo cáo Tuần 2 |
| 2 | Tải/EDA ViQuAD, preprocessing và EM/F1 | Xem `docs/qa_week2_A_evidence.md` và `src/` / `tests/` |
| 2 | QA head + training loop + PhoBERT baseline | Có code và training evidence |
| 2 | Viết 33 Gold, gán nhãn 33 câu của B | Người thực hiện xác nhận đã xong; bộ 100 câu nhóm đã chốt |
| 2 | W&B và B1 sơ bộ | W&B đã có; B1 chỉ coi là lịch sử/chưa khớp cấu hình cuối |
| 3 | PhoBERT tuning 3 LR, n-best, no-answer | Đã có evidence và bảng số liệu trong `docs/results/qa_week3_A_training_summary.md` |
| 3 | Phân tích lỗi sơ bộ | Có các báo cáo lỗi phát triển sau này |


- PhoBERT LR=5e-5, seed=12, epoch 3: **Dev EM 55,78%; Dev F1 69,94%**.
- Clean Dev: **2.768 câu; 13 titles; 3.155 features; 29 unalignable; 0 skipped**.

