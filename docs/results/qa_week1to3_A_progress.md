# Tiến độ A — Extractive QA

| Hạng mục | Trạng thái |
|---|---|
| Chia ViQuAD Train/Dev theo title | Hoàn thành; 13 title Dev, không trùng context Train/Dev |
| EDA, xử lý answer offsets | Đã thực hiện; xem `docs/qa_week2_A_evidence.md` |
| PhoBERT preprocessing và QA head | Đã triển khai, có kiểm thử |
| Huấn luyện và đánh giá PhoBERT | V2: LR 5e-5, seed 12, epoch 3; **EM 54.91%, F1 70.33%**; W&B j212ro7f |
| Gold miền ứng dụng | A viết 33 câu và gán nhãn 33 câu của B; **30 Dev / 70 Test** có trong repo và là bản Gold chính thức |
| B1 — BM25 + PhoBERT | V2: 216 chunks, Top-3, 30 Domain Dev; threshold 0: EM 63.33%, F1 78.98%; tuned: EM 66.67%, F1 85.28% |

Kết quả chi tiết PhoBERT: `docs/results/qa_week3_A_training_summary.md`. Chỉ dùng Dev để chọn tham số, không dùng Public Test hoặc Domain Test.

## Cập nhật evidence Tuần 4 (lịch sử)

- PhoBERT **3 seed Legacy Recipe**: F1 **70,52 ± 0,83%**; đây là nhóm run độc lập, không thay checkpoint B1.
- XLM-R subword decoder: **EM 48,48%, F1 61,82%**; PhoBERT NoSeg GoldBoundary v3: **EM 36,31%, F1 43,04%** (có thay đổi preprocessing/align ngoài segmentation).
- B1 trên Domain Dev: **216 chunks**, EM **70,00%**, F1 **86,73%**. Không có kết quả Domain Test; Gold 30 Dev / 70 Test là bản chính thức
- ViQuAD checkpoint lịch sử đánh giá lại: EM **55,24%**, F1 **70,10%** (CI F1 [68,51%; 71,64%]); chênh kết quả Tuần 3 **chưa xác định nguyên nhân**.

Chi tiết các kết quả, nguồn file và caveats ở `docs/results/qa_week4_A_progress.md`.

## Kết quả QA V2 — 26/11/2026

| Thí nghiệm | EM (%) | F1 (%) |
|---|---:|---:|
| PhoBERT LR 5e-5, seed 12 | 54.91 | 70.33 |
| XLM-R | 34.57 | 41.73 |
| PhoBERT NoSeg | 36.13 | 43.81 |
| B1 Domain Dev, threshold 0 | 63.33 | 78.98 |
| B1 Domain Dev, tuned | 66.67 | 85.28 |

PhoBERT 3 seed (12/42/123):
EM 55.43 +/- 0.56%;
F1 70.01 +/- 0.40%.

W&B evaluation: [j212ro7f](https://forge.coreweave.com/wandb/nhom09-vi-qa-rag/vi-qa-rag/runs/j212ro7f).

Các kết quả đều trên Dev, không dùng Test để tuning.

Chi tiết: [QA V2](qa_v2/README.md).


Phân tích lỗi hiện hành: [CSV V2](viquad_66_error.csv). Nguồn chuẩn: [QA V2](qa_v2/README.md).
