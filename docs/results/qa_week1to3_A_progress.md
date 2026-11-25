# Tiến độ A — Extractive QA

| Hạng mục | Trạng thái |
|---|---|
| Chia ViQuAD Train/Dev theo title | Hoàn thành; 13 title Dev, không trùng context Train/Dev |
| EDA, xử lý answer offsets | Đã thực hiện; xem `docs/qa_week2_A_evidence.md` |
| PhoBERT preprocessing và QA head | Đã triển khai, có kiểm thử |
| Huấn luyện và đánh giá PhoBERT | Checkpoint `5e-5`, seed `12`, epoch `3`: **Clean Dev EM 55,24%, F1 70,10%** |
| Gold miền ứng dụng | A viết 33 câu và gán nhãn 33 câu của B; **30 Dev / 70 Test** có trong repo, việc chốt nhãn chưa được xác minh |
| B1 — BM25 + PhoBERT | **Đã chạy Domain Dev 30 câu, BM25 Top-3, index 216 chunks:** EM **70,00%**, F1 **86,73%** tại threshold 0; có summary SHA-256 ở `docs/results/b1_domain_dev/summary.json` |

Kết quả chi tiết PhoBERT: `docs/results/qa_week3_A_training_summary.md`. Chỉ dùng Dev để chọn tham số, không dùng Public Test hoặc Domain Test.

## Cập nhật evidence Tuần 4

- PhoBERT **3 seed Legacy Recipe**: F1 **70,52 ± 0,83%**; đây là nhóm run độc lập, không thay checkpoint B1.
- XLM-R subword decoder: **EM 48,48%, F1 61,82%**; PhoBERT NoSeg GoldBoundary v3: **EM 36,31%, F1 43,04%** (có thay đổi preprocessing/align ngoài segmentation).
- B1 trên Domain Dev: **216 chunks**, EM **70,00%**, F1 **86,73%**. Không có kết quả Domain Test; Gold 30 Dev / 70 Test đã có dữ liệu
- ViQuAD checkpoint lịch sử đánh giá lại: EM **55,24%**, F1 **70,10%** (CI F1 [68,51%; 71,64%]); chênh kết quả Tuần 3 **chưa xác định nguyên nhân**.

Chi tiết các kết quả, nguồn file và caveats ở `docs/results/qa_week4_A_progress.md`.
