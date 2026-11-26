# Kết quả QA lịch sử — đối chiếu evidence đến 25/11/2026

> **Cập nhật 26/11/2026:** Các kết quả trước đợt V2 được giữ làm evidence lịch sử. Nguồn kết quả mới: [QA V2](qa_v2/README.md).

> **Mốc dữ liệu:** ViQuAD Clean Dev và Domain Dev. **Không có số liệu Test để tuning.** Đây là số liệu đã kiểm tra từ `qa_results_last7days.zip` và `outputs/b1_domain_dev/summary.json`. Kết quả lịch sử của các lần chạy khác chỉ nằm trong tài liệu theo thời gian; không dùng thay kết quả V2.

## 1. B1 — cấu hình lịch sử

| Hạng mục | Kết quả |
|---|---|
| Index BM25 | **216 chunks**, cặp BM25/metadata khớp |
| Reader | PhoBERT checkpoint lịch sử LR `5e-5`, seed `12` |
| Dữ liệu | **30 Domain Dev**: 27 in-scope, 3 out-of-scope |
| Retrieval / threshold cố định | **Top-3 / 0** |
| **B1 EM / F1** | **70,00% / 86,73%** |
| In-scope EM / F1 | 66,67% / 85,25% (27 câu) |
| Out-of-scope | 3/3 câu từ chối đúng |
| Threshold tune trên Domain Dev | `-0.4774336814880371` → **không cải thiện EM/F1**, in-sample |
| Bằng chứng nhỏ trong Git | `docs/results/b1_domain_dev/summary.json` và `docs/results/b1_domain_dev.md` |

**Lưu ý:** B1 và ViQuAD Clean Dev là hai benchmark khác nhau. B1 không được đánh giá trên Domain Test để chọn threshold. Gold 30 Dev / 70 Test là bản chính thức của nhóm.

## 2. PhoBERT — ViQuAD Clean Dev (2.768 câu)

| Nhóm evidence | Seed | Dev EM (%) | Dev F1 (%) | Ý nghĩa |
|---|---|---:|---:|---|
| **Checkpoint lịch sử Set 1, đánh giá lại** | 12 | **55,24** | **70,10** | CI bootstrap 95% F1 [68,51; 71,64], threshold ≈ −0,644425 |
| Pipeline mới | 12 | 54,73 | 69,86 | 3 epoch, LR 5e-5 |
| Pipeline mới | 21 | 54,91 | 69,31 | Cùng recipe pipeline mới |
| Pipeline mới | 42 | 55,67 | 69,96 | Tốt nhất **trong pipeline mới**, không phải freeze cuối |
| **Pipeline mới — mean ± std** | 12/21/42 | **55,11 ± 0,50** | **69,71 ± 0,35** | Bootstrap F1 CI của trung bình [68,33; 71,16] |
| Legacy Recipe | 12 | 56,07 | **71,43** | Tốt nhất **trong Legacy**, recipe riêng |
| Legacy Recipe | 21 | 55,06 | 69,81 | Cùng Legacy Recipe |
| Legacy Recipe | 42 | 55,17 | 70,32 | Cùng Legacy Recipe |
| **Legacy Recipe — mean ± std** | 12/21/42 | **55,43 ± 0,56** | **70,52 ± 0,83** | Chưa tìm thấy bootstrap CI độc lập cho nhóm này |

- **Chưa có quyết định experiment freeze**; không tự đổi checkpoint B1 sang Legacy seed 12 hoặc pipeline mới seed 42 chỉ vì một chỉ số Dev.
- **Tuning 3 LR ở Tuần 3** được giữ tại `docs/results/qa_week3_A_training_summary.md` như **evidence lịch sử**, không dùng nhầm làm kết quả chạy gần nhất.
- Re-evaluation checkpoint lịch sử đạt F1 70,10% so với kết quả lưu từ Tuần 3. **Nguyên nhân chênh lệch chưa xác định**; khác biệt features/decoder/phiên bản chỉ là giả thuyết.

## 3. XLM-R, PhoBERT không tách từ và error analysis

| Thí nghiệm trên Clean Dev | EM | F1 | Ghi chú |
|---|---:|---:|---|
| XLM-R + subword decoder | **48,48%** | **61,82%** | Cùng checkpoint với decoder ablation; không retrain |
| PhoBERT NoSeg GoldBoundary v3 | **36,31%** | **43,04%** | Có thay đổi alignment/preprocessing, không phải single-variable segmentation ablation |
- Các CI của pipeline mới sử dụng 3.000 bootstrap resamples theo câu; threshold đã chọn được giữ cố định. Std giữa seed khác với bootstrap CI theo câu hỏi.

## 4. Nguồn và việc còn lại

- B1: `outputs/b1_domain_dev/summary.json` (local) và `docs/results/b1_domain_dev/summary.json` (evidence nhỏ).
- PhoBERT 3 seed mới: `docs/results/qa_week4_phobert_3seed_bootstrap_ci.json`.
- Checkpoint lịch sử CI: `docs/results/qa_week4_A_phobert_bootstrap_ci.json`.
- Legacy Recipe, XLM-R, NoSeg, phân tích lỗi: xem `docs/results/qa_week4_A_progress.md`, nguồn là ZIP thí nghiệm trong Downloads (bằng chứng gốc chưa được đưa toàn bộ vào Git).

## 5. Cập nhật V2

PhoBERT V2, LR 5e-5, seed 12: Dev EM 54.9133%, F1 70.3350%, threshold 0.2479505539; [W&B evaluation j212ro7f](https://forge.coreweave.com/wandb/nhom09-vi-qa-rag/vi-qa-rag/runs/j212ro7f).

Nguồn mới: [QA V2](qa_v2/README.md).

## Đối chiếu các checkpoint lịch sử

| Nguồn | W&B | EM (%) | F1 (%) | Threshold |
|---|---|---:|---:|---:|
| Tuần 3 LR 5e-5 | jrduerjp (training history) | 55,78 | 69,94 | -1,489326 |
| Set 1 re-evaluation | Chưa xác định | 55,23844 | 70,09862 | -0,6444211006164551 |
| Set 2 re-evaluation | Chưa xác định | 53,61272 | 68,76198 | -0,00926065444946289 |
| Checkpoint Kaggle lịch sử | i0pttl2l | 56,68 | 70,80 | -1,6078 |

Set 1 QA head SHA256:
`9cd8c5f92ef9a7af8665b9664801bc58ae8b97b323633321cac8e5ada1ed3e95`.

Set 2 QA head SHA256:
`7db817d542f9c0021e93a39a540624268ca0ccbbabf8aa572469811690e1be69`.

CSV lịch sử khớp 66/66 ID với Set 2, 3/66 ID với Set 1.
Chưa xác minh độc lập toàn bộ provenance lịch sử.
Không gán audit của V2 hoặc i0pttl2l cho CSV Set 2.

W&B training run lịch sử:
2e-5: w81nmhkx; 3e-5: o977pz63; 5e-5: jrduerjp.
Các lượt chạy thiếu ID/log được giữ là evidence lịch sử.
Không suy diễn nguyên nhân chênh lệch giữa các run.


Phân tích lỗi hiện hành: [CSV V2](viquad_66_error.csv). Nguồn chuẩn: [QA V2](qa_v2/README.md).
