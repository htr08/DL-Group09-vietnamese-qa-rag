# Tuần 4 — A: tổng hợp kết quả QA đã chạy (19–25/11/2026)

> **Cập nhật 26/11/2026:** Các kết quả trước đợt V2 được giữ làm evidence lịch sử. Nguồn kết quả mới: [QA V2](qa_v2/README.md).

> Tổng hợp **kết quả Dev đã có evidence** từ Downloads; không phải báo cáo Test. Giữ nguyên B1 216 chunks và checkpoint lịch sử. Experiment freeze chưa chốt.

## 1. Kết quả PhoBERT

| Nhóm thí nghiệm | Seed / LR | EM Dev (%) | F1 Dev (%) | Ghi chú |
|---|---|---:|---:|---|
| Tuning 3 LR | 2e-5, seed 12 | 52,42 | 66,39 | 3 epoch |
| Tuning 3 LR | 3e-5, seed 12 | 53,76 | 67,93 | 3 epoch |
| **Tuning 3 LR** | **5e-5, seed 12** | **55,78** | **69,94** | Best lịch sử theo Dev F1 |
| Đánh giá lại checkpoint lịch sử | 5e-5, seed 12 | 55,24 | 70,10 | CI F1 [68,51; 71,64] |

**Chênh lệch re-eval với lịch sử 55,78/69,94:** nguyên nhân **chưa xác định** (feature file, decoder, thư viện có thể khác, nhưng chưa chứng minh yếu tố nào gây ra sai khác). `outputs/week3_original_checkpoint_eval/summary.json` là **evidence ngoài Git**. Không gộp hai kết quả như cùng inference run.

### 3 seed pipeline mới (same recipe)

| Seed | EM Dev (%) | F1 Dev (%) | F1 bootstrap 95% CI (%) | Tuned threshold |
|---|---:|---:|---|---:|
| 12 | 54,73 | 69,86 | [68,32; 71,39] | 0,025114 |
| 21 | 54,91 | 69,31 | [67,77; 70,94] | −1,359228 |
| 42 | 55,67 | 69,96 | [68,43; 71,53] | −2,695232 |
| **Mean ± sample std (across seeds)** | **55,11 ± 0,50** | **69,71 ± 0,35** | [68,33; 71,16] (CI của trung bình) | — |

Cùng 2.768 Dev / 3.155 features. Bootstrap theo câu hỏi **3.000 resamples**, giữ fixed predictions/threshold của từng seed, **không tune threshold lại**. Độ lệch chuẩn giữa seed và bootstrap CI theo câu hỏi là hai đại lượng khác nhau. Source: `qa_week4_phobert_3seed_bootstrap_ci.json`, `qa_week4_A_phobert_3seed_verified.md`.

### 3 seed Legacy Recipe — nhóm riêng, không hoán đổi checkpoint B1

| Seed | EM Dev (%) | F1 Dev (%) | Tuned threshold |
|---|---:|---:|---:|
| 12 | 56,07 | **71,43** | −1,087250 |
| 21 | 55,06 | 69,81 | −1,060884 |
| 42 | 55,17 | 70,32 | −0,940445 |
| **Mean ± sample std** | **55,43 ± 0,56** | **70,52 ± 0,83** | — |

Source: `phobert_legacy_recipe_seed12_evidence.zip` và `phobert_legacy_recipe_21_42_evidence (1).zip` → `selected_best.json`. Đây là **training recipe khác** với 3 seed pipeline mới, chọn epoch theo Dev F1@threshold=0 rồi tune threshold trên Dev. Không so sánh như single-variable ablation, không dùng Test để chọn recipe.

## 2. XLM-R và PhoBERT NoSeg — Dev

| Cấu hình | EM (%) | F1 (%) | Phạm vi |
|---|---:|---:|---|
| XLM-R, whitespace decoder | 35,55 | 50,71 | Checkpoint chung, decoder ban đầu |
| **XLM-R, subword decoder** | **48,48** | **61,82** | Cùng checkpoint, decoder ablation (không train lại) |
| PhoBERT NoSeg all-in-one | 35,40 | 42,61 | Nhiều lỗi alignment khi train |
| PhoBERT NoSeg GoldBoundary v3 | 36,31 | 43,04 | Slow BPE + train-only Gold boundary refinement |

Sources: `qa_all_in_one_viquad_week4_evidence.zip/decoder_ablation/decoder_ablation_summary.json`, `.../evaluations/phobert_noseg/summary.json`, `phobert_noseg_GOLDBOUNDARY_v3_evidence.zip/evaluation_goldboundary_v3/summary.json`. **NoSeg v3 thay nhiều yếu tố cùng lúc**, nên không thể quy F1 thấp hoàn toàn cho việc bỏ VnCoreNLP.


## 3. Phân tích lỗi ViQuAD

| Evidence | Số mẫu | Nội dung |
|---|---:|---|
| Set 1 (QA_5e5_66_preliminary_review.csv, ngoài repo) | 66 | Lịch sử; 14 mẫu cần xem xét Gold |
| [Set 2 lịch sử](viquad_66_error_legacy_set2.csv) | 66 | 10 cột; 11 mẫu cần xem xét Gold |
| [CSV V2](viquad_66_error.csv) | 66 | 10 cột; có taxonomy và giải thích; 18 mẫu cần xem xét Gold |

V2 lấy mẫu seed 12, chia đều 22/22/22 theo ba loại lỗi.
Mẫu không đại diện cho phân bố lỗi toàn Dev.

Set 1 và Set 2 lịch sử chỉ trùng 3 ID.
Không cộng dồn hoặc chuyển nhãn giữa các checkpoint.

Giữ nguyên dữ liệu Gold, prediction và F1.
Xem [danh sách nghi vấn Gold](qa_v2/gold_checks.md).
Các nhãn phân tích chưa có xác nhận độc lập.

Hai ví dụ được chỉnh trong Set 2:
- 0117-0011-0004: sai so sánh số liệu GDP.
- 0117-0012-0003: thiếu từ "cách" và thừa mệnh đề.

Audit V2: [audit_summary.json](qa_v2/audit_summary.json).

## 4. B1 Domain Dev — baseline lịch sử

- BM25 index/metadata: **216 chunks**, Top-3, PhoBERT LR 5e-5 seed 12 checkpoint lịch sử.
- **30 Domain Dev** (27 trong phạm vi / 3 ngoài phạm vi), fixed threshold 0: **EM 70,00%, F1 86,73%**. Tuned threshold −0,477434
- Provenance: `docs/results/b1_domain_dev/summary.json` và `docs/results/b1_domain_dev.md`. **Không dùng kết quả 213 chunks làm baseline chính.**


## 5. QA rerun V2

PhoBERT V2, LR 5e-5, seed 12: Dev EM 54.9133%, F1 70.3350%, threshold 0.2479505539; [W&B evaluation j212ro7f](https://forge.coreweave.com/wandb/nhom09-vi-qa-rag/vi-qa-rag/runs/j212ro7f).

Xem [QA V2](qa_v2/README.md).

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
