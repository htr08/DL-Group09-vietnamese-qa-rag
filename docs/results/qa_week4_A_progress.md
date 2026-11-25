# Tuần 4 — A: tổng hợp kết quả QA đã chạy (19–25/11/2026)

> Tổng hợp **kết quả Dev đã có evidence** từ Downloads; không phải báo cáo Test. Giữ nguyên B1 216 chunks và checkpoint lịch sử. Trạng thái `FINAL_REVIEWED` cho 50+ lỗi và experiment freeze **chưa xác minh**.

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

## 3. Phân tích lỗi ViQuAD — 66 câu (chưa chốt manual final)

| Evidence | Số câu | Trạng thái |
|---|---:|---|
| `QA_5e5_66_preliminary_review.csv` | **66** | 22 wrong span, 22 false negative, 22 false positive; 66 `manual_label` có dữ liệu; **66 `PRELIMINARY_MANUAL_REVIEW`** |


Hai bộ chỉ trùng **3 ID**; không gộp thành 132 mẫu đã xác minh. **14/66** bản ghi của bộ đầu có nhãn `QUESTION_GOLD_REVIEW` / `POSSIBLE_GOLD_REVIEW`, cần kiểm tra Gold. Kế hoạch yêu cầu ≥50 mẫu lỗi phân loại: **đã có nhãn sơ bộ cho 66**, nhưng cần người làm xác nhận `FINAL_REVIEWED`, ghi người kiểm tra và notes để gọi là review thủ công hoàn tất.

## 4. B1 Domain Dev — baseline đang giữ nguyên

- BM25 index/metadata: **216 chunks**, Top-3, PhoBERT LR 5e-5 seed 12 checkpoint lịch sử.
- **30 Domain Dev** (27 trong phạm vi / 3 ngoài phạm vi), fixed threshold 0: **EM 70,00%, F1 86,73%**. Tuned threshold −0,477434
- Provenance: `docs/results/b1_domain_dev/summary.json` và `docs/results/b1_domain_dev.md`. **Không dùng kết quả 213 chunks làm baseline chính.**
