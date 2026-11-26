# QA rerun V2 — 26/11/2026

PhoBERT V2, LR 5e-5, seed 12: Dev EM 54.9133%, F1 70.3350%, threshold 0.2479505539; [W&B evaluation j212ro7f](https://forge.coreweave.com/wandb/nhom09-vi-qa-rag/vi-qa-rag/runs/j212ro7f).

- QA head SHA256: `9d322cc56b09f43526f29e6925db1d6f8950162c6d6b259291f46d4a90c078e3`
- Backbone SHA256: `91da676e6e344893739a6b65aa251b11d82a390d3617ee12081edfcd4089fbf5`
- Git commit: `1251e4b4ffae6401126abf58be2b9bc10e2c7541`
- Evaluation run ID không phải training run ID.
- Dev: 2.768 câu. Best epoch: 3.

## Learning-rate grid

| LR | EM (%) | F1 (%) | Threshold | W&B evaluation |
|---|---:|---:|---:|---|
| 2e-5 | 51.1922 | 66.2341 | -0.07685518 | [y3uf1ks9](https://forge.coreweave.com/wandb/nhom09-vi-qa-rag/vi-qa-rag/runs/y3uf1ks9) |
| 3e-5 | 54.0101 | 67.9759 | -1.43005562 | [4gpwf2ba](https://forge.coreweave.com/wandb/nhom09-vi-qa-rag/vi-qa-rag/runs/4gpwf2ba) |
| 5e-5 | 54.9133 | 70.3350 | 0.24795055 | [j212ro7f](https://forge.coreweave.com/wandb/nhom09-vi-qa-rag/vi-qa-rag/runs/j212ro7f) |

## Seed stability

- Seeds: [12, 42, 123].
- EM mean +/- sample std: 55.4311 +/- 0.5647%.
- F1 mean +/- sample std: 70.0109 +/- 0.4044%.
- Bootstrap: 10.000 resamples, seed 42, đơn vị câu hỏi.
- Xem JSON evidence và thư mục metrics.

## B1 Domain Dev

30 câu, BM25 216 chunks, Top-3, checkpoint V2.

| Threshold | EM (%) | F1 (%) |
|---|---:|---:|
| 0 | 63.33 | 78.98 |
| 1.17805576 | 66.67 | 85.28 |

Threshold Domain Dev được tune riêng, kết quả tuned là in-sample.

## Error analysis

- Tổng lỗi EM: 1248/2768 câu Dev.
- Toàn Dev: 725 wrong_span, 218 false_negative, 305 false_positive.
- Mẫu: 66 câu, seed 12, chia đều 22/22/22.
- [CSV chính thức](../viquad_66_error.csv): 66 dòng, 10 cột.
- 8 trường dữ liệu nguồn, taxonomy_label và analysis_note.
- 66 mẫu có phân loại và giải thích.
- 18 mẫu cần xem xét Gold; xem [danh sách](gold_checks.md).
- Nhãn phân tích chưa có xác nhận độc lập.
- Mẫu không đại diện cho phân bố lỗi toàn Dev.
- Audit: [audit_summary.json](audit_summary.json).

### SHA256

- CSV gốc do script xuất: `66_dev_errors_to_review.csv`
  `ddf886965c7bb44854bdff0ed9cf6e91f731dc6b23eeec655d0c333eba3c900f`
- CSV phân tích 10 cột trong working tree:
  `c64c7969ebc3ed31f183b9b00afe96c80eba77f1e894561973f5ff50c428f89a`

Checksum đầu vào và đầu ra khác nhau do cấu trúc CSV.
Đã đối chiếu 8 trường dữ liệu của toàn bộ 66 mẫu.
Hash đầu ra tính trên byte của file local;
line ending do Git có thể làm hash byte khác.

### Phân bố taxonomy

| Taxonomy | Số mẫu |
|---|---:|
| false_refusal_descriptive | 5 |
| false_refusal_factoid | 13 |
| gold_answerability_dispute | 7 |
| gold_context_ambiguity | 3 |
| question_premise_mismatch | 6 |
| question_type_mismatch | 3 |
| span_extra_clause | 4 |
| span_extra_modifier | 5 |
| span_too_short | 5 |
| wrong_actor_or_entity | 4 |
| wrong_entity_or_relation | 4 |
| wrong_event_or_scope | 3 |
| wrong_event_or_semantics | 2 |
| wrong_time_reference | 2 |
| **Tổng** | **66** |

Không thay đổi Gold hoặc kết quả đánh giá.
Không dùng Public Test hoặc Domain Test để tuning.
Experiment freeze chưa chốt.

## So sánh mô hình

| Model | EM (%) | F1 (%) | W&B evaluation |
|---|---:|---:|---|
| PhoBERT | 54.91 | 70.33 | [j212ro7f](https://forge.coreweave.com/wandb/nhom09-vi-qa-rag/vi-qa-rag/runs/j212ro7f) |
| XLM-R | 34.57 | 41.73 | [65osbily](https://forge.coreweave.com/wandb/nhom09-vi-qa-rag/vi-qa-rag/runs/65osbily) |
| PhoBERT NoSeg | 36.13 | 43.81 | [fsrcrbhv](https://forge.coreweave.com/wandb/nhom09-vi-qa-rag/vi-qa-rag/runs/fsrcrbhv) |

Cùng Clean Dev, LR 5e-5, seed 12, 3 epoch.
Khác biệt preprocessing và tokenizer cần được xem xét
khi diễn giải chênh lệch giữa các mô hình.

## N-best ablation

| N-best | EM fixed (%) | F1 fixed (%) | EM tuned (%) | F1 tuned (%) |
|---|---:|---:|---:|---:|
| 1 | 54.55 | 69.00 | 54.52 | 69.14 |
| 5 | 54.88 | 70.12 | 54.91 | 70.33 |
| 10 | 54.88 | 70.12 | 54.91 | 70.33 |
| 20 | 54.88 | 70.12 | 54.91 | 70.33 |

Cùng checkpoint PhoBERT V2.
Kết quả n-best 5, 10 và 20 giống nhau trên Dev.

## Giới hạn evidence

- Các ID W&B trong bảng là evaluation run ID.
- Training run ID cần đối chiếu riêng với log huấn luyện.
- Các thí nghiệm lịch sử thiếu log được ghi nhận riêng.
- Không trộn metrics từ các checkpoint khác nhau.


W&B: [Danh sách evaluation run](wandb_runs.md).
