# W&B và provenance thí nghiệm

Các ID sau được trích từ evaluation_record.json.
Đây là evaluation run ID, không phải training run ID.

| Thí nghiệm | EM (%) | F1 (%) | Threshold | W&B |
|---|---:|---:|---:|---|
| A_v2_phobert_lr2e-5_seed12_nbest20 | 51.1922 | 66.2341 | -0.076855183 | [y3uf1ks9](https://forge.coreweave.com/wandb/nhom09-vi-qa-rag/vi-qa-rag/runs/y3uf1ks9) |
| A_v2_phobert_lr3e-5_seed12_nbest20 | 54.0101 | 67.9759 | -1.430055618 | [4gpwf2ba](https://forge.coreweave.com/wandb/nhom09-vi-qa-rag/vi-qa-rag/runs/4gpwf2ba) |
| A_v2_phobert_lr5e-5_seed123_nbest20 | 56.0332 | 70.1402 | -1.219979763 | [5kw90932](https://forge.coreweave.com/wandb/nhom09-vi-qa-rag/vi-qa-rag/runs/5kw90932) |
| A_v2_phobert_lr5e-5_seed12_nbest1 | 54.5159 | 69.1367 | 0.229227781 | [9ipkxxgm](https://forge.coreweave.com/wandb/nhom09-vi-qa-rag/vi-qa-rag/runs/9ipkxxgm) |
| A_v2_phobert_lr5e-5_seed12_nbest10 | 54.9133 | 70.3350 | 0.247950554 | [bqznjic1](https://forge.coreweave.com/wandb/nhom09-vi-qa-rag/vi-qa-rag/runs/bqznjic1) |
| A_v2_phobert_lr5e-5_seed12_nbest20 | 54.9133 | 70.3350 | 0.247950554 | [j212ro7f](https://forge.coreweave.com/wandb/nhom09-vi-qa-rag/vi-qa-rag/runs/j212ro7f) |
| A_v2_phobert_lr5e-5_seed12_nbest5 | 54.9133 | 70.3350 | 0.247950554 | [oe245lvr](https://forge.coreweave.com/wandb/nhom09-vi-qa-rag/vi-qa-rag/runs/oe245lvr) |
| A_v2_phobert_lr5e-5_seed42_nbest20 | 55.3468 | 69.5577 | -2.023808718 | [macyyccm](https://forge.coreweave.com/wandb/nhom09-vi-qa-rag/vi-qa-rag/runs/macyyccm) |
| A_v2_phobert_noseg_lr5e-5_seed12_nbest20 | 36.1272 | 43.8055 | 0.948032141 | [fsrcrbhv](https://forge.coreweave.com/wandb/nhom09-vi-qa-rag/vi-qa-rag/runs/fsrcrbhv) |
| A_v2_xlmr_lr5e-5_seed12_nbest20 | 34.5737 | 41.7256 | 1.627282381 | [65osbily](https://forge.coreweave.com/wandb/nhom09-vi-qa-rag/vi-qa-rag/runs/65osbily) |


## Evidence lịch sử và phần còn thiếu

- Training Tuần 3: w81nmhkx, o977pz63, jrduerjp.
- Checkpoint Kaggle lịch sử: i0pttl2l.
- Checkpoint V2 dùng cho báo cáo Dev: evaluation j212ro7f.
- Training run ID của V2 chưa được liên kết đầy đủ.
- B1, bootstrap và error analysis có JSON evidence,
  nhưng chưa xác nhận được W&B run ID tương ứng.
- Chưa xác định nguyên nhân thiếu log lịch sử sau 21/11.
- Không tạo ID hoặc log huấn luyện giả.
- Chênh lệch các run lịch sử cùng cấu hình không phải
  độ lệch chuẩn của thí nghiệm 3 seed V2.
