# B1 — BM25 (216 chunks) + PhoBERT trên Domain Dev

> **Trạng thái 25/11/2026:** đã chạy inference trên **30 câu Dev**. Đây là kết quả hệ thống tại cấu hình đã chạy, dùng bộ Gold chính thức của nhóm, không phải Test. Không rebuild BM25, không đổi checkpoint.

## Cấu hình

| Thành phần | Giá trị |
|---|---|
| Corpus | Quy chế ĐHQGHN 2014, **216 chunks** |
| BM25 | Cặp `data/index/domain/bm25.pkl` và `chunks_meta.json` đã kiểm tra cùng 216 ID, đúng thứ tự |
| Reader | PhoBERT QA, LR `5e-5`, seed `12`, best checkpoint lịch sử |
| Retrieval | BM25 **Top-3** |
| Câu hỏi | **30 Domain Dev**: 27 in-scope, 3 out-of-scope |
| Decode | Word/char offsets, best span so với null score trên các windows; max answer length 30 |
| Thiết bị | CPU |

## Kết quả Dev

| Thiết lập | EM | F1 |
|---|---:|---:|
| **Fixed threshold = 0 (baseline chính)** | **70,00% (21/30)** | **86,73%** |
| Tuned threshold ≈ −0,477434 (Dev) | **70,00%** | **86,73%** |

Phân nhóm tại threshold 0: **in-scope** (27 câu): EM **66,67%**, F1 **85,25%**; **out-of-scope** (3 câu): EM/F1 **100%**. Điều chỉnh threshold trên cùng Dev không cải thiện điểm.

## Provenance: hash từ lần chạy local

| Artifact | SHA-256 |
|---|---|
| Gold Dev | `80b434b045f2f09253c40ca277b723fa2715e6cab0f3855ea2a3c8a353b1e0e3` |
| BM25 index | `b804ca6aab83a0c266ddbf1a113d1c3b2684c93c40bf0f015e5bc77a92a76d73` |
| Chunk metadata | `46e3ff18b9217c6f8078a16266e99134aab9fa3651750a012e8657da19a29f43` |
| PhoBERT QA head | `9cd8c5f92ef9a7af8665b9664801bc58ae8b97b323633321cac8e5ada1ed3e95` |

Bản JSON nguyên vẹn từ output local: [`docs/results/b1_domain_dev/summary.json`](b1_domain_dev/summary.json). File gốc trên máy nằm tại `outputs/b1_domain_dev/summary.json`. Không đưa BM25 pickle, checkpoint, raw predictions hay 70 câu Domain Test lên Git.

## Hạn chế và quyết định

- Các lần chạy B1 trước đây với index khác 216 chunks không được dùng làm baseline của cấu hình hiện tại.
- Score/logits của spans ở các chunk khác nhau **chưa được calibration**, kết quả là baseline.
- Dev chỉ 30 câu nên việc tune threshold dễ overfit; giữ **threshold 0** làm điểm tham chiếu chính.
- Dữ liệu Gold Dev/Test gồm 30/70 bản ghi và là bản chính thức. Không tune/chạy Test trước experiment freeze.

## Chạy B1 (nếu cần tái hiện trên máy có artifact)

```powershell
$CKPT = "$HOME\Downloads\phobert_5e5_extracted\phobert_qa_lr5e-5_seed12\best_model"
python scripts/run_b1_domain_dev.py --checkpoint "$CKPT" --top-k 3
```

Điều kiện: index BM25 **216 chunks** đã tồn tại tại `data/index/domain` và script B1 ở `scripts/`. Không rebuild hay thay checkpoint để làm tài liệu.
