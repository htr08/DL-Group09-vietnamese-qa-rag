# Tuần 2 – A: EDA ViQuAD và VnCoreNLP/PhoBERT
- Ngày tạo: 2026-11-19T21:23:35
- Dữ liệu: data/raw/viquad/{train,dev,public_test}.jsonl
- Split: data/splits/viquad_dev_titles.txt (13 title, PASS; Train/Dev title overlap = 0)
- Khuyến nghị: chỉ dùng Dev để tuning, không đánh giá public_test trước mốc test cuối.

## EDA
| Split | Questions | Titles | Contexts | Answerable | Impossible | Impossible % | Median context chars |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 25686 | 125 | 3711 | 17352 | 8334 | 32.45 | 741 |
| dev | 2768 | 13 | 390 | 1886 | 882 | 31.86 | 760 |
| public_test | 3814 | 19 | 557 | 2653 | 1161 | 30.44 | 721 |

## Answer offsets (RAW)
```json
{
  "train": {
    "matching_spans": 17206,
    "mismatched_spans": 146,
    "missing_answer_records": 0,
    "example_error_ids": [
      "0004-0002-0003",
      "0004-0005-0003",
      "0006-0002-0001",
      "0011-0005-0001",
      "0011-0007-0005"
    ]
  },
  "dev": {
    "matching_spans": 1878,
    "mismatched_spans": 8,
    "missing_answer_records": 0,
    "example_error_ids": [
      "0059-0019-0004",
      "0083-0020-0004",
      "0093-0006-0001",
      "0117-0002-0004",
      "0117-0057-0005"
    ]
  }
}
```

## Answer offsets (PROCESSED)
```json
{
  "train": {
    "matching_spans": 17309,
    "mismatched_spans": 0,
    "missing_answer_records": 0,
    "example_error_ids": []
  },
  "dev": {
    "matching_spans": 1929,
    "mismatched_spans": 0,
    "missing_answer_records": 0,
    "example_error_ids": []
  }
}
```

## VnCoreNLP và PhoBERT tokenizer
```json
{
  "status": "unavailable",
  "success": 0,
  "failed": 0,
  "failures": [],
  "reason": "ModuleNotFoundError: No module named 'py_vncorenlp'"
}
```

## Kết luận
- Kiểm tra 13 title và không overlap: PASS
- VnCoreNLP/PhoBERT: CHƯA XÁC MINH trên môi trường này
- Kiểm tra log tiền xử lý toàn bộ: xem data/processed/qa_features/pipeline_summary.json (nếu đã chạy).
