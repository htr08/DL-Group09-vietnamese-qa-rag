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


## Đối chiếu và xác minh dữ liệu — 21/11/2026

Sau khi sửa preprocessing, nhóm thực hiện kiểm tra lại các tập ViQuAD Train và Dev.

| Chỉ số | Train | Dev |
|---|---:|---:|
| Tổng số câu hỏi | 25.686 | 2.768 |
| Answerable | 17.352 | 1.886 |
| Impossible | 8.334 | 882 |
| QA features được tạo | 29.333 | 3.155 |
| Câu bị skip | 0 | 0 |

**Kết quả đối chiếu Raw và Processed:**

- Tập ID của Raw và Processed giống nhau trong từng split.
- Không có thay đổi `title` hoặc `context` của các ID tương ứng.
- Không phát hiện ID chuyển chéo giữa Train và Dev.
- Không có context xuất hiện đồng thời trong Train và Dev.

**Giải trình số liệu lịch sử:**

Báo cáo cũ ghi nhận `matching_spans` của Train là 17.309 và Dev là 1.929, khác với số lượng answerable hiện tại là 17.352 và 1.886.

`matching_spans` là chỉ số kiểm tra ánh xạ đáp án, không đồng nhất về định nghĩa với `answerable_examples`.

Dữ liệu hiện tại không cho thấy sự chuyển đổi 43 câu giữa hai split. Chưa tái tạo được nguyên nhân tạo ra số liệu lịch sử; vì vậy, giữ các giá trị cũ để truy vết nhưng sử dụng thống kê mới cho các thí nghiệm tiếp theo.

**Lưu ý:** Dev preprocessing hiện tại có 29 câu answerable không align được nhãn vị trí, nhưng các câu này vẫn được giữ để đánh giá EM/F1.

## Gold Leakage Audit — 21/11/2026

**Mục tiêu:** Định lượng mức ảnh hưởng của việc sử dụng Gold answer trong preprocessing Dev trước khi sửa.

**Phương pháp:**
- Sử dụng preprocessing cũ từ `origin/main:src/preprocessing_qa.py`.
- So sánh với Dev features sạch được tạo bằng `mode=eval`.
- Sử dụng cùng PhoBERT tokenizer.
- Thực hiện trên toàn bộ 2.768 câu Dev.

**Kết quả:**

| Chỉ số | Số câu |
|---|---:|
| Tổng câu Dev | 2.768 |
| Gold làm thay đổi context tokens | 29 |
| Gold làm thay đổi sliding windows | 0 |
| Input features cũ khác Dev sạch | 29 |
| Tổng số câu bị ảnh hưởng | 29 |
| Pipeline cũ không xử lý được | 0 |

**Kết luận:** Pipeline cũ có Gold Leakage, ảnh hưởng đến 29/2.768 câu Dev (1,05%). Tác động được phát hiện ở bước tinh chỉnh token theo Gold answer, không phát hiện thay đổi vị trí cửa sổ trong bộ dữ liệu này.

Pipeline mới tách `mode=train` và `mode=eval`, không sử dụng Gold answer để thay đổi đầu vào Dev.

Kết quả audit chỉ phản ánh số câu có representation khác nhau, chưa đo trực tiếp thay đổi EM/F1.

**Evidence:** `outputs/gold_leakage_audit/summary.json` và `affected_examples.csv`.

Gold Leakage Audit: Pipeline cũ làm thay đổi token của 29/2.768 câu Dev (1,05%). Cả 29 ID đều trùng với các câu unalignable trong Dev sạch. Không phát hiện thay đổi sliding windows do Gold. Chi tiết xem docs/results/gold_leakage_audit/.