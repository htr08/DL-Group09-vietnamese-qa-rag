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

Các số trong bảng khớp `docs/qa_week2_A_eda_reproduced.md`. Số QA features (29.333 cho Train, 3.155 cho Dev) khớp `train_features` và `dev_features` trong config các run fine-tune PhoBERT trên W&B (project `nhom09-vi-qa-rag/vi-qa-rag`, commit `b202d3dc`).

**Tái tạo Dev ngày 22/11/2026** trên Kaggle bằng `preprocess_qa.py` (VnCoreNLP 1.2, `transformers==4.57.6`): 2.768 câu được xử lý, 0 câu bị skip, 3.155 features, trong đó 1.857 `aligned`, 882 `impossible` và 29 `unalignable`. SHA-256 của `dev.jsonl`, tính sau khi chuẩn hóa dấu xuống dòng sang CRLF (tệp sinh trên Windows), trùng giá trị ghi trong `docs/results/qa_week3_A_training_summary.md`. Số features Train (29.333) chưa được tái tạo; lần chạy trên 2.000 câu Train đầu tiên có 0 câu bị skip.

**Đối chiếu Raw và Processed** (tái tạo ngày 22/11/2026 bằng `prepare_viquad_qa.py --overwrite` rồi `fix_answer_start.py`, so sánh từng bản ghi):
- ID, `title` và `context` giống nhau giữa Raw và Processed ở cả ba split.
- Không có context nào nằm ở cả Train lẫn Dev, khớp `data/splits/viquad_split_meta.json` (`context_overlap_count = 0`).
- Danh sách title Dev tạo lại trùng `data/splits/viquad_dev_titles.txt` đã commit.
- Số span sai offset ở Raw là 146 (Train), 8 (Dev) và 22 (Public Test); sau `fix_answer_start.py` còn 0 ở cả ba split.

Đoạn mã so sánh không nằm trong repository.

**Giải trình số liệu lịch sử:**

Báo cáo ngày 19/11 (phần "Answer offsets (PROCESSED)" phía trên) ghi `matching_spans` của Train là 17.309 và Dev là 1.929. Báo cáo tái tạo ngày 21/11 (`docs/qa_week2_A_eda_reproduced.md`) ghi 17.352 và 1.886, bằng số câu answerable, cùng tên chỉ số.

Tổng hai split bằng nhau (17.309 + 1.929 = 17.352 + 1.886 = 19.238); Train thấp hơn 43 và Dev cao hơn 43 so với số hiện tại. Nguyên nhân chưa xác định vì dữ liệu của lần chạy cũ không còn trong repository. Số mới được dùng cho các thí nghiệm tiếp theo.

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
| Gold làm thay đổi overlap của sliding window | 0 |
| Input features cũ khác Dev sạch | 29 |
| Tổng số câu bị ảnh hưởng | 29 |
| Pipeline cũ không xử lý được | 0 |

**Kết luận:** Pipeline cũ có Gold Leakage, ảnh hưởng đến 29/2.768 câu Dev (1,05%). Tác động được phát hiện ở bước tinh chỉnh token theo Gold answer, không phát hiện thay đổi tham số overlap của cửa sổ trong bộ dữ liệu này.

Pipeline mới tách `mode=train` và `mode=eval`, không sử dụng Gold answer để thay đổi đầu vào Dev.

Kết quả audit chỉ phản ánh số câu có representation khác nhau, chưa đo trực tiếp thay đổi EM/F1.

**Đo lại ngày 22/11/2026** (Kaggle; so sánh `mode=eval` với `mode=train` của mã hiện tại trên 1.886 câu Dev có đáp án): 29 câu đổi token và đổi `input_ids`, trùng đúng 29 câu `unalignable` của Dev sạch; ranh giới cửa sổ (`context_start`, `context_end`) khác nhau ở 25/29 câu. Phép đo này đếm ranh giới cửa sổ, còn audit đếm tham số overlap (cột `overlap_changed`, bằng 0 ở cả 29 dòng); hai chỉ số có định nghĩa khác nhau. Chế độ `train` của mã hiện tại chỉ ước lượng hành vi cũ, không phải lần chạy trên mã cũ.

**Evidence:** `docs/results/gold_leakage_audit/summary.json` và `affected_examples.csv`.

Gold Leakage Audit: Pipeline cũ làm thay đổi token của 29/2.768 câu Dev (1,05%). Không phát hiện thay đổi overlap của sliding window do Gold. Phép đo lại ngày 22/11/2026 cho 29 câu đổi đầu vào, trùng 29/29 với các câu `unalignable` trong Dev sạch; danh sách ID của audit gốc chưa được đối chiếu trực tiếp vì `affected_examples.csv` không có cột `label_status`.