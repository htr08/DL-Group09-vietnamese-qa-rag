# Tuần 2 — A: EDA ViQuAD, tiền xử lý và bằng chứng PhoBERT

> **Bản hợp nhất tài liệu khi merge `origin/main` vào `feat/extractive-qa-pipeline`.** Giữ lại EDA/audit lịch sử trên `main` và phần tổng kết tiếng Việt của nhánh QA. Đây là báo cáo dữ liệu và tiến độ, không phải báo cáo kết quả Test.
>
> **Phạm vi:** Công việc của thành viên A theo `docs/project_plan.md`: ViQuAD 2.0, tách Train/Dev theo bài viết, VnCoreNLP–PhoBERT, QA features, EM/F1, QA head/training loop, đóng góp nhãn Gold và baseline B1 sơ bộ. Các kết quả tuning chính nằm trong `docs/results/qa_week3_A_training_summary.md`.

## 1. Nguồn dữ liệu và quy tắc chia

- Nguồn ViQuAD 2.0: `taidng/UIT-ViQuAD2.0`. Dữ liệu raw/processed được chuẩn bị bằng `scripts/download_data.py`, `scripts/prepare_viquad_qa.py`, `scripts/fix_answer_start.py` (theo phiên bản pipeline thích hợp).
- Tách **Dev từ Train theo `title`**, cố định danh sách tại `data/splits/viquad_dev_titles.txt` (**13 title**). Kiểm tra: **Train/Dev title overlap = 0**, context overlap = 0 (theo báo cáo đối chiếu).
- Split `validation` do nguồn Hugging Face cung cấp được dành làm **Public Test**. Chỉ thống kê dữ liệu; **không dùng Public Test để chọn mô hình, tune threshold hoặc đánh giá trước mốc test cuối**.

### 1.1. EDA ngày 19/11/2026

| Split | Số câu | Số title | Số context | Có đáp án | Không đáp án | Không đáp án (%) | Trung vị độ dài context (ký tự) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 25.686 | 125 | 3.711 | 17.352 | 8.334 | 32,45 | 741 |
| Dev nội bộ | 2.768 | 13 | 390 | 1.886 | 882 | 31,86 | 760 |
| Public Test (chỉ thống kê) | 3.814 | 19 | 557 | 2.653 | 1.161 | 30,44 | 721 |

*Các số này phản ánh dữ liệu của nhóm theo lần chạy EDA, không phải số trích nguyên từ bài báo.*

## 2. Kiểm tra answer offsets

### 2.1. Raw — báo cáo ngày 19/11/2026

| Split | `matching_spans` | `mismatched_spans` | `missing_answer_records` |
|---|---:|---:|---:|
| Train | 17.206 | 146 | 0 |
| Dev | 1.878 | 8 | 0 |

- Ví dụ ID có offset lỗi ở Train: `0004-0002-0003`, `0004-0005-0003`, `0006-0002-0001`, `0011-0005-0001`, `0011-0007-0005`.
- Ví dụ ID có offset lỗi ở Dev: `0059-0019-0004`, `0083-0020-0004`, `0093-0006-0001`, `0117-0002-0004`, `0117-0057-0005`.

### 2.2. Processed — báo cáo lịch sử ngày 19/11/2026

| Split | `matching_spans` | `mismatched_spans` | `missing_answer_records` |
|---|---:|---:|---:|
| Train | 17.309 | 0 | 0 |
| Dev | 1.929 | 0 | 0 |

**Cảnh báo về số liệu lịch sử:** Hai chỉ số `matching_spans` này **khác phân bố answerable** trong EDA tái tạo (Train 17.352 / Dev 1.886), nhưng tổng bằng nhau: `17.309 + 1.929 = 17.352 + 1.886 = 19.238`. Chênh **43 câu chuyển giữa hai split** chưa có nguyên nhân xác định do dữ liệu trung gian của lần chạy cũ không còn. **Kết quả tái tạo mới** được dùng cho thí nghiệm tiếp theo; không âm thầm thay số liệu EDA cũ.

### 2.3. Đối chiếu tái tạo ngày 21–22/11/2026

- Sau khi chạy `prepare_viquad_qa.py --overwrite` rồi `fix_answer_start.py`, báo cáo đối chiếu ghi: ID, `title`, `context` không đổi giữa raw và processed; danh sách title Dev khớp tệp split đã commit.
- Theo phép kiểm tra này, số offset lỗi trong Raw là **146 Train, 8 Dev, 22 Public Test**; sau sửa còn **0** ở cả ba split. Việc kiểm tra offset Public Test **không tương đương sử dụng Test để tune mô hình**.
- Không có context trùng Train/Dev; tham chiếu `data/splits/viquad_split_meta.json` (`context_overlap_count = 0`).
- Đoạn mã so sánh từng bản ghi của lần chạy này **không được lưu trong repository**, là hạn chế tái lập cần ghi nhận.

## 3. Kiểm tra VnCoreNLP/PhoBERT và QA features

### 3.1. Môi trường

- Ngày **19/11/2026**: kiểm tra VnCoreNLP/tokenizer báo `unavailable` vì `ModuleNotFoundError: No module named 'py_vncorenlp'` (**0 thành công / 0 thất bại** vì chưa thực thi mẫu).
- Ngày **21/11/2026**: chạy lại có môi trường phù hợp, **10/10 mẫu thành công**. Xem `docs/qa_week2_A_eda_reproduced.md`. Vì vậy không được dùng dòng `unavailable` của lần đầu để mô tả trạng thái cuối của pipeline.

### 3.2. Preprocessing Clean Dev

- Pipeline: VnCoreNLP word segmentation → PhoBERT tokenizer → sliding windows (`max_length=256`, `doc_stride=128`) → QA features, nhãn huấn luyện start/end. Đối với Dev, **các input/window phải độc lập với Gold** (`mode=eval`).
- Đợt tái tạo ngày **22/11/2026** trên Kaggle dùng VnCoreNLP 1.2 và `transformers==4.57.6`.

| Chỉ số Clean Dev | Giá trị |
|---|---:|
| Tổng/đã xử lý | 2.768 / 2.768 |
| Features | 3.155 |
| Bỏ qua (`skipped`) | 0 |
| `aligned` | 1.857 |
| `impossible` | 882 |
| `unalignable` | 29 |

- Kết quả `unalignable=29` được **giữ để đánh giá**, không tự loại các câu khó.
- Số lượng Train features **29.333** được nêu trong cấu hình của các run PhoBERT; báo cáo ngày 22/11 ghi **chưa tái tạo toàn bộ** Train features. Kiểm tra 2.000 câu đầu của Train không có mẫu bị bỏ qua.
- SHA-256 của Dev feature file được ghi trong manifest thí nghiệm cũ: `b27046196cd92f0ede419737b155f7f36a6bb1f2c54c563b3bafa31c82d8d3c3`. Theo báo cáo tái tạo ngày 22/11, file được chuẩn hóa dấu xuống dòng CRLF khi đối chiếu hash.
- **Chú ý phiên bản:** Từ ngày 22/11/2026, schema feature thêm `preprocessing_mode`. File được sinh bằng code mới có hash khác file lưu trước thay đổi schema. **Hash khớp không thay cho kiểm tra `mode=eval` và tính độc lập Gold.**

## 4. Gold Leakage Audit — 21–22/11/2026

**Mục tiêu:** so sánh representation Dev của pipeline cũ (có tinh chỉnh token theo Gold) với pipeline độc lập Gold, **không dùng audit để ước lượng trực tiếp chênh lệch F1**.

| Chỉ số audit lịch sử | Kết quả |
|---|---:|
| Tổng Dev | 2.768 |
| Context token thay đổi do Gold | 29 |
| Window boundary thay đổi chỉ do overlap khi giữ nguyên token | 0 |
| Feature input khác Clean Dev | 29 |
| Mẫu pipeline cũ không xử lý được | 0 |

- Trong phép audit lịch sử: Gold-dependent token refinement ảnh hưởng **29/2.768 câu (1,05%)**; không phát hiện thay đổi window boundary do overlap **khi giữ nguyên token đã tinh chỉnh**.
- Đo lại **22/11/2026** trên 1.886 câu Dev có đáp án, so sánh `mode=eval` với `mode=train` trong code hiện tại: **29 câu thay đổi token và `input_ids`** (trùng 29 câu `unalignable`), **25/29** câu có `context_start/context_end` khác. Hai phép đo **không mâu thuẫn**: phép đo sau không giữ cố định token; phép đo trước chỉ quy thay đổi window cho riêng yếu tố overlap.
- `mode=train` trong code hiện tại chỉ **ước lượng** hành vi Gold-dependent cũ, không phải chạy trực tiếp bằng phiên bản code cũ.
- Danh sách ID ở audit cũ chưa được đối chiếu trực tiếp theo `label_status`, vì `affected_examples.csv` không có cột này.
- **Evidence trong repo:** `docs/results/gold_leakage_audit/summary.json` và `affected_examples.csv`.

## 5. Thí nghiệm, truy vết nguồn và phạm vi đóng góp

- Mô hình PhoBERT QA: tự triển khai QA head/loss, training loop, EM/F1 và n-best decoder theo kế hoạch; tài liệu chi tiết 3 LR, threshold và chỉ số Dev tại `docs/results/qa_week3_A_training_summary.md`.
- Tư liệu thí nghiệm lịch sử: `phobert_3lr_clean_dev_EVIDENCE.zip` (manifest và summary), **không nằm trong Git repo**, không bao gồm model weights/predictions đầy đủ; không coi có ZIP là đã tái chạy inference.
- Báo cáo lần chạy W&B: project `nhom09-vi-qa-rag/vi-qa-rag`; commit huấn luyện `b202d3dc9fb4bcdd02131d213d4b1b8d26f1dde8`. **Chưa có URL/ID run riêng được đối chiếu trong tài liệu này**; cần dẫn link cụ thể khi tổng hợp báo cáo cuối.
- Đóng góp nhãn của A: **viết 33 câu**, **gán nhãn độc lập 33 câu của B**. Bộ Gold nhóm có **30 Dev / 70 Test** (trên `main` theo lần kiểm tra Git); việc chốt bất đồng và bằng chứng đồng thuận hai vòng cần đối chiếu với hồ sơ gán nhãn trước khi khẳng định toàn bộ đã *finalized*.
- B1 sơ bộ: script `scripts/run_b1_domain_dev.py` hiện có. Kết quả chạy local theo người thực hiện trên **30 Domain Dev, BM25 Top-3, 213 chunks**, fixed threshold 0: **EM 66,67%, F1 84,89%**. Đây là **kết quả Dev của cấu hình 213 chunks**, không phải Test; để tái lập cần đúng checkpoint, BM25 index và chunks metadata tương ứng. Output từng câu/summary chưa được đính kèm trong tài liệu hợp nhất này.

## 6. Giới hạn và vị trí bằng chứng

- EDA cũ và kết quả tái tạo có lệch `matching_spans` giữa Train/Dev; nguyên nhân chưa xác minh.
- Manifest 3 LR không cung cấp checkpoint hoặc predictions đầy đủ để tự tính bootstrap CI của thí nghiệm lịch sử; việc xác minh load/inference checkpoint cần có evidence riêng.
- Bằng chứng EDA ngày 21/11: `docs/qa_week2_A_eda_reproduced.md`.
- Bằng chứng dữ liệu/split: `data/splits/viquad_dev_titles.txt`, `data/splits/viquad_split_meta.json`.
- Bằng chứng leakage: `docs/results/gold_leakage_audit/`.
- Bằng chứng PhoBERT 3 LR: `docs/results/qa_week3_A_training_summary.md` và ZIP ngoài repo.

**Không sử dụng Public Test hoặc Domain Test để chọn tham số trong các nội dung trên.**
