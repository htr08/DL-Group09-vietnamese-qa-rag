
## 1. Cấu trúc thư mục

| Đường dẫn | Mục đích |
|---|---|
| `src/` | Mã nguồn tiền xử lý QA, huấn luyện, truy hồi, sinh câu trả lời, metrics và pipeline |
| `scripts/` | Các script tải dữ liệu, xây dựng index và đánh giá |
| `tests/` | Unit tests; CI chạy khi tạo/cập nhật Pull Request |
| `notebooks/` | EDA, phân tích và Kaggle smoke test (`00_kaggle_qwen_smoke_test.ipynb`) |
| `data/raw/` | Dữ liệu tải về; được loại khỏi Git |
| `data/domain/` | Tài liệu miền ứng dụng: Quy chế đào tạo ĐHQGHN 2014 (miền chính) và Samsung Galaxy S24 (nghiên cứu truy hồi liên ngôn ngữ) |
| `data/gold/` | Vị trí lưu nhãn Dev/Test miền ứng dụng sau khi nhóm chốt; các file trong bản source đã rà soát còn là placeholder, **không phải Gold đã thống nhất** |
| `data/splits/` | Danh sách chia dữ liệu và seed |
| `kaggle/smoke_test_dataset/` | Bộ dữ liệu nhỏ cho Kaggle smoke test |
| `app/` | Ứng dụng demo |
| `docs/` | Kế hoạch, Charter, khai báo dùng AI và tài liệu phân biệt phần tái sử dụng/phần tự cài đặt |
| `report/` | Báo cáo LaTeX theo mẫu ACL: `proposal/`, `milestone/`, `final/` |

## 2. Cài đặt môi trường

Yêu cầu **Python 3.12**. Để tách từ bằng VnCoreNLP, cần cài **JDK (Java 8 trở lên)** và cấu hình Java trong `PATH`.

```bash
python -m venv venv
# Windows:      venv\Scripts\activate
# macOS/Linux:  source venv/bin/activate
pip install -r requirements.txt
```

`torch` không được cố định phiên bản trong `requirements.txt`; cần cài bản PyTorch phù hợp với hệ điều hành, CPU/GPU và CUDA của máy (xem chú thích đầu file yêu cầu thư viện).

## 3. Tiến độ của A — Extractive QA 

### 3.1. Công việc đã thực hiện và còn lại

| Tuần | Nhiệm vụ của A | Tình trạng tại ngày 22/11/2026 |
|---|---|---|
| 1 | Chia ViQuAD theo tiêu đề bài viết (title), chống trùng Train/Dev | **Hoàn thành:** seed `42`, 13 title Dev, 125 title Train, 0 context trùng; xem `data/splits/viquad_split_meta.json` |
| 1 | Kiểm tra môi trường tách từ và tokenizer | Smoke test VnCoreNLP + PhoBERT đạt 10/10 mẫu; xem `docs/qa_week2_A_eda_reproduced.md` |
| 2 | Tải ViQuAD, EDA, EM/F1, sliding window, offsets, lớp QA head | **Đã triển khai và có tests**; xem `notebooks/01_eda_viquad_A.ipynb` và `docs/qa_week2_A_evidence.md` |
| 2 | Huấn luyện PhoBERT ban đầu; viết câu hỏi và gán nhãn miền ứng dụng | **Đã huấn luyện**; A viết 33 câu và có bản gán nhãn độc lập 33 câu của B
| 2 | B1 sơ bộ (BM25 + PhoBERT); chưa phải kết quả chính thức chấm theo Gold miền ứng dụng đã được thống nhất |
| 3 | Tuning PhoBERT với LR `2e-5`, `3e-5`, `5e-5`; n-best và no-answer threshold; kết quả ở mục 4 |
| 3 | Chốt B1, phân tích lỗi và viết phần QA cho Milestone | B1 Dev và mẫu lỗi **còn mang tính sơ bộ**; nội dung Milestone QA mới có bản nháp

Việc thống nhất nhãn qua hai lượt gán nhãn và chốt tập miền ứng dụng **30 Dev / 70 Test
## 4. Dữ liệu và kết quả PhoBERT

### 4.1. Chia dữ liệu UIT-ViQuAD 2.0

Split `validation` từ Hugging Face được **giữ lại làm Public Test**, **không** dùng để lựa chọn hyperparameters. Dev được tách theo **title của bài viết** từ split `train` với seed `42`; danh sách được lưu tại `data/splits/viquad_dev_titles.txt`.

Các số liệu sau lấy từ dữ liệu đã xử lý và báo cáo EDA:

| Chỉ số | Train | Dev nội bộ | Public Test (chưa dùng để tuning) |
|---|---:|---:|---:|
| Số câu hỏi | 25.686 | 2.768 | 3.814 |
| Số title | 125 | 13 | 19 |
| Số context | 3.711 | 390 | 557 |
| Câu có đáp án | 17.352 | 1.886 | 2.653 |
| Câu không có đáp án | 8.334 | 882 | 1.161 |
| Số QA features sạch | 29.333 | 3.155 | Chưa tạo cho báo cáo |

**Luồng tiền xử lý:** VnCoreNLP tách từ → PhoBERT tokenizer → sliding window (`max_len=256`, `doc_stride=128`) → lưu ánh xạ về vị trí ký tự trong context gốc.

- Chế độ `mode=train` tính nhãn huấn luyện. Khi đánh giá Dev dùng `mode=eval`, không điều chỉnh token hoặc ranh giới window theo Gold answer.

### 4.2. Kết quả huấn luyện PhoBERT — Dev 

Ba cấu hình đều sử dụng **seed 12**, **3 epoch**, cùng bộ **ViQuAD Clean Dev**. Các số dưới đây được tính bằng **n-best decoding và no-answer threshold tối ưu trên Dev**; cần phân biệt với Dev F1 ở ngưỡng cố định trong vòng lặp huấn luyện. EM/F1 tính theo phần trăm.

| Learning rate | Epoch tốt nhất | Dev EM (%) | Dev F1 (%) | Ngưỡng null đã tune |
|---|---:|---:|---:|---:|
| `2e-5` | 3 | 52,42 | 66,39 | -1,483816 |
| `3e-5` | 3 | 53,76 | 67,93 | -0,657978 |
| `5e-5` | 3 | **55,78** | **69,94** | -1,489326 |

Cấu hình tốt nhất theo F1 là `5e-5` (epoch 3). 

Nguồn đối chiếu là bộ evidence ngoài repository `phobert_3lr_clean_dev_EVIDENCE.zip` (không commit dữ liệu lớn hoặc checkpoint). Bản tóm tắt nhỏ nằm tại `docs/results/qa_week3_A_training_summary.md`.

## 5. Hướng dẫn chạy lại pipeline QA

Cần Python 3.12, Transformers, `datasets`, Java và tài nguyên VnCoreNLP đặt trong `data/raw/vncorenlp/`. `requirements-ci.txt` chỉ phục vụ unit tests với các thư viện tối thiểu


```bash
# Tải/chuẩn bị dữ liệu ViQuAD; giữ lại title-based split đã có.
# Không dùng --overwrite nếu không chủ động tạo lại dữ liệu.
python scripts/prepare_viquad_qa.py

# Tạo QA features cho Train và Dev nội bộ bằng VnCoreNLP.
# Không tiền xử lý Public Test trong Tuần 1–3.
python scripts/preprocess_qa.py --splits train dev

# Ví dụ train PhoBERT, LR 3e-5 (không bảo đảm tái hiện byte-for-byte lịch sử).
python -m src.train_qa --model_name vinai/phobert-base --epochs 3 --learning_rate 3e-5 --lr_label 3e-5 --seed 12 --output_dir outputs/phobert_qa_lr3e-5

# Chỉ đánh giá trên Dev nội bộ với best_model đã lưu.
python scripts/eval_qa_nbest.py --split dev --checkpoint outputs/phobert_qa_lr3e-5/best_model --features data/processed/qa_features/dev.jsonl --gold data/processed/viquad/dev.jsonl --output-dir outputs/qa_nbest_clean_dev
```

Các thí nghiệm lịch sử được ghi với commit gốc `b202d3dc9fb4bcdd02131d213d4b1b8d26f1dde8`; sau đó mã nguồn có các thay đổi sửa review/CI. Chỉ tuning threshold trên **Dev nội bộ**. Theo plan, **Public Test chỉ được đánh giá một lần trong Tuần 5**, không dùng để tuning.
