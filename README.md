# Vietnamese QA: RAG mã nguồn mở và đọc hiểu trích xuất

Đồ án môn Học sâu, Nhóm 9, Đề tài 10. Hệ thống thay các thành phần dùng API thương mại của một hệ thống RAG trước đó (Gemini embedding và Gemini 2.5 Flash) bằng BGE-M3 và Qwen2.5-7B-Instruct, đồng thời bổ sung mô-đun đọc hiểu trích xuất (PhoBERT, so sánh với XLM-R) được đánh giá trên UIT-ViQuAD 2.0 và trên tập 100 câu hỏi miền ứng dụng do nhóm tự gán nhãn.

Mã của hệ thống nền (AI Marketing Assistant) không được sao chép vào repository này; liên kết tới repository gốc nằm trong `docs/reuse_vs_new.md`, tài liệu này cũng là nguồn cho bảng "tự cài đặt so với dùng sẵn".

## 1. Cấu trúc thư mục

| Đường dẫn | Mục đích |
|---|---|
| `src/` | Mã nguồn tiền xử lý QA, huấn luyện, truy hồi, sinh câu trả lời, metrics và pipeline |
| `scripts/` | Các script tải dữ liệu, xây dựng index và đánh giá |
| `tests/` | Unit tests; CI chạy khi tạo/cập nhật Pull Request và khi push lên `main` |
| `notebooks/` | EDA, phân tích và Kaggle smoke test (`00_kaggle_qwen_smoke_test.ipynb`) |
| `data/raw/` | Dữ liệu tải về; được loại khỏi Git |
| `data/domain/` | Tài liệu miền ứng dụng: Quy chế đào tạo ĐHQGHN 2014 (miền chính) và Samsung Galaxy S24 (nghiên cứu truy hồi liên ngôn ngữ) |
| `data/gold/` | Vị trí lưu nhãn Dev/Test miền ứng dụng sau khi nhóm chốt; các file trong bản source đã rà soát còn là placeholder, **không phải Gold đã thống nhất** |
| `data/splits/` | Danh sách chia dữ liệu và seed |
| `kaggle/smoke_test_dataset/` | Bộ dữ liệu nhỏ cho Kaggle smoke test |
| `app/` | Ứng dụng demo |
| `docs/` | Kế hoạch, Charter, khai báo dùng AI và tài liệu phân biệt phần tái sử dụng/phần tự cài đặt |
| `report/` | Báo cáo LaTeX theo mẫu ACL; hiện có `proposal/`, các thư mục `milestone/` và `final/` theo kế hoạch chưa được tạo |

## 2. Cài đặt môi trường

Yêu cầu **Python 3.12**. Để tách từ bằng VnCoreNLP, cần cài **JDK (Java 8 trở lên)** và cấu hình Java trong `PATH`.

```bash
python -m venv venv
# Windows:      venv\Scripts\activate
# macOS/Linux:  source venv/bin/activate
pip install -r requirements.txt
```

`torch` không được cố định phiên bản trong `requirements.txt`; cần cài bản PyTorch phù hợp với hệ điều hành, CPU/GPU và CUDA của máy (xem chú thích đầu file yêu cầu thư viện).

### 2.1. Theo dõi thí nghiệm (Weights & Biases)

Mọi lần chạy tạo số liệu cho báo cáo (huấn luyện, đánh giá, dựng index) được ghi vào project W&B dùng chung: team `nhom09-vi-qa-rag`, project `vi-qa-rag`. Tên run theo dạng `<vai trò>_<mô hình>_<cấu hình>_seed<k>`.

```bash
pip install wandb==0.31.0
wandb login            # dùng khóa cá nhân từ https://wandb.ai/authorize; không commit khóa
```

Trên Kaggle, khóa được lưu thành secret `WANDB_API_KEY` (Add-ons, Secrets) và đặt vào biến môi trường trong notebook:

```python
import os
from kaggle_secrets import UserSecretsClient
os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
```

`src/tracking.py` cung cấp `init_run`, giữ entity, project và commit hash thống nhất giữa các lần chạy:

```python
from src.tracking import init_run
run = init_run("A_phobert_lr3e-5_seed42", job_type="train", config={"lr": 3e-5})
```

`WANDB_MODE=disabled` tắt ghi log (CI, kiểm tra nhanh trên máy cục bộ).

### 2.2. Kiểm thử

```bash
pip install -r requirements-ci.txt
pytest
```

## 3. Tiến độ của A — Extractive QA 

### 3.1. Công việc đã thực hiện và còn lại

| Tuần | Nhiệm vụ của A | Tình trạng tại ngày 22/11/2026 |
|---|---|---|
| 1 | Chia ViQuAD theo tiêu đề bài viết (title), chống trùng Train/Dev | **Hoàn thành:** seed `42`, 13 title Dev, 125 title Train, 0 context trùng; xem `data/splits/viquad_split_meta.json` |
| 1 | Kiểm tra môi trường tách từ và tokenizer | Smoke test VnCoreNLP + PhoBERT đạt 10/10 mẫu; xem `docs/qa_week2_A_eda_reproduced.md` |
| 2 | Tải ViQuAD, EDA, EM/F1, sliding window, offsets, lớp QA head | **Đã triển khai và có tests**; xem `notebooks/01_eda_viquad_A.ipynb` và `docs/qa_week2_A_evidence.md` |
| 2 | Huấn luyện PhoBERT ban đầu | **Đã huấn luyện**; số liệu lấy từ bộ evidence ngoài repository (mục 4.2) |
| 2 | Viết câu hỏi và gán nhãn miền ứng dụng | A viết 33 câu và gán nhãn 33 câu của B; các tệp nháp và bản đối soát đã được bổ sung từ `main` trong `data/gold/drafts/`. |
| 2 | B1 sơ bộ (BM25 + PhoBERT) | Đã có `scripts/run_b1_domain_dev.py`; kết quả Dev 30 câu, BM25 Top-3, corpus 213 chunks: EM 66,67%; F1 84,89%. |
| 3 | Tuning PhoBERT với LR `2e-5`, `3e-5`, `5e-5`; n-best và no-answer threshold | **Đã thực hiện**; kết quả ở mục 4.2 |
| 3 | Chốt B1, phân tích lỗi và viết phần QA cho Milestone | Đã chạy B1 trên Domain Dev; có script phân tích lỗi. Milestone chung của nhóm chưa được xác minh trong repository. |

Bộ Gold gồm 30 câu Dev và 70 câu Test đã có dữ liệu trên main. Việc đối chiếu mức độ đồng thuận gán nhãn và biên bản chốt nhãn cần được xác minh riêng.

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

- Chế độ `mode=train` tính nhãn huấn luyện. Khi đánh giá Dev dùng `mode=eval`, không điều chỉnh token hoặc ranh giới window theo Gold answer. Mỗi feature lưu `preprocessing_mode`; `scripts/eval_qa_nbest.py` và vòng lặp Dev của `src/train_qa.py` từ chối feature có giá trị khác `eval`/`predict`, kể cả feature tạo trước khi trường này tồn tại.

### 4.2. Kết quả huấn luyện PhoBERT — Dev 

Ba cấu hình đều sử dụng **seed 12**, **3 epoch**, cùng bộ **ViQuAD Clean Dev**. Các số dưới đây được tính bằng **n-best decoding và no-answer threshold tối ưu trên Dev**; cần phân biệt với Dev F1 ở ngưỡng cố định trong vòng lặp huấn luyện. EM/F1 tính theo phần trăm.

| Learning rate | Epoch tốt nhất | Dev EM (%) | Dev F1 (%) | Ngưỡng null đã tune |
|---|---:|---:|---:|---:|
| `2e-5` | 3 | 52,42 | 66,39 | -1,483816 |
| `3e-5` | 3 | 53,76 | 67,93 | -0,657978 |
| `5e-5` | 3 | **55,78** | **69,94** | -1,489326 |

Cấu hình tốt nhất theo F1 là `5e-5` (epoch 3). 

Nguồn đối chiếu là bộ evidence ngoài repository `phobert_3lr_clean_dev_EVIDENCE.zip` (không commit dữ liệu lớn hoặc checkpoint). Bản tóm tắt nhỏ nằm tại `docs/results/qa_week3_A_training_summary.md`.

## 5. Quy trình chạy lại pipeline QA

Cần Python 3.12, Transformers, `datasets`, Java và tài nguyên VnCoreNLP đặt trong `data/raw/vncorenlp/`. `requirements-ci.txt` chỉ phục vụ unit tests với các thư viện tối thiểu.


```bash
# Tải ViQuAD, tách Train/Dev/Public Test theo title và ghi vào data/raw/viquad/.
# Khi chỉ một phần tệp đã tồn tại (bản clone mới chỉ có data/splits/viquad_dev_titles.txt),
# script ném FileExistsError và cần --overwrite. Với --overwrite, danh sách title Dev
# trùng tệp đã commit (git diff data/splits rỗng).
# scripts/download_data.py ghi data/raw/viquad_*.jsonl cho phần truy hồi
# (build_index.py, run_eval.py); luồng QA dùng prepare_viquad_qa.py.
python scripts/prepare_viquad_qa.py

# Sửa answer_start lệch: data/raw/viquad -> data/processed/viquad; data/raw không bị ghi đè.
python scripts/fix_answer_start.py

# Tạo QA features cho Train và Dev nội bộ bằng VnCoreNLP.
# Không tiền xử lý Public Test trong Tuần 1–3.
python scripts/preprocess_qa.py --splits train dev

# Ví dụ train PhoBERT, LR 3e-5 (không bảo đảm tái hiện byte-for-byte lịch sử).
python -m src.train_qa --model_name vinai/phobert-base --epochs 3 --learning_rate 3e-5 --lr_label 3e-5 --seed 12 --output_dir outputs/phobert_qa_lr3e-5

# Chỉ đánh giá trên Dev nội bộ với best_model đã lưu.
python scripts/eval_qa_nbest.py --split dev --checkpoint outputs/phobert_qa_lr3e-5/best_model --features data/processed/qa_features/dev.jsonl --gold data/processed/viquad/dev.jsonl --output-dir outputs/qa_nbest_clean_dev
```

Các thí nghiệm lịch sử được ghi với commit gốc `b202d3dc9fb4bcdd02131d213d4b1b8d26f1dde8`; sau đó mã nguồn có các thay đổi sửa review/CI. Chỉ tuning threshold trên **Dev nội bộ**. Theo plan, **Public Test chỉ được đánh giá một lần trong Tuần 5**, không dùng để tuning.
