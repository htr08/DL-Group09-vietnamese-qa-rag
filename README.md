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
| `data/gold/` | Tập câu hỏi miền ứng dụng đã có **30 Dev / 70 Test**; bản Gold chính thức của nhóm (nhãn tác giả, nhãn độc lập và resolution nằm trong `drafts/`) |
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

Theo quy định, mọi lần chạy tạo số liệu cho báo cáo (huấn luyện, đánh giá, dựng index) phải được ghi vào project W&B dùng chung: team `nhom09-vi-qa-rag`, project `vi-qa-rag`. Tên run theo dạng `<vai trò>_<mô hình>_<cấu hình>_seed<k>`.

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

| Tuần | Nhiệm vụ của A | Tình trạng cập nhật V2 ngày 27/11/2026 |
|---|---|---|
| 1 | Chia ViQuAD theo tiêu đề bài viết (title), chống trùng Train/Dev | **Hoàn thành:** seed `42`, 13 title Dev, 125 title Train, 0 context trùng; xem `data/splits/viquad_split_meta.json` |
| 1 | Kiểm tra môi trường tách từ và tokenizer | Smoke test VnCoreNLP + PhoBERT đạt 10/10 mẫu; xem `docs/qa_week2_A_eda_reproduced.md` |
| 2 | Tải ViQuAD, EDA, EM/F1, sliding window, offsets, lớp QA head | **Đã triển khai và có tests**; xem `notebooks/01_eda_viquad_A.ipynb` và `docs/qa_week2_A_evidence.md` |
| 2 | Huấn luyện PhoBERT ban đầu | **Đã huấn luyện**; số liệu lấy từ bộ evidence ngoài repository (mục 4.2) |
| 2 | Viết câu hỏi và gán nhãn miền ứng dụng | A viết 33 câu và gán nhãn 33 câu của B. Tập **30 Dev / 70 Test** là bản Gold chính thức, đã có trong repository |
| 2–3 | B1 (BM25 + PhoBERT) — Domain Dev | **V2: 216 chunks, Top-3, 30 câu; EM 63.33%, F1 78.98% tại threshold 0.** Tuned: EM 66.67%, F1 85.28%. |
| 3 | Tuning PhoBERT với LR `2e-5`, `3e-5`, `5e-5`; n-best và no-answer threshold | **Đã thực hiện**; kết quả ở mục 4.2 |
| 3–4 | Phân tích lỗi, ablation và evidence | **V2: 66 mẫu có phân loại lỗi và giải thích; 18 mẫu cần xem xét Gold.** Có XLM-R, NoSeg, n-best và 3 seed. Experiment freeze chưa chốt. |

**Gold miền ứng dụng:** `data/gold/gold_dev.jsonl` (**30 câu**) và `data/gold/gold_test.jsonl` (**70 câu**) là bản chính thức của nhóm. Mỗi câu có nhãn của tác giả và nhãn độc lập của người gán nhãn; bất đồng duy nhất (A04) được xử lý bằng resolution `rewritten` trong `data/gold/drafts/resolutions.jsonl`. Phép chia Dev/Test dùng seed 1102 và không để Khoản nào xuất hiện ở cả hai tập (`data/splits/gold_split.json`).

## 4. Dữ liệu và kết quả PhoBERT

### Kết quả QA V2 — 26/11/2026

| Thí nghiệm | EM (%) | F1 (%) |
|---|---:|---:|
| PhoBERT LR 5e-5, seed 12 | 54.91 | 70.33 |
| XLM-R | 34.57 | 41.73 |
| PhoBERT NoSeg | 36.13 | 43.81 |
| B1 Domain Dev, threshold 0 | 63.33 | 78.98 |
| B1 Domain Dev, tuned | 66.67 | 85.28 |

PhoBERT 3 seed (12/42/123):
EM 55.43 +/- 0.56%;
F1 70.01 +/- 0.40%.

W&B evaluation: [j212ro7f](https://forge.coreweave.com/wandb/nhom09-vi-qa-rag/vi-qa-rag/runs/j212ro7f).

Các kết quả đều trên Dev, không dùng Test để tuning.

Chi tiết: [QA V2](docs/results/qa_v2/README.md).

Các kết quả tại mục 4.2-4.4 là evidence lịch sử.

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

### 4.2. Kết quả huấn luyện PhoBERT Tuần 3 (lịch sử) — Dev

Ba cấu hình đều sử dụng **seed 12**, **3 epoch**, cùng bộ **ViQuAD Clean Dev**. Các số dưới đây được tính bằng **n-best decoding và no-answer threshold tối ưu trên Dev**; cần phân biệt với Dev F1 ở ngưỡng cố định trong vòng lặp huấn luyện. EM/F1 tính theo phần trăm.

| Learning rate | Epoch tốt nhất | Dev EM (%) | Dev F1 (%) | Ngưỡng null đã tune |
|---|---:|---:|---:|---:|
| `2e-5` | 3 | 52,42 | 66,39 | -1,483816 |
| `3e-5` | 3 | 53,76 | 67,93 | -0,657978 |
| `5e-5` | 3 | **55,78** | **69,94** | -1,489326 |

Trong **nhóm ba learning rate lịch sử**, cấu hình tốt nhất theo Dev F1 là `5e-5` (epoch 3); **không phải tuyên bố đây là kết quả tốt nhất trong toàn bộ thí nghiệm Tuần 4**.

Nguồn đối chiếu là bộ evidence ngoài repository `phobert_3lr_clean_dev_EVIDENCE.zip` (không commit dữ liệu lớn hoặc checkpoint). Bản tóm tắt nhỏ nằm tại `docs/results/qa_week3_A_training_summary.md`.

### 4.3. Đánh giá lại checkpoint và B1 lịch sử — Dev (25/11/2026)

| Thí nghiệm | Tập | EM | F1 | Diễn giải |
|---|---|---:|---:|---|
| PhoBERT LR 5e-5 seed 12, đánh giá lại checkpoint lịch sử | ViQuAD Clean Dev (2.768 câu) | 55,24% | 70,10% | N-best, Dev-tuned threshold ≈ −0,64443; **chưa xác định** nguyên nhân chênh kết quả với lịch sử 55,78/69,94 |
| **B1: BM25 216 chunks + checkpoint PhoBERT lịch sử**, Top-3, threshold 0 | Domain Dev (30 câu) | **70,00%** | **86,73%** | 27 câu in-scope / 3 out-of-scope; tuned threshold −0,477434 **không cải thiện** |

- **Nguồn B1 local:** `outputs/b1_domain_dev/summary.json` (đã chạy trên máy) và bản evidence nhỏ `docs/results/b1_domain_dev/summary.json` để commit. Hash Gold, BM25, chunks và QA head nằm trong summary.
- B1 lịch sử trong mục này sử dụng kết quả từ **index 216 chunks**; các đánh giá với index không đồng nhất trước đây không được coi là baseline.
- Hai bộ Dev khác nhau; không so EM/F1 của ViQuAD với Domain như cùng benchmark.
- B1 chưa phải kết quả Test; đây là kết quả trên Domain Dev (30 câu) của bộ Gold chính thức.

### 4.4. Kết quả ablation Tuần 4 (lịch sử)

- **PhoBERT pipeline mới, 3 seed (12/21/42):** F1 lần lượt **69,86% / 69,31% / 69,96%**; mean ± std **69,71 ± 0,35%**. Bootstrap F1 95% CI của trung bình: **[68,33%; 71,16%]**.
- **PhoBERT Legacy Recipe, 3 seed:** F1 **71,43% / 69,81% / 70,32%**; mean ± std **70,52 ± 0,83%**. Đây là *training recipe khác*, không gộp kết quả với pipeline mới và không đổi checkpoint của B1.
- **XLM-R:** EM **48,48%**, F1 **61,82%** với subword decoder; **PhoBERT NoSeg GoldBoundary v3:** EM **36,31%**, F1 **43,04%**. NoSeg v3 còn thay cách căn chỉnh/preprocessing, nên **không** quy chênh lệch riêng cho việc không tách từ.
- **Phân tích lỗi V2:** [66 mẫu, 10 cột](docs/results/viquad_66_error.csv); [các trường hợp cần xem xét Gold](docs/results/qa_v2/gold_checks.md). Set 2 được giữ riêng làm evidence lịch sử.
- Bảng kết quả lịch sử ngày 25/11: [`docs/results/qa_results_current_2026-11-25.md`](docs/results/qa_results_current_2026-11-25.md). Chi tiết evidence và các hạn chế: `docs/results/qa_week4_A_progress.md`. **Chưa chạy Public Test hoặc Domain Test để tune.**

### 4.5. QA rerun V2

PhoBERT V2, LR 5e-5, seed 12: Dev EM 54.9133%, F1 70.3350%, threshold 0.2479505539; [W&B evaluation j212ro7f](https://forge.coreweave.com/wandb/nhom09-vi-qa-rag/vi-qa-rag/runs/j212ro7f).

Nguồn chuẩn mới: [QA V2](docs/results/qa_v2/README.md).
Audit: [audit_summary.json](docs/results/qa_v2/audit_summary.json).

B1 V2 threshold 0: EM 63.33%, F1 78.98%.
Các số trước mục này được giữ làm evidence lịch sử.
Chưa chốt experiment freeze.

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

### 5.1. Tái hiện B1 lịch sử trên Domain Dev (không dựng lại BM25)

```powershell
$CKPT = "$HOME\Downloads\phobert_5e5_extracted\phobert_qa_lr5e-5_seed12\best_model"
python scripts/run_b1_domain_dev.py --checkpoint "$CKPT" --top-k 3
```

B1 nạp cặp `data/index/domain/bm25.pkl` và `chunks_meta.json` của **216 chunks** đã kiểm tra khớp ID/thứ tự. Chỉ chạy **Dev 30 câu**, không chạy lại Train/Test. Cặp index và checkpoint không commit vào Git; chỉ commit script, tài liệu và JSON summary nhỏ.
