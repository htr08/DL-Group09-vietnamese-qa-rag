# Tuần 3 — A: PhoBERT trên tập ViQuAD Dev nội bộ sạch

**Phạm vi và nguồn bằng chứng.** Tài liệu tổng hợp kết quả thực nghiệm lịch sử từ `phobert_3lr_clean_dev_EVIDENCE.zip` (15 tệp, gồm lịch sử huấn luyện, tóm tắt huấn luyện, manifest từng lần chạy và kết quả n-best). Gói bằng chứng này được lưu **ngoài repository**; tài liệu không bao gồm trọng số mô hình hoặc bộ dữ liệu huấn luyện.

- **Dữ liệu:** ViQuAD Dev nội bộ tách theo `title`, gồm **2.768 câu hỏi**, 13 title và 3.155 features; 1.886 câu có đáp án / 882 câu không có đáp án. Quá trình tiền xử lý sạch không bỏ qua câu nào; giữ lại 29 trường hợp không căn chỉnh được nhãn (`unalignable`) để đánh giá.
- **Mô hình:** `vinai/phobert-base`; 3 epoch; `seed=12`; ba learning rate: `2e-5`, `3e-5`, `5e-5`.
- **Toàn vẹn dữ liệu:** Train và Dev không trùng title hoặc context. Dev sử dụng `mode=eval`; không điều chỉnh đầu vào hoặc cửa sổ trượt dựa trên đáp án Gold.
- **SHA-256** của tệp Dev features sạch theo manifest đã cung cấp: `b27046196cd92f0ede419737b155f7f36a6bb1f2c54c563b3bafa31c82d8d3c3`.
- **Dấu vân tay mã nguồn (source fingerprint)** trong manifest huấn luyện: `3e04aedb82c69908f19b1cdcda6499ab51ac497b656ba7826ef90b0b6bd82deb`.

## 1. Kết quả — N-best trên Dev với ngưỡng không có đáp án được tối ưu trên Dev

| Learning rate | Epoch tốt nhất | Dev F1 trong vòng lặp huấn luyện (%) | Dev EM sau tối ưu (%) | Dev F1 sau tối ưu (%) | Ngưỡng null |
|---|---:|---:|---:|---:|---:|
| `2e-5` | 3 | 65,59 | 52,42 | 66,39 | -1,483816 |
| `3e-5` | 3 | 67,35 | 53,76 | 67,93 | -0,657978 |
| `5e-5` | 3 | 69,53 | **55,78** | **69,94** | -1,489326 |

**Cấu hình tốt nhất trong phép thử ba LR của Tuần 3:** learning rate `5e-5`, epoch 3. Đây là kết quả **lựa chọn trên Dev**, không phải Public Test. Không được đánh đồng F1 trong vòng lặp huấn luyện (ở ngưỡng cố định) với F1 sau hậu xử lý và tối ưu ngưỡng.

| Learning rate | Dev F1 — câu có đáp án (%) | Dev EM/F1 — câu không có đáp án (%) |
|---|---:|---:|
| `2e-5` | 67,16 | 64,74 |
| `3e-5` | **70,33** | 62,81 |
| `5e-5` | 68,08 | **73,92** |

## 2. Khả năng tái lập và giới hạn

- Commit mã nguồn dùng trong lần huấn luyện gốc theo manifest: `b202d3dc9fb4bcdd02131d213d4b1b8d26f1dde8`. Các bản sửa CI/review sau đó đã thay đổi mã nguồn. Muốn tái lập chính xác kết quả lịch sử cần cố định commit, phiên bản thư viện và xác minh hash của đầu vào/checkpoint.
- Gói evidence chỉ chứa metrics và manifest, **không chứa trọng số mô hình đã huấn luyện**. Riêng tài liệu Markdown này không chứng minh được việc nạp lại checkpoint hoặc chạy inference thành công.
- **Gói evidence lịch sử của ba learning rate không chứa bootstrap CI.** Tuy nhiên, lần đánh giá lại checkpoint và nhóm 3 seed Tuần 4 đã có bootstrap CI riêng; xem mục 3 và tài liệu Tuần 4. Không áp CI mới cho các run cũ.
- Kiểm toán rò rỉ Gold lịch sử: quá trình tiền xử lý Dev cũ làm thay đổi features của **29/2.768 câu**, được ghi trong `docs/results/gold_leakage_audit/`. Đây **không phải** chênh lệch F1 đã đo được.
- B1 miền ứng dụng **đã có kết quả đánh giá kỹ thuật trên Domain Dev với index 216 chunks** (EM **70,00%**, F1 **86,73%**), trên bộ Gold chính thức của nhóm. Không có kết quả Domain Test trong tài liệu này.
- **Không đánh giá hoặc tối ưu tham số trên Public Test** cho các kết quả trên. Kế hoạch dành một lần đánh giá Test cuối cùng cho Tuần 5.

Xem thêm `docs/qa_week2_A_evidence.md`, `docs/qa_week2_A_eda_reproduced.md` và `data/splits/viquad_split_meta.json` để kiểm tra dữ liệu.

## 3. Đánh giá bổ sung — 25/11/2026 (chỉ trên Dev)

### 3.1. Đánh giá lại checkpoint lịch sử (`5e-5`, seed 12)

| Chỉ số | Kết quả lịch sử Tuần 3 | Đánh giá lại checkpoint lịch sử | Chênh lệch |
|---|---:|---:|---:|
| ViQuAD Clean Dev EM (%) | 55,78 | 55,24 | −0,54 điểm phần trăm |
| ViQuAD Clean Dev F1 (%) | 69,94 | 70,10 | +0,16 điểm phần trăm |
| Số câu Dev | 2.768 | 2.768 | 0 |
| Ngưỡng null tối ưu trên Dev | −1,489326 | ≈ −0,644425 | Khác nhau |

Lần đánh giá lại sử dụng checkpoint `phobert_qa_lr5e-5_seed12/best_model`, bộ features `data/processed/qa_features_week4_eval/dev.jsonl` và `scripts/eval_qa_nbest.py`. Kết quả chi tiết tại `outputs/week3_original_checkpoint_eval/summary.json` là **artifact lưu ngoài repository/trên máy thực nghiệm**, chưa được commit vào repository này.

**Chưa xác định được nguyên nhân chênh lệch** giữa hai lần đánh giá. Khác biệt về features, decoder hoặc phiên bản mã nguồn mới chỉ là **giả thuyết**, chưa phải nguyên nhân đã được chứng minh.

Kết quả bootstrap cho lần đánh giá lại: **EM 95% CI [53,36%; 57,08%]**, **F1 95% CI [68,51%; 71,64%]**; 3.000 lần lấy mẫu lại, giữ cố định ngưỡng đã chọn.

### 3.2. B1 — Corpus hiện tại có thể tái lập

| Thành phần | Kết quả |
|---|---|
| Corpus miền ứng dụng | **216 chunks** (BM25 index khớp với metadata của chunks) |
| Mô hình đọc hiểu | Checkpoint PhoBERT lịch sử `5e-5`, seed 12 |
| Domain Dev | 30 câu hỏi: 27 trong phạm vi, 3 ngoài phạm vi |
| Truy hồi | BM25 Top-3 |
| Ngưỡng cố định = 0 | **EM 70,00%; F1 86,73%** |
| Ngưỡng tối ưu trên Domain Dev = −0,477434 | **EM 70,00%; F1 86,73%** (không cải thiện) |
| Bằng chứng | `docs/results/b1_domain_dev/summary.json`; có thông tin hash để truy vết nguồn |

Các lần chạy B1 với **index khác 216 chunks** không phải baseline của mã nguồn hiện tại; chỉ dùng kết quả B1 216 chunks đã xác minh. Kết quả hiện tại vẫn chỉ được đánh giá trên **Domain Dev**. Không dùng Domain Test để tối ưu tham số.

### 3.3. Bổ sung kết quả Tuần 4

Các bằng chứng Tuần 4 (đều trên Dev) cho thấy:

- **PhoBERT pipeline mới — 3 seed:** F1 trung bình **69,71 ± 0,35%**.
- **PhoBERT Legacy Recipe — 3 seed có kiểm soát:** F1 trung bình **70,52 ± 0,83%**.
- **XLM-R với subword decoder:** Dev F1 **61,82%**.
- **PhoBERT NoSeg v3:** Dev F1 **43,04%**.

Các cấu hình trên sử dụng **training recipe/decoder khác nhau**, vì vậy không được diễn giải chúng như những phép so sánh nhân quả chỉ thay đổi một yếu tố. Chi tiết xem `docs/results/qa_week4_A_progress.md`.

**Trạng thái lựa chọn mô hình:** checkpoint seed 42 có F1 cao nhất **trong nhóm pipeline mới**; Legacy Recipe seed 12 có F1 71,43% trong **nhóm recipe riêng**. Chưa có biên bản so sánh/freeze cuối cùng; B1 vẫn dùng checkpoint seed 12 lịch sử. Xem `docs/results/qa_results_current_2026-11-25.md`.
