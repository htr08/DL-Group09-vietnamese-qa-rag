# Tuần 3 — A: PhoBERT trên ViQuAD Dev nội bộ đã làm sạch

> **Bản hợp nhất tài liệu của `origin/main` và nhánh QA.** Kết quả trong các bảng dưới đây là **evidence lịch sử Tuần 3**, không phải kết quả re-inference Tuần 4. Nguồn ngoài repository: `phobert_3lr_clean_dev_EVIDENCE.zip` (15 file: manifest, history, training summaries và n-best summaries). ZIP **không chứa trained weights hoặc predictions đầy đủ**.

## 1. Cấu hình huấn luyện và dữ liệu

- Backbone: `vinai/phobert-base`; **3 epoch**, `seed=12`; ba learning rates: `2e-5`, `3e-5`, `5e-5`.
- Dev nội bộ tách theo **title** từ train, **13 title**, **2.768 câu**, **3.155 features**; gồm **1.886 answerable** và **882 impossible**; **0 skipped**; **29 unalignable** vẫn giữ để đánh giá.
- Theo kiểm tra split: không overlap title/context giữa Train và Dev. Clean Dev dùng tiền xử lý suy luận không phụ thuộc Gold (`mode=eval`) để tránh Gold-driven input/window refinement.
- SHA-256 Dev features ghi trong manifest lịch sử: `b27046196cd92f0ede419737b155f7f36a6bb1f2c54c563b3bafa31c82d8d3c3`.
- Source fingerprint ghi trong training manifests: `3e04aedb82c69908f19b1cdcda6499ab51ac497b656ba7826ef90b0b6bd82deb`.
- **Lưu ý:** Hash trên thuộc tệp features lịch sử (trước khi schema thêm `preprocessing_mode`); việc `mode=eval` của pipeline thí nghiệm và metadata hiện diện trong file là hai vấn đề khác nhau. Không dùng hash đơn lẻ để chứng minh không leakage.

## 2. Kết quả Dev: n-best + ngưỡng không có đáp án tune trên Dev

| Learning rate | Best epoch | Train-loop Dev F1 (%) | Dev EM sau tune (%) | Dev F1 sau tune (%) | Null threshold |
|---|---:|---:|---:|---:|---:|
| `2e-5` | 3 | 65,59 | 52,42 | 66,39 | -1,483816 |
| `3e-5` | 3 | 67,35 | 53,76 | 67,93 | -0,657978 |
| **`5e-5`** | **3** | **69,53** | **55,78** | **69,94** | **-1,489326** |

**Cấu hình được chọn:** LR `5e-5`, epoch 3 theo **Dev F1 tổng thể**. Đây là **lựa chọn dựa trên Dev**, không phải đánh giá Public Test. F1 trong training loop tại ngưỡng của lần đánh giá đó và F1 sau decoder n-best + tune threshold là **hai phép đo khác nhau**.

### 2.1. Kết quả theo nhóm câu hỏi

| Learning rate | F1 câu có đáp án (%) | EM/F1 câu không có đáp án (%) |
|---|---:|---:|
| `2e-5` | 67,16 | 64,74 |
| `3e-5` | **70,33** | 62,81 |
| `5e-5` | 68,08 | **73,92** |

Theo Dev F1 tổng thể, `5e-5` là cấu hình tốt nhất, mặc dù F1 riêng nhóm answerable của `3e-5` cao hơn. Không chọn mô hình bằng cách chỉ nhìn một nhóm câu.

## 3. Hậu xử lý, leakage và tính tái lập

- Decoder n-best chọn các span start/end hợp lệ trên sliding windows, giới hạn chiều dài và ánh xạ về văn bản gốc; ngưỡng không có đáp án được **tối ưu trên Dev**, không trên Test.
- Audit preprocessing cũ ghi **29/2.768 câu** có input features thay đổi bởi Gold-dependent token refinement; đây **không phải** phần tăng/giảm EM hoặc F1 đã được đo. Chi tiết ở `docs/results/gold_leakage_audit/` và `docs/qa_week2_A_evidence.md`.
- **Base commit lịch sử** từ manifest: `b202d3dc9fb4bcdd02131d213d4b1b8d26f1dde8`. Các sửa đổi CI/QA sau đó thay đổi code; muốn tái lập đúng lần chạy cũ cần pin commit, dependencies và kiểm tra hash input/checkpoint.
- ZIP evidence gồm kết quả và manifest, **không có trained weights/predictions từng câu**, do đó bản báo cáo lịch sử này **không chứng minh đã reload checkpoint hay chạy lại inference**.
- **Bootstrap 95% CI của PhoBERT chính chưa được xác minh trong evidence ZIP lịch sử**, không tự tạo giá trị CI.
- W&B được ghi nhận ở project `nhom09-vi-qa-rag/vi-qa-rag`, liên hệ với commit nêu trên; URL của từng run cần được bổ sung vào báo cáo có thể kiểm tra độc lập.
- **Public Test chưa được sử dụng** để chọn cấu hình hoặc threshold trong các kết quả này; theo kế hoạch chỉ dùng ở đợt đánh giá cuối.

## 4. B1 Domain Dev — tách biệt với ViQuAD

- Baseline B1 = BM25 Top-3 + PhoBERT QA, đánh giá trên **30 câu Domain Dev** (27 in-scope, 3 out-of-scope), **không phải** 2.768 câu ViQuAD Dev.
- Người thực hiện đã chạy local với **213 chunks**: fixed threshold = `0`, **EM 66,67% / F1 84,89%**; Dev-tuned threshold ≈ `-0,700919` cho cùng EM/F1.
- **Giới hạn:** số *tuned* chọn và chấm trên chính 30 câu Dev (**in-sample**); threshold có tính cấu hình theo `top_k`, corpus và checkpoint. Việc chốt nhãn Gold cần truy vết hồ sơ đối soát trước khi tuyên bố điểm B1 cuối cùng; không có kết quả Domain Test ở đây.
- Kết quả này được ghi theo log local do người thực hiện cung cấp sau Tuần 3; **không thuộc archive PhoBERT 3 LR**. Để tái lập cần lưu cùng hash Gold/index/metadata/checkpoint và predictions tương ứng.

## 5. Đối chiếu với báo cáo khác và phần chưa xác minh

- EDA/offsets, Dev features và Gold leakage: `docs/qa_week2_A_evidence.md`, `docs/qa_week2_A_eda_reproduced.md`, `data/splits/viquad_split_meta.json`.
- PhoBERT n-best và 3 LR: `phobert_3lr_clean_dev_EVIDENCE.zip` (lưu ngoài repo).
- **Không gộp kết quả re-evaluation Tuần 4 vào bảng lịch sử Tuần 3.** Nếu cấu hình mới cho EM/F1 khác, ghi thành bảng đối chiếu riêng với hash checkpoint, preprocessing, decoder và ngưỡng.
- **Chưa xác minh** CI 95% từ predictions đầy đủ của lần chạy PhoBERT lịch sử; không coi kết quả tuning trên Dev là kết quả tổng quát hóa trên Test.
