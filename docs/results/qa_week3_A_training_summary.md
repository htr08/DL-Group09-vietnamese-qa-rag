# Tuần 3 — A: PhoBERT trên tập ViQuAD Dev nội bộ đã làm sạch

**Nguồn:** `phobert_3lr_clean_dev_EVIDENCE.zip` (15 file: manifest, history, training summary, n-best summary). Đây là **kết quả thí nghiệm lịch sử**; ZIP không có trọng số mô hình, predictions từng câu hay dữ liệu train.

## 1. Cấu hình và tính toàn vẹn dữ liệu

- Mô hình: `vinai/phobert-base`; 3 epoch; `seed=12`.
- Thử **ba learning rates**: `2e-5`, `3e-5`, `5e-5`.
- Dev nội bộ tách theo title: **2.768 câu / 13 titles / 3.155 features**.
- Dev: **1.886 answerable**, **882 impossible**; **0 skipped**, **29 unalignable** vẫn được giữ để đánh giá.
- Không trùng title/context giữa Train và Dev. Dev tạo bằng `mode=eval`, không điều chỉnh input/windows dựa trên Gold.

## 2. Kết quả Dev sau n-best và tối ưu ngưỡng không có đáp án

| Learning rate | Best epoch | Train-loop Dev F1 (%) | Dev EM (%) | Dev F1 (%) | Null threshold |
|---|---:|---:|---:|---:|---:|
| `2e-5` | 3 | 65,59 | 52,42 | 66,39 | -1,483816 |
| `3e-5` | 3 | 67,35 | 53,76 | 67,93 | -0,657978 |
| **`5e-5`** | **3** | **69,53** | **55,78** | **69,94** | **-1,489326** |

**Cấu hình được chọn:** LR `5e-5`, epoch 3 theo **Dev F1 tổng thể**. Đây là lựa chọn trên Dev, **không phải** Public Test.

| Learning rate | Answerable F1 (%) | Impossible EM/F1 (%) |
|---|---:|---:|
| `2e-5` | 67,16 | 64,74 |
| `3e-5` | **70,33** | 62,81 |
| `5e-5` | 68,08 | **73,92** |

**Giải thích:** F1 từ training loop (đánh giá ở ngưỡng mặc định) và F1 sau hậu xử lý + tối ưu threshold là **hai chỉ số khác nhau**, không được gộp.

## 3. Hậu xử lý và kiểm soát leakage

- Decoder n-best lấy candidate hợp lệ qua các sliding windows, giới hạn chiều dài đáp án và ánh xạ về văn bản gốc.
- Ngưỡng no-answer được tối ưu **trên Dev**, không dùng Test.
- Audit pipeline Dev cũ cho thấy có **29/2.768 câu** mà input features thay đổi do Gold-dependent token refinement; đây là **số mẫu bị ảnh hưởng**, **không phải** mức cải thiện F1. Xem `docs/results/gold_leakage_audit/`.
