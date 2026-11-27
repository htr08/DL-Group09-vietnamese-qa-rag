# Đánh giá Truy hồi Lệch ngôn ngữ — Miền phụ Samsung Galaxy S24

Tài liệu này ghi nhận kết quả đánh giá năng lực truy hồi liên ngôn ngữ (Cross-lingual Retrieval: Truy vấn tiếng Việt -> Kho văn bản tiếng Anh) trên miền phụ Samsung Galaxy S24 theo cam kết trong Proposal & Project Plan (§3.1, §4.3 T4) và xử lý đầy đủ các phản biện của Reviewer (PR #27).

---

## 1. Thông tin thí nghiệm và Quy trình tái lập

- **Tập dữ liệu:** `data/gold/samsung_queries.jsonl` (27 câu hỏi tiếng Việt, nhãn đã kiểm tra chéo `reviewed`).
- **Kho văn bản đánh chỉ mục:** 2 tài liệu thông số kỹ thuật PDF tiếng Anh của Samsung Galaxy S24:
  1. `samsung_galaxy_s24_128gb_device_specs.pdf` (5 trang -> 5 chunks).
  2. `samsung_galaxy_s24_specifications.pdf` (3 trang -> 4 chunks sau khi áp dụng cửa sổ trượt 200 từ, bước nhảy 180 từ).
  - **Quy mô kho:** **9 chunks** (chuẩn hóa tái dựng sạch bằng `scripts/build_index.py --corpus samsung`).
- **Mô hình Dense:** `BAAI/bge-m3` (chạy trên CPU / PyTorch).
- **Thuật toán Hybrid:** Reciprocal Rank Fusion (RRF), thử nghiệm cả k=15 và k=60.
- **Đánh giá đa chunk (`relevant_ids`):** 8/27 câu (S09, S10, S11, S12, S13, S25, S26, S27) có thông tin xuất hiện đồng thời ở cả hai tài liệu hoặc nằm trong vùng giao thoa của 2 cửa sổ trượt. Pipeline đánh giá tính trúng đích (Hit) khi truy hồi được bất kỳ chunk nào trong danh sách `relevant_ids`.

---

## 2. Kết quả Tổng thể (N = 27)

Bảng kết quả chạy trên kho 9 chunks tái dựng sạch với nhãn đa chunk `relevant_ids`:

| Phương pháp | R@1 | R@3 | R@5 | R@10 | MRR@10 | Độ trễ (ms) | N |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BM25** | 0.5185 | 0.5926 | 0.9259 | 1.0000 | 0.6370 | 0.1 | 27 |
| **Dense (BGE-M3)** | **0.5556** | **0.8889** | **0.9630** | 1.0000 | **0.7185** | 132.2 | 27 |
| **Hybrid RRF (k=60)** | 0.5185 | **0.8889** | 0.9259 | 1.0000 | 0.7121 | 136.9 | 27 |
| **Hybrid RRF (k=15)** | 0.5185 | **0.8889** | 0.9259 | 1.0000 | 0.7121 | 128.8 | 27 |

> **Về hiện tượng bão hòa và ngưỡng ngẫu nhiên trên kho 9 chunks:**
> - Do toàn bộ kho chỉ có 9 chunk, **Recall@10 = 1.000 (100%)** với mọi phương pháp một cách hiển nhiên (bão hòa hoàn toàn).
> - Ngưỡng chọn ngẫu nhiên lý thuyết:
>   Random R@1 ≈ 1/9 ≈ 11.1%, Random R@5 ≈ 5/9 ≈ 55.6%, Random R@10 = 100.0%
> - Vì vậy, **loại bỏ R@10 khỏi kết luận so sánh**. Hai chỉ số phân biệt chính có ý nghĩa đo lường là **R@1** và **MRR@10**.

### Khoảng tin cậy Bootstrap (Paired Bootstrap CI 95%, 10.000 lần lấy mẫu lại)

| So sánh cặp | Chỉ số | Ước lượng | 95% CI | Ý nghĩa thống kê? |
| :--- | :---: | :---: | :---: | :---: |
| **Hybrid (k=60) vs Dense** | MRR@10 | -0.0064 | [-0.1336, +0.1273] | Không (0 in CI) |
| | R@1 | -0.0370 | [-0.2593, +0.2222] | Không (0 in CI) |
| | R@5 | -0.0370 | [-0.1111, +0.0000] | Không |
| **Hybrid (k=60) vs BM25** | MRR@10 | +0.0750 | [-0.0198, +0.1654] | Không (0 in CI) |
| | R@1 | +0.0000 | [-0.1481, +0.1481] | Không (0 in CI) |
| | R@5 | +0.0000 | [+0.0000, +0.0000] | Không |

---

## 3. Phân tầng kết quả theo mức độ trùng lặp từ vựng (Ablation Study)

Để kiểm chứng nghiêm túc giả thuyết §3.5b của Đề cương ("BM25 suy giảm mạnh trong bài toán lệch ngôn ngữ so với Dense"), 27 câu hỏi được phân tầng thành 2 tập con:

### Nhóm 1: Thuần Việt & Chỉ trùng tên máy (n = 12)
*(Gồm 4 câu không trùng bất kỳ từ tiếng Anh nào: S09, S16, S24, S25; và 8 câu chỉ chứa tên định danh thương hiệu `Samsung/Galaxy/S24`: S01, S03, S04, S06, S07, S14, S15, S19).*

| Phương pháp | R@1 | R@3 | R@5 | R@10 | MRR@10 | Độ trễ (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BM25** | **0.0000** (0/12) | 0.0833 (1/12) | 0.8333 | 1.0000 | **0.2250** | 0.2 |
| **Dense (BGE-M3)** | **0.5833** (7/12) | **0.8333** | **0.9167** | 1.0000 | **0.7250** | 116.0 |
| **Hybrid RRF (k=60)** | 0.0833 (1/12) | 0.7500 | 0.8333 | 1.0000 | 0.4494 | 117.7 |

### Nhóm 2: Chứa từ mượn kỹ thuật Latinh (n = 15)
*(Chứa ít nhất 1 thuật ngữ kỹ thuật viết bằng ký tự Latinh ngoài tên máy: `nits`, `UFS`, `GPU`, `SAR`, `AMOLED`, `FHD`, `DeX`, `camera`, `zoom`, `3.5mm`...: S02, S05, S08, S10, S11, S12, S13, S17, S18, S20, S21, S22, S23, S26, S27).*

| Phương pháp | R@1 | R@3 | R@5 | R@10 | MRR@10 | Độ trễ (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BM25** | **0.9333** (14/15) | **1.0000** | **1.0000** | 1.0000 | **0.9667** | 0.2 |
| **Dense (BGE-M3)** | 0.5333 (8/15) | 0.9333 | **1.0000** | 1.0000 | 0.7133 | 121.5 |
| **Hybrid RRF (k=60)** | 0.8667 (13/15) | **1.0000** | **1.0000** | 1.0000 | 0.9222 | 117.6 |

---

## 4. Nhận định Khoa học và Thảo luận

1. **Giả thuyết §3.5b được ủng hộ mạnh mẽ trên tập câu hỏi thuần Việt:**
   - Khi câu hỏi hoàn toàn bằng tiếng Việt và không chứa các từ khóa kỹ thuật Latinh, **BM25 thất bại hoàn toàn ở Rank 1 (R@1 = 0%, MRR@10 = 0.225)** do không thể bắc cầu qua sự sai khác ngôn ngữ giữa câu hỏi tiếng Việt và tài liệu tiếng Anh.
   - Ngược lại, **Dense BGE-M3 thể hiện năng lực hiểu liên ngôn ngữ xuất sắc (R@1 = 58.33%, MRR@10 = 0.725)**, vượt trội cách biệt so với BM25.
   - Ở nhóm này, việc kết hợp Hybrid thậm chí làm giảm hiệu năng của Dense (từ 0.725 xuống 0.449) do BM25 xếp ngẫu nhiên các chunk chỉ chứa tên `Samsung Galaxy` lên đầu bảng xếp hạng.

2. **Giả thuyết §3.5b KHÔNG được ủng hộ trên nhóm câu hỏi có từ mượn kỹ thuật:**
   - Khi câu hỏi chứa từ mượn tiếng Anh đặc thù (ví dụ: `AMOLED`, `UFS`, `DeX`, `nits`, `GPU`), **BM25 đạt hiệu năng gần như tuyệt đối (R@1 = 93.33%, MRR@10 = 0.9667)**, vượt xa Dense (R@1 = 53.33%).
   - Nguyên nhân: Các thuật ngữ kỹ thuật này có tần suất xuất hiện rất hiếm trong kho (IDF cao), giúp BM25 ghim chính xác chunk mục tiêu ở Rank 1 mà không cần hiểu ngữ nghĩa.

3. **Về thứ hạng phương pháp trên toàn bộ tập Samsung (N=27):**
   - **Thứ hạng thực tế:** Dense (MRR 0.7185) > Hybrid (MRR 0.7121) > BM25 (MRR 0.6370).
   - **Không kết luận "Hybrid tốt nhất"**: Trên tập dữ liệu này, Dense là phương pháp đạt điểm cao nhất, và Hybrid không đem lại cải thiện so với Dense đơn lẻ.
   - Do kích thước mẫu khiêm tốn (N=27), tất cả các khoảng tin cậy Bootstrap 95% giữa các phương pháp đều chứa 0 (không có ý nghĩa thống kê ở mức alpha=0.05). Nhóm ghi nhận đây là kết quả định tính phản ánh tính chất của dữ liệu thực tế hơn là một khẳng định thống kê tuyệt đối.
