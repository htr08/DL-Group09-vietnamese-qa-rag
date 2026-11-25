# Tuần 4 — PhoBERT: kết quả 3 seed trên ViQuAD Clean Dev

## Thiết lập

- Mô hình `vinai/phobert-base`; seed **12, 21, 42**; **3 epoch**, learning rate **5e-5**, batch size **8**, gradient accumulation **2**.
- Dữ liệu: **29.333 Train features**; **2.768 câu Dev / 3.155 Dev features**; cùng Train/Dev/Gold và source code theo kiểm tra fingerprint.
- Đánh giá: n-best **20**, max answer length **30**, cùng quy tắc tối ưu ngưỡng null trên Dev của từng seed; không sử dụng Public Test.

## Kết quả trên Dev

| Seed | EM (%) | EM 95% CI (%) | F1 (%) | F1 95% CI (%) | Null threshold |
|---|---:|---|---:|---|---:|
| 12 | 54,73 | [52,85; 56,58] | **69,86** | [68,32; 71,39] | 0,025114 |
| 21 | 54,91 | [53,03; 56,83] | **69,31** | [67,77; 70,94] | -1,359228 |
| 42 | 55,67 | [53,76; 57,55] | **69,96** | [68,43; 71,53] | -2,695232 |
| **Mean ± Std** | **55,11 ± 0,50** | — | **69,71 ± 0,35** | — | — |

**Khoảng tin cậy bootstrap 95% của trung bình 3 seed:** EM **[53,38%; 56,88%]**; F1 **[68,33%; 71,16%]**.

- **Bootstrap:** 3.000 lần resample theo câu hỏi, sử dụng predictions cố định và threshold đã chọn của từng seed; **không tune lại threshold trong bootstrap**.
- **Std:** độ lệch chuẩn *giữa ba lần huấn luyện với các seed khác nhau*; khác với CI bootstrap theo câu hỏi. CI trung bình không phản ánh toàn bộ bất định do chỉ có 3 seed huấn luyện.
- Các CI chồng lấn; không thể kết luận seed 42 tốt hơn về mặt thống kê chỉ từ bảng này.

## Kết quả tốt nhất trong nhóm pipeline mới — chưa chốt experiment freeze

| Thành phần | Giá trị |
|---|---|
| Checkpoint Dev F1 cao nhất **trong 3 seed pipeline mới** | **PhoBERT seed 42** |
| Dev EM / F1 | **55,67% / 69,96%** |
| Null threshold | `-2.6952319145202637` |
| Decoder | N-best 20, max answer length 30, offsets độc lập với Gold |
| Dữ liệu đánh giá | ViQuAD Clean Dev, 2.768 câu |
| Nguồn evidence | `qa_week4_phobert_3seed_bootstrap_ci.json`, `phobert_week4_evidence_only.zip`, `seed12_evidence_only.zip` |

**Không coi seed 42 là checkpoint final của toàn bộ nhánh QA:** nhóm **Legacy Recipe** có kết quả riêng (Dev F1 lần lượt 71,43% / 69,81% / 70,32%), không phải cùng recipe. B1 216 chunks vẫn dùng checkpoint PhoBERT lịch sử seed 12. Chỉ so sánh các recipe trên cùng protocol trước khi freeze. **Trước khi freeze chính thức:** lưu checkpoint `seed42/best_model`, manifest có hash của checkpoint và source, bản cấu hình eval, threshold và kết quả Dev vào evidence; xác nhận PR liên quan đã review và không thay đổi training/eval pipeline sau freeze. Chỉ thực hiện Public Test sau khi khóa cấu hình theo quy trình của nhóm.

Đối chiếu kết quả mới nhất và giới hạn: `docs/results/qa_results_current_2026-11-25.md`.
