## 1. Phạm vi

Theo `docs/project_plan.md`, tải và khảo sát ViQuAD, xây pipeline tiền xử lý, metric, QA head và training loop, fine-tune PhoBERT lần đầu, cùng nhóm tạo/gán nhãn tập 100 câu và chạy B1 sơ bộ.

## 2. Dữ liệu và cách chia

- Nguồn: `taidng/UIT-ViQuAD2.0`; raw/processed được tạo qua `scripts/download_data.py`, `scripts/prepare_viquad_qa.py`, `scripts/fix_answer_start.py` (tùy bản commit).
- **Dev nội bộ:** tách theo title từ tập Train, **13 titles / 2.768 questions**; tránh train/dev trùng title/context. Trong Dev: **1.886 answerable / 882 impossible**.
- **Train sau chia theo title:** **25.686 questions** (số từ manifest evidence huấn luyện sạch).
- Split `validation` của bản dữ liệu nhóm dùng được giữ làm **Public Test**; không được dùng để tuning.
- **Tiền xử lý Dev sạch ở thí nghiệm Week 3:** 2.768/2.768 mẫu xử lý, **3.155 features**, 0 skipped; 29 nhãn `unalignable` được giữ để đánh giá.


- Đã **viết 33 câu Gold** và **gán nhãn độc lập 33 câu của thành viên B**.
- Nhóm đã chốt bộ 100 câu Gold với 30 Dev / 70 Test; việc đối chiếu bản ghi và hash thực tế của lần chốt vẫn cần lưu vào provenance của dự án.

## 3. Kết quả

`phobert_3lr_clean_dev_EVIDENCE.zip` (manifest nguồn và số liệu Dev sau làm sạch)
