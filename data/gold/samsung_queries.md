# Bộ câu hỏi tiếng Việt - Miền phụ Samsung Galaxy S24 (Thí nghiệm Lệch ngôn ngữ)

Bộ dữ liệu gồm **27 câu hỏi tiếng Việt** được xây dựng dựa trên 2 tài liệu thông số kỹ thuật tiếng Anh của Samsung Galaxy S24:
1. `data/domain/samsung_galaxy_s24_128gb_device_specs.pdf` (5 trang, 5 chunks)
2. `data/domain/samsung_galaxy_s24_specifications.pdf` (3 trang, 4 chunks sau khi chia cửa sổ trượt max 200 từ, stride 180)
Tổng số chunk được đánh chỉ mục trong kho: **9 chunks**.

## 1. Thông tin kiểm duyệt và phân công (Theo Project Plan §4.3 T4)
- **Người tạo câu hỏi và gắn nhãn (Role B - Retrieval Engineer):** Nguyễn Bá Hiển
- **Người kiểm tra chéo và duyệt nhãn (Role A - Lead & Role C - QA Engineer):** Bùi Huyền Trang & Nguyễn Quang Hòa
- **Trạng thái:** Đã kiểm tra chéo và chuẩn hóa chuỗi đáp án nguyên văn (`reviewed`).
- **Mục đích thí nghiệm:** Đánh giá năng lực truy hồi liên ngôn ngữ (Cross-lingual Retrieval) giữa BM25, Dense (BGE-M3) và Hybrid RRF theo cam kết trong Proposal & Project Plan (§3.1, §4.3 T4).

---

## 2. Phân tầng câu hỏi theo mức độ trùng lặp từ vựng (Token Overlap Stratification)
Nhằm kiểm chứng chính xác giả thuyết §3.5b ("BM25 suy giảm mạnh trong bài toán lệch ngôn ngữ so với Dense"), 27 câu hỏi được phân thành 3 nhóm căn cứ vào mức độ trùng token tiếng Anh / Latinh với chunk chứa đáp án:

1. **Nhóm thuần Việt hoàn toàn (`none` - 4 câu):** S09, S16, S24, S25. Câu hỏi không chứa bất kỳ từ tiếng Anh hay tên máy nào trùng với chunk gold. Đây là nhóm kiểm thử năng lực hiểu ngữ nghĩa liên ngôn ngữ thuần túy.
2. **Nhóm chỉ trùng tên thương hiệu (`brand_only` - 8 câu):** S01, S03, S04, S06, S07, S14, S15, S19. Chỉ chứa các token định danh `Samsung`, `Galaxy`, `S24` (vốn xuất hiện ở hầu hết mọi chunk trong kho).
3. **Nhóm chứa từ mượn kỹ thuật (`technical_loanword` - 15 câu):** S02, S05, S08, S10, S11, S12, S13, S17, S18, S20, S21, S22, S23, S26, S27. Chứa các thuật ngữ kỹ thuật Latinh (`nits`, `UFS`, `GPU`, `SAR`, `AMOLED`, `FHD`, `DeX`, `camera`, `zoom`, `3.5mm`...). Ở nhóm này, BM25 có thể tận dụng lợi thế khớp từ khóa đặc thù có IDF cao.

> **Lưu ý về các câu hỏi có đáp án ở nhiều chunk (`relevant_ids`):**
> Có 8/27 câu (S09, S10, S11, S12, S13, S25, S26, S27) có thông tin xuất hiện ở cả hai tài liệu hoặc nằm ở vùng giao nhau giữa 2 cửa sổ trượt. Pipeline đánh giá hỗ trợ tính hit (Recall@k, MRR@k) khi trúng bất kỳ chunk hợp lệ nào trong danh sách `relevant_ids`.

---

## 3. Danh sách 27 câu hỏi và nhãn căn cứ chi tiết

| ID | Nhóm từ vựng | Câu hỏi tiếng Việt | Chunk ID nguồn hợp lệ (`relevant_ids`) | Đoạn trích dẫn nguyên văn trong chunk (`evidence_text`) | Đáp án chuẩn (`answer`) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **S01** | Chỉ trùng tên máy | Samsung Galaxy S24 được phát hành chính thức vào ngày tháng năm nào? | `s24_samsung_galaxy_s24_specifications_p001_w0` | LAUNCH Status Release 2024, January 24 | 2024, January 24 |
| **S02** | Có từ mượn kỹ thuật | Màn hình của Samsung Galaxy S24 có độ sáng tối đa là bao nhiêu nits? | `s24_samsung_galaxy_s24_specifications_p001_w0` | DISPLAY Type Dynamic LTPO AMOLED 2X, 120Hz, HDR10+, 2600 nits (peak) | 2600 nits (peak) |
| **S03** | Chỉ trùng tên máy | Màn hình và mặt lưng của Samsung Galaxy S24 được bảo vệ bởi loại kính cường lực nào? | `s24_samsung_galaxy_s24_specifications_p001_w0` | Protection Corning Gorilla Glass Victus 2 | Corning Gorilla Glass Victus 2 |
| **S04** | Chỉ trùng tên máy | Samsung Galaxy S24 đạt tiêu chuẩn chống bụi và kháng nước cấp độ mấy? | `s24_samsung_galaxy_s24_specifications_p001_w0` | IP68 dust/water resistant (up to 1.5m for 30 min) | IP68 dust/water resistant (up to 1.5m for 30 min) |
| **S05** | Có từ mượn kỹ thuật | Tại thị trường Mỹ, Canada và Trung Quốc, máy sử dụng vi xử lý chipset nào? | `s24_samsung_galaxy_s24_specifications_p001_w0` | Chipset Qualcomm SM8650-AC Snapdragon 8 Gen 3 (4 nm) - USA/Canada/China | Qualcomm SM8650-AC Snapdragon 8 Gen 3 (4 nm) |
| **S06** | Chỉ trùng tên máy | Phiên bản quốc tế của Samsung Galaxy S24 được trang bị bộ vi xử lý nào? | `s24_samsung_galaxy_s24_specifications_p001_w0` | Exynos 2400 (4 nm) - International | Exynos 2400 (4 nm) |
| **S07** | Chỉ trùng tên máy | Khung kim loại của Galaxy S24 sử dụng vật liệu gì để tăng khả năng chống rơi và trầy xước? | `s24_samsung_galaxy_s24_specifications_p001_w0` | Armor aluminum 2 frame with tougher drop and scratch resistance (advertised) | Armor aluminum 2 frame |
| **S08** | Có từ mượn kỹ thuật | Galaxy S24 phiên bản bộ nhớ trong 128GB sử dụng chuẩn chip nhớ UFS nào? | `s24_samsung_galaxy_s24_specifications_p001_w1` | UFS 3.1 - 128GB only | UFS 3.1 |
| **S09** | Thuần Việt (0 trùng) | Samsung Galaxy S24 có hỗ trợ khe cắm thẻ nhớ mở rộng MicroSD không? | `s24_samsung_galaxy_s24_specifications_p001_w0`<br>`s24_samsung_galaxy_s24_specifications_p001_w1` | MEMORY Card slot No | No (Card slot No) |
| **S10** | Có từ mượn kỹ thuật | Bộ xử lý đồ họa GPU trên phiên bản quốc tế của Galaxy S24 có tên là gì? | `s24_samsung_galaxy_s24_specifications_p001_w0`<br>`s24_samsung_galaxy_s24_specifications_p001_w1` | Xclipse 940 - International | Xclipse 940 |
| **S11** | Có từ mượn kỹ thuật | Camera tele của Samsung Galaxy S24 hỗ trợ mức zoom quang học bao nhiêu x? | `s24_samsung_galaxy_s24_specifications_p002`<br>`s24_samsung_galaxy_s24_128gb_device_specs_p002` | CAMERA 10 MP, f/2.4, 67mm (telephoto), 1/3.94", 1.0µm, PDAF, OIS, 3x optical zoom | 3x optical zoom |
| **S12** | Có từ mượn kỹ thuật | Camera góc rộng chính phía sau của Galaxy S24 có độ phân giải bao nhiêu megapixel? | `s24_samsung_galaxy_s24_specifications_p002`<br>`s24_samsung_galaxy_s24_128gb_device_specs_p002` | MAIN Triple 50 MP, f/1.8, 24mm (wide), 1/1.56", 1.0µm, Dual Pixel PDAF, OIS | 50 MP |
| **S13** | Có từ mượn kỹ thuật | Camera trước dùng để chụp selfie của máy có độ phân giải bao nhiêu? | `s24_samsung_galaxy_s24_specifications_p002`<br>`s24_samsung_galaxy_s24_128gb_device_specs_p002` | SELFIE Single 12 MP, f/2.2, 26mm (wide), Dual Pixel PDAF | 12 MP |
| **S14** | Chỉ trùng tên máy | Galaxy S24 hỗ trợ công nghệ sạc nhanh có dây công suất tối đa bao nhiêu Watt? | `s24_samsung_galaxy_s24_specifications_p002` | Charging 25W wired, PD3.0, 50% in 30 min (advertised) | 25W wired |
| **S15** | Chỉ trùng tên máy | Công suất tính năng chia sẻ pin không dây sạc ngược của Galaxy S24 là bao nhiêu? | `s24_samsung_galaxy_s24_specifications_p002` | 4.5W reverse wireless | 4.5W reverse wireless |
| **S16** | Thuần Việt (0 trùng) | Cảm biến vân tay của điện thoại sử dụng công nghệ gì và nằm ở vị trí nào? | `s24_samsung_galaxy_s24_specifications_p002` | FEATURES Sensors Fingerprint (under display, ultrasonic) | Fingerprint (under display, ultrasonic) |
| **S17** | Có từ mượn kỹ thuật | Hệ thống loa ngoài stereo của Galaxy S24 được tinh chỉnh âm thanh bởi hãng nào? | `s24_samsung_galaxy_s24_specifications_p002` | Tuned by AKG | Tuned by AKG |
| **S18** | Có từ mượn kỹ thuật | Điện thoại Samsung Galaxy S24 có cổng cắm tai nghe 3.5mm truyền thống không? | `s24_samsung_galaxy_s24_specifications_p002` | 3.5mm jack No | No (3.5mm jack No) |
| **S19** | Chỉ trùng tên máy | Mức giá khởi điểm bằng USD của Samsung Galaxy S24 được niêm yết là bao nhiêu tiền? | `s24_samsung_galaxy_s24_specifications_p003` | Price $ 799.99 / C$ 1,099.99 / £ 799.00 / € 899.00 / ₹ 79,999 | $ 799.99 |
| **S20** | Có từ mượn kỹ thuật | Theo chuẩn Châu Âu SAR EU, tỷ lệ hấp thụ bức xạ ở phần đầu của thiết bị là bao nhiêu? | `s24_samsung_galaxy_s24_specifications_p003` | SAR EU 0.87 W/kg (head) 1.35 W/kg (body) | 0.87 W/kg (head) |
| **S21** | Có từ mượn kỹ thuật | Màn hình Dynamic AMOLED của máy có khả năng hiển thị bao nhiêu triệu màu? | `s24_samsung_galaxy_s24_128gb_device_specs_p001` | 2340 x 1080 (FHD+) 16M Dynamic AMOLED 2 | 16M |
| **S22** | Có từ mượn kỹ thuật | Đối với phiên bản bộ nhớ 128GB, dung lượng lưu trữ thực tế còn trống khả dụng là bao nhiêu GB? | `s24_samsung_galaxy_s24_128gb_device_specs_p002` | 128GB 104.6GB 8GB | 104.6GB |
| **S23** | Có từ mượn kỹ thuật | Tính năng quay phim chuyển động chậm Slow Motion ở độ phân giải FHD hỗ trợ tốc độ bao nhiêu khung hình/giây? | `s24_samsung_galaxy_s24_128gb_device_specs_p002` | Yes 240fps @FHD, 120fps @UHD | 240fps @FHD |
| **S24** | Thuần Việt (0 trùng) | Thời lượng phát lại âm thanh không dây tối đa của pin Galaxy S24 có thể kéo dài bao nhiêu giờ? | `s24_samsung_galaxy_s24_128gb_device_specs_p004` | 4000 Up to 72 Up to 28 (Audio Playback Time: Up to 72) | Up to 72 |
| **S25** | Thuần Việt (0 trùng) | Pin của Samsung Galaxy S24 có thể tự tháo rời nắp lưng để thay thế được không? | `s24_samsung_galaxy_s24_128gb_device_specs_p004`<br>`s24_samsung_galaxy_s24_specifications_p002` | Removable ... No / BATTERY Type Li-Ion 4000 mAh, non-removable | No (Removable No / non-removable) |
| **S26** | Có từ mượn kỹ thuật | Samsung Galaxy S24 có hỗ trợ không gian làm việc giao diện máy tính Samsung DeX không? | `s24_samsung_galaxy_s24_128gb_device_specs_p005`<br>`s24_samsung_galaxy_s24_specifications_p002` | Samsung DeX Support ... Yes / Samsung DeX, Samsung Wireless DeX (desktop experience support) | Yes (Samsung DeX Support: Yes) |
| **S27** | Có từ mượn kỹ thuật | Trọng lượng vật lý của điện thoại tính theo đơn vị gram là bao nhiêu g? | `s24_samsung_galaxy_s24_128gb_device_specs_p005`<br>`s24_samsung_galaxy_s24_specifications_p001_w0` | 147.0 x 70.6 x 7.6 168 / Weight 167 g or 168 g (5.89 oz) | 168 (167 g or 168 g) |
