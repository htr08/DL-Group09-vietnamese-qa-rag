# Project Charter

Nhóm 9, Đề tài 10: hỏi đáp tiếng Việt bằng extractive QA và RAG với mô hình mã nguồn mở.

| Mục | Nội dung |
|---|---|
| Trạng thái | Bản 1.1 (2026-11-12). Bản 1.0 chốt ở Tuần 1; chỉnh một lần nữa sau khi có kết quả baseline ở Tuần 2. |
| Nguồn | Proposal và Kế hoạch dự án 6 tuần (mục 1.3, 1.4). Charter tóm tắt các tiêu chí của dự án; khi có khác biệt, Proposal được ưu tiên. |
| Thành viên | A: Nguyễn Hữu Hoà (Extractive QA). B: Nguyễn Bá Hiển (Retrieval). C: Nguyễn Thị Hiền Trang (RAG và sản phẩm). |

## 1. Mục tiêu

1. Thay hai thành phần phụ thuộc API đóng của hệ thống nền (Gemini embedding và Gemini 2.5 Flash) bằng BGE-M3 và Qwen2.5-7B-Instruct (4-bit).
2. Bổ sung mô-đun extractive QA (PhoBERT; XLM-R để đối chiếu), đánh giá trên UIT-ViQuAD 2.0.
3. Đánh giá hệ thống có kiểm định thống kê, gồm tập 100 câu hỏi miền ứng dụng do nhóm tự gán nhãn.
4. Báo cáo so sánh với baseline B1, B2, B3, kèm phân tích lỗi và ảo giác.

Đối chiếu yêu cầu tối thiểu của đề bài (Đề tài 10): fine-tune PhoBERT hoặc XLM-R cho extractive QA (EM, F1) và xây hệ thống RAG gồm retriever BM25 và dense cùng một LLM mã nguồn mở. Các mục so sánh BM25/dense/hybrid, đánh giá ảo giác và demo chatbot thuộc phần mở rộng.

## 2. Baseline

| Mã | Mô tả | Cách lấy số liệu |
|---|---|---|
| B1 | BM25 + mô hình đọc hiểu | Tự chạy |
| B2 | Hệ thống nền (Gemini), cấu hình gốc `RRF_K = 15` | Tự chạy trên miền chính (mục 4.1) |
| B3 | mBERT trên ViQuAD 2.0 | Lấy từ bài báo gốc, không tự chạy; đối chiếu lại với bài gốc ở Tuần 1 |

## 3. Tiêu chí thành công (KPI)

### 3.1. Tiêu chí theo Proposal (KPI chính)

| # | Tiêu chí | Dữ liệu |
|---|---|---|
| K1 | F1 của PhoBERT cao hơn B3 (mBERT) | Test công khai ViQuAD 2.0 |
| K2 | Hybrid có Recall@10 cao hơn BM25 đơn lẻ | Kho ViQuAD. Proposal không nêu kho; miền chính báo kèm R@1, R@3, MRR@10 (mục 4.1) |
| K3 | RAG mã nguồn mở có tỷ lệ ảo giác không cao hơn B2 | 70 câu test của miền chính (mục 4.1) |

Tập dữ liệu "miền chính" của K3 được xác định ở mục 4.1.

Nếu một tiêu chí không đạt, báo cáo phân tích nguyên nhân (đề bài cho phép: kết quả không vượt baseline vẫn được điểm tốt nếu thí nghiệm đúng và phân tích được nguyên nhân).

### 3.2. Kiểm tra chặt hơn (báo cáo thêm, không thay K1-K3)

- Hybrid so với retriever đơn tốt nhất (không chỉ BM25) về Recall@5 trên kho ViQuAD, kèm bootstrap 95% CI của chênh lệch.
- Kết quả trên tập 100 câu chỉ được báo kèm CI vì cỡ mẫu nhỏ.

### 3.3. Mức tham chiếu (đề xuất, điều chỉnh sau baseline Tuần 2; không phải tiêu chí chấm)

- Ảo giác ≤ 10% trên 70 câu test.
- Từ chối đúng ≥ 70% số câu ngoài phạm vi.

## 4. Dữ liệu và quy tắc kiểm soát đánh giá

| Dữ liệu | Quy tắc |
|---|---|
| ViQuAD 2.0 (bản Hugging Face `taidng/UIT-ViQuAD2.0`) | Split `train` (28.454 dòng) dùng để huấn luyện và tách dev. Split `validation` (3.814 dòng, có nhãn) chính là test công khai, chỉ dùng làm test. Split `test` của bản này không có nhãn và có đoạn trùng với train nên không dùng. |
| Dev của ViQuAD | 10% số bài viết của train, tách theo `title` (không tách theo câu hỏi, vì tách theo câu làm cùng một đoạn nằm ở cả train và dev). Cố định seed; danh sách lưu ở `data/splits/`. Có unit test kiểm tra không đoạn nào nằm ở cả train và dev. |
| Kho truy hồi ViQuAD | Gộp mọi đoạn văn thành một kho. Đoạn gốc của câu hỏi có đáp án là nhãn đúng. Câu không có đáp án bị loại khỏi Recall@k. |
| Tập 100 câu miền chính (tài liệu tiếng Việt, xem 4.1) | Khoảng 90 câu trong phạm vi và 10 câu ngoài phạm vi. Chia 30 dev (tuning k, trọng số, prompt) và 70 test; câu ngoài phạm vi chia 3 dev / 7 test. A và B viết mỗi người 33 câu, C viết 34 câu. Đoạn nguồn gán ở mức Điều và Khoản. Mỗi câu có hai lượt gán nhãn độc lập (xoay vòng A→B→C→A); bất đồng được thống nhất ở Tuần 3 T2; báo cáo tỉ lệ đồng thuận (định nghĩa ở mục 5). Không chỉnh gì dựa trên câu test. |
| Miền phụ Samsung Galaxy S24 (xem 4.1) | Chỉ dùng cho thí nghiệm lệch ngôn ngữ; không tính vào tập 100 câu và không dùng cho KPI K2, K3. |
| Test | Chạy **đúng một lần**, ở Tuần 5 T2. |

Dữ liệu LoRA (nếu thực hiện) lấy từ ViQuAD train (đã loại các bài thuộc dev), không dùng văn bản Quy chế; tập miền ứng dụng không dùng để huấn luyện.

### 4.1. Quyết định về miền ứng dụng (2026-11-08)

Hệ thống nền dùng hai PDF thông số Samsung Galaxy S24. Kho này chỉ khoảng 8 trang (vài chục đoạn), nên Recall@10 dễ bão hòa ở mức gần 100% với mọi phương pháp và khó phân biệt BM25, dense, hybrid. Vì vậy nhóm dùng thêm một kho tiếng Việt cỡ vừa làm miền chính. Quyết định:

- **Miền chính:** Quy chế đào tạo đại học của ĐHQGHN, bản 2014 (Quyết định 5115/QĐ-ĐHQGHN), file `data/domain/dhqghn_quy_che_dao_tao_dai_hoc_2014.pdf`. Văn bản tiếng Việt, 48 trang, 47 điều; trang 46-48 là mục lục và bị loại khỏi kho. Kho cắt hai cấp Điều → Khoản, khoảng 164 chunk (162 Khoản và 2 Điều không chia Khoản); Khoản dài được tách tiếp theo Điểm hoặc cửa sổ trượt. Nếu chỉ cắt theo Điều thì kho có 47 chunk và Recall@10/20 bão hòa như kho S24. Tập 100 câu và KPI K3 áp dụng cho miền này; K2 đo trên kho ViQuAD, còn miền này báo kèm R@1, R@3, MRR@10. Đề bài gợi ý loại kho này ("quy chế đào tạo").
- **Miền phụ:** hai PDF S24, chỉ dùng cho thí nghiệm lệch ngôn ngữ (câu hỏi tiếng Việt, tài liệu tiếng Anh). Dự kiến 20-30 câu hỏi cho thí nghiệm này. Giả thuyết "dense đa ngữ khắc phục lệch ngôn ngữ" và ablation "có/không dịch câu hỏi sang tiếng Anh" chỉ thực hiện trên miền này.

**Hệ quả cần xử lý**
- Dựng index cho kho mới (BM25 và dense); loại mục lục và cắt đoạn theo Điều → Khoản; thêm 1-2 văn bản khác của ĐHQGHN làm nhiễu nếu muốn kho trên 200 chunk.
- Viết hướng dẫn gán nhãn và bộ 100 câu theo miền mới.
- Trích văn bản bằng `pdfplumber` (đã đo: ít lỗi tách chữ tiếng Việt hơn `pypdf`).
- Chạy B2 trên miền chính. Hệ thống nền phải nạp tài liệu mới, và prompt gốc viết cho trợ lý bán hàng (có nhánh viết bài quảng cáo). Mọi chỉnh sửa prompt cho B2 phải được ghi lại và nêu trong báo cáo. B2 giữ cách cắt đoạn gốc (MarkItDown, theo heading); nên chạy thêm B2-cùng-chunk (dùng bộ chunk Điều → Khoản của hệ mới) để tách hiệu ứng LLM khỏi hiệu ứng cắt đoạn.
- Ở miền chính BM25 không còn bị bất lợi do lệch ngôn ngữ, nên kết quả BM25/dense/hybrid có thể khác với giả thuyết ban đầu; kết quả nào cũng được báo cáo và phân tích.
- Văn bản là bản 2014: báo cáo ghi rõ năm và tránh câu hỏi dạng "quy định hiện hành".
- **Hệ quả cần xử lý:** index lại kho cho cả BM25 và dense; chạy B2 (Gemini) trên miền chính; viết lại hướng dẫn gán nhãn và bộ 100 câu theo miền mới.

## 5. Chỉ số

| Phần | Chỉ số |
|---|---|
| Extractive QA | EM và F1 theo SQuAD 2.0; báo riêng câu có và không có đáp án; bootstrap 95% CI theo câu hỏi |
| Retrieval | Recall@{1, 3, 5, 10, 20}, MRR@10, độ trễ mỗi truy vấn; so sánh cặp bằng bootstrap CI của chênh lệch Recall@5. Miền chính: chỉ số chính là R@1, R@3, MRR@10; đoạn đúng tính ở mức Khoản (báo thêm mức Điều); câu ngoài phạm vi bị loại khỏi Recall@k và MRR |
| Toàn hệ thống | EM/F1 mức câu trả lời so với đáp án gold trên câu trong phạm vi (theo Proposal); với câu sinh bằng LLM chỉ để tham khảo |
| RAG (chấm tay) | Tỷ lệ ảo giác, từ chối đúng, từ chối sai, độ trễ, bộ nhớ GPU. Chấm mù: trộn câu trả lời của mọi cấu hình, ẩn tên hệ thống; 70 câu test chia cho ba người; khoảng 30% được người kế chấm lại để tính Cohen's kappa |
| Đồng thuận gán nhãn | Tỉ lệ cùng Khoản (và cùng Điều) cho đoạn nguồn; EM và F1 token giữa hai span đáp án; tỉ lệ cùng gắn nhãn ngoài phạm vi |
| Kiểm tra phụ | Tỷ lệ câu trả lời lẫn ký tự CJK |

Phân loại lỗi RAG: **Đúng**, **Đúng một phần**, **Sai-bịa (ảo giác)**, **Từ chối đúng**, **Từ chối sai**. Ảo giác là loại lỗi nặng nhất.

## 6. Điều kiện dừng và phạm vi

### 6.1. Điều kiện dừng tuning

- Mỗi mô hình tối đa 3 cấu hình (PhoBERT: lr ∈ {2e-5, 3e-5, 5e-5}).
- Early stopping với patience = 1-2 epoch theo dev F1.
- Thời gian một epoch được đo ở Tuần 2 để ước lượng ngân sách GPU.
- **Experiment freeze:** cuối Tuần 4 (Thứ Năm).
- Mọi tham số đã chọn được ghi lại; giá trị `k` của RRF được nêu rõ ở từng phần của báo cáo.

### 6.2. Mức ưu tiên

| Mức | Hạng mục |
|---|---|
| Bắt buộc (MVP) | PhoBERT và XLM-R có EM/F1 (lớp đầu ra, vòng lặp huấn luyện và hậu xử lý n-best tự cài, theo Proposal); RAG gồm BM25, dense (BGE-M3), RRF và Qwen2.5-7B; tập 100 câu và Recall@k; B1, B2, B3; các ablation đã nêu trong Proposal (BM25/dense/hybrid; có/không xếp hạng lại; có/không tách từ; có/không dịch câu hỏi; k ∈ {3, 5, 10}; chế độ sinh so với chế độ trích xuất); phân tích lỗi; chấm ảo giác; demo |
| Nên làm | 3 seed cho PhoBERT tốt nhất; B2-cùng-chunk |
| Chỉ làm nếu thừa thời gian | LoRA do C làm ở Tuần 4 T2-T4, trước experiment freeze, dữ liệu từ ViQuAD train (quyết định go/no-go ở họp cuối Tuần 3) |

### 6.3. Thứ tự cắt khi trễ tiến độ

Cắt từ trên xuống, dừng khi kịp tiến độ. Phân tích lỗi, ablation cốt lõi, kiểm soát dev/test và tập vấn đáp không bị cắt. Hạng mục đã hứa trong Proposal mà bị cắt phải được ghi chú trong báo cáo.

1. LoRA
2. B2-cùng-chunk
3. Hai seed thêm cho PhoBERT
4. PhoBERT không tách từ
5. Dịch truy vấn sang tiếng Anh
6. XLM-R đối chiếu
7. Rút tập 100 câu xuống 60 (20 dev, 40 test), giữ hai người gán nhãn

## 7. Chính sách công cụ

- Hệ thống chính chỉ dùng mô hình mã nguồn mở. Gemini chỉ dùng để chạy baseline B2.
- Mọi lần dùng API và công cụ AI được ghi lại (`docs/ai_usage.md`) và khai báo trong báo cáo.
- Phần dùng lại từ hệ thống nền được ghi lại (`docs/reuse_vs_new.md`) và nêu trong bảng "tự cài đặt so với dùng sẵn" của báo cáo.
- Mã hệ thống nền không chép vào repo; chỉ dẫn link tới repo gốc ở một commit cố định.
- Repository công khai: không commit khóa API, token hay file `.env`.

## 8. Definition of Done

Một mốc được coi là hoàn thành khi thỏa các điều kiện sau.

**Mọi Pull Request**
- [ ] Ít nhất 1 người khác approve; CI xanh.
- [ ] Phần dễ sai có unit test (ánh xạ offset QA, chuẩn hóa EM/F1, Recall@k, RRF, bootstrap).
- [ ] Không commit dữ liệu thô, index, checkpoint; output notebook đã xóa.

**Mỗi thí nghiệm**
- [ ] Log lên W&B với tên run `<role>_<model>_<cấu hình>_seed<k>` và commit hash.
- [ ] Seed và cấu hình được lưu; bảng trong báo cáo sinh từ script hoặc log, không sửa tay.

**Bản nộp cuối**
- [ ] Mục tiêu K1-K3 đã được báo cáo là đạt hoặc không đạt, kèm phân tích.
- [ ] Bảng 1-5 của kế hoạch có số liệu; bootstrap CI cho các so sánh chính.
- [ ] Báo cáo đúng template ACL (`\usepackage[final]{acl}`), tối đa 8 trang nội dung, có Limitations, đóng góp của thành viên, khai báo công cụ AI, bảng tự cài đặt so với dùng sẵn; tên file `Nhom09_DeTai10.pdf`.
- [ ] Repository chạy lại được từ môi trường sạch; README hướng dẫn cài đặt, tải dữ liệu, huấn luyện, đánh giá; có `requirements.txt`.
- [ ] Log W&B chia sẻ công khai hoặc với giảng viên.
- [ ] Slide 10-12 phút và demo hai chế độ (sinh và trích xuất) chạy được, có video dự phòng.
- [ ] Mỗi thành viên có tóm tắt 1 trang về phần mình và phần của người khác cho vấn đáp.

## 9. Lịch sử chỉnh sửa

| Ngày | Nội dung |
|---|---|
| 2026-11-08 | Bản 1.0, lập từ kế hoạch dự án. |
| 2026-11-12 | Bản 1.1, theo góp ý rà soát kế hoạch: cắt kho Quy chế theo Điều → Khoản; K2 đo trên kho ViQuAD; chấm ảo giác mù và chia cho ba người; LoRA dời sang Tuần 4 (C, dữ liệu ViQuAD train); câu ngoài phạm vi chia 3 dev / 7 test; định nghĩa tỉ lệ đồng thuận; thêm EM/F1 toàn hệ thống và B2-cùng-chunk; không chép mã hệ thống nền vào repo. |

