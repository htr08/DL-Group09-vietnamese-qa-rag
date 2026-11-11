# Kế hoạch dự án (bản 6 tuần)

*Hỏi đáp tiếng Việt: cải tiến hệ thống RAG bằng mô hình mã nguồn mở và đọc hiểu trích xuất (Nhóm 9, Đề tài 10)*

| **Mục** | **Nội dung** |
|---|---|
| **Nhóm** | Nhóm 9, 3 thành viên: **Nguyễn Hữu Hoà (A)**, **Nguyễn Bá Hiển (B)**, **Nguyễn Thị Hiền Trang (C)**. Phân vai khớp với Proposal (C1, C2, C3). |
| **Căn cứ** | Bám theo Proposal: dùng lại hệ thống nền AI Marketing Assistant, thay thành phần API thương mại bằng mô hình mã nguồn mở, thêm extractive QA, tự xây tập đánh giá có nhãn người. |
| **Thời gian và mốc nộp** | 6 tuần theo lịch môn học. Proposal (≤ 2 trang) cuối Tuần 1; Milestone report (≤ 4 trang) cuối Tuần 3; bản cuối Tuần 6 (báo cáo ≤ 8 trang nội dung, mã nguồn, slide, thuyết trình; file báo cáo nộp theo mẫu NhomXX_DeTaiYY.pdf, tức Nhom09_DeTai10.pdf). Tuần 1 là tuần Proposal nên chỉ có checklist; lịch theo ngày bắt đầu từ Tuần 2. |
| **Nguyên tắc của bản kế hoạch** | Nguồn lực tập trung vào các thành phần có trọng số chấm cao (phân tích, ablation, vấn đáp, tiến độ đều); tuning và độ chính xác tuyệt đối chỉ ở mức vừa đủ. Mỗi hạng mục có mức ưu tiên: **Bắt buộc**, **Nên làm**, **Chỉ làm nếu thừa thời gian**. |

## 1. Mục tiêu và sản phẩm đầu ra

### 1.1. Mục tiêu

Thay các thành phần phụ thuộc API đóng của hệ thống nền (Gemini embedding và Gemini 2.5 Flash) bằng BGE-M3 và Qwen2.5-7B-Instruct; bổ sung mô-đun **extractive QA** (PhoBERT, đối chiếu XLM-R); đánh giá có kiểm định thống kê trên UIT-ViQuAD 2.0 và trên tập 100 câu hỏi miền ứng dụng (Quy chế đào tạo đại học của ĐHQGHN) do nhóm tự gán nhãn; thí nghiệm lệch ngôn ngữ dùng thêm hai tài liệu Samsung Galaxy S24. Mục tiêu cuối: bảng so sánh với hệ thống nền (B2) và baseline B1, B3 như trong Proposal, kèm phân tích lỗi và ảo giác.

### 1.2. Sản phẩm cuối (Deliverables)

- **Source code:** repository mới của nhóm (public), tái chạy được, log W&B chia sẻ. File docs/reuse_vs_new.md ghi rõ phần dùng lại từ hệ thống nền và phần làm mới.
- **Kết quả Extractive QA:** EM/F1 trên ViQuAD 2.0 (dev nội bộ và test công khai), tách câu có và không có đáp án; so sánh PhoBERT với XLM-R.
- **Kết quả Retrieval:** Recall@{1,5,10,20} và MRR@10 cho BM25, dense, hybrid, hybrid + xếp hạng lại theo tiêu đề, trên kho ViQuAD và miền ứng dụng.
- **Kết quả RAG:** tỷ lệ ảo giác, từ chối đúng, độ trễ, bộ nhớ GPU, so với hệ thống nền (B2) trên tập 100 câu; so sánh chế độ sinh bằng LLM với chế độ trích xuất trên đoạn truy hồi.
- **Phân tích lỗi** có phân loại cho cả extractive QA và retrieval.
- **Báo cáo ACL (≤ 8 trang nội dung):** theo cấu trúc của đề bài (Abstract 150–250 từ, Introduction, Related Work, Method có hình kiến trúc, Experiments, Analysis, Conclusion), kèm Limitations, phần Đóng góp của thành viên (tỉ lệ đóng góp ước tính), khai báo công cụ AI và bảng "tự cài đặt vs dùng sẵn". Limitations, Đóng góp của thành viên và References không tính vào 8 trang.
- **Demo chatbot** (hai chế độ) và **slide** (10–12 phút).

### 1.3. Project Charter: KPI và Definition of Done

Charter do C soạn, cả nhóm duyệt, được chốt ở Tuần 1 và chỉnh một lần sau baseline ở Tuần 2 (bản hiện hành: docs/charter.md):

- **Tiêu chí thành công theo Proposal (dùng làm KPI chính):** (i) F1 của PhoBERT cao hơn baseline B3 (mBERT, số liệu từ bài báo; đối chiếu lại với bài gốc ở Tuần 1) trên test công khai; (ii) hybrid có Recall@10 cao hơn BM25 đơn lẻ; (iii) RAG mã nguồn mở có tỷ lệ ảo giác không cao hơn hệ thống nền B2 trên tập miền ứng dụng. Nếu không đạt thì phân tích nguyên nhân (đề cho phép).
- **Kiểm tra chặt hơn (báo cáo thêm, không thay tiêu chí của Proposal):** hybrid so với retriever đơn tốt nhất (không chỉ BM25) về Recall@5 trên kho ViQuAD, kèm bootstrap 95% CI của chênh lệch. Kết quả trên tập 100 câu chỉ báo kèm CI vì mẫu nhỏ.
- **Mức tham chiếu (đề xuất, điều chỉnh sau baseline Tuần 2; không phải tiêu chí chấm):** ảo giác ≤ 10% trên 70 câu test; từ chối đúng ≥ 70% số câu ngoài phạm vi.
- **Điều kiện dừng tuning:** mỗi mô hình tối đa 3 cấu hình; early stopping patience = 1–2 epoch trên dev F1; **experiment freeze cuối T5 của Tuần 4**; **test chạy đúng một lần ở Tuần 5, T2**. Đo thời gian 1 epoch ở Tuần 2 để ước lượng ngân sách GPU.
- **Phân loại lỗi RAG:** Đúng / Đúng một phần / Sai-bịa (ảo giác) / Từ chối đúng / Từ chối sai. Ảo giác là lỗi nặng nhất.
- **Chính sách công cụ:** hệ thống chính chỉ dùng mô hình mã nguồn mở. Gemini chỉ dùng để chạy baseline B2; mọi lần dùng API và công cụ AI ghi vào docs/ai_usage.md và khai báo trong báo cáo.
- **Mức ưu tiên. Bắt buộc (MVP, đúng những gì Proposal đã hứa, trừ LoRA):** PhoBERT và XLM-R có EM/F1; RAG gồm BM25, dense (BGE-M3), RRF và Qwen2.5-7B; tập 100 câu và Recall@k; B1, B2 (tự chạy), B3 (số từ bài báo); các ablation nêu trong Proposal (BM25/dense/hybrid; có/không xếp hạng lại; có/không tách từ; có/không dịch câu hỏi; k ∈ {3, 5, 10}; chế độ sinh so với chế độ trích xuất); phân tích lỗi; chấm ảo giác; demo. **Nên làm:** 3 seed cho PhoBERT tốt nhất. **Chỉ làm nếu thừa thời gian:** LoRA (quyết định go/no-go ở họp cuối Tuần 3). Các hạng mục có/không tách từ, có/không dịch câu hỏi và XLM-R thuộc nhóm Bắt buộc theo Proposal nhưng đồng thời nằm trong thứ tự cắt ở mục 4.7; chúng chỉ bị cắt khi tiến độ chậm, và khi đó thay đổi được ghi chú trong báo cáo.

### 1.4. Phân bổ công sức theo thang điểm

| **Thành phần chấm** | **Tỉ trọng** | **Kế hoạch dành cho phần này** |
|---|---|---|
| **Kỹ thuật và thí nghiệm (nhóm)** | 25% | Phần tự cài (tiền xử lý QA, metric, đánh giá) có unit test; thí nghiệm có kiểm soát (dev/test tách bạch, test một lần, seed). Tuning được giới hạn vì đề không chấm theo độ chính xác. |
| **Phân tích và báo cáo (nhóm)** | 25% | Ablation cốt lõi, phân tích lỗi có phân loại (Tuần 5), bootstrap CI, đúng format ACL, trích dẫn đủ. Khung báo cáo được dựng từ Milestone để việc viết phân bổ đều qua các tuần. |
| **Quá trình (nhóm)** | 10% | Proposal và Milestone đúng hạn; commit đều đặn, mỗi lần chạy có log W&B suốt 6 tuần; repo và W&B chia sẻ cho giảng viên. |
| **Thuyết trình và demo (nhóm)** | 10% | Demo hai chế độ chạy ổn, có video dự phòng; tổng duyệt ở Tuần 6. |
| **Vấn đáp (cá nhân)** | 30% | Review chéo vòng tròn; mỗi người có tóm tắt 1 trang về phần mình và phần người khác; 3 buổi tập vấn đáp (cuối Tuần 4, Tuần 5, giữa Tuần 6). |

### 1.5. Đối chiếu với yêu cầu của đề bài

| **Yêu cầu của đề bài (Đề tài 10 và quy định chung)** | **Nơi đáp ứng trong kế hoạch** |
|---|---|
| Tối thiểu: fine-tune PhoBERT hoặc XLM-R cho extractive QA, đánh giá EM và F1 | Mục 4.2–4.5 (A); Bảng 1 |
| Tối thiểu: RAG gồm retriever BM25 và dense cùng một LLM mã nguồn mở | Mục 3.3; Bảng 2, 3 (B, C) |
| Mở rộng: so sánh BM25, dense, hybrid; đánh giá ảo giác; chatbot demo | Bảng 2, 3, 4; demo ở Tuần 4 T4 và Tuần 6 |
| Mở rộng: fine-tune LLM nhỏ bằng LoRA | Hạng mục tùy chọn, go/no-go ở họp cuối Tuần 3 (mục 1.3) |
| Milestone ≤ 4 trang: kết quả baseline, tiến độ, khó khăn, kế hoạch còn lại | Mục 4.3 (T5) |
| Báo cáo cuối: cấu trúc Abstract đến Conclusion, Limitations, Đóng góp của thành viên, References; Method ghi rõ phần tự cài đặt và phần dùng sẵn, có hình kiến trúc | Mục 1.2, 4.5, 4.6; docs/reuse_vs_new.md |
| Analysis: ablation, phân tích lỗi, ví dụ đầu ra tốt và xấu | Bảng 4, 5; Tuần 4 T4, Tuần 5 T4 |
| Định dạng: ACL `[final]`, A4 hai cột, tiếng Việt (XeLaTeX/LuaLaTeX), trích dẫn tác giả–năm bằng BibTeX, tên file Nhom09_DeTai10.pdf | Mục 8.3 (checklist trước mỗi mốc nộp) |
| Mã nguồn: README (cài đặt, tải dữ liệu, huấn luyện, đánh giá), requirements, script tái tạo bảng chính, không đưa dữ liệu lớn hoặc trọng số lớn vào repo | Mục 6, 8.1; Tuần 6 T4 |
| Công cụ AI, mã nguồn mở và mô hình pretrained phải được khai báo hoặc trích dẫn; code được kiểm tra trùng lặp (MOSS/JPlag) | docs/ai_usage.md, docs/reuse_vs_new.md; mã hệ thống nền tách riêng ở legacy/ |
| Vấn đáp 5–7 phút, có câu hỏi về phần của bạn khác, có thể chọn ngẫu nhiên một đoạn code | Mục 2.2; ba buổi tập vấn đáp |
| Kết quả không vượt baseline vẫn được điểm nếu thí nghiệm đúng và phân tích được nguyên nhân | Tiêu chí thành công ở mục 1.3 |

## 2. Phân công và nguyên tắc phân vai

### 2.1. Phân công trách nhiệm

| **Thành viên** | **Vai trò chính** | **Trách nhiệm chi tiết** |
|---|---|---|
| **A: Nguyễn Hữu Hoà** | Extractive QA Engineer (C1) | Dữ liệu ViQuAD 2.0; tách từ VnCoreNLP, tokenize, sliding window; fine-tune PhoBERT (và XLM-R nếu còn thời gian); tuning; EM/F1; n-best và ngưỡng không-có-đáp-án; baseline B1 (BM25 + reader) và B3; phân tích lỗi trích xuất. |
| **B: Nguyễn Bá Hiển** | Retrieval Engineer (C2) | Tách module truy hồi từ hệ thống nền (Qdrant, BM25); thay embedding bằng BGE-M3; RRF; xếp hạng lại theo tiêu đề; đánh giá truy hồi (Recall@k, MRR@10) trên ViQuAD và miền ứng dụng; phân tích lỗi truy hồi. |
| **C: Nguyễn Thị Hiền Trang** | RAG & Product Lead (C3) | Repo, CI, W&B, Charter; thay Gemini bằng Qwen2.5-7B (4-bit); prompt, ràng buộc ngữ cảnh, từ chối; chạy baseline B2; giao thức chấm ảo giác; demo chatbot; hợp nhất báo cáo ACL và slide. |

**Việc chung:** mỗi người viết 33 câu (kèm đoạn nguồn và đáp án), sau đó gán nhãn độc lập 33 câu của người kế (A gán câu của B, B gán câu của C, C gán câu của A); bất đồng thống nhất ở Tuần 3 T2. Việc chấm ảo giác do C thực hiện toàn bộ; A và B chấm lại ngẫu nhiên khoảng 30% để tính kappa (mục 3.4).

### 2.2. Quy định phối hợp chung

- Review chéo theo vòng **A → B → C → A** qua Pull Request. Người review đọc đủ kỹ để giải thích được phần đó.
- Mỗi người tự viết phần báo cáo và slide của mình; C hợp nhất và thống nhất văn phong.
- **Vấn đáp (30% điểm, cá nhân):** có thể hỏi phần của bạn khác và đoạn code ngẫu nhiên; dưới 4/10 thì tổng không quá 5/10. Có 3 buổi tập vấn đáp chéo (cuối Tuần 4, cuối Tuần 5, Tuần 6 T5) và mỗi người có tóm tắt 1 trang về phần mình và phần của người khác.
- Việc dùng công cụ AI được ghi vào docs/ai_usage.md tại thời điểm sử dụng; phần tái sử dụng từ hệ thống nền được ghi vào docs/reuse_vs_new.md từ Tuần 1.

## 3. Lựa chọn kỹ thuật và công cụ (Stack)

### 3.1. Dữ liệu

- **UIT-ViQuAD 2.0:** dùng để fine-tune extractive QA và làm kho truy hồi (gộp mọi đoạn văn thành một kho, đoạn gốc của câu hỏi có đáp án là nhãn đúng; **câu không có đáp án bị loại khỏi Recall@k**). Tải bằng scripts/download_data.py. Theo bài báo: 35.990 câu = train 28.457 + test công khai 3.821 + test riêng của cuộc thi 3.712. Nhóm dùng bản trên Hugging Face (taidng/UIT-ViQuAD2.0; trường license để trống, ghi rõ nguồn này trong báo cáo). Kiểm tra của nhóm cho thấy: split train có 28.454 dòng (4.101 đoạn, 138 bài); split **validation** (3.814 dòng, 557 đoạn, 19 bài, có nhãn) **chính là test công khai**, chỉ dùng làm test; split test của bản này không có nhãn và có đoạn trùng với train nên **không dùng**. Bộ dữ liệu không có dev riêng, nên **dev = 10% số bài viết của train, tách theo tiêu đề bài (title)**, không tách theo câu hỏi vì mỗi đoạn có khoảng 7 câu và tách theo câu sẽ làm đoạn nằm ở cả train lẫn dev. Báo cáo cuối ghi số đếm thực tế của file nhóm dùng (28.454 và 3.814), nêu rõ lệch nhẹ so với số của bài báo.
- **Miền ứng dụng (miền chính):** Quy chế đào tạo đại học của ĐHQGHN, bản 2014 (Quyết định 5115/QĐ-ĐHQGHN), tệp data/domain/dhqghn_quy_che_dao_tao_dai_hoc_2014.pdf. Văn bản tiếng Việt, 48 trang, 47 điều; ba trang cuối là mục lục và bị loại khỏi kho. Kho được cắt đoạn theo Điều. Lý do chọn: kho Samsung Galaxy S24 của hệ thống nền chỉ khoảng 8 trang nên Recall@10 bão hòa với mọi phương pháp và không phân biệt được BM25, dense, hybrid; đề bài cũng nêu quy chế đào tạo như một loại kho phù hợp. Quyết định và hệ quả được ghi ở docs/charter.md, mục 4.1.
- **Miền phụ (lệch ngôn ngữ):** hai tệp PDF tiếng Anh về Samsung Galaxy S24 của hệ thống nền (data/domain/). Câu hỏi bằng tiếng Việt, nên truy hồi là **liên ngôn ngữ**; BM25 dự kiến yếu (ít từ khóa chung ngoài tên model và số liệu). Một bộ 20–30 câu riêng phục vụ thí nghiệm lệch ngôn ngữ và ablation dịch câu hỏi; bộ này không thuộc tập 100 câu và không dùng cho tiêu chí (ii), (iii) của mục 1.3.
- **Tập 100 câu tự gán nhãn (miền chính):** gồm khoảng 90 câu trong phạm vi và 10 câu ngoài phạm vi (để đo từ chối đúng). Mỗi câu có đoạn nguồn và đáp án, gán bởi hai thành viên độc lập (tác giả câu hỏi và một người gán nhãn không xem nhãn của tác giả, xoay vòng A→B→C→A), bất đồng được thống nhất lại và báo cáo tỉ lệ đồng thuận. Chia **30 câu dev** (tuning) và **70 câu test** (chạy một lần ở Tuần 5, T2). Văn bản là bản 2014 nên câu hỏi không dùng dạng "quy định hiện hành"; năm ban hành được nêu trong báo cáo.
- **Đặc điểm cần xử lý:** PhoBERT chỉ có 258 vị trí nhúng nên đoạn dài cần cắt bằng sliding window; PDF Samsung S24 có bảng thông số dễ vỡ khi cắt đoạn; PDF Quy chế cần loại phần mục lục, cắt theo Điều và trích văn bản bằng pdfplumber (ít lỗi tách chữ tiếng Việt hơn pypdf); VnCoreNLP cần Java.

### 3.2. Công nghệ sử dụng

- **Ngôn ngữ và thư viện:** Python, PyTorch, transformers, datasets, accelerate, VnCoreNLP (py_vncorenlp), sentence-transformers/FlagEmbedding (BGE-M3), qdrant-client, rank_bm25, pdfplumber, bitsandbytes (4-bit), peft (nếu làm LoRA), pytest.
- **Mô hình:** PhoBERT-base và XLM-R-base (đối chiếu) cho extractive QA; BAAI/bge-m3 (embedding); Qwen2.5-7B-Instruct 4-bit (Apache-2.0) làm LLM; Qwen2.5-3B chỉ dùng dự phòng khi hết VRAM hoặc quá chậm (kiểm tra giấy phép của bản 3B trước khi dùng).
- **Quản lý thí nghiệm và triển khai:** Weights & Biases, Docker/Qdrant (dùng lại từ hệ thống nền), Gradio, GitHub, GitHub Actions chạy pytest.
- **Tài nguyên:** Kaggle/Colab GPU (T4 trở lên). Lưu checkpoint thường xuyên vì phiên miễn phí có thể bị ngắt.
- **Môi trường ảo (bắt buộc):** mỗi người dùng một virtual environment riêng, cài đúng phiên bản trong requirements.txt:

```bash
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

Thư mục venv/ không được commit. Python 3.12 là phiên bản yêu cầu. requirements-ci.txt là tập phụ thuộc tối thiểu cho GitHub Actions (chỉ chạy unit test, không cần GPU). Mỗi lần thêm hoặc nâng cấp thư viện, phiên bản cụ thể được cập nhật trong requirements.txt và thông báo cho cả nhóm.

### 3.3. Mô hình và cấu hình so sánh

- **Baseline theo Proposal:** B1 = BM25 + mô hình đọc hiểu; B2 = hệ thống nền (Gemini) trên tập 100 câu; B3 = mBERT (số liệu từ bài báo, không tự chạy).
- **Extractive QA:** PhoBERT-base (tách từ) là mô hình chính; XLM-R-base để đối chiếu; PhoBERT không tách từ làm ablation nhỏ.
- **Retrieval:** BM25 / dense (BGE-M3) / hybrid (RRF) / hybrid + xếp hạng lại theo tiêu đề; số đoạn đưa vào LLM k ∈ {3, 5, 10}.
- **RAG:** Qwen2.5-7B với prompt có ràng buộc "chỉ dựa trên ngữ cảnh, nếu không có thì nói không biết". Chế độ trích xuất (PhoBERT đọc các đoạn đã truy hồi) để so sánh với chế độ sinh. Dịch truy vấn sang tiếng Anh trước khi truy hồi (giải quyết lệch ngôn ngữ; ablation theo Proposal, thực hiện trên miền phụ S24). Chỉ làm nếu thừa thời gian: LoRA một LLM nhỏ.

### 3.4. Chỉ số đánh giá (Evaluation Metrics)

- **Extractive QA:** EM và F1 theo SQuAD 2.0, báo cáo riêng câu có và không có đáp án; 95% CI bằng bootstrap theo câu hỏi.
- **Retrieval:** Recall@{1, 5, 10, 20} và MRR@10; so sánh cặp bằng bootstrap CI của chênh lệch Recall@5; độ trễ mỗi truy vấn.
- **RAG (chấm tay):** tỷ lệ ảo giác, từ chối đúng, độ trễ, bộ nhớ GPU. Một người chấm toàn bộ và người thứ hai chấm lại ngẫu nhiên khoảng 30% để tính Cohen’s kappa.
- **Kiểm tra phụ:** tỷ lệ câu trả lời lẫn ký tự CJK (tiếng Trung).

### 3.5. Quy ước kỹ thuật chính

#### a) Chia dữ liệu

- **ViQuAD:** tách ngẫu nhiên 10% số bài viết (theo title) của train làm dev, cố định seed, lưu danh sách bài và chỉ số vào data/splits/;  báo kết quả cuối trên test công khai, **chạy một lần** ở Tuần 5, T2.
- **Tập 100 câu:** 30 dev (tuning k, trọng số, prompt) và 70 test (chạy một lần ở Tuần 5, T2). Không chỉnh gì dựa trên câu test. Vì dev chỉ có 30 câu nên kết quả tuning trên dev nhiễu; ghi rõ trong Limitations.

#### b) Chống rò rỉ dữ liệu

- Dữ liệu LoRA (nếu làm) không được sinh từ chunk chứa đáp án của câu test. Tập kiểm tra miền ứng dụng không dùng để huấn luyện.

#### c) Hybrid retrieval

- Lấy danh sách xếp hạng từ BM25 và dense rồi hợp nhất bằng RRF (score = Σ 1/(k + rank)); coi k là siêu tham số, thử k ∈ {15, 60} và chọn theo dev. Hệ thống nền (B2) giữ nguyên cấu hình gốc (RRF_K = 15) vì là baseline; ghi rõ giá trị k dùng ở từng phần của báo cáo; thử xếp hạng lại theo độ tương đồng tiêu đề (của hệ thống nền) như một thành phần ablation. Ghi lại mọi tham số đã chọn.

#### d) Lệch ngôn ngữ (câu hỏi tiếng Việt, tài liệu tiếng Anh)

- Đo riêng BM25, dense và hybrid trên miền phụ (PDF Samsung S24 tiếng Anh, câu hỏi tiếng Việt). Giả thuyết: BM25 yếu, dense đa ngữ và hybrid khá hơn. Thí nghiệm mở rộng: dịch hoặc viết lại truy vấn sang tiếng Anh bằng Qwen trước khi truy hồi. Trên miền chính (Quy chế, cùng ngôn ngữ với câu hỏi) BM25 không chịu bất lợi này nên kết quả có thể khác giả thuyết; mọi kết quả đều được báo cáo.

#### e) Kiểm soát sinh câu trả lời

- Prompt yêu cầu trả lời bằng tiếng Việt, chỉ dựa trên ngữ cảnh và nêu nguồn; tự động kiểm tra ký tự CJK và sinh lại nếu có. Nếu Qwen2.5-7B 4-bit quá chậm hoặc hết VRAM thì giảm số đoạn k hoặc dùng bản nhỏ hơn và ghi lại trong báo cáo.

## 4. Kế hoạch triển khai theo từng tuần

### 4.1. Tuần 1: Proposal và thiết lập (checklist)

Tuần 1 là tuần Proposal. Hạng mục trong danh sách dưới đây chưa hoàn thành được chuyển sang T2 của Tuần 2.

| **Việc** | **Người phụ trách** | **Ghi chú** |
|---|---|---|
| Proposal ≤ 2 trang theo template ACL, nộp cuối tuần | Cả nhóm (C hợp nhất) | Bản nộp lưu tại report/proposal/ (proposal.tex, proposal.pdf) |
| Repo mới DL-Group09-vietnamese-qa-rag; W&B; CI pytest; requirements.txt | C |  |
| docs/reuse_vs_new.md và docs/ai_usage.md | C | Bắt đầu ghi ngay, bổ sung dần |
| Project Charter (mục 1.3) | C | Chỉnh một lần sau baseline Tuần 2 |
| Cài môi trường: VnCoreNLP (cần Java), transformers, Qdrant; thử Qwen2.5-7B 4-bit trên Kaggle/Colab | A, B, C | Phát hiện sớm lỗi cài đặt và giới hạn VRAM |
| Khóa quy ước tách dữ liệu: dev = 10% bài viết (title) của train; split validation của Hugging Face = test công khai; không dùng split test (không nhãn). Chạy lại thống kê trên file nhóm tải và ghi số đếm thực tế vào README | A | Tránh rò rỉ dữ liệu; có unit test kiểm tra không đoạn nào nằm ở cả train và dev |
| Hỏi giảng viên: dùng lại hệ thống nền có được không; trang tham khảo có tính vào 2 trang không | C | Ghi lại câu trả lời vào repo |

### 4.2. Tuần 2: Dữ liệu, baseline và tập đánh giá

**Mục tiêu tuần:** pipeline QA và truy hồi chạy được, tập 100 câu được viết và gán nhãn hai lượt, baseline B2 đã chạy.

| **Ngày** | **Hoà (A): Extractive QA** | **Hiển (B): Retrieval** | **Trang (C): RAG & Product** |
|---|---|---|---|
| **T2** | Tải ViQuAD (download_data.py); EDA nhanh; kiểm tra VnCoreNLP với tokenizer PhoBERT; viết EM/F1 kèm unit test. | Tách code truy hồi khỏi hệ thống nền; dựng Qdrant; index BGE-M3 và BM25 cho kho đoạn ViQuAD, kho Quy chế ĐHQGHN (cắt theo Điều, loại mục lục) và hai PDF S24. | Các mục còn lại của checklist Tuần 1. Viết bootstrap CI dùng chung (unit test) và hướng dẫn gán nhãn (docs/annotation_guide.md) cho miền Quy chế. |
| **T3** | Tiền xử lý QA: tách từ, tokenize, sliding window (max_len 256, doc_stride), ánh xạ offset; unit test. | Viết Recall@k, MRR@10 (unit test); chạy BM25 vs dense trên kho ViQuAD dev. | Viết prompt RAG v1 cho Qwen 7B (tiếng Việt, ràng buộc ngữ cảnh, nêu nguồn); kiểm tra thử 10 câu. |
| **T4** | Fine-tune PhoBERT lần 1 (lr 3e-5, 3 epoch); log W&B; đo thời gian 1 epoch để chốt ngân sách GPU. | Hybrid RRF (thử k ∈ {15, 60}); tách module xếp hạng lại theo tiêu đề; bảng retrieval sơ bộ. | Nối truy hồi của B với Qwen 7B thành RAG v0 chạy được; viết 15 câu gán nhãn đầu tiên. |
| **T5** | Viết 33 câu gán nhãn (đoạn nguồn, đáp án). | Viết 33 câu gán nhãn. | Viết 18 câu gán nhãn còn lại (gồm khoảng 10 câu ngoài phạm vi cho cả nhóm). |
| **T6** | Gán nhãn độc lập 33 câu của B (không xem nhãn của tác giả); đánh giá dev EM/F1 của PhoBERT lần 1. Review PR (A→B). | Gán nhãn độc lập 33 câu của C; chạy BM25/dense/hybrid trên tập 100 câu (chấm sau khi chốt nhãn). Review PR (B→C). | Gán nhãn độc lập 33 câu của A; chạy baseline B2 (Gemini, giữ cấu hình gốc RRF_K = 15) trên tập 100 câu (hệ thống nền nạp kho Quy chế; mọi chỉnh sửa prompt được ghi lại), theo dõi hạn mức API; họp review, chỉnh Charter. Review PR (C→A). |

**Deliverables Tuần 2:** ViQuAD và index chạy được, tiền xử lý QA và metric có test, tập 100 câu (đã gán nhãn hai lượt, chốt ở Tuần 3 T2), B2 đã chạy (chấm khi nhãn chốt), PhoBERT lần 1, bảng retrieval sơ bộ.

### 4.3. Tuần 3: Phương pháp chính và Milestone report

**Mục tiêu tuần:** PhoBERT đã tuning, retrieval v1 chốt, RAG v1 và chế độ trích xuất chạy được; nộp Milestone report (≤ 4 trang).

| **Ngày** | **Hoà (A): Extractive QA** | **Hiển (B): Retrieval** | **Trang (C): RAG & Product** |
|---|---|---|---|
| **T2** | Họp 30 phút cả nhóm: thống nhất bất đồng nhãn, chốt tập 100 câu. Tuning PhoBERT: tối đa 3 cấu hình (lr {2e-5, 3e-5, 5e-5}); chọn theo dev F1. | Dự họp chốt nhãn. Tune k của RRF và trọng số hybrid trên dev (kho ViQuAD và 30 câu dev miền ứng dụng). | Chủ trì họp chốt nhãn, tính tỉ lệ đồng thuận. RAG v1 = hybrid + Qwen 7B; kiểm tra ký tự CJK và sinh lại; log độ trễ và bộ nhớ GPU. |
| **T3** | Chốt PhoBERT; n-best và ngưỡng không-có-đáp-án trên dev. | **Chốt retrieval v1**; đo độ trễ và Recall@k ở cả hai miền. | Viết giao thức chấm ảo giác (nhãn, ví dụ từng nhãn); chạy RAG v1 trên 30 câu dev để thử giao thức. |
| **T4** | Baseline B1 (BM25 + PhoBERT reader đã tuning) trên tập 100 câu; phân tích lỗi sơ bộ PhoBERT trên dev. | Viết 20–30 câu hỏi tiếng Việt cho hai PDF S24 (C kiểm tra nhãn). Thí nghiệm lệch ngôn ngữ: BM25/dense/hybrid trên PDF tiếng Anh với các câu hỏi này. | Nối chế độ trích xuất (PhoBERT đọc các đoạn truy hồi) vào pipeline và demo khung. |
| **T5** | Viết phần Phương pháp và Kết quả extractive QA cho Milestone. | Viết phần Retrieval cho Milestone. | Viết phần RAG, tiến độ, khó khăn và kế hoạch còn lại; dựng khung LaTeX ACL (report/milestone/ và report/final/, mỗi phần một file .tex) và hợp nhất. |
| **T6** | Review chéo; nộp Milestone. | Review chéo; nộp Milestone. | Họp chốt phạm vi Tuần 4–6; **quyết định go/no-go LoRA**. |

**Deliverables Tuần 3:** PhoBERT đã tuning, retrieval v1, RAG v1, chế độ trích xuất, giao thức chấm ảo giác, tập 100 câu đã chốt nhãn, B1, **Milestone report (nộp)**.

### 4.4. Tuần 4: Thí nghiệm và ablation

**Mục tiêu tuần:** hoàn thành toàn bộ ablation và chốt cấu hình cuối. **Experiment freeze cuối T5.**

| **Ngày** | **Hoà (A): Extractive QA** | **Hiển (B): Retrieval** | **Trang (C): RAG & Product** |
|---|---|---|---|
| **T2** | XLM-R-base cùng cấu hình với PhoBERT; PhoBERT không tách từ (ít epoch, 1 seed). | Ablation retrieval: BM25 / dense / hybrid / hybrid + xếp hạng lại; k ∈ {3, 5, 10} đưa vào LLM. | Ablation prompt: có/không ràng buộc ngữ cảnh; đo ảo giác và từ chối trên 30 câu dev. |
| **T3** | Chạy thêm 2 seed cho PhoBERT tốt nhất (nếu đủ GPU); mean ± std và bootstrap CI. | Dịch truy vấn sang tiếng Anh trước khi truy hồi (ablation theo Proposal, trên miền phụ S24); bootstrap CI chênh lệch Recall@5. | So sánh chế độ sinh bằng LLM và chế độ trích xuất trên 30 câu dev; ghi độ trễ. |
| **T4** | Phân tích lỗi có phân loại cho extractive QA trên dev (≥ 50 mẫu sai, gán loại lỗi). | Phân tích lỗi retrieval (≥ 30 truy vấn trượt: cắt đoạn, bảng, lệch ngôn ngữ, từ đồng nghĩa). | Hoàn thiện demo hai chế độ, hiển thị nguồn; chuẩn bị script chạy test. |
| **T5** | **Chốt cấu hình cuối và experiment freeze.** Review PR chéo. | **Chốt cấu hình retrieval cuối.** Review PR chéo. | **Chốt cấu hình RAG cuối.** Họp tuần. |
| **T6** | Tập vấn đáp #1 (mỗi người 5–7 phút, bị hỏi phần người khác). | Tập vấn đáp #1. | Tập vấn đáp #1; viết tóm tắt 1 trang phần mình (cả nhóm làm). |

**Deliverables Tuần 4:** kết quả ablation trên dev, CI, mean ± std, phân loại lỗi, demo hai chế độ, cấu hình cuối đóng băng.

### 4.5. Tuần 5: Đánh giá cuối trên test và bản nháp báo cáo

**Mục tiêu tuần:** chạy test **đúng một lần**, chấm ảo giác, viết bản nháp đầy đủ.

| **Ngày** | **Hoà (A): Extractive QA** | **Hiển (B): Retrieval** | **Trang (C): RAG & Product** |
|---|---|---|---|
| **T2** | **Test cuối một lần:** PhoBERT (và XLM-R) trên ViQuAD test công khai. | **Test cuối một lần:** retrieval trên kho ViQuAD và 70 câu test miền ứng dụng. | **Test cuối một lần:** RAG (B2, RAG cơ bản, RAG cuối) trên 70 câu test; bắt đầu chấm tay. |
| **T3** | Chấm lại ngẫu nhiên khoảng 30% câu RAG để tính kappa; tổng hợp Bảng 1. | Chấm lại phần còn lại của 30% để tính kappa; tổng hợp Bảng 2. | Hoàn tất chấm ảo giác; tính kappa; tổng hợp Bảng 3 và 4. |
| **T4** | Phân tích lỗi trên test (bổ sung vào phân tích dev); hình, bảng và các ví dụ đầu ra tốt/xấu; (nếu go) LoRA nhỏ. | Phân tích lỗi trên test; hình Recall@k. | Bản nháp phần RAG, Limitations và hình kiến trúc hệ thống; ghép bản nháp toàn báo cáo. |
| **T5** | Viết bản nháp phần extractive QA (Phương pháp, Kết quả, Phân tích lỗi). | Viết bản nháp phần Retrieval (kể cả lệch ngôn ngữ). | Hoàn thiện bản nháp ACL ≤ 8 trang; kiểm tra trích dẫn. |
| **T6** | Tập vấn đáp #2; review chéo bản nháp. | Tập vấn đáp #2; review chéo bản nháp. | Tập vấn đáp #2; gửi bản nháp cho giảng viên/bạn đọc thử nếu được. |

**Deliverables Tuần 5:** kết quả test cuối (Bảng 1–4), kappa, phân loại lỗi trên test, bản nháp báo cáo đầy đủ (không chấm điểm).

### 4.6. Tuần 6: Hoàn thiện, thuyết trình và nộp bài

**Mục tiêu tuần:** báo cáo, repo, slide và demo hoàn chỉnh; nộp đúng hạn.

| **Ngày** | **Hoà (A): Extractive QA** | **Hiển (B): Retrieval** | **Trang (C): RAG & Product** |
|---|---|---|---|
| **T2** | Sửa báo cáo theo review; hoàn thiện phần của mình. | Sửa báo cáo; kiểm tra số liệu bảng khớp log W&B. | Hoàn thiện báo cáo; Limitations; phần Đóng góp của thành viên (tỉ lệ ước tính, cả nhóm thống nhất); khai báo công cụ AI; bảng tự cài vs dùng sẵn; BibTeX. |
| **T3** | Slide phần extractive QA; chuẩn bị kịch bản demo. | Slide phần Retrieval. | Slide tổng hợp và demo; quay video demo dự phòng. |
| **T4** | Dọn repo: chạy lại từ đầu trên môi trường sạch; xóa output notebook. | README hướng dẫn cài đặt, tải dữ liệu, huấn luyện, đánh giá và tái tạo index, bảng kết quả. | Kiểm tra CI xanh, W&B và repo đã chia sẻ; kiểm tra format ACL final và giới hạn trang. |
| **T5** | Tập vấn đáp #3 và tổng duyệt thuyết trình. | Tập vấn đáp #3 và tổng duyệt. | Tập vấn đáp #3; kiểm tra lần cuối số liệu và trích dẫn. |
| **T6** | Nộp: Nhom09_DeTai10.pdf, mã nguồn, slide (dự phòng thời gian). | Hỗ trợ nộp; kiểm tra file mở được. | Nộp và xác nhận đã nộp đủ sản phẩm. |

**Deliverables Tuần 6:** báo cáo cuối ACL, repo hoàn chỉnh, slide, demo chạy được.

### 4.7. Thứ tự cắt khi bị trễ

Các hạng mục bị cắt theo thứ tự từ trên xuống cho đến khi kịp tiến độ. Phân tích lỗi, ablation cốt lõi, kiểm soát dev/test và tập vấn đáp không thuộc danh sách cắt. Hạng mục đã nêu trong Proposal mà bị cắt được ghi chú thay đổi trong báo cáo.

| **Thứ tự** | **Hạng mục bị cắt** | **Cách xử lý trong báo cáo** |
|---|---|---|
| 1 | LoRA | Ghi vào "Hướng phát triển". |
| 2 | 2 seed thêm cho PhoBERT | Báo 1 seed và nêu rõ trong Limitations. |
| 3 | PhoBERT không tách từ | Ghi chú thay đổi so với Proposal. |
| 4 | Dịch truy vấn sang tiếng Anh | Ghi chú thay đổi so với Proposal; nêu như hướng cải thiện cho lệch ngôn ngữ. |
| 5 | XLM-R đối chiếu | Đề chỉ yêu cầu PhoBERT hoặc XLM-R; ghi chú thay đổi so với Proposal. |
| 6 | Rút tập 100 câu xuống 60 (20 dev, 40 test) | Giữ hai người gán nhãn; ghi rõ CI rộng hơn. |

## 5. Mẫu bảng so sánh

Các bảng dưới đây là khung; số liệu được lấy từ log W&B, không nhập tay.

#### Bảng 1. Extractive QA (ViQuAD dev/test)

| **Mô hình** | **Tách từ** | **EM** | **F1** | **F1 câu không đáp án** | **Bootstrap 95% CI của F1** |
|---|---|---|---|---|---|
| B3: mBERT (từ bài báo) | Không áp dụng |  |  |  |  |
| PhoBERT-base | Có |  |  |  |  |
| XLM-R-base (mở rộng) | Không |  |  |  |  |

#### Bảng 2. Retrieval (kho ViQuAD và tập 100 câu miền Quy chế; miền phụ S24 báo riêng)

| **Phương pháp** | **R@1** | **R@5** | **R@10** | **R@20** | **MRR@10** | **Độ trễ / truy vấn** |
|---|---|---|---|---|---|---|
| BM25 |  |  |  |  |  |  |
| Dense (BGE-M3) |  |  |  |  |  |  |
| Hybrid (RRF) |  |  |  |  |  |  |
| Hybrid + xếp hạng lại theo tiêu đề |  |  |  |  |  |  |

#### Bảng 3. RAG (70 câu test, chấm tay)

| **Cấu hình** | **Ảo giác** | **Từ chối đúng** | **Từ chối sai** | **Lẫn CJK** | **Độ trễ / GPU** |
|---|---|---|---|---|---|
| B2: hệ thống nền (Gemini) |  |  |  |  |  |
| Qwen2.5-7B, prompt cơ bản |  |  |  |  |  |
| Qwen2.5-7B, prompt ràng buộc (cuối) |  |  |  |  |  |

#### Bảng 4. Ablation tổng hợp

| **Thay đổi** | **Chỉ số bị ảnh hưởng** | **Kết quả (± CI)** |
|---|---|---|
| BM25 vs dense vs hybrid | Recall@5 |  |
| Có/không xếp hạng lại theo tiêu đề | Recall@5, độ trễ |  |
| Có/không ràng buộc ngữ cảnh | Ảo giác, từ chối đúng |  |
| Số đoạn k ∈ {3, 5, 10} | Ảo giác, độ trễ |  |

#### Bảng 5. Phân loại lỗi (≥ 50 mẫu sai extractive QA, ≥ 30 truy vấn trượt)

| **Nhóm lỗi** | **Mô tả ngắn** | **Số mẫu** | **Ví dụ (id)** |
|---|---|---|---|
| Extractive: sai biên span |  |  |  |
| Extractive: nhầm câu không có đáp án |  |  |  |
| Extractive: sai số liệu/thực thể |  |  |  |
| Retrieval: đoạn bị cắt/mất ngữ cảnh |  |  |  |
| Retrieval: lệch ngôn ngữ |  |  |  |
| Retrieval: bảng biểu |  |  |  |

## 6. Cấu trúc Repository chuẩn

```text
DL-Group09-vietnamese-qa-rag/
├── data/
│   ├── raw/                  # gitignore
│   ├── domain/               # commit: PDF Quy chế ĐHQGHN (miền chính), 2 PDF Samsung S24 (miền phụ)
│   ├── gold/                 # commit: gold_dev.jsonl, gold_test.jsonl
│   └── splits/               # commit: chỉ số/seed các lần chia
├── notebooks/
│   ├── 00_kaggle_qwen_smoke_test.ipynb
│   ├── 01_eda_viquad.ipynb
│   ├── 02_retrieval_analysis.ipynb
│   └── 03_error_analysis.ipynb
├── kaggle/smoke_test_dataset/  # dữ liệu nhỏ cho notebook smoke test
├── scripts/
│   ├── download_data.py
│   ├── build_index.py
│   └── run_eval.py
├── src/
│   ├── preprocessing_qa.py   # tách từ, tokenize, sliding window
│   ├── train_qa.py           # fine-tune PhoBERT/XLM-R
│   ├── retrieval.py          # BM25, dense, RRF, xếp hạng lại
│   ├── generate.py           # prompt, Qwen, kiểm tra CJK
│   ├── metrics.py            # EM/F1, Recall@k, MRR, bootstrap
│   └── pipeline.py
├── tests/
├── app/app.py                # demo
├── docs/
│   ├── project_plan.md       # kế hoạch dự án (tài liệu này)
│   ├── charter.md
│   ├── ai_usage.md
│   ├── reuse_vs_new.md       # phần dùng lại từ hệ thống nền
│   ├── annotation_guide.md
│   └── results/              # kết quả nhỏ đã chốt (ví dụ qwen_smoke_test/)
├── legacy/                   # mã hệ thống nền, chỉ để tham khảo, không import từ src/
├── report/                   # LaTeX ACL, mỗi mốc nộp một thư mục tự chứa
│   ├── proposal/             # bản đã nộp
│   ├── milestone/
│   └── final/                # main.tex, sections/, figures/, references.bib
├── .github/workflows/ci.yml
├── .gitignore
├── pytest.ini
├── requirements.txt          # pin version cụ thể
├── requirements-ci.txt       # tập tối thiểu cho CI
└── README.md                 # cách chạy + bảng tự cài vs dùng sẵn
```

CI (GitHub Actions) chạy pytest ở mỗi Pull Request. Mỗi mốc nộp có một thư mục riêng trong report/ (proposal/, milestone/, final/), tự chứa template ACL và references.bib; bản đã nộp được giữ nguyên làm bản lưu. Trong report/final/, mỗi phần là một file .tex trong sections/ để hạn chế xung đột khi merge; .gitignore bỏ file build (*.aux, *.log, *.out, *.bbl, *.blg, *.synctex.gz). Thư mục legacy/ chứa mã hệ thống nền chỉ để tham khảo.

## 7. Quản lý rủi ro và phương án xử lý

| **Rủi ro** | **Mô tả** | **Phương án xử lý** |
|---|---|---|
| **Trễ tiến độ hoặc thiếu GPU** | GPU miễn phí giới hạn giờ; một nhánh kéo dài làm trễ cả nhóm. | Giữ MVP (mục 1.3); đo 1 epoch ở Tuần 2 để chốt ngân sách; cắt theo thứ tự ở mục 4.7; họp review cuối mỗi tuần. |
| **Lệch ngôn ngữ (câu hỏi tiếng Việt, tài liệu tiếng Anh)** | BM25 gần như không khớp từ khóa; Recall thấp trên miền phụ S24. | Đo riêng trên miền phụ và nêu như một kết quả; thử dịch truy vấn; dựa vào dense đa ngữ và hybrid. |
| **Kho miền ứng dụng quá nhỏ** | Kho S24 khoảng 8 trang làm Recall@k bão hòa, không phân biệt được các phương pháp. | Miền chính là Quy chế ĐHQGHN (47 điều); S24 chỉ dùng cho thí nghiệm lệch ngôn ngữ. |
| **Tập đánh giá nhỏ và do nhóm tự viết** | 100 câu, dev chỉ 30 câu; có thiên lệch và CI rộng. | Hai người gán nhãn độc lập, kiểm tra chéo; báo CI; nêu trong Limitations; KPI tin cậy đặt trên kho ViQuAD. |
| **Rò rỉ dữ liệu đánh giá** | Chỉnh prompt hay tham số theo câu test; tách dev theo câu hỏi làm đoạn văn nằm ở cả train và dev; dùng nhầm split validation (là test công khai) làm dev. | Dev tách theo bài viết (title) và có test kiểm tra không trùng đoạn; split validation chỉ dùng làm test; test chạy một lần ở Tuần 5, T2. |
| **Qwen2.5-7B 4-bit chậm hoặc hết VRAM** | Phiên Kaggle/Colab giới hạn; sinh chậm. | Giảm số đoạn k; dùng bản nhỏ hơn dự phòng (kiểm tra giấy phép); ghi độ trễ trong báo cáo. |
| **Hạn mức API cho B2** | Gemini có giới hạn, có thể không chạy hết. | Chỉ chạy B2 trên 100 câu (không chạy toàn ViQuAD); chạy sớm khi nhãn đã đủ. |
| **VnCoreNLP cần Java, PhoBERT giới hạn độ dài** | Cài đặt môi trường lỗi; đoạn dài bị cắt mất đáp án. | Cài đặt và kiểm tra môi trường từ Tuần 1; sliding window có unit test cho ánh xạ offset. |
| **Một thành viên không nắm phần của người khác** | Điểm vấn đáp cá nhân thấp kéo tổng điểm xuống. | Review chéo vòng tròn; 3 buổi tập vấn đáp chéo; mỗi người tự chạy lại pipeline từ README. |

## 8. Quy trình làm việc và quy tắc phát triển

### 8.1. Quy định sử dụng Git

- **Branch:** `<role>/<task>`, ví dụ qa/phobert-finetune, retrieval/hybrid, rag/prompt-v1. Không làm trực tiếp trên main.
- **Push/Pull:** mỗi người chỉ push branch của mình; pull/rebase từ main đầu mỗi ngày.
- **Merge:** qua Pull Request, ít nhất 1 người khác approve và CI xanh; PR nhỏ, mỗi PR một việc.
- **Tập tin:** không commit dữ liệu thô, index, checkpoint; xóa output notebook trước khi commit.
- **Tên file:** tiếng Anh, không dấu, viết thường, snake_case; nội dung báo cáo, comment, docstring viết bằng tiếng Việt.
- **Commit message**: Theo chuẩn Conventional Commits (ví dụ: feat: add IV-based feature selection, fix: leakage in scaler fit).

### 8.2. Quy định code và quản lý thí nghiệm

- Mọi lần chạy log lên W&B với tên run theo mẫu `<role>_<model>_<cấu hình>_seed<k>`, kèm commit hash.
- Cố định seed, lưu cấu hình từng thí nghiệm; mọi bảng trong báo cáo sinh từ script hoặc log, không sửa tay.
- Có unit test cho phần dễ sai: ánh xạ offset QA, chuẩn hóa EM/F1, Recall@k, RRF.

### 8.3. Cơ chế giao tiếp và quản lý nhóm

- **Họp:** kickoff đầu tuần (T2), daily sync 10 phút trên nhóm chat, họp review cuối tuần (T6) ghi biên bản ngắn.
- **Theo dõi việc:** GitHub Projects; mỗi dòng trong bảng tuần ở mục 4 là một issue có người phụ trách.
- **Khi chậm tiến độ hoặc gặp vướng mắc:** vấn đề được thông báo cho cả nhóm trong ngày; hạng mục mở rộng được cắt trước hạng mục đánh giá.
- **Trước mỗi mốc nộp:** checklist gồm: template ACL với `\usepackage[final]{acl}`; họ tên và MSSV của ba thành viên; khổ A4 hai cột, giữ nguyên font, cỡ chữ và lề; biên dịch bằng XeLaTeX hoặc LuaLaTeX với gói tiếng Việt; trích dẫn tác giả–năm (`\citet`, `\citep`) với BibTeX; đúng giới hạn trang (phần vượt quá không được chấm); số liệu khớp log; trích dẫn đủ; khai báo công cụ AI; repo chạy lại được.
