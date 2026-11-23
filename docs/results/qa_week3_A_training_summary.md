Phạm vi báo cáo: Tổng hợp kết quả huấn luyện PhoBERT QA trong Tuần 3, đánh giá trên ViQuAD Dev nội bộ, xây dựng baseline B1 (BM25 + PhoBERT)

Báo cáo phân biệt hai nhóm kết quả:
- Kết quả lịch sử Tuần 3: được ghi nhận trong phobert_3lr_clean_dev_EVIDENCE.zip, gồm 15 tệp manifest, lịch sử huấn luyện, tổng kết huấn luyện và đánh giá n-best. Gói evidence này không chứa trọng số mô hình hoặc predictions đầy đủ từng câu.
- Kết quả kiểm toán bổ sung: được tính từ các tệp checkpoint, predictions và kết quả đánh giá đã lưu trên máy thực nghiệm. Việc kiểm toán không bao gồm huấn luyện lại mô hình hoặc đánh giá trên Test.
1. Cấu hình huấn luyện và dữ liệu
1.1. Cấu hình PhoBERT
- Mô hình nền: vinai/phobert-base.
- Số epoch: 3.
- Random seed: 12.
- Các learning rate được thử nghiệm: 2e-5, 3e-5, 5e-5.
- Bài toán: Extractive Question Answering, gồm câu hỏi có đáp án và không có đáp án.
- Phương pháp hậu xử lý: n-best span decoding kết hợp ngưỡng phát hiện câu hỏi không có đáp án.
1.2. Dữ liệu ViQuAD Dev nội bộ
Tập Dev được tách từ dữ liệu Train theo title, nhằm hạn chế việc cùng bài viết xuất hiện đồng thời trong Train và Dev.
Chỉ số	Giá trị
Số title	13
Tổng số câu hỏi	2.768
Câu có đáp án	1.886
Câu không có đáp án	882
Tổng số features	3.155
Câu bị bỏ qua	0
Câu không căn chỉnh được đáp án (unalignable)	29


Theo kết quả kiểm tra dữ liệu, không phát hiện trùng lặp title hoặc context giữa Train và Dev.

Trong giai đoạn đánh giá, các features được tạo bằng chế độ mode=eval, không sử dụng đáp án Gold để điều chỉnh đầu vào hoặc vị trí cửa sổ trượt.
2. Kết quả thực nghiệm PhoBERT Tuần 3
2.1. So sánh ba learning rates
Mỗi cấu hình được huấn luyện trong 3 epoch và đánh giá trên cùng tập Dev nội bộ.

Kết quả dưới đây được lấy từ evidence lịch sử Tuần 3.
Learning rate	Best epoch	Train-loop Dev F1 (%)	Dev EM sau tune (%)	Dev F1 sau tune (%)	Null threshold
2e-5	3	65,59	52,42	66,39	-1,483816
3e-5	3	67,35	53,76	67,93	-0,657978
5e-5	3	69,53	55,78	69,94	-1,489326


Cấu hình được lựa chọn: learning rate 5e-5, epoch 3, dựa trên F1 tổng thể cao nhất trên Dev.

Điểm F1 trong vòng lặp huấn luyện và F1 sau n-best decoding kết hợp tối ưu ngưỡng là hai phép đo khác nhau, không nên so sánh trực tiếp nếu chưa thống nhất cách tính.

Việc lựa chọn cấu hình được thực hiện trên Dev, không sử dụng Public Test.
2.2. Kết quả theo nhóm câu hỏi
Learning rate	F1 câu có đáp án (%)	EM/F1 câu không có đáp án (%)
2e-5	67,16	64,74
3e-5	70,33	62,81
5e-5	68,08	73,92


Nhận xét:
- Learning rate 3e-5 cho F1 cao nhất trên nhóm câu hỏi có đáp án.
- Learning rate 5e-5 cho kết quả tốt nhất trên nhóm câu hỏi không có đáp án.
- Xét F1 tổng thể, 5e-5 vẫn là cấu hình tốt nhất trong ba cấu hình được thử nghiệm.
Kết quả cho thấy cần đánh giá đồng thời khả năng trích xuất đáp án và khả năng xác định câu hỏi không có đáp án, thay vì chỉ tối ưu một nhóm.
3. Hậu xử lý, kiểm tra rò rỉ dữ liệu và khả năng tái lập
3.1. N-best decoding và ngưỡng không có đáp án
Quá trình hậu xử lý gồm:
1. Thu thập các vị trí bắt đầu và kết thúc đáp án từ các cửa sổ trượt.
2. Loại những span không hợp lệ hoặc vượt quá độ dài cho phép.
3. Ánh xạ vị trí đáp án về văn bản gốc.
4. Chọn đáp án từ các ứng viên hợp lệ.
5. So sánh điểm số với ngưỡng để quyết định trả về đáp án hay chuỗi rỗng.
Ngưỡng không có đáp án được tối ưu trên Dev. Không sử dụng Test để lựa chọn ngưỡng.
3.2. Kiểm toán Gold leakage
Chỉ số	Kết quả
Tổng số câu Dev	2.768
Câu có input features thay đổi do phụ thuộc Gold	29
Tỷ lệ bị ảnh hưởng	1,05%
Câu bị bỏ qua	0


4. Baseline B1 — BM25 kết hợp PhoBERT trên Domain Dev
4.1. Mục tiêu
B1 kết hợp hai thành phần:
- BM25: truy hồi các đoạn văn bản có khả năng chứa đáp án.
- PhoBERT QA: đọc các đoạn được truy hồi và dự đoán span đáp án.
Khác với việc đánh giá PhoBERT trực tiếp trên ViQuAD Dev, B1 được đánh giá trên bộ câu hỏi thuộc miền Quy chế đào tạo ĐHQGHN.
4.2. Cấu hình và kết quả
Tham số	Giá trị
Tập đánh giá	Domain Dev
Tổng số câu	30
Trong phạm vi	27
Ngoài phạm vi	3
Corpus	213 chunks
Phương pháp truy hồi	BM25
Top-k	3
Mô hình đọc hiểu	PhoBERT QA


Kết quả:
Cấu hình ngưỡng	EM (%)	F1 (%)
Fixed threshold = 0	66,67	84,89
Tuned threshold = -0,700919	66,67	84,89


Hai ngưỡng cho cùng kết quả trên 30 câu Domain Dev.
4.3. Giới hạn đánh giá B1
Ngưỡng được tối ưu và đánh giá trên cùng 30 câu Dev nên kết quả tuned là in-sample.

Ngưỡng còn phụ thuộc vào:
- Giá trị top_k.
- Tập chunks và chỉ mục BM25.
- Checkpoint PhoBERT.
- Cách tính score và tổng hợp các cửa sổ dự đoán.
Vì vậy, không mặc định áp dụng ngưỡng đã tối ưu ở Top-3 cho một cấu hình Top-k khác.
5. Đánh giá lại PhoBERT trên ViQuAD Clean Dev
Kết quả đánh giá lại checkpoint gốc LR 5e-5, seed 12 bằng pipeline hiện tại được đối chiếu với kết quả lịch sử Tuần 3.
Chỉ số	Lịch sử Tuần 3	Đánh giá lại checkpoint gốc	Chênh lệch
EM (%)	55,78	55,24	-0,54 điểm %
F1 (%)	69,94	70,10	+0,16 điểm %
Số câu Dev	2.768	2.768	0
Null threshold	-1,489326	≈ -0,64443	Khác


Kết quả đánh giá lại đạt:
- EM = 55,24%.
- F1 = 70,10%.
Số lượng câu hỏi:
- Có đáp án: 1.886 câu.
- Không có đáp án: 882 câu.
- Tổng cộng: 2.768 câu (3.155 features).
Đánh giá lại sử dụng checkpoint gốc tại phobert_5e5_extracted/phobert_qa_lr5e-5_seed12/best_model, bộ features Dev ở data/processed/qa_features_week4_eval/dev.jsonl và decoder hiện tại trong scripts/eval_qa_nbest.py. Kết quả được lưu tại outputs/week3_original_checkpoint_eval/summary.json.