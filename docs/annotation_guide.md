# Annotation Guide — Vietnamese QA RAG

## 1. Mục tiêu

Xây dựng bộ 100 câu hỏi đánh giá hệ thống hỏi đáp trên miền chính: Quy chế đào tạo đại học của ĐHQGHN năm 2014.

Mỗi câu hỏi cần được gán nhãn độc lập bởi hai người:
- Người thứ nhất: người viết câu hỏi.
- Người thứ hai: người được phân công theo lịch xoay vòng của nhóm.

Hai người không trao đổi đáp án trước khi hoàn tất gán nhãn độc lập.

## 2. Phân bố câu hỏi

Tổng cộng 100 câu:
- Khoảng 90 câu có thể trả lời từ tài liệu.
- Khoảng 10 câu ngoài phạm vi tài liệu.

Phân chia:
- Dev: 30 câu, gồm 27 câu có đáp án và 3 câu ngoài phạm vi.
- Test: 70 câu, gồm 63 câu có đáp án và 7 câu ngoài phạm vi.

Không sử dụng test để điều chỉnh hệ thống. Chỉ chạy đánh giá test theo đúng mốc đã thống nhất trong kế hoạch.

## 3. Quy tắc viết câu hỏi

1. Câu hỏi phải tự nhiên, rõ nghĩa và có một yêu cầu cụ thể.
2. Không hỏi những nội dung cần kiến thức bên ngoài tài liệu.
3. Không dùng cách diễn đạt mơ hồ như "quy định hiện hành".
4. Khi hỏi về quy chế, phải xác định rõ đây là Quy chế đào tạo đại học ĐHQGHN năm 2014 nếu ngữ cảnh chưa đủ rõ.
5. Tránh nhiều câu hỏi có cùng nội dung và cùng đáp án.
6. Không viết câu hỏi theo cách tiết lộ trực tiếp đáp án.
7. Câu hỏi ngoài phạm vi phải hợp lý, có liên quan đến chủ đề nhưng không thể tìm được câu trả lời trong tài liệu.

## 4. Quy tắc gán nhãn câu có đáp án

- Ghi đáp án dựa trên nội dung nguyên văn của tài liệu.
- Ghi chính xác Điều và Khoản chứa đáp án.
- Lưu nội dung Khoản làm nguồn đối chiếu.
- Không tự suy diễn hoặc bổ sung thông tin không có trong nguồn.
- Nếu câu hỏi cần nhiều đoạn để trả lời, ghi rõ tất cả nguồn liên quan.
- Không lấy mục lục ở ba trang cuối PDF làm nguồn nội dung.

## 5. Quy tắc gán nhãn câu ngoài phạm vi

- Đặt answerable = false.
- Không gán đáp án.
- Không tự tạo nguồn Điều/Khoản.
- Ghi lý do câu hỏi nằm ngoài phạm vi tài liệu.

## 6. Xử lý bất đồng

1. Hai người hoàn thành nhãn độc lập trước.
2. So sánh answerable, đáp án và nguồn.
3. Nếu bất đồng, cùng kiểm tra lại PDF gốc.
4. Chốt nhãn cuối cùng sau khi thống nhất.
5. Không tự sửa nhãn của người còn lại trước khi ghi nhận bất đồng.

## 7. Cấu trúc dữ liệu câu hỏi

Mỗi dòng JSONL là một câu hỏi, theo cấu trúc:

{
  "id": "A001",
  "question": "Câu hỏi",
  "answerable": true,
  "gold_answer": "Đáp án trích từ tài liệu",
  "source_article": 4,
  "source_clause": "1",
  "source_text": "Nội dung Khoản chứa đáp án",
  "author": "A",
  "second_annotator": "C",
  "second_annotation": null,
  "final_answer": null,
  "final_source_article": null,
  "final_source_clause": null,
  "adjudication_status": "pending"
}

Với câu ngoài phạm vi:
- answerable = false
- gold_answer = null
- source_article = null
- source_clause = null
- source_text = null
- Ghi lý do vào trường out_of_scope_reason.

## 8. Kiểm tra chất lượng

Trước khi đưa vào bộ gold:
- Không trùng ID.
- Không bỏ trống câu hỏi.
- Đáp án phải xuất hiện trong source_text.
- Điều và Khoản phải tồn tại trong PDF.
- Câu ngoài phạm vi không được có đáp án hoặc nguồn giả.
- Hai lượt gán nhãn phải được lưu riêng.
- Chỉ sử dụng nhãn đã thống nhất để đánh giá.
