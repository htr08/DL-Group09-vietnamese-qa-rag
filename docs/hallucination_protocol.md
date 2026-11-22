# Giao thức chấm ảo giác (chấm mù)

Tài liệu quy định cách chấm tay câu trả lời của các cấu hình RAG trên tập 100 câu miền Quy chế đào tạo ĐHQGHN. Căn cứ: `docs/project_plan.md` (mục 3.4, 4.3, 7), `docs/charter.md` (phân loại lỗi RAG) và `docs/annotation_guide.md` (định dạng gold). Công cụ: `scripts/grading_tool.py`, `src/grading.py`.

| Mục | Nội dung |
|---|---|
| Đối tượng chấm | Câu trả lời của mọi cấu hình trên 70 câu test (63 trong phạm vi, 7 ngoài phạm vi); trên 30 câu dev khi thử giao thức |
| Cấu hình được chấm | B1, B2 (và B2-cùng-chunk nếu có), RAG cơ bản, RAG cuối, chế độ trích xuất (bảng 3 của kế hoạch) |
| Người chấm | Ba thành viên; mỗi người chấm chính khoảng 1/3 số câu hỏi, ở mọi cấu hình |
| Chấm lại | Khoảng 30% câu trả lời của mỗi người chấm chính do người kế chấm lại: A chấm lại phần của B, B của C, C của A |
| Chỉ số | Tỷ lệ ảo giác, từ chối đúng, từ chối sai, Cohen's kappa giữa lượt chấm chính và lượt chấm lại |

## 1. Nhãn

Mỗi câu trả lời nhận đúng một nhãn. Người chấm dùng đáp án gold, mã đoạn nguồn và văn bản Quy chế (PDF) để đối chiếu.

| Nhãn (`label`) | Dùng cho | Điều kiện |
|---|---|---|
| `dung` | Câu trong phạm vi | Nội dung chính khớp đáp án gold hoặc một đáp án chấp nhận; không có thông tin sai kèm theo |
| `dung_mot_phan` | Câu trong phạm vi | Có ý đúng nhưng thiếu ý chính, hoặc đúng kèm chi tiết thừa không mâu thuẫn với văn bản |
| `sai_bia` | Câu trong và ngoài phạm vi | Có khẳng định sai so với văn bản, hoặc đưa ra thông tin không có trong văn bản như thể có (số liệu, điều kiện, tên đơn vị, trích dẫn Điều/Khoản không tồn tại); với câu ngoài phạm vi, mọi câu trả lời nội dung thay cho từ chối |
| `tu_choi_dung` | Câu ngoài phạm vi | Nêu rõ không có thông tin trong văn bản, không kèm nội dung bịa |
| `tu_choi_sai` | Câu trong phạm vi | Từ chối hoặc nói không có thông tin trong khi văn bản có đáp án |

Ảo giác là nhãn `sai_bia`; đây là loại lỗi nặng nhất trong phân loại của Charter.

Quy tắc quyết định, theo thứ tự:

1. Câu trả lời là từ chối (không đưa nội dung trả lời): `tu_choi_dung` nếu câu ngoài phạm vi, `tu_choi_sai` nếu câu trong phạm vi.
2. Câu ngoài phạm vi mà hệ thống đưa nội dung trả lời: `sai_bia`, kể cả khi nội dung đúng theo kiến thức chung, vì mục tiêu của hệ thống là chỉ dựa trên văn bản.
3. Câu trong phạm vi: có khẳng định mâu thuẫn với văn bản hoặc không có trong văn bản → `sai_bia`; nếu không, khớp đầy đủ đáp án → `dung`, khớp một phần → `dung_mot_phan`.
4. Câu trả lời vừa có phần đúng vừa có phần bịa: `sai_bia`.
5. Trích dẫn nguồn (số Điều, Khoản) sai hoặc không tồn tại là thông tin bịa; trích dẫn thiếu thì không bị trừ.
6. Câu trả lời lẫn ký tự tiếng Trung hoặc bằng ngôn ngữ khác: chấm theo nội dung; việc lẫn ký tự được thống kê riêng bằng kiểm tra tự động, không ghi vào nhãn.

Ví dụ minh họa (không thuộc tập gold; nội dung là giả định):

| Câu hỏi | Câu trả lời | Nhãn | Lý do |
|---|---|---|---|
| Sinh viên phải nộp đơn cho bộ phận nào? (gold: phòng X) | "Sinh viên nộp đơn cho phòng X." | `dung` | Khớp đáp án |
| Thời hạn nộp là bao lâu? (gold: 2 tuần) | "Có thời hạn, thường vào đầu học kỳ." | `dung_mot_phan` | Có ý đúng, thiếu số liệu |
| Thời hạn nộp là bao lâu? (gold: 2 tuần) | "Thời hạn là 4 tuần." | `sai_bia` | Số liệu sai |
| Lệ phí là bao nhiêu? (ngoài phạm vi) | "Lệ phí là 200.000 đồng." | `sai_bia` | Bịa thông tin cho câu ngoài phạm vi |
| Lệ phí là bao nhiêu? (ngoài phạm vi) | "Văn bản không nêu thông tin này." | `tu_choi_dung` | Từ chối đúng |
| Thời hạn nộp là bao lâu? (gold: 2 tuần) | "Văn bản không nêu thông tin này." | `tu_choi_sai` | Văn bản có đáp án |

## 2. Cấu hình trích xuất

B1 và chế độ trích xuất trả về span nằm trong đoạn truy hồi nên theo định nghĩa không có ảo giác (bảng 3 của kế hoạch ghi "không áp dụng"). Các cấu hình này vẫn được chấm bằng cùng bảng nhãn; câu trả lời sai được gắn `sai_bia` trong bảng chấm, và bước tổng hợp báo số câu sai (`wrong_count`) thay cho tỷ lệ ảo giác (`--extractive`).

## 3. Quy trình

| Bước | Việc | Người thực hiện |
|---|---|---|
| 1 | Mọi cấu hình chạy xong trên cùng danh sách câu hỏi và lưu kết quả thô (JSON Lines, khóa `id`, `answer`). Không chạy lại hay chỉnh prompt sau khi đã xuất bảng chấm | Người chạy cấu hình |
| 2 | `python scripts/grading_tool.py export --split test --system ten=duong_dan ... --out-dir data/grading` | C |
| 3 | Gửi `sheet_A.csv`, `sheet_B.csv`, `sheet_C.csv` cho từng người; `key.json` giữ riêng, không mở và không gửi | C |
| 4 | Mỗi người điền cột `label` (một trong năm nhãn) và cột `note` khi cần; không trao đổi về các mục cụ thể trong lúc chấm | A, B, C |
| 5 | Thu ba bảng, chạy `python scripts/grading_tool.py summarize --split test --dir data/grading --extractive b1 --extractive extractive_mode` | C |
| 6 | Mở khóa ánh xạ, đối chiếu các mục bất đồng giữa lượt chính và lượt hai, báo cáo kappa và các chỉ số | C |

Quy ước của công cụ xuất bảng:

- Mã mục (`X0001`, ...) được gán ngẫu nhiên, không theo hệ thống hay thứ tự câu hỏi; thứ tự dòng trong mỗi bảng được trộn.
- Tất cả câu trả lời của mọi cấu hình cho cùng một câu hỏi thuộc cùng một người chấm chính.
- Một mục được chấm lại xuất hiện trong bảng của người chấm chính và của người chấm lại với cùng mã mục; mỗi người không biết mục nào là chấm lại.
- Bảng có câu hỏi, đáp án gold, mã đoạn nguồn và câu trả lời. Bảng không có đoạn truy hồi, tên cấu hình hay độ trễ.
- Seed cố định (mặc định 2026); cùng đầu vào cho cùng bảng chấm.

## 4. Tổng hợp và báo cáo

- Tỷ lệ ảo giác = số câu `sai_bia` / số câu của cấu hình (cả câu trong và ngoài phạm vi); từ chối đúng tính trên câu ngoài phạm vi; từ chối sai tính trên câu trong phạm vi. Mỗi tỷ lệ kèm bootstrap CI 95% lấy mẫu lại theo câu hỏi (`bootstrap_ci` trong `src/metrics.py`).
- Cohen's kappa tính trên các mục được chấm lại, báo cả năm nhãn và nhãn nhị phân ảo giác (`sai_bia` hay không). Nhãn dùng để báo cáo là nhãn của người chấm chính; bất đồng giữa hai lượt được liệt kê trong phân tích lỗi.
- Mẫu nhỏ (63 câu trong phạm vi, 7 câu ngoài phạm vi trên test): báo cáo ghi số câu cùng CI và nêu hạn chế trong Limitations.
- Công cụ từ chối ghi nhãn không hợp lệ (ví dụ `tu_choi_dung` cho câu trong phạm vi) và báo mục thiếu nhãn.

## 5. Hạn chế

- Ẩn tên cấu hình không che hết phong cách trả lời (ví dụ cách trích dẫn nguồn, độ dài, biểu tượng); người chấm có thể đoán cấu hình. Giao thức giảm chứ không loại bỏ thiên lệch này.
- Người viết prompt và dựng RAG cũng là người chấm; chia việc chấm cho cả ba người và chấm lại 30% là biện pháp hạn chế thiên lệch.
- Kappa trên khoảng 40 mục có khoảng tin cậy rộng.

## 6. Lịch sử chỉnh sửa

| Ngày | Nội dung |
|---|---|
| 2026-11-22 | Bản đầu. |
