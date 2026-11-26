# Các trường hợp cần xem xét Gold

Các trường hợp dưới đây được nhận diện khi phân tích lỗi.
Đây không phải kết luận rằng nhãn dataset bị sai.

Giữ nguyên Gold, context, prediction và F1.
Không loại các mẫu này khỏi EM/F1 chuẩn.
Không tính chúng là lỗi mô hình đã xác nhận.

## V2: 18/66 mẫu

| ID | Taxonomy | Nhận xét |
|---|---|---|
| 0073-0016-0002 | gold_context_ambiguity | Context quy việc mở đường gây cả chia cắt sinh cảnh và nguy cơ mất loài. Prediction nêu hậu quả được hỗ trợ, nhưng Gold sát đối tượng “sinh cảnh” hơn; cần xác định mức chấp nhận đáp án thay thế. |
| 0099-0007-0005 | gold_context_ambiguity | Context đồng thời nói hai nhóm là sản phẩm của chủ nghĩa tư bản và xác định nguồn gốc là thất bại cá nhân. Gold theo vế sau, prediction theo vế trước; cần làm rõ quan điểm và cách diễn đạt mâu thuẫn. |
| 0087-0052-0001 | span_too_short | Context nêu cổ phần giảm từ 56% xuống 27%. Prediction đúng mức cuối theo từ “xuống bao nhiêu”, nhưng thiếu mức ban đầu theo Gold; cần xem lại phạm vi Gold. |
| 0087-0018-0003 | question_type_mismatch | Câu hỏi “ai” đòi tên người (Trump); Gold và prediction đều là mệnh đề về khả năng tranh cử. Cần xác minh kiểu câu hỏi và Gold trước khi quy lỗi span thuần túy. |
| 0117-0026-0002 | question_premise_mismatch | Đoạn chỉ khẳng định tỷ lệ nghèo tương đối vượt tỷ lệ của các quốc gia giàu từ thập niên 1980, không nói tốc độ tăng nhanh; tiền đề về xu hướng tăng cần xác minh. |
| 0087-0041-0002 | question_type_mismatch | Câu hỏi “Khi nào dmdj?” không xác định sự kiện, còn Gold là Quốc vương Abdullah (người), không phải thời điểm. Không thể chứng minh model từ chối sai. |
| 0087-0015-0001 | question_premise_mismatch | Người gọi Trump là “kẻ thua cuộc ba lần” là cựu Thủ tướng Alex Salmond, không phải Chính phủ Scotland với tư cách tổ chức. Gold có span nhưng tiền đề chủ thể sai lệch. |
| 0117-0056-0004 | question_premise_mismatch | Context chỉ nói David Audretsch ghi lại vai trò của sáng tạo kinh doanh, không nói ông phổ biến các tiến bộ. Gold có tên nhưng quan hệ hành động bị đổi. |
| 0087-0032-0007 | gold_answerability_dispute | Context nêu dù bỏ cuộc Trump vẫn thắng bầu cử sơ bộ ở California và Michigan. Prediction đúng địa điểm; câu hỏi lỗi cụm “đứng thứ lượng phiếu bầu”, cần xác minh ý định và Gold rỗng. |
| 0093-0024-0006 | question_premise_mismatch | Context nói NIBRS là hệ thống nhằm bổ sung hạn chế UCR, không nói được bán ra. Prediction đúng mục đích nếu “bán ra” là lỗi chữ; cần xác minh tiền đề và Gold rỗng. |
| 0059-0004-0007 | gold_answerability_dispute | Context nêu đại hội chủ trương tạm thời rút bỏ khẩu hiệu “Tất cả chính quyền về tay xô viết”. Prediction trích đúng khẩu hiệu (thừa dấu ngoặc kép cuối), Gold rỗng đáng xem lại. |
| 0059-0015-0006 | gold_answerability_dispute | Context nói Cách mạng Tháng Mười do giai cấp vô sản tiến hành; prediction phù hợp nếu “lãnh đạo” được hiểu gần nghĩa. Hai vai trò không hoàn toàn đồng nhất; cần xét Gold rỗng. |
| 0041-0005-0004 | gold_answerability_dispute | Context liệt kê rõ các giải thưởng, khái niệm và địa điểm mang tên Fermi. Prediction nêu bốn mục đầu nhưng thiếu nhà máy điện và fermium; Gold rỗng đáng xét, prediction vẫn chưa đầy đủ. |
| 0117-0025-0006 | gold_answerability_dispute | Khảo sát tháng 8/2017 về lương vừa đủ chi phí sinh hoạt được nêu do CareerBuilder thực hiện. Prediction đúng tổ chức theo context; Gold rỗng cần xác minh. |
| 0059-0019-0006 | gold_answerability_dispute | Context nói Liên Xô thúc đẩy chủ nghĩa cộng sản lan rộng, buộc các nước phương Tây cải cách. Prediction đúng hành động nếu “nước đế quốc” được hiểu là các nước phương Tây trong đoạn. |
| 0087-0008-0006 | gold_answerability_dispute | Context nêu công ty bất động sản của cha Trump tập trung vào nhà cho thuê trung lưu và Trump quản lý căn hộ của công ty. Prediction đúng lĩnh vực, dù “đang quản lý công ty” rộng hơn quản lý căn hộ. |
| 0093-0010-0008 | wrong_actor_or_entity | Context nói FBI tán thành hầu hết đề nghị, còn nhận xét thay đổi hình thức đến từ thành viên Ủy ban. Prediction đảo người đưa nhận xét; đồng thời Gold rỗng cần xét vì câu đầu có đáp án. |
| 0031-0005-0005 | gold_context_ambiguity | Context nêu bất lợi của thay đổi công nghệ là thế hệ trước khó bắt kịp. Câu hỏi lại nói “thay đổi của con người”; prediction hợp lý nếu ý hỏi là công nghệ, cần làm rõ Gold rỗng. |

## Legacy Set 2: 11/66 mẫu

| ID | Taxonomy | Nhận xét |
|---|---|---|
| 0117-0055-0005 | gold_context_ambiguity | Context đặt Nikola Tesla trong lịch sử công nghệ Hoa Kỳ, không nói trực tiếp ông mang quốc tịch nào. Gold 'Hoa Kỳ' xuất hiện ở câu khác nhưng không được gắn ngữ nghĩa với Tesla; prediction rỗng chưa đủ chứng minh model sai. Cần xác minh Gold. |
| 0117-0020-0002 | gold_context_ambiguity | Câu hỏi có lỗi 'bên vững' (nên là 'bền vững'). Context nói bất bình đẳng thu nhập đe dọa ổn định xã hội và tài chính; 'sự bất công' là một cách mô tả, không nêu trực tiếp nguyên nhân 'phát triển kém bền vững'. Gold liên quan nhưng chưa thật sát câu hỏi. |
| 0087-0030-0004 | gold_answerability_dispute | Gold gốc đánh dấu không thể trả lời, song context nêu trực tiếp Trump ủng hộ dỡ bỏ 'chế độ dự trữ và xếp hàng' sau bê bối chăm sóc cựu chiến binh. Prediction trích đúng đoạn có thể trả lời; cần adjudication Gold, chưa tự sửa nhãn. |
| 0130-0010-0008 | gold_answerability_dispute | Context nói 'chuyến đi dự kiến kéo dài tám tháng', sau đó kết thúc sau hai tháng. Prediction 'tám tháng' phù hợp thời lượng dự kiến, nhưng question dùng 'chiến dịch được khẳng định' không trùng diễn đạt 'chuyến đi dự kiến'; cần xác định chủ thể và hàm ý trước khi kết luận Gold rỗng sai. |
| 0095-0037-0006 | gold_answerability_dispute | Context nêu 'Đảng Cách mạng Tây Tạng do Pandatsang Rapga thành lập'. Prediction nêu đúng người sáng lập theo ngữ cảnh, nhưng question viết 'Đảng Cách mạng' thiếu định danh Tây Tạng, có thể mơ hồ. Cần adjudication tính trả lời được của Gold rỗng. |
| 0094-0006-0006 | gold_answerability_dispute | Context liệt kê bài chòi, hát bội, nhạc võ Tây Sơn, hò bá trạo là các loại hình nghệ thuật ở Bình Định; không trực tiếp khẳng định từng loại đã được công nhận là 'di sản văn hóa phi vật thể'. Prediction trả lời danh sách gần nghĩa nhưng điều kiện trong question chưa được chứng minh. |
| 0055-0006-0007 | gold_answerability_dispute | Context nêu John Law thuyết phục giới tư sản/quý tộc Paris đầu tư vào Công ty Mississippiy 'để phục vụ cho việc chiếm đóng thuộc địa Louisiana'. Prediction trích đúng mục đích; Gold rỗng đáng kiểm tra lại. Question có lỗi diễn đạt nhưng phần hỏi mục đích vẫn đối chiếu được. |
| 0117-0051-0002 | gold_context_ambiguity | Context nói tỷ trọng chi tiêu người tiêu dùng trong GDP là 62% năm 1960 và 'giữ nguyên tới 1981', không trực tiếp nói giá trị chi tiêu tuyệt đối 'chững lại'. Gold gán mốc cho tỷ trọng, question có thể nhầm mức chi tiêu với tỷ trọng GDP; cần xác minh câu hỏi/Gold. |
| 0093-0008-0004 | question_premise_mismatch | Context nói Đội Giải cứu con tin xử lý sự việc liên quan khủng bố; CART được thành lập song song năm 1984 nhưng không nêu mục đích chống 'khủng bố công nghệ cao'. Question có thể ghép nhầm chức năng hai lực lượng; không quy chắc chắn là từ chối sai. |
| 0117-0060-0001 | gold_context_ambiguity | Context liệt kê New Mexico cùng các bang trong thống kê chi R&D, không đưa bất kỳ thông tin nào về đường biên với Mexico. Question đòi kiến thức địa lý bên ngoài context; Gold 'New Mexico' không được đoạn hỗ trợ. |
| 0130-0005-0009 | question_type_mismatch | Question hỏi 'châu lục' trong khi context nói Argentina phản đối định nghĩa lại Nam Đại Dương (một đại dương, không phải châu lục). Gold 'Nam Đại Dương' có trong đoạn nhưng không cùng loại đối tượng câu hỏi; cần xác minh cách đặt câu. |


Hai bộ thuộc các lượt đánh giá khác nhau.
Không cộng dồn số mẫu hoặc chuyển nhãn giữa hai bộ.

Các nhãn phân tích chưa có xác nhận độc lập.
V2 có 48 mẫu không nằm trong danh sách nghi vấn Gold;
Set 2 có 55 mẫu không nằm trong danh sách tương ứng.
Không đồng nhất các số này với số mẫu đã xác nhận.
