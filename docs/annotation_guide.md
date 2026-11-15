# Hướng dẫn gán nhãn tập 100 câu (miền Quy chế đào tạo ĐHQGHN)

Tài liệu quy định cách viết câu hỏi, gán nhãn hai lượt độc lập và thống nhất nhãn cho tập 100 câu của miền chính. Căn cứ: `docs/project_plan.md` (mục 2.1, 3.1, 3.4, 3.5) và `docs/charter.md` (mục 4, 5). Khi có khác biệt, kế hoạch dự án và Charter được ưu tiên.

| Mục | Nội dung |
|---|---|
| Văn bản nguồn | Quy chế đào tạo đại học của ĐHQGHN, ban hành kèm Quyết định 5115/QĐ-ĐHQGHN ngày 25/12/2014; tệp `data/domain/dhqghn_quy_che_dao_tao_dai_hoc_2014.pdf` |
| Phạm vi văn bản | 47 Điều (trang 1–45); ba trang cuối là mục lục, không dùng làm nguồn |
| Quy mô | 100 câu: 90 trong phạm vi, 10 ngoài phạm vi |
| Chia tập | 30 dev (27 trong phạm vi, 3 ngoài phạm vi) và 70 test (63 trong phạm vi, 7 ngoài phạm vi) |
| Tác giả | A và B mỗi người 33 câu, C 34 câu |
| Người gán nhãn độc lập | A gán câu của B, B gán câu của C, C gán câu của A |
| Hạn | Viết xong ở Tuần 2 T5; gán nhãn độc lập xong ở Tuần 2 T6; thống nhất nhãn ở Tuần 3 T2 |

## 1. Mục đích của tập 100 câu

Tập câu hỏi phục vụ ba phép đo: (i) truy hồi (R@1, R@3, MRR@10) với đoạn đúng ở mức Khoản; (ii) RAG chấm tay (ảo giác, từ chối đúng, từ chối sai) và EM/F1 mức câu trả lời; (iii) KPI K3 so với hệ thống nền B2. Vì vậy mỗi câu trong phạm vi cần có đúng một Khoản nguồn và một đáp án là đoạn nguyên văn của Khoản đó; mỗi câu ngoài phạm vi cần được xác nhận là văn bản không trả lời được.

## 2. Quy ước đoạn nguồn

Kho truy hồi được cắt hai cấp Điều → Khoản (kế hoạch, mục 3.1). Đoạn nguồn của một câu được gán ở mức Khoản bằng hai số nguyên: `article` (số Điều, 1–47) và `clause` (số Khoản).

- **Khoản** là mục đánh số `1.`, `2.`, `3.` trong một Điều. Các Điểm `a)`, `b)`, `c)` và các gạch đầu dòng thuộc Khoản chứa chúng; không gán nhãn ở mức Điểm.
- **Điều không chia Khoản** (Điều 15 và Điều 18): `clause` = `null`; đoạn nguồn là toàn bộ Điều.
- **Phần dẫn trước Khoản 1** (đoạn văn đứng ngay dưới tiêu đề Điều, chưa có số Khoản, ví dụ ba dòng đầu của Điều 41): không dùng làm đoạn nguồn, vì không thuộc Khoản nào. Câu hỏi mà đáp án chỉ có trong phần dẫn được thay bằng câu khác.
- **Một nguồn duy nhất:** đáp án phải nằm trọn trong một Khoản. Câu hỏi cần ghép thông tin từ hai Khoản hoặc hai Điều (nhiều bước suy luận) không thuộc tập này.
- **Tham chiếu chéo:** nếu Khoản chỉ viện dẫn Khoản khác (ví dụ "theo quy định tại khoản 1, Điều 20 của Quy chế này") và không tự nêu nội dung, Khoản đó không dùng làm đoạn nguồn.
- **Bảng và dạng bảng:** nội dung dạng bảng (ví dụ bảng quy đổi điểm chữ ở Điều 38, khoản 2) dễ vỡ khi trích văn bản từ PDF. Tối đa 3 câu của cả tập có đáp án nằm trong nội dung dạng bảng; các câu này được gắn `question_type = "bang"` để phân tích riêng.

Mã đoạn nguồn viết ngắn là `D<article>.K<clause>`, ví dụ `D21.K2` (Điều 21, Khoản 2) và `D15` (Điều 15 không chia Khoản). Mã này dùng trong báo cáo và trong phân tích lỗi; trong tệp dữ liệu hai trường số được lưu riêng.

Khi đánh giá truy hồi, mọi chunk con của Khoản gold (do tách theo Điểm hoặc cửa sổ trượt) đều tính là đúng; mức Điều được báo thêm (kế hoạch, mục 3.4).

## 3. Quy tắc viết câu hỏi

### 3.1. Câu hỏi trong phạm vi

1. **Tự đứng được.** Câu hỏi hiểu được mà không cần đọc văn bản; không dùng cụm "theo điều này", "ở đây", "như trên".
2. **Một ý hỏi.** Mỗi câu hỏi một thông tin; không ghép hai câu hỏi bằng "và".
3. **Không nêu số Điều hoặc Khoản** trong câu hỏi. Câu hỏi dạng "Điều 21 quy định gì?" làm lộ nguồn và biến phép đo truy hồi thành so khớp từ khóa.
4. **Diễn đạt lại.** Câu hỏi viết bằng lời của người hỏi thực tế (sinh viên, cán bộ đào tạo), không chép nguyên cụm từ dài từ văn bản; không sao chép quá bốn từ liên tiếp của câu trong văn bản, trừ thuật ngữ cố định (ví dụ "cảnh báo học vụ", "điểm trung bình chung tích luỹ"). Quy tắc này tránh thiên lệch có lợi cho BM25.
5. **Không dùng dạng "hiện hành".** Văn bản là bản năm 2014; không viết "quy định hiện hành", "hiện nay", "mới nhất". Câu hỏi cần đúng với nội dung của văn bản này.
6. **Hạn chế câu hỏi có/không.** Đáp án phải là đoạn trích nguyên văn nên câu hỏi có/không chỉ dùng khi văn bản có mệnh đề trả lời trực tiếp (ví dụ câu quy định "được phép" hoặc "không được phép"); tối đa 10 câu có/không của cả tập.
7. **Đáp án xác định.** Nội dung đáp án phải đủ rõ để hai người đọc độc lập chọn cùng một span (xem mục 4). Câu hỏi mơ hồ được viết lại.

Phân bố mục tiêu của mỗi tác giả (mức hướng dẫn, không phải ràng buộc cứng):

| Tiêu chí | Mục tiêu |
|---|---|
| Độ phủ | Rải trên nhiều Chương; mỗi Điều không quá 3 câu trong phạm vi của toàn tập |
| Loại câu hỏi (`question_type`) | Cân đối giữa `so_lieu` (số tín chỉ, điểm, thời hạn, số tuần), `dieu_kien` (điều kiện, tiêu chuẩn), `thu_tuc` (ai làm gì, trình tự), `dinh_nghia` (khái niệm, đơn vị), `liet_ke` (danh mục, trường hợp), `bang` (tối đa 3 câu toàn tập), `co_khong` (tối đa 10 câu toàn tập) |
| Độ khó | Gồm câu có từ khóa trùng với văn bản và câu cần hiểu đồng nghĩa (ví dụ "thôi học" so với "buộc thôi học", "rút môn" so với "rút bớt học phần") |

Ví dụ minh họa (không nằm trong tập):

| Câu hỏi | Đoạn nguồn | Đáp án |
|---|---|---|
| Khi học kỳ chính bắt đầu, sinh viên có tối đa bao lâu để rút bớt học phần mà vẫn được hoàn học phí? | `D21.K2` | 2 tuần kể từ đầu học kỳ chính |
| Một năm học chia thành mấy học kỳ chính? | `D15` | hai học kỳ chính |

### 3.2. Câu hỏi ngoài phạm vi

Câu ngoài phạm vi là câu hỏi hợp lý của người dùng nhưng văn bản không có thông tin để trả lời; đáp án đúng của hệ thống là từ chối. Mỗi tác giả viết: A 3 câu, B 3 câu, C 4 câu (tổng 10). Ba nhóm, mỗi nhóm có ít nhất 2 câu trong tập:

| Nhóm (giá trị `question_type`) | Mô tả | Ví dụ minh họa |
|---|---|---|
| `oos_gan_mien` | Cùng chủ đề đào tạo đại học nhưng văn bản không nêu | Điều kiện để sinh viên được ở ký túc xá |
| `oos_tien_de_sai` | Câu hỏi chứa tiền đề không có trong văn bản, dễ khiến mô hình bịa | Hỏi "điểm sàn xét tuyển" của từng ngành, khi văn bản không có khái niệm đó |
| `oos_ngoai_mien` | Không liên quan đến văn bản | Câu hỏi kiến thức chung |

Quy tắc xác nhận:

1. Tác giả tìm toàn bộ 45 trang văn bản bằng từ khóa chính và từ đồng nghĩa, rồi xác nhận không có Khoản nào trả lời.
2. Câu hỏi không được trả lời một phần bằng nội dung có sẵn trong văn bản; nếu có một phần, câu hỏi được viết lại hoặc thay. Ví dụ: câu hỏi về mức học phí mỗi tín chỉ của một ngành cụ thể không thuộc nhóm này, vì Điều 8 nêu công thức tính học phí và giao đơn vị đào tạo xây dựng định mức, tức văn bản trả lời được một phần.
3. Quy tắc 1, 2, 3, 5 của mục 3.1 áp dụng tương tự.
4. Người gán nhãn độc lập kiểm tra lại bằng cách tìm trong văn bản (mục 4).

## 4. Quy trình gán nhãn hai lượt

### 4.1. Các bước

| Bước | Người thực hiện | Việc | Thời điểm |
|---|---|---|---|
| 1 | Tác giả | Viết câu hỏi và nhãn (đoạn nguồn, đáp án, loại) theo mục 5.1; **chỉ gửi danh sách câu hỏi** (id và `question`) cho người gán nhãn | Tuần 2 T4–T5 |
| 2 | Tác giả | Bốc ngẫu nhiên tập dev (mục 6) | Tuần 2 T5, sau khi viết đủ câu |
| 3 | Người gán nhãn | Đọc văn bản, với mỗi câu xác định: có nằm trong phạm vi không; nếu có thì Điều, Khoản và đáp án (mục 5.2). **Không xem nhãn của tác giả** | Tuần 2 T6 |
| 4 | Cả hai | Sau khi người gán nhãn nộp xong, nhãn của tác giả mới được đưa vào repository; tính ba chỉ số đồng thuận (mục 7) trên nhãn lượt đầu | Tuần 2 T6 – Tuần 3 T2 |
| 5 | Cả nhóm | Họp thống nhất nhãn các câu bất đồng (mục 8) | Tuần 3 T2 |

Điều kiện độc lập: người gán nhãn không được thấy nhãn, đoạn nguồn, đáp án hoặc phân chia dev/test của tác giả trước khi nộp nhãn của mình; không trao đổi về nội dung câu hỏi trong thời gian gán nhãn. Câu hỏi quá mơ hồ để gán nhãn được đánh dấu bằng `note` thay vì hỏi lại tác giả.

### 4.2. Cách xác định đáp án

- Đáp án là **một đoạn liên tục, nguyên văn** của đúng Khoản nguồn, ngắn nhất mà vẫn trả lời đủ câu hỏi (thường 1–20 từ). Không diễn đạt lại, không ghép các đoạn rời nhau.
- Đáp án so khớp với văn bản sau khi chuẩn hóa: Unicode NFC, gộp ký tự xuống dòng và khoảng trắng liên tiếp thành một dấu cách. Việc xuống dòng do trình bày của PDF không làm đáp án sai.
- Không đưa dấu chấm, dấu chấm phẩy ở cuối Khoản vào đáp án.
- Câu liệt kê (nhiều trường hợp): đáp án là đoạn chứa toàn bộ danh sách khi danh sách nằm liền nhau; nếu các mục rời nhau (a, b, c ở các dòng khác nhau), chọn đoạn từ mục đầu đến mục cuối và chấp nhận đáp án dài hơn 20 từ.
- Câu có nhiều đáp án hợp lệ (cùng nội dung ở hai cách viết trong một Khoản): ghi thêm ở `alt_answers`.

## 5. Định dạng tệp

Tất cả tệp là JSON Lines, mã hóa UTF-8, mỗi dòng một đối tượng, tên khóa bằng tiếng Anh viết thường. Giá trị `question`, `answer` và `note` viết tiếng Việt có dấu.

### 5.1. Nhãn của tác giả

Tệp `data/gold/drafts/author_<A|B|C>.jsonl`.

| Khóa | Kiểu | Ý nghĩa |
|---|---|---|
| `id` | chuỗi | Chữ cái tác giả và số thứ tự hai chữ số: `A01`–`A33`, `B01`–`B33`, `C01`–`C34` |
| `question` | chuỗi | Câu hỏi |
| `in_scope` | bool | `true` nếu văn bản trả lời được |
| `article` | số nguyên hoặc `null` | Số Điều (1–47); `null` nếu ngoài phạm vi |
| `clause` | số nguyên hoặc `null` | Số Khoản; `null` nếu ngoài phạm vi hoặc Điều không chia Khoản (Điều 15, 18) |
| `answer` | chuỗi hoặc `null` | Đáp án nguyên văn; `null` nếu ngoài phạm vi |
| `alt_answers` | mảng chuỗi | Đáp án hợp lệ khác trong cùng Khoản (có thể rỗng) |
| `question_type` | chuỗi | `so_lieu`, `dieu_kien`, `thu_tuc`, `dinh_nghia`, `liet_ke`, `bang`, `co_khong` cho câu trong phạm vi; `oos_gan_mien`, `oos_tien_de_sai`, `oos_ngoai_mien` cho câu ngoài phạm vi |
| `note` | chuỗi | Ghi chú (có thể rỗng) |

Ví dụ:

```json
{"id": "B07", "question": "Khi học kỳ chính bắt đầu, sinh viên có tối đa bao lâu để rút bớt học phần mà vẫn được hoàn học phí?", "in_scope": true, "article": 21, "clause": 2, "answer": "2 tuần kể từ đầu học kỳ chính", "alt_answers": [], "question_type": "so_lieu", "note": ""}
{"id": "B31", "question": "Sinh viên cần đáp ứng điều kiện gì để được ở ký túc xá?", "in_scope": false, "article": null, "clause": null, "answer": null, "alt_answers": [], "question_type": "oos_gan_mien", "note": ""}
```

Hai dòng trên chỉ minh họa định dạng; `id` và nội dung không thuộc tập thật.

### 5.2. Nhãn của người gán nhãn độc lập

Tệp `data/gold/drafts/annot_<người gán nhãn>_of_<tác giả>.jsonl`, ví dụ `annot_A_of_B.jsonl`. Khóa gồm `id`, `in_scope`, `article`, `clause`, `answer`, `alt_answers`, `note` với ý nghĩa như mục 5.1. Không có `question`, `question_type` và `split`. Câu không tìm được đoạn nguồn thì `in_scope = false`.

### 5.3. Tệp gold cuối

Hai tệp `data/gold/gold_dev.jsonl` (30 dòng) và `data/gold/gold_test.jsonl` (70 dòng), sinh sau khi thống nhất nhãn ở Tuần 3 T2.

| Khóa | Ý nghĩa |
|---|---|
| `id`, `question`, `in_scope`, `article`, `clause`, `question_type` | Như mục 5.1, sau khi thống nhất |
| `answers` | Mảng đáp án chấp nhận: đáp án thống nhất, cộng các đáp án hợp lệ khác của hai người (không trùng lặp); mảng rỗng nếu ngoài phạm vi |
| `author`, `annotator` | Chữ cái `A`, `B` hoặc `C` |
| `split` | `dev` hoặc `test` |
| `agreement` | Kết quả so khớp lượt đầu: `source` (`same_clause`, `same_article`, `different`, `n/a`), `answer_f1` (số thực hoặc `null`), `scope` (`same`, `different`) |
| `resolution` | Cách thống nhất: `none` (đồng thuận từ đầu), `author_label`, `annotator_label`, `merged`, `rewritten` |

Nhãn mô tả ở đây là nhãn vàng cho đánh giá; hệ thống không được đọc các tệp này trong quá trình huấn luyện hoặc tuning prompt (tuning chỉ dùng tệp dev).

## 6. Chia dev và test

Quota của mỗi tác giả:

| Tác giả | Tổng | Trong phạm vi | Ngoài phạm vi | Dev (trong / ngoài phạm vi) | Test (trong / ngoài phạm vi) |
|---|---|---|---|---|---|
| A | 33 | 30 | 3 | 9 / 1 | 21 / 2 |
| B | 33 | 30 | 3 | 9 / 1 | 21 / 2 |
| C | 34 | 30 | 4 | 9 / 1 | 21 / 3 |
| Tổng | 100 | 90 | 10 | 27 / 3 | 63 / 7 |

Quy tắc:

1. Phân chia được thực hiện **một lần**, bằng bốc ngẫu nhiên, ngay sau khi tác giả viết đủ câu (Tuần 2 T5) và **trước khi bất kỳ hệ thống nào chạy trên tập**. Tác giả không chọn tay câu nào vào dev.
2. Bốc độc lập trong từng nhóm của từng tác giả (câu trong phạm vi, câu ngoài phạm vi), cố định seed; seed và danh sách `id` của dev được lưu ở `data/splits/gold_split.json`. Ví dụ với câu trong phạm vi của A: `random.Random(9).sample(ids_in_scope, 9)` là các câu dev; phần còn lại là test.
3. Các lần chạy ở Tuần 2 T6 (BM25, dense, hybrid, B2 trên 100 câu) chỉ **lưu kết quả thô**; không tính hoặc xem điểm của các câu test. Điểm trên test chỉ được tính ở Tuần 5 T2.
4. Việc thống nhất nhãn ở Tuần 3 T2 có thể thay một câu bị loại hoặc viết lại; câu mới giữ nguyên `split` và nhóm của câu bị thay.
5. Mọi tuning (k, trọng số hybrid, prompt, ngưỡng) chỉ dùng 30 câu dev. Không chỉnh gì dựa trên câu test.

Kích thước dev chỉ 30 câu nên kết quả tuning nhiễu; hạn chế này được ghi trong Limitations của báo cáo.

## 7. Chỉ số đồng thuận

Tính trên **nhãn lượt đầu** của tác giả và người gán nhãn, trước khi thống nhất (kế hoạch, mục 3.4):

| Chỉ số | Định nghĩa | Mẫu số |
|---|---|---|
| (a) Đoạn nguồn | Tỉ lệ câu mà hai người chọn cùng Khoản (cùng `article` và `clause`); báo thêm tỉ lệ cùng Điều | Câu mà cả hai người cùng cho là trong phạm vi |
| (b) Đáp án | EM và F1 token giữa hai span (chuẩn hóa như SQuAD: chữ thường, bỏ dấu câu; tách token theo khoảng trắng), lấy trung bình; với câu có `alt_answers`, lấy cặp cho điểm cao nhất | Câu mà cả hai người cùng cho là trong phạm vi |
| (c) Phạm vi | Tỉ lệ câu mà hai người cùng gắn nhãn trong phạm vi hoặc cùng gắn ngoài phạm vi | Cả 100 câu |

Quy ước bổ sung:

- Báo cáo thêm số câu hai người bất đồng về phạm vi (một người gán ngoài phạm vi, người kia gán trong phạm vi); các câu này không vào (a) và (b).
- Số liệu được tính bằng script do C viết trước cuộc họp Tuần 3 T2 và ghi vào khóa `agreement` của mục 5.3; báo cáo cuối nêu ba chỉ số kèm số câu của mẫu số.
- Bootstrap CI 95% cho các tỉ lệ đồng thuận được tính bằng `bootstrap_ci` trong `src/metrics.py` khi báo cáo.

## 8. Thống nhất nhãn (Tuần 2 T6 – Tuần 3 T2)

Cuộc họp 30 phút ở Tuần 3 T2 do C chủ trì. Mỗi câu bất đồng được xử lý theo thứ tự sau:

1. **Đối chiếu văn bản.** Cả nhóm đọc Khoản được hai bên nêu. Nếu một bên sai (chọn nhầm Khoản, trích sai nguyên văn, bỏ sót nội dung), nhãn đúng được chọn và `resolution` ghi `author_label` hoặc `annotator_label`.
2. **Hai nhãn đều hợp lệ.** Hai Khoản cùng trả lời được câu hỏi: câu hỏi được viết lại để chỉ còn một nguồn (`rewritten`); nếu không viết lại được, câu bị thay (mục 6, quy tắc 4). Hai đáp án cùng nằm trong một Khoản, đều đúng: giữ cả hai trong `answers` (`merged`).
3. **Bất đồng về phạm vi.** Cả nhóm tìm lại trong văn bản. Nếu có Khoản trả lời được thì câu là trong phạm vi; nếu không thì câu là ngoài phạm vi. Câu có nội dung chỉ trả lời một phần được viết lại hoặc thay.
4. **Câu mơ hồ.** Câu mà nhóm không chọn được một đáp án xác định được viết lại hoặc thay.

Tác giả của câu hỏi không có quyền quyết định riêng; quyết định theo văn bản và được cả ba thành viên chấp thuận. Sau cuộc họp, C ghi `resolution` và sinh `gold_dev.jsonl`, `gold_test.jsonl`. Tổng cộng vẫn là 100 câu, giữ đúng quota ở mục 6.

## 9. Kiểm tra trước khi nộp

Mỗi tác giả kiểm tra tệp của mình trước khi gửi:

- [ ] Đủ số câu và đúng quota trong phạm vi, ngoài phạm vi (mục 6); `id` không trùng.
- [ ] Không câu hỏi nào nêu số Điều hoặc Khoản; không có "hiện hành", "hiện nay".
- [ ] Với mọi câu trong phạm vi: `answer` là đoạn nguyên văn của đúng Khoản `article`/`clause` (mở PDF và đối chiếu); không rơi vào phần dẫn hoặc Khoản chỉ có tham chiếu chéo.
- [ ] Mỗi Điều không quá 3 câu trong phạm vi (xét cả ba tác giả khi hợp nhất); tối đa 3 câu `bang`, tối đa 10 câu `co_khong` toàn tập.
- [ ] Với mọi câu ngoài phạm vi: đã tìm trong toàn văn và không có Khoản trả lời, kể cả một phần.
- [ ] Tệp là JSON Lines hợp lệ, UTF-8, tiếng Việt có dấu, khóa đúng mục 5.1.
- [ ] Nhãn của tác giả chưa được commit trước khi người gán nhãn nộp xong (mục 4.1).

## 10. Lịch sử chỉnh sửa

| Ngày | Nội dung |
|---|---|
| 2026-11-13 | Bản đầu, lập ở Tuần 2 T2 theo kế hoạch dự án bản sau rà soát (ngày 2026-11-12). |
