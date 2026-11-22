"""Sửa `answer_start` lệch trong ViQuAD thô, ghi ra data/processed/viquad.

Với mỗi đáp án mà `context[start:start+len(text)] != text`, chọn vị trí xuất hiện
của `text` trong context gần `start` cũ nhất (chỉ xét các lần xuất hiện không chồng
lấn, theo `re.finditer`); ném ValueError nếu `text` không có trong context. Bỏ qua
câu `is_impossible` và câu không có đáp án. Chỉ `answer_start` thay đổi; không
thêm hoặc bớt bản ghi. Xử lý cả train, dev và public_test; dữ liệu thô
data/raw/viquad không bị ghi đè. Tiền xử lý public_test (preprocess_qa.py) chỉ
chạy khi chọn `--splits public_test`. Đường dẫn tính theo thư mục gốc repository,
không phụ thuộc thư mục làm việc.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source_dir = ROOT / "data" / "raw" / "viquad"
output_dir = ROOT / "data" / "processed" / "viquad"
output_dir.mkdir(parents=True, exist_ok=True)

for filename in ["train.jsonl", "dev.jsonl", "public_test.jsonl"]:
    source = source_dir / filename
    output = output_dir / filename

    total = 0
    fixed = 0

    with source.open(encoding="utf-8") as f_in, output.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            record = json.loads(line)
            answers = record["answers"]

            if not record["is_impossible"] and answers:
                for i, (text, start) in enumerate(
                    zip(answers["text"], answers["answer_start"])
                ):
                    if record["context"][start:start + len(text)] != text:
                        total += 1
                        positions = [
                            m.start()
                            for m in re.finditer(re.escape(text), record["context"])
                        ]

                        if not positions:
                            raise ValueError(f"Cannot find answer: {record['id']}")

                        new_start = min(positions, key=lambda p: abs(p - start))
                        answers["answer_start"][i] = new_start
                        fixed += 1

            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(filename, "| Số vị trí sai ban đầu:", total, "| Đã hiệu chỉnh:", fixed)

print("Đã lưu dữ liệu tại:", output_dir)
