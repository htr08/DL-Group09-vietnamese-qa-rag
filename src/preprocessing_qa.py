"""Tiền xử lý extractive QA.

Quy tắc quan trọng: feature cho dev/test/inference KHÔNG phụ thuộc vào đáp án gold.
Gold chỉ được dùng để tạo nhãn (nếu có) SAU khi token và cửa sổ đã cố định.

Để tương thích checkpoint/các unit test trước đó, chế độ train (mặc định)
giữ lại hành vi tinh chỉnh token theo answer span và overlap theo độ dài answer.
"""
import re
import unicodedata

MAX_LEN = 256
MAX_QUESTION_LEN = 64
DOC_STRIDE = 128
IGNORE_INDEX = -100


def normalize_with_mapping(text):
    normalized = []
    mapping = []
    for i, char in enumerate(text):
        for c in unicodedata.normalize("NFD", char.lower()):
            if unicodedata.category(c) != "Mn":
                normalized.append(c)
                mapping.append(i)
    return "".join(normalized), mapping


def align_tokens_to_text(text, tokens):
    normalized_text, mapping = normalize_with_mapping(text)
    spans = []
    cursor = 0
    for token in tokens:
        parts = [normalize_with_mapping(part)[0] for part in token.split("_")]
        if not parts or any(not part for part in parts):
            raise ValueError(f"Invalid token: {token}")
        pattern_parts = []
        for i, part in enumerate(parts):
            if i == 0:
                pattern_parts.append(re.escape(part))
                continue
            previous = parts[i - 1]
            previous_is_punctuation = all(
                unicodedata.category(c).startswith(("P", "S")) for c in previous
            )
            current_is_punctuation = all(
                unicodedata.category(c).startswith(("P", "S")) for c in part
            )
            separator = r"\s*" if (previous_is_punctuation or current_is_punctuation) else r"\s+"
            pattern_parts.append(separator + re.escape(part))
        match = re.search("".join(pattern_parts), normalized_text[cursor:])
        if match is None:
            raise ValueError(f"Cannot align token: {token}")
        position = cursor + match.start()
        match_end = cursor + match.end()
        spans.append((mapping[position], mapping[match_end - 1] + 1))
        cursor = match_end
    return spans


def tokenize_words(tokenizer, tokens):
    input_ids = []
    subword_spans = []
    for token in tokens:
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        start = len(input_ids)
        input_ids.extend(token_ids)
        subword_spans.append((start, len(input_ids)))
    return input_ids, subword_spans


def map_answer_to_subwords(answer_start, answer_text, word_spans, subword_spans):
    answer_end = answer_start + len(answer_text)
    overlapping = [i for i, (start, end) in enumerate(word_spans)
                   if start < answer_end and end > answer_start]
    if not overlapping:
        raise ValueError("Answer does not overlap any token")
    first = overlapping[0]
    last = overlapping[-1]
    if word_spans[first][0] != answer_start or word_spans[last][1] != answer_end:
        raise ValueError("Answer does not align with token boundaries")
    return subword_spans[first][0], subword_spans[last][1] - 1


def create_sliding_windows(
    question_ids,
    context_ids,
    max_len=MAX_LEN,
    max_question_len=MAX_QUESTION_LEN,
    doc_stride=DOC_STRIDE,
    answer_start=None,
    answer_end=None,
):
    question_ids = question_ids[:max_question_len]
    context_capacity = max_len - len(question_ids) - 4
    if context_capacity <= 0:
        raise ValueError("Question is too long")
    if not 0 <= doc_stride < context_capacity:
        raise ValueError("doc_stride must be smaller than context capacity")
    overlap = doc_stride
    if answer_start is not None or answer_end is not None:
        if answer_start is None or answer_end is None:
            raise ValueError("Both answer_start and answer_end are required")
        if answer_start < 0 or answer_end < answer_start:
            raise ValueError("Invalid answer subword span")
        if answer_end >= len(context_ids):
            raise ValueError("Answer span exceeds context length")
        answer_length = answer_end - answer_start + 1
        if answer_length > context_capacity:
            raise ValueError(
                f"answer_too_long: answer has {answer_length} subwords, "
                f"context capacity is {context_capacity}"
            )
        overlap = max(overlap, answer_length - 1)
    step = context_capacity - overlap
    windows = []
    start = 0
    while start < len(context_ids):
        end = min(start + context_capacity, len(context_ids))
        windows.append({
            "input_ids": [0] + question_ids + [2, 2] + context_ids[start:end] + [2],
            "context_start": start,
            "context_end": end,
        })
        if end == len(context_ids):
            break
        start += step
    return windows


def label_sliding_windows(windows, answer_start, answer_end,
                          question_length, is_impossible=False):
    context_offset = question_length + 3
    features = []
    for window in windows:
        if is_impossible:
            start_position = end_position = 0
        elif answer_start >= window["context_start"] and answer_end < window["context_end"]:
            start_position = context_offset + answer_start - window["context_start"]
            end_position = context_offset + answer_end - window["context_start"]
        else:
            start_position = end_position = 0
        features.append({
            **window,
            "start_positions": start_position,
            "end_positions": end_position,
        })
    return features


def read_vncorenlp_output(output_path):
    from pathlib import Path
    tokens = []
    for line in Path(output_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        columns = line.split("\t")
        if len(columns) >= 2:
            tokens.append(columns[1])
    return tokens


def refine_tokens_for_answer(text, tokens, answer_start, answer_text):
    answer_end = answer_start + len(answer_text)
    spans = align_tokens_to_text(text, tokens)
    refined = []
    for token, (start, end) in zip(tokens, spans):
        boundaries = sorted(boundary for boundary in (answer_start, answer_end)
                            if start < boundary < end)
        if not boundaries:
            refined.append(token)
            continue
        positions = [start] + boundaries + [end]
        for left, right in zip(positions, positions[1:]):
            part = text[left:right].strip()
            if part:
                refined.extend(part.split())
    return refined


def _answer_from_gold(row):
    """Đọc gold chỉ để tạo nhãn; tuyệt đối không gọi cho chế độ predict."""
    if row["is_impossible"]:
        return None
    answers = row["answers"]
    texts = answers["text"]
    starts = answers["answer_start"]
    if len(texts) != len(starts):
        raise ValueError("Answer texts and starts have different lengths")
    if not texts:
        raise ValueError("Answerable example has no answer")
    text, start = texts[0], starts[0]
    if not text or start < 0:
        raise ValueError("Invalid answer")
    if row["context"][start:start + len(text)] != text:
        raise ValueError("Answer span does not match context")
    return start, text


def _attach_word_offsets(feature, word_spans, subword_spans, question_length):
    """Vị trí từ trong context gốc + chỉ số subword *trong input_ids* của window.

    Chỉ giữ những từ nằm trọn trong window. Decoder sau này phải chọn start/end
    ở biên của những khoảng này, không dùng regex tìm lần xuất hiện đầu tiên.
    """
    word_offsets = []
    subword_offsets = []
    offset = question_length + 3
    for (char_start, char_end), (sub_start, sub_end) in zip(word_spans, subword_spans):
        if sub_start >= feature["context_start"] and sub_end <= feature["context_end"]:
            word_offsets.append([char_start, char_end])
            subword_offsets.append([
                offset + sub_start - feature["context_start"],
                offset + sub_end - feature["context_start"],
            ])
    feature["word_spans"] = word_offsets
    feature["subword_spans"] = subword_offsets


def preprocess_example(row, question_tokens, context_tokens, tokenizer, *, mode="train"):
    """Sinh feature QA.

    mode=train: hành vi cũ, cho phép dùng gold để chỉnh token/overlap (giữ tương thích).
    mode=eval: input_ids/windows độc lập với gold; gold chỉ dùng gán nhãn.
    mode=predict: không đọc gold, nhãn = -100 cho mọi window.

    Nếu câu có đáp án nhưng span không align được với biên từ, không tạo nhãn
    CLS giả; gắn -100 (= ignore_index của torch cross entropy) và ghi status.
    Cần evaluator dùng gold độc lập để tính EM/F1 và không tính loss cho nhãn -100.
    """
    if mode not in {"train", "eval", "predict"}:
        raise ValueError(f"Invalid mode: {mode}")
    if not row["context"]:
        raise ValueError("Context is empty")

    question_ids, _ = tokenize_words(tokenizer, question_tokens)
    question_ids = question_ids[:MAX_QUESTION_LEN]

    # Trong train cũ, token được tinh chỉnh để gold answer nằm đúng biên từ.
    # Trong eval/predict tuyệt đối không được dùng gold để tạo context_ids/window.
    if mode == "train" and not row["is_impossible"]:
        answer_start, answer_text = _answer_from_gold(row)
        context_tokens = refine_tokens_for_answer(
            row["context"], context_tokens, answer_start, answer_text
        )

    context_word_spans = align_tokens_to_text(row["context"], context_tokens)
    context_ids, context_subword_spans = tokenize_words(tokenizer, context_tokens)
    if not context_ids:
        raise ValueError("Context tokens are empty")

    # Cố định cửa sổ trước khi đọc gold trong mode eval; predict không đọc gold.
    if mode == "train" and not row["is_impossible"]:
        sub_start, sub_end = map_answer_to_subwords(
            answer_start, answer_text, context_word_spans, context_subword_spans
        )
        windows = create_sliding_windows(
            question_ids, context_ids, answer_start=sub_start, answer_end=sub_end
        )
    else:
        windows = create_sliding_windows(question_ids, context_ids)

    if mode == "predict":
        features = [{**w, "start_positions": IGNORE_INDEX,
                     "end_positions": IGNORE_INDEX} for w in windows]
        label_status = "unlabeled"
        is_impossible = False
    elif row["is_impossible"]:
        features = label_sliding_windows(
            windows, 0, 0, len(question_ids), is_impossible=True
        )
        label_status = "impossible"
        is_impossible = True
    else:
        is_impossible = False
        if mode == "eval":
            answer_start, answer_text = _answer_from_gold(row)
            try:
                sub_start, sub_end = map_answer_to_subwords(
                    answer_start, answer_text, context_word_spans, context_subword_spans
                )
            except ValueError as exc:
                # Việc gold nằm giữa một từ là giới hạn của decoder theo biên từ;
                # không được biến gold answerable thành nhãn impossible giả.
                if str(exc) not in {
                    "Answer does not overlap any token",
                    "Answer does not align with token boundaries",
                }:
                    raise
                sub_start = sub_end = None
        if sub_start is None:
            features = [{**w, "start_positions": IGNORE_INDEX,
                         "end_positions": IGNORE_INDEX} for w in windows]
            label_status = "unalignable"
        else:
            features = label_sliding_windows(
                windows, sub_start, sub_end, len(question_ids)
            )
            if not any(f["start_positions"] != 0 for f in features):
                if mode == "train":
                    raise ValueError("Answerable example has no positive window")
                for f in features:
                    f["start_positions"] = IGNORE_INDEX
                    f["end_positions"] = IGNORE_INDEX
                label_status = "not_covered"
            else:
                label_status = "aligned"

    for feature in features:
        feature["id"] = row["id"]
        feature["is_impossible"] = is_impossible
        feature["label_status"] = label_status
        _attach_word_offsets(
            feature, context_word_spans, context_subword_spans, len(question_ids)
        )
    return features


def read_qa_vncorenlp_output(output_path, num_examples=100):
    from pathlib import Path
    examples = []
    context_blocks = []
    current_block = []

    def save_block():
        return current_block.copy() if current_block else []

    for line in Path(output_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            if current_block:
                block = save_block()
                if block[0].startswith("ZZZQABOUNDARY"):
                    if len(context_blocks) < 2:
                        raise ValueError("Invalid QA example")
                    examples.append({
                        "question_tokens": context_blocks[0],
                        "context_tokens": [token for sentence in context_blocks[1:]
                                           for token in sentence],
                    })
                    context_blocks = []
                else:
                    context_blocks.append(block)
                current_block = []
            continue
        columns = line.split("\t")
        if len(columns) >= 2:
            current_block.append(columns[1].strip())
    if current_block:
        context_blocks.append(current_block)
    if len(examples) != num_examples:
        raise ValueError(f"Expected {num_examples} examples, got {len(examples)}")
    return examples
