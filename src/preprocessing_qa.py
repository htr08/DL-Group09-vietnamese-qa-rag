import unicodedata
import re
MAX_LEN = 256
MAX_QUESTION_LEN = 64
DOC_STRIDE = 128

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

    positions = [
        i for i, char in enumerate(normalized_text)
        if not char.isspace()
    ]
    searchable = "".join(
        normalized_text[i] for i in positions
    )

    spans = []
    search_cursor = 0

    for token in tokens:
        normalized_token, _ = normalize_with_mapping(
            token.replace("_", " ")
        )
        target = "".join(normalized_token.split())

        position = searchable.find(target, search_cursor)

        if position >= 0:
            start_position = positions[position]
            end_position = positions[position + len(target) - 1]

        else:
            raw_position = (
                mapping[positions[search_cursor]]
                if search_cursor < len(positions)
                else len(text)
            )

            if (
                len(target) == 1
                and raw_position < len(text)
                and ord(text[raw_position]) > 0xFFFF
                and (ord(text[raw_position]) & 0xFFFF) == ord(target)
            ):
                start_position = positions[search_cursor]
                end_position = start_position
            else:
                raise ValueError(
                    f"Cannot align token: {token}; "
                    f"search_cursor={search_cursor}"
                )

        start = mapping[start_position]
        end = mapping[end_position] + 1

        spans.append((start, end))
        search_cursor = position + len(target) if position >= 0 else search_cursor + 1

    return spans



def tokenize_words(tokenizer, tokens):
    input_ids = []
    subword_spans = []

    for token in tokens:
        token_ids = tokenizer.encode(
            token,
            add_special_tokens=False
        )

        start = len(input_ids)
        input_ids.extend(token_ids)
        end = len(input_ids)

        subword_spans.append((start, end))

    return input_ids, subword_spans
def map_answer_to_subwords(
    answer_start,
    answer_text,
    word_spans,
    subword_spans
):
    answer_end = answer_start + len(answer_text)

    overlapping = [
        i
        for i, (start, end) in enumerate(word_spans)
        if start < answer_end and end > answer_start
    ]

    if not overlapping:
        raise ValueError("Answer does not overlap any token")

    start_subword = subword_spans[overlapping[0]][0]
    end_subword = subword_spans[overlapping[-1]][1] - 1

    return start_subword, end_subword
def create_sliding_windows(
    question_ids,
    context_ids,
    max_len=MAX_LEN,
    max_question_len=MAX_QUESTION_LEN,
    doc_stride=DOC_STRIDE
):
    question_ids = question_ids[:max_question_len]

    special_tokens = 4
    context_capacity = max_len - len(question_ids) - special_tokens

    if context_capacity <= 0:
        raise ValueError("Question is too long")

    if not 0 <= doc_stride < context_capacity:
        raise ValueError("doc_stride must be smaller than context capacity")

    windows = []
    start = 0

    while start < len(context_ids):
        end = min(start + context_capacity, len(context_ids))

        window_context = context_ids[start:end]

        input_ids = [0] + question_ids + [2, 2] + window_context + [2]

        windows.append({
            "input_ids": input_ids,
            "context_start": start,
            "context_end": end
        })

        if end == len(context_ids):
            break

        start += context_capacity - doc_stride
    return windows
def label_sliding_windows(
    windows,
    answer_start,
    answer_end,
    question_length,
    is_impossible=False
):
    context_offset = question_length + 3

    features = []

    for window in windows:
        if is_impossible:
            start_position = 0
            end_position = 0

        elif (
            answer_start >= window["context_start"]
            and answer_end < window["context_end"]
        ):
            start_position = (
                context_offset
                + answer_start
                - window["context_start"]
            )

            end_position = (
                context_offset
                + answer_end
                - window["context_start"]
            )

        else:
            start_position = 0
            end_position = 0

        features.append({
            **window,
            "start_positions": start_position,
            "end_positions": end_position
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
def preprocess_example(row, question_tokens, context_tokens, tokenizer):
    if not row["context"]:
        raise ValueError("Context is empty")

    question_ids, _ = tokenize_words(tokenizer, question_tokens)
    question_ids = question_ids[:MAX_QUESTION_LEN]

    context_word_spans = align_tokens_to_text(
        row["context"],
        context_tokens
    )

    context_ids, context_subword_spans = tokenize_words(
        tokenizer,
        context_tokens
    )

    if not context_ids:
        raise ValueError("Context tokens are empty")

    windows = create_sliding_windows(
        question_ids,
        context_ids
    )

    is_impossible = row["is_impossible"]

    if is_impossible:
        features = label_sliding_windows(
            windows,
            0,
            0,
            len(question_ids),
            is_impossible=True
        )
    else:
        answers = row["answers"]
        answer_texts = answers["text"]
        answer_starts = answers["answer_start"]

        if len(answer_texts) != len(answer_starts):
            raise ValueError("Answer texts and starts have different lengths")

        if not answer_texts:
            raise ValueError("Answerable example has no answer")

        answer_text = answer_texts[0]
        answer_start = answer_starts[0]

        if not answer_text or answer_start < 0:
            raise ValueError("Invalid answer")

        if row["context"][answer_start:answer_start + len(answer_text)] != answer_text:
            raise ValueError("Answer span does not match context")

        start_subword, end_subword = map_answer_to_subwords(
            answer_start,
            answer_text,
            context_word_spans,
            context_subword_spans
        )

        features = label_sliding_windows(
            windows,
            start_subword,
            end_subword,
            len(question_ids)
        )

    for feature in features:
        feature["id"] = row["id"]
        feature["is_impossible"] = is_impossible

    return features

def read_qa_vncorenlp_output(output_path, num_examples=100):
    from pathlib import Path

    examples = []
    context_blocks = []
    current_block = []

    def save_block():
        if current_block:
            return current_block.copy()
        return []

    for line in Path(output_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            if current_block:
                block = save_block()

                if block[0].startswith("ZZZQABOUNDARY"):
                    if len(context_blocks) < 2:
                        raise ValueError("Invalid QA example")

                    examples.append({
                        "question_tokens": context_blocks[0],
                        "context_tokens": [
                            token
                            for sentence in context_blocks[1:]
                            for token in sentence
                        ]
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
        raise ValueError(
            f"Expected {num_examples} examples, got {len(examples)}"
        )

    return examples