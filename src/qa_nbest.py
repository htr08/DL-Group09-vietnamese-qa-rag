"""Giải mã QA n-best và chọn ngưỡng từ chối trên Dev.

Khi có word_spans/subword_spans, kết quả được trích đúng vị trí ký tự từ
context gốc; không tìm bằng regex (vốn có thể chọn sai lần xuất hiện).
"""
from __future__ import annotations

import math

# mode=train dùng gold để chỉnh token và sliding window nên không được chấm Dev/Test.
GOLD_INDEPENDENT_MODES = frozenset({"eval", "predict"})


def require_gold_independent_features(features, source="features"):
    """Từ chối feature không có `preprocessing_mode` thuộc eval/predict.

    Feature tạo trước khi trường này tồn tại cũng bị từ chối vì không chứng minh
    được chúng độc lập với gold; cần tạo lại bằng preprocess_qa.py (mode=eval).
    """
    for index, feature in enumerate(features):
        mode = feature.get("preprocessing_mode")
        if mode not in GOLD_INDEPENDENT_MODES:
            raise ValueError(
                f"{source}: feature #{index} (id={feature.get('id')}) có "
                f"preprocessing_mode={mode!r}; chỉ chấp nhận "
                f"{sorted(GOLD_INDEPENDENT_MODES)}. Tạo lại Dev/Test bằng mode=eval."
            )


def decode_window(input_ids, start_logits, end_logits, tokenizer,
                  n_best_size=20, max_answer_length=30, sep_token_id=2,
                  *, word_spans=None, subword_spans=None, context=None):
    """Trả (candidates, null_score). Chỉ số end_token là inclusive.

    Metadata là cặp tương ứng theo từ: word_spans=[char_start,char_end] trên
    context gốc; subword_spans=[token_start,token_end) trên input_ids window.
    Nếu cung cấp metadata, chỉ chọn biên từ hợp lệ; không đoán văn bản bằng decode.
    Không cung cấp metadata: giữ giải mã legacy để các test cũ vẫn hoạt động;
    tuyệt đối không dùng legacy mode để báo cáo kết quả Dev mới.
    """
    if n_best_size <= 0 or max_answer_length <= 0:
        raise ValueError("n_best_size and max_answer_length must be positive")
    if len(input_ids) != len(start_logits) or len(input_ids) != len(end_logits):
        raise ValueError("logits/input_ids length mismatch")
    if not input_ids:
        raise ValueError("Empty QA window")
    null_score = float(start_logits[0]) + float(end_logits[0])
    context_start = next(
        (i + 2 for i in range(len(input_ids) - 1)
         if input_ids[i] == sep_token_id and input_ids[i + 1] == sep_token_id),
        None,
    )
    if context_start is None:
        raise ValueError("Cannot find pair separator </s></s> in QA input")
    context_end = len(input_ids) - 1
    if context_start >= context_end:
        return [], null_score

    use_offsets = any(x is not None for x in (word_spans, subword_spans, context))
    if use_offsets and any(x is None for x in (word_spans, subword_spans, context)):
        raise ValueError("word_spans, subword_spans and context must be supplied together")

    if use_offsets:
        if len(word_spans) != len(subword_spans):
            raise ValueError("Word/subword offsets count mismatch")
        valid_starts = []
        valid_ends = []
        start_char = {}
        end_char = {}
        for (a, b), (s, t) in zip(word_spans, subword_spans):
            if not (0 <= a < b <= len(context)):
                raise ValueError("Invalid character offsets")
            if not (context_start <= s < t <= context_end):
                raise ValueError("Invalid token offsets")
            valid_starts.append(int(s))
            valid_ends.append(int(t) - 1)
            start_char[int(s)] = int(a)
            end_char[int(t) - 1] = int(b)
    else:
        valid_starts = list(range(context_start, context_end))
        valid_ends = valid_starts

    starts = sorted(valid_starts, key=lambda i: float(start_logits[i]), reverse=True)[:n_best_size]
    ends = sorted(valid_ends, key=lambda i: float(end_logits[i]), reverse=True)[:n_best_size]
    pairs = []
    for s in starts:
        for e in ends:
            if e < s or e - s + 1 > max_answer_length:
                continue
            pairs.append((float(start_logits[s]) + float(end_logits[e]), s, e))
    pairs.sort(reverse=True)
    unique = []
    seen = set()
    for score, s, e in pairs:
        if use_offsets:
            a, b = start_char[s], end_char[e]
            if a >= b:
                continue
            text = context[a:b]
            raw_text = text
            # Dedupe theo vị trí, không theo nội dung (có thể lặp 'Hà Nội').
            identity = (a, b)
        else:
            raw_text = tokenizer.decode(
                input_ids[s:e + 1], skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()
            text = raw_text.replace("_", " ")
            identity = text
            a = b = None
        if not text.strip() or identity in seen:
            continue
        seen.add(identity)
        candidate = {
            "text": text.strip(), "raw_text": raw_text, "score": score,
            "start_token": s, "end_token": e,
        }
        if use_offsets:
            candidate.update(char_start=a, char_end=b)
        unique.append(candidate)
        if len(unique) >= n_best_size:
            break
    return unique, null_score


def scores_at_threshold(items, threshold):
    """Trả EM/F1 trung bình từ các điểm số Dev theo từng câu hỏi."""
    if not items:
        raise ValueError("Cannot evaluate an empty dev set")
    em = f1 = 0.0
    for item in items:
        answer = item["score_diff"] <= threshold
        em += item["answer_em"] if answer else item["null_em"]
        f1 += item["answer_f1"] if answer else item["null_f1"]
    return em / len(items), f1 / len(items)


def best_dev_threshold(items):
    """Quét mọi ngưỡng trên Dev; tối ưu F1, rồi EM, rồi gần 0 nhất.

    score_diff = null_score - best_span_score; chọn span nếu <= threshold.
    Không gọi trên public test.
    """
    if not items:
        raise ValueError("Cannot tune on an empty dev set")
    rows = sorted(items, key=lambda r: r["score_diff"])
    em = sum(r["null_em"] for r in rows)
    f1 = sum(r["null_f1"] for r in rows)
    initial = math.nextafter(float(rows[0]["score_diff"]), -math.inf)
    best = (initial, em / len(rows), f1 / len(rows))
    i = 0
    while i < len(rows):
        boundary = rows[i]["score_diff"]
        while i < len(rows) and rows[i]["score_diff"] == boundary:
            row = rows[i]
            em += row["answer_em"] - row["null_em"]
            f1 += row["answer_f1"] - row["null_f1"]
            i += 1
        cur = (float(boundary), em / len(rows), f1 / len(rows))
        key_cur = (round(cur[2], 12), round(cur[1], 12), -abs(cur[0]))
        key_best = (round(best[2], 12), round(best[1], 12), -abs(best[0]))
        if key_cur > key_best:
            best = cur
    return best
