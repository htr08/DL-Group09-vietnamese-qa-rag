"""Unit test cho phần không cần GPU của src/generate.py: prompt, CJK, từ chối, sinh lại."""
from src.generate import REFUSAL_TEXT, SYSTEM_PROMPT, build_messages, generate_answer, has_cjk, is_refusal


def test_build_messages_numbers_contexts_from_one():
    messages = build_messages("Học kỳ chính là gì?", ["đoạn thứ nhất", "đoạn thứ hai"])
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == SYSTEM_PROMPT
    user = messages[1]["content"]
    assert "[1] đoạn thứ nhất" in user and "[2] đoạn thứ hai" in user
    assert user.endswith("CÂU HỎI: Học kỳ chính là gì?")
    assert user.index("[1]") < user.index("[2]") < user.index("CÂU HỎI")


def test_system_prompt_contains_refusal_sentence():
    assert REFUSAL_TEXT in SYSTEM_PROMPT


def test_has_cjk_detects_han_kana_hangul_but_not_vietnamese():
    assert has_cjk("Sinh viên 学生 phải đóng học phí")
    assert has_cjk("ありがとう")
    assert has_cjk("안녕")
    assert not has_cjk("Sinh viên được rút bớt học phần trong 2 tuần.")


def test_is_refusal_ignores_case_and_unicode_form():
    assert is_refusal(REFUSAL_TEXT)
    assert is_refusal("KHÔNG TÌM THẤY THÔNG TIN trong tài liệu.")
    # Dạng NFD (dấu tách rời) vẫn được nhận diện.
    import unicodedata
    assert is_refusal(unicodedata.normalize("NFD", REFUSAL_TEXT))
    assert not is_refusal("Sinh viên được rút bớt học phần trong 2 tuần [1].")


def _scripted(outputs):
    """generate_fn giả: trả lần lượt các chuỗi trong `outputs`, ghi lại tham số lấy mẫu."""
    calls = []

    def fn(messages, sampling):
        calls.append(sampling)
        return outputs[len(calls) - 1]

    return fn, calls


def test_generate_answer_no_retry_when_clean():
    fn, calls = _scripted(["Trả lời sạch [1]."])
    result = generate_answer(fn, "câu hỏi", ["đoạn"])
    assert result["answer"] == "Trả lời sạch [1]."
    assert result["attempts"] == 1 and not result["cjk"] and not result["refusal"]
    assert calls == [{}]  # lần đầu dùng greedy (không tham số lấy mẫu)


def test_generate_answer_retries_with_sampling_after_cjk():
    fn, calls = _scripted(["Câu trả lời 学生", "Câu trả lời sạch."])
    result = generate_answer(fn, "câu hỏi", ["đoạn"])
    assert result["answer"] == "Câu trả lời sạch."
    assert result["attempts"] == 2 and not result["cjk"]
    assert calls[0] == {} and calls[1]["do_sample"] is True


def test_generate_answer_flags_cjk_when_retries_exhausted():
    fn, calls = _scripted(["学生 một", "学生 hai", "学生 ba"])
    result = generate_answer(fn, "câu hỏi", ["đoạn"], max_retries=2)
    assert len(calls) == 3  # 1 lần đầu + 2 lần sinh lại
    assert result["attempts"] == 3 and result["cjk"] is True
    assert result["answer"] == "学生 ba"


def test_generate_answer_marks_refusal():
    fn, _ = _scripted([REFUSAL_TEXT])
    assert generate_answer(fn, "câu hỏi", ["đoạn"])["refusal"] is True
