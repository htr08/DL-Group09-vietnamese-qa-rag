"""Sinh câu trả lời bằng Qwen2.5-7B (4-bit): prompt, ràng buộc ngữ cảnh, từ chối, kiểm tra CJK."""
from __future__ import annotations

import re
import time
import unicodedata
from typing import Callable, Sequence

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

REFUSAL_TEXT = "Không tìm thấy thông tin trong tài liệu."

# Prompt v1: trả lời bằng tiếng Việt, chỉ dựa trên ngữ cảnh, nêu nguồn theo số thứ tự
# đoạn, từ chối bằng một câu cố định. Phần ràng buộc ngữ cảnh và từ chối kế thừa từ
# prompt của notebook smoke test; phần nêu nguồn là bổ sung của v1.
SYSTEM_PROMPT = (
    "Bạn là trợ lý hỏi đáp tiếng Việt. "
    "Chỉ trả lời dựa trên phần NGỮ CẢNH được cung cấp, không dùng kiến thức bên ngoài. "
    "Luôn trả lời bằng tiếng Việt, ngắn gọn, chỉ dùng chữ Latinh (không dùng chữ Hán). "
    "Sau mỗi ý, nêu nguồn bằng số thứ tự của đoạn ngữ cảnh trong ngoặc vuông, ví dụ [1]. "
    "Nếu ngữ cảnh không chứa thông tin để trả lời, chỉ trả lời đúng một câu: "
    f"'{REFUSAL_TEXT}'"
)

# Dấu câu CJK (U+3000-303F), Hiragana, Katakana, chữ Hán (gồm mở rộng A), Hangul và các dạng toàn chiều rộng
# (U+FF00-FFEF); hai nhóm đầu thường đi kèm khi Qwen chuyển sang tiếng Trung, như dấu ， và 。.
_CJK_RE = re.compile(r"[\u3000-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uff00-\uffef]")

_CITATION_RE = re.compile(r"\[\d+\]")

# Tham số lấy mẫu cho các lần sinh lại: greedy cho cùng kết quả nên lần sinh lại phải lấy mẫu.
_RETRY_SAMPLING = {"do_sample": True, "temperature": 0.7, "top_p": 0.9}


def build_messages(question: str, contexts: Sequence[str]) -> list[dict]:
    """Ghép system prompt và user prompt; các đoạn ngữ cảnh được đánh số từ 1."""
    context = "\n\n".join(f"[{i}] {text}" for i, text in enumerate(contexts, start=1))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"NGỮ CẢNH:\n{context}\n\nCÂU HỎI: {question}"},
    ]


def has_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text))


def is_refusal(text: str) -> bool:
    """Câu trả lời chỉ gồm câu từ chối cố định (có thể kèm số nguồn và dấu câu cuối).

    Câu vừa từ chối một phần vừa nêu thông tin khác, ví dụ "Không tìm thấy thông tin về X, nhưng
    theo [1] ...", không tính là từ chối; câu như vậy được để cho bước chấm tay.
    """
    normalized = unicodedata.normalize("NFC", _CITATION_RE.sub("", text)).lower().strip(" .\n")
    return normalized == unicodedata.normalize("NFC", REFUSAL_TEXT).lower().rstrip(".")


def find_invalid_citations(answer: str, num_contexts: int) -> list[int]:
    """Các số nguồn [n] trong câu trả lời nằm ngoài khoảng 1..`num_contexts` (không trùng, tăng dần)."""
    cited = {int(m[1:-1]) for m in _CITATION_RE.findall(answer)}
    return sorted(n for n in cited if not 1 <= n <= num_contexts)


def generate_answer(
    generate_fn: Callable[[list[dict], dict], str],
    question: str,
    contexts: Sequence[str],
    max_retries: int = 2,
) -> dict:
    """Sinh câu trả lời và sinh lại (có lấy mẫu) tối đa `max_retries` lần nếu lẫn CJK.

    `generate_fn(messages, sampling_kwargs)` trả về văn bản; xem `make_generate_fn`.
    Nếu hết số lần sinh lại mà vẫn lẫn CJK, trả về lần sinh cuối với `cjk=True`
    để chỉ số "lẫn CJK" của báo cáo đếm được trường hợp này.
    Nếu không có đoạn ngữ cảnh nào (rỗng hoặc chỉ gồm khoảng trắng) thì trả về ngay câu từ chối
    cố định, không gọi mô hình (`attempts=0`), vì không có thông tin để trả lời.
    """
    if not any(text.strip() for text in contexts):
        return {"answer": REFUSAL_TEXT, "attempts": 0, "cjk": False, "refusal": True, "seconds": 0.0}
    messages = build_messages(question, contexts)
    start = time.perf_counter()
    answer = generate_fn(messages, {})
    attempts = 1
    while has_cjk(answer) and attempts <= max_retries:
        answer = generate_fn(messages, _RETRY_SAMPLING)
        attempts += 1
    return {
        "answer": answer,
        "attempts": attempts,
        "cjk": has_cjk(answer),
        "refusal": is_refusal(answer),
        "seconds": round(time.perf_counter() - start, 2),
    }


def load_model(model_id: str = MODEL_ID):
    """Nạp Qwen2.5-7B-Instruct ở 4-bit (NF4); cần GPU. Trả về (model, tokenizer)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map={"": 0}, dtype=torch.float16
    )
    model.eval()
    return model, tokenizer


def make_generate_fn(model, tokenizer, max_new_tokens: int = 256) -> Callable[[list[dict], dict], str]:
    """Tạo `generate_fn` cho `generate_answer` từ mô hình và tokenizer của Hugging Face."""
    import torch

    @torch.inference_mode()
    def generate_fn(messages: list[dict], sampling: dict) -> str:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        # Mặc định greedy; đặt temperature/top_p/top_k = None để không kèm cảnh báo của transformers.
        kwargs = {"do_sample": False, "temperature": None, "top_p": None, "top_k": None}
        kwargs.update(sampling)
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, **kwargs)
        new_ids = out[0, inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    return generate_fn
