"""Chạy baseline B2: hệ thống nền (mã gốc, Gemini) trên 100 câu hỏi miền Quy chế.

Mã gốc không nằm trong repo; script nạp nó từ một thư mục ngoài repo. Chuẩn bị:
    git archive <commit-cố-định-trong-docs/reuse_vs_new.md> legacy | tar -x -C <B2_BASE_DIR>
    pip install langchain-google-genai langchain-core langchain-text-splitters qdrant-client \
        "markitdown[pdf]" pdfplumber pymupdf pydantic-settings   # nên dùng venv riêng
    export GEMINI_API_KEY=...   # đặt trong .env (đã bị .gitignore chặn), không commit

Chạy (cấu hình giữ nguyên bản gốc: RRF_K = 15, top_k = 5, nhiệt độ 0,2, cắt đoạn bằng MarkItDown;
đổi model sinh bằng biến môi trường GEMINI_CHAT_MODEL, giá trị được ghi vào từng dòng kết quả):
    python scripts/run_b2.py --base-dir <B2_BASE_DIR> --pdf data/domain/<quy_che>.pdf \
        --gold-dir data/gold --out <kết_quả>.jsonl --embed-cache <đệm>.pkl

Bảo vệ hạn mức (gói miễn phí: sinh văn bản 20 lượt/ngày/dự án/model, embedding 1000 lượt/ngày):
- Mỗi câu gửi đúng MỘT request sinh: không thử lại (`max_retries=1`), không tự nối tiếp
  (`automatic_continuation=False`), có timeout HTTP thật. Lỗi 429/503 vẫn có thể bị tính vào hạn mức.
- Gặp bất kỳ lỗi nào thì ghi dòng lỗi và DỪNG cả lần chạy (không sang câu kế).
- Ngân sách lượt sinh trong ngày được trừ TRƯỚC khi gọi và lưu bền (`<kết_quả>.budget.json`).
- Mỗi request sinh được ghi vào `<kết_quả>.calls.jsonl` (giờ, câu hỏi, trạng thái, thời gian, token).
- Khóa `<kết_quả>.lock` chặn hai lần chạy chồng nhau; bộ canh giờ kết thúc cả tiến trình nếu một câu treo.
- Embedding được điều tiết (100 lượt/phút) và lưu ra --embed-cache để lần sau không tốn lại.
Chỉ lưu kết quả thô, không tính điểm.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import threading
import time
import types
from pathlib import Path


class QuotaStop(Exception):
    """Ngân sách lượt sinh trong ngày hạn mức đã hết: dừng lần chạy."""


def _install_settings_stub(api_key: str) -> None:
    """Thay `legacy/config.py` (cần PostgreSQL, MinIO... của backend) bằng cấu hình tối thiểu."""
    stub = types.ModuleType("legacy.config")
    stub.settings = types.SimpleNamespace(
        GEMINI_API_KEY=api_key, QDRANT_URL=":memory:", QDRANT_API_KEY=""
    )
    sys.modules["legacy.config"] = stub


def _call_once(fn, *args, **kwargs):
    """Gọi đúng một lần, không thử lại: lỗi 429/503 vẫn có thể bị tính vào hạn mức ngày."""
    return fn(*args, **kwargs)


def _quota_day() -> str:
    """Ngày hạn mức: hồi lúc nửa đêm giờ Thái Bình Dương (UTC-8 mùa đông = 15:00 giờ Việt Nam)."""
    return str((dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=8)).date())


def _spend_generate_budget(path: Path, cap: int) -> int:
    """Trừ một lượt sinh vào ngân sách trong ngày (lưu bền); dừng khi hết."""
    state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if state.get("day") != _quota_day():
        state = {"day": _quota_day(), "used": 0}
    if state["used"] >= cap:
        raise QuotaStop(f"đã dùng {state['used']}/{cap} lượt sinh trong ngày hạn mức {state['day']}")
    state["used"] += 1
    path.write_text(json.dumps(state), encoding="utf-8")
    return state["used"]


def _throttle_embeddings(cache_path: Path, max_per_minute: int = 80, slice_size: int = 40) -> None:
    """Điều tiết và lưu đệm lời gọi embedding để không vượt hạn mức miễn phí (100 lượt/phút).

    Chỉ thay cách gọi, không đổi kết quả: embedding của cùng một văn bản là xác định, nên
    văn bản đã gọi được dùng lại (tiêu đề chunk lặp lại nhiều lần trong bước xếp hạng lại).
    """
    import pickle

    from legacy.rag.embeddings import gemini as g

    cls = g.GeminiEmbeddingService
    orig_docs, orig_query = cls.embed_documents, cls.embed_query
    cache: dict[str, list[float]] = pickle.load(open(cache_path, "rb")) if cache_path.exists() else {}
    print(f"[embed] đệm trên đĩa: {len(cache)} văn bản", flush=True)
    window: list[float] = []

    def wait(n: int) -> None:
        while True:
            now = time.time()
            while window and now - window[0] > 60:
                window.pop(0)
            if len(window) + n <= max_per_minute:
                window.extend([now] * n)
                return
            time.sleep(max(1.0, 60 - (now - window[0]) + 0.5))

    def embed_documents(self, texts):
        missing = list(dict.fromkeys(t for t in texts if t not in cache))
        for i in range(0, len(missing), slice_size):
            part = missing[i : i + slice_size]
            wait(len(part))
            vecs = _call_once(orig_docs, self, part)
            cache.update(zip(part, vecs))
            pickle.dump(cache, open(cache_path, "wb"))
        return [cache[t] for t in texts]

    def embed_query(self, text):
        wait(1)
        return _call_once(orig_query, self, text)

    cls.embed_documents, cls.embed_query = embed_documents, embed_query


def _make_llm(st, api_key: str, calls_log: Path | None = None, cur: dict | None = None,
              timeout: float = 120, base_url: str | None = None):
    """LLM sinh: đúng một request HTTP mỗi lần gọi, có timeout thật, không tự nối tiếp.

    Mỗi request được ghi vào `calls_log` (nếu có) kèm mã câu hỏi `cur["qid"]`.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    kwargs = {"base_url": base_url} if base_url else {}
    llm = ChatGoogleGenerativeAI(
        model=st.GEMINI_CHAT_MODEL, google_api_key=api_key,
        temperature=st.GEMINI_TEMPERATURE, max_output_tokens=st.GEMINI_MAX_OUTPUT_TOKENS,
        max_retries=1,  # 1 = không thử lại (0 bị SDK hiểu là mặc định 5)
        timeout=timeout, **kwargs)
    if calls_log is not None:
        models = llm.client.models
        orig = models.generate_content

        def logged(*a, **kw):
            rec, t0 = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "qid": (cur or {}).get("qid")}, time.time()
            try:
                r = orig(*a, **kw)
                u = getattr(r, "usage_metadata", None)
                rec.update(status=200, thoughts=getattr(u, "thoughts_token_count", None),
                           out=getattr(u, "candidates_token_count", None))
                return r
            except Exception as exc:  # noqa: BLE001
                rec.update(status=getattr(exc, "code", None), err=str(exc)[:200])
                raise
            finally:
                rec["s"] = round(time.time() - t0, 1)
                with open(calls_log, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

        models.generate_content = logged
    return llm.bind(automatic_continuation=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=os.environ.get("B2_BASE_DIR"), required=not os.environ.get("B2_BASE_DIR"))
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--gold-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sleep", type=float, default=7.0, help="giây nghỉ giữa hai câu")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=120, help="giây chờ tối đa cho request sinh (timeout HTTP thật)")
    ap.add_argument("--gen-budget", type=int, default=18, help="số lượt sinh tối đa mỗi ngày hạn mức (hạn mức thật là 20)")
    ap.add_argument("--embed-cache", required=True, help="tệp pickle lưu embedding đã gọi (dùng lại khi chạy tiếp)")
    args = ap.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("Thiếu GEMINI_API_KEY", file=sys.stderr)
        return 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lock = out.with_suffix(".lock")
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print(f"Đã có phiên chạy khác hoặc phiên trước chưa dọn khóa: {lock}. Kiểm tra không còn tiến trình "
              "run_b2.py rồi xóa khóa.", file=sys.stderr)
        return 1
    try:
        return _run(args, api_key, out)
    finally:
        os.close(lock_fd)
        lock.unlink(missing_ok=True)


def _run(args, api_key: str, out: Path) -> int:
    sys.path.insert(0, args.base_dir)
    _install_settings_stub(api_key)

    from qdrant_client import QdrantClient

    from legacy.rag.pipeline import MarketingRAGEngine

    _throttle_embeddings(Path(args.embed_cache))

    engine = MarketingRAGEngine(qdrant_client=QdrantClient(":memory:"))
    cur: dict = {"qid": None}
    calls_log = out.with_suffix(".calls.jsonl")
    engine.generator.llm = _make_llm(engine.settings, api_key, calls_log, cur, timeout=args.timeout)

    t0 = time.time()
    result = engine.processor.process_document(file_path=args.pdf, filename=Path(args.pdf).name)
    print(f"[ingest] status={result.status} chunks={result.chunks_count} msg={result.message} ({time.time() - t0:.0f}s)")
    if result.status != "success" and not result.chunks_count:
        return 2

    questions = []
    for split in ("dev", "test"):
        with open(Path(args.gold_dir) / f"gold_{split}.jsonl", encoding="utf-8") as f:
            questions += [json.loads(line) | {"split": split} for line in f if line.strip()]
    questions.sort(key=lambda x: x["id"])
    if args.limit:
        questions = questions[: args.limit]

    done: set[str] = set()
    if out.exists():  # chạy tiếp: giữ các câu đã có câu trả lời, bỏ các dòng lỗi để chạy lại
        kept = [line for line in open(out, encoding="utf-8") if line.strip() and not json.loads(line).get("error")]
        out.write_text("".join(kept), encoding="utf-8")
        done = {json.loads(line)["id"] for line in kept}
    top_k = engine.settings.DEFAULT_TOP_K
    budget_file = out.with_suffix(".budget.json")
    watchdog_s = args.timeout + 180  # truy hồi (embedding) + sinh; quá giờ thì kết thúc cả tiến trình

    with open(out, "a", encoding="utf-8", newline="\n") as f:
        for q in questions:
            if q["id"] in done:
                continue
            cur["qid"] = q["id"]
            row = {"id": q["id"], "split": q["split"], "question": q["question"], "system": "B2_original_chunks",
                   "config": {"chat_model": engine.settings.GEMINI_CHAT_MODEL, "rrf_k": engine.settings.RRF_K,
                              "top_k": top_k, "chunk_size": engine.settings.CHUNK_SIZE,
                              "temperature": engine.settings.GEMINI_TEMPERATURE,
                              "max_retries": 1, "automatic_continuation": False}}
            start = time.time()

            def _kill():
                print(f"[canh giờ] {q['id']} quá {watchdog_s:.0f}s: kết thúc tiến trình", flush=True)
                os._exit(4)

            timer = threading.Timer(watchdog_s, _kill)
            timer.daemon = True
            timer.start()
            try:
                docs = _call_once(engine.retriever.retrieve, question=q["question"], top_k=top_k)
                t_ret = time.time() - start
                used = _spend_generate_budget(budget_file, args.gen_budget)
                resp = engine.generator.generate(question=q["question"], documents=docs)
            except Exception as exc:  # noqa: BLE001
                timer.cancel()
                row.update(answer=None, contexts=[], error=repr(exc)[:500], total_s=round(time.time() - start, 3))
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                print(f"[dừng] {q['id']}: {type(exc).__name__}: giữ hạn mức, không thử câu kế", flush=True)
                return 3
            timer.cancel()
            row.update(
                answer=resp.answer, retrieval_s=round(t_ret, 3), total_s=round(time.time() - start, 3),
                contexts=[{"headings": (d.metadata or {}).get("headings"), "chunk_id": (d.metadata or {}).get("chunk_id"),
                           "text": d.page_content} for d in docs],
                error=None, generate_calls_today=used)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"{q['id']} ok {row['total_s']}s (lượt sinh hôm nay: {used})", flush=True)
            time.sleep(args.sleep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
