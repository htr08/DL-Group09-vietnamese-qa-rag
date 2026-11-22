"""Chạy baseline B2: hệ thống nền (mã gốc, Gemini) trên 100 câu hỏi miền Quy chế.

Mã gốc không nằm trong repo; script nạp nó từ một thư mục ngoài repo. Chuẩn bị:
    git archive <commit-cố-định-trong-docs/reuse_vs_new.md> legacy | tar -x -C <B2_BASE_DIR>
    pip install langchain-google-genai langchain-core langchain-text-splitters qdrant-client         "markitdown[pdf]" pdfplumber pymupdf pydantic-settings   # nên dùng venv riêng
    export GEMINI_API_KEY=...   # đặt trong .env (đã bị .gitignore chặn), không commit

Chạy (cấu hình giữ nguyên bản gốc: RRF_K = 15, top_k = 5, nhiệt độ 0,2, cắt đoạn bằng MarkItDown;
đổi model sinh bằng biến môi trường GEMINI_CHAT_MODEL, giá trị được ghi vào từng dòng kết quả):
    python scripts/run_b2.py --base-dir <B2_BASE_DIR> --pdf data/domain/<quy_che>.pdf         --gold-dir data/gold --out <kết_quả>.jsonl --embed-cache <đệm>.pkl

Gói miễn phí giới hạn embedding 100 lượt/phút và 1000 lượt/ngày: embedding được điều tiết và
lưu ra --embed-cache để lần chạy sau không tốn lại hạn mức. Chỉ lưu kết quả thô, không tính điểm.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path


def _install_settings_stub(api_key: str) -> None:
    """Thay `legacy/config.py` (cần PostgreSQL, MinIO... của backend) bằng cấu hình tối thiểu."""
    stub = types.ModuleType("legacy.config")
    stub.settings = types.SimpleNamespace(
        GEMINI_API_KEY=api_key, QDRANT_URL=":memory:", QDRANT_API_KEY=""
    )
    sys.modules["legacy.config"] = stub


def _with_retry(fn, *args, tries: int = 6, **kwargs):
    for attempt in range(tries):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # hạn mức, lỗi mạng tạm thời
            text = str(exc).lower()
            if "perday" in text.replace(" ", "").replace("_", "") or "daily" in text:
                raise  # hết hạn mức ngày: chờ vô ích, để nơi gọi đổi key
            transient = any(k in text for k in ("429", "quota", "resource", "503", "unavailable", "timeout", "overload"))
            if attempt == tries - 1 or not transient:
                raise
            time.sleep(15 * (attempt + 1))


def _throttle_embeddings(cache_path: Path, max_per_minute: int = 80, slice_size: int = 40) -> None:
    """Điều tiết và lưu đệm lời gọi embedding để không vượt hạn mức miễn phí (100 lượt/phút).

    Chỉ thay cách gọi, không đổi kết quả: embedding của cùng một văn bản là xác định, nên
    văn bản đã gọi được dùng lại (tiêu đề chunk lặp lại nhiều lần trong bước xếp hạng lại).
    """
    from legacy.rag.embeddings import gemini as g

    cls = g.GeminiEmbeddingService
    orig_docs, orig_query = cls.embed_documents, cls.embed_query
    import pickle

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
            vecs = _with_retry(orig_docs, self, part)
            cache.update(zip(part, vecs))
            pickle.dump(cache, open(cache_path, "wb"))
        return [cache[t] for t in texts]

    def embed_query(self, text):
        wait(1)
        return _with_retry(orig_query, self, text)

    cls.embed_documents, cls.embed_query = embed_documents, embed_query


def _swap_key(engine, api_key: str) -> None:
    """Đổi key của bước sinh câu trả lời sang key dự phòng (cấu hình giữ nguyên)."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    st = engine.settings
    engine.generator.llm = ChatGoogleGenerativeAI(
        model=st.GEMINI_CHAT_MODEL, google_api_key=api_key,
        temperature=st.GEMINI_TEMPERATURE, max_output_tokens=st.GEMINI_MAX_OUTPUT_TOKENS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=os.environ.get("B2_BASE_DIR"), required=not os.environ.get("B2_BASE_DIR"))
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--gold-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sleep", type=float, default=7.0, help="giây nghỉ giữa hai câu (hạn mức miễn phí)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--embed-cache", required=True, help="tệp pickle lưu embedding đã gọi (dùng lại khi chạy tiếp)")
    args = ap.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("Thiếu GEMINI_API_KEY", file=sys.stderr)
        return 1
    sys.path.insert(0, args.base_dir)
    _install_settings_stub(api_key)

    from qdrant_client import QdrantClient

    from legacy.rag.pipeline import MarketingRAGEngine

    _throttle_embeddings(Path(args.embed_cache))

    client = QdrantClient(":memory:")
    engine = MarketingRAGEngine(qdrant_client=client)

    t0 = time.time()
    result = engine.processor.process_document(file_path=args.pdf, filename=Path(args.pdf).name)
    print(f"[ingest] status={result.status} chunks={result.chunks_count} msg={result.message} ({time.time() - t0:.0f}s)")
    if result.status != "success" and not result.chunks_count:
        return 2

    questions = []
    for split in ("dev", "test"):
        with open(Path(args.gold_dir) / f"gold_{split}.jsonl", encoding="utf-8") as f:
            questions += [json.loads(l) | {"split": split} for l in f if l.strip()]
    questions.sort(key=lambda x: x["id"])
    if args.limit:
        questions = questions[: args.limit]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)["id"] for l in open(out, encoding="utf-8") if l.strip()}
    top_k = engine.settings.DEFAULT_TOP_K
    backup_key = os.environ.get("GEMINI_API_KEY_2", "")
    state = {"swapped": False}

    with open(out, "a", encoding="utf-8", newline="\n") as f:
        for q in questions:
            if q["id"] in done:
                continue
            row = {"id": q["id"], "split": q["split"], "question": q["question"], "system": "B2_original_chunks",
                   "config": {"chat_model": engine.settings.GEMINI_CHAT_MODEL, "rrf_k": engine.settings.RRF_K,
                              "top_k": top_k, "chunk_size": engine.settings.CHUNK_SIZE,
                              "temperature": engine.settings.GEMINI_TEMPERATURE}}
            start = time.time()
            swapped_now = False
            while True:
                try:
                    docs = _with_retry(engine.retriever.retrieve, question=q["question"], top_k=top_k)
                    t_ret = time.time() - start
                    resp = _with_retry(engine.generator.generate, question=q["question"], documents=docs)
                    row.update(
                        answer=resp.answer,
                        retrieval_s=round(t_ret, 3),
                        total_s=round(time.time() - start, 3),
                        contexts=[{"headings": (d.metadata or {}).get("headings"), "chunk_id": (d.metadata or {}).get("chunk_id"),
                                   "text": d.page_content} for d in docs],
                        error=None,
                    )
                    break
                except Exception as exc:
                    text = str(exc).lower().replace(" ", "").replace("_", "")
                    if ("perday" in text or "daily" in text) and backup_key and not state["swapped"]:
                        print("[key] hết hạn mức ngày của key chính, chuyển sang key dự phòng", flush=True)
                        _swap_key(engine, backup_key)
                        state["swapped"] = True
                        continue
                    row.update(answer=None, contexts=[], error=f"{type(exc).__name__}: {exc}"[:500],
                               total_s=round(time.time() - start, 3))
                    break
            row["key"] = "backup" if state["swapped"] else "primary"
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"{q['id']} {'ERR' if row['error'] else 'ok'} {row['total_s']}s", flush=True)
            time.sleep(args.sleep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
