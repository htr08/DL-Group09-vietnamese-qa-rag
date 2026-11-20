"""Dựng index BM25 (pickle) và dense (Qdrant) cho các kho đoạn văn.

Sử dụng:
    python scripts/build_index.py --corpus viquad   # kho ViQuAD (cần download_data.py trước)
    python scripts/build_index.py --corpus domain   # kho Quy chế ĐHQGHN (miền chính)
    python scripts/build_index.py --corpus samsung  # kho Samsung S24 (miền phụ)
    python scripts/build_index.py --all             # tất cả
"""
import argparse
import json
import pickle
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import List, Dict, Tuple

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tracking import init_run

RAW_DIR = ROOT / "data" / "raw"
DOMAIN_DIR = ROOT / "data" / "domain"
INDEX_DIR = ROOT / "data" / "index"  # gitignore

# Kích thước cửa sổ trượt cho Khoản quá dài (tính theo token BM25 xấp xỉ)
SLIDING_MAX_WORDS = 200
SLIDING_STRIDE = 180  # Chồng lặp nhẹ 10% ở biên để giữ mạch văn, tránh lặp 50% làm phình chunk


# ---------------------------------------------------------------------------
# Tiện ích chung
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Chuẩn hóa Unicode NFC, bỏ ký tự điều khiển."""
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)


def sliding_window(text: str, chunk_id: str, max_words: int = SLIDING_MAX_WORDS,
                   stride: int = SLIDING_STRIDE) -> List[Tuple[str, str]]:
    """
    Cắt đoạn dài thành các cửa sổ trượt, bảo tồn nguyên vẹn dấu xuống dòng (\\n).
    Cắt theo character slice dựa trên vị trí token, không dùng ' '.join() làm mất cấu trúc dòng.
    """
    tokens = list(re.finditer(r"\S+", text))
    if len(tokens) <= max_words:
        return [(chunk_id, text.strip())]

    chunks = []
    start = 0
    part = 0
    while start < len(tokens):
        end = min(start + max_words, len(tokens))
        char_start = tokens[start].start()
        char_end = tokens[end - 1].end()
        sub_text = text[char_start:char_end].strip()
        sub_id = f"{chunk_id}_w{part}"
        chunks.append((sub_id, sub_text))
        if end >= len(tokens):
            break
        start += stride
        part += 1
    return chunks


def save_bm25_index(doc_ids: List[str], corpus_texts: List[str], out_path: Path):
    """Dựng BM25Okapi và lưu vào file pickle."""
    from rank_bm25 import BM25Okapi

    def tokenize(t):
        return re.findall(r"\w+", t.lower())

    tokenized = [tokenize(t) for t in corpus_texts]
    bm25 = BM25Okapi(tokenized)
    index_obj = {"doc_ids": doc_ids, "bm25": bm25}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(index_obj, f)
    print(f"  BM25 → {out_path} ({len(doc_ids)} docs)")


def save_dense_index(doc_ids: List[str], corpus_texts: List[str],
                     collection: str, out_path: Path):
    """Encode bằng BGE-M3 và lưu numpy array; upsert vào Qdrant."""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("BAAI/bge-m3")
    print(f"  Encoding {len(doc_ids)} chunks với BGE-M3...")
    t0 = time.time()
    embeddings = model.encode(
        corpus_texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    encode_time = time.time() - t0
    print(f"  Thời gian encode: {encode_time:.2f}s ({encode_time/len(doc_ids):.4f}s/chunk)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path) + ".npy", embeddings)
    with open(str(out_path) + "_ids.json", "w", encoding="utf-8") as f:
        json.dump(doc_ids, f, ensure_ascii=False)
    print(f"  Dense file lưu tại: {out_path}.npy + {out_path}_ids.json")

    # Ghi vào Qdrant (tùy chọn — eval dùng numpy brute-force, không dùng Qdrant).
    # Qdrant chỉ phục vụ demo trực tiếp sau này; không ảnh hưởng kết quả đánh giá.
    # Nếu Qdrant chưa chạy hoặc không kết nối được, cảnh báo rõ và tiếp tục bình thường.
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct

        client = QdrantClient(host="localhost", port=6333)
        dim = embeddings.shape[1]

        # Nếu collection đã tồn tại, xóa đi tạo lại để số point khớp đúng 100%
        existing_cols = [c.name for c in client.get_collections().collections]
        if collection in existing_cols:
            client.delete_collection(collection_name=collection)

        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

        batch_size = 500
        total_points = len(doc_ids)
        for b_start in range(0, total_points, batch_size):
            b_end = min(b_start + batch_size, total_points)
            points = [
                PointStruct(
                    id=i,
                    vector=embeddings[i].tolist(),
                    payload={"doc_id": doc_ids[i], "text": corpus_texts[i]}
                )
                for i in range(b_start, b_end)
            ]
            client.upsert(collection_name=collection, points=points)

        info = client.get_collection(collection_name=collection)
        points_count = info.points_count if hasattr(info, 'points_count') else info.vectors_count
        print(f"  Qdrant '{collection}': {points_count} vectors (khớp {total_points} chunks)")
    except Exception as exc:
        print(f"  [Qdrant Warning] Không thể kết nối hoặc ghi vào Qdrant ({type(exc).__name__}: {exc}).")
        print(f"  → Toàn bộ vector đã được lưu an toàn vào {out_path}.npy để phục vụ evaluation pipeline (dùng numpy).")
    return encode_time


def log_index_run(corpus: str, n_chunks: int, encode_time: float):
    """Log số chunk và thời gian encode của một lần dựng index lên W&B (bật bằng --wandb, cần đăng nhập)."""
    run = init_run(
        name=f"B_bge-m3_index-{corpus}",
        job_type="build_index",
        config={"corpus": corpus, "sliding_max_words": SLIDING_MAX_WORDS, "sliding_stride": SLIDING_STRIDE},
    )
    run.summary["n_chunks"] = n_chunks
    run.summary["encode_time_s"] = round(encode_time, 2)
    run.finish()


# ---------------------------------------------------------------------------
# Kho ViQuAD
# ---------------------------------------------------------------------------

def build_viquad_index(log_wandb: bool = False):
    """Dựng index cho kho ViQuAD: gộp mọi đoạn văn, loại câu không có đáp án."""
    jsonl_path = RAW_DIR / "viquad_train.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"{jsonl_path} chưa tồn tại. Chạy scripts/download_data.py trước."
        )

    print("\n=== Kho ViQuAD ===")
    seen_contexts: Dict[str, str] = {}  # context_text -> doc_id

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            ctx = normalize_text(row["context"])
            if ctx not in seen_contexts:
                doc_id = f"viquad_{len(seen_contexts):05d}"
                seen_contexts[ctx] = doc_id

    doc_ids = list(seen_contexts.values())
    corpus_texts = list(seen_contexts.keys())
    print(f"  Tổng {len(doc_ids)} đoạn văn duy nhất từ ViQuAD train")

    save_bm25_index(doc_ids, corpus_texts, INDEX_DIR / "viquad" / "bm25.pkl")
    encode_time = save_dense_index(doc_ids, corpus_texts, "viquad",
                                   INDEX_DIR / "viquad" / "dense")
    if log_wandb:
        log_index_run("viquad", len(doc_ids), encode_time)

    # Lưu ánh xạ doc_id → text để dùng khi evaluate
    mapping = {did: txt for did, txt in zip(doc_ids, corpus_texts)}
    mp_path = INDEX_DIR / "viquad" / "docid_to_text.json"
    mp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(mp_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False)
    print(f"  Ánh xạ → {mp_path}")


# ---------------------------------------------------------------------------
# Kho Quy chế ĐHQGHN (miền chính)
# ---------------------------------------------------------------------------

_DIEU_HEADER = re.compile(
    r"(?m)^(Điều\s+\d+[.\s]*[^\n]{0,80})\s*$"
)
_KHOAN_HEADER = re.compile(
    r"(?m)^(\d+\.\s)"  # "1. ", "2. " …
)


def chunk_quy_che(text: str) -> List[Tuple[str, str, str]]:
    """
    Cắt văn bản Quy chế theo cấu trúc Điều → Khoản (và Điểm nếu Khoản dài).
    Trả về [(chunk_id, title, chunk_text), ...].
    Loại bỏ phần mục lục (3 trang cuối = sau "MỤC LỤC" hoặc "Điều 47").
    Đoạn dẫn trước Khoản 1 được đặt ID là dieuN_intro.
    Số Khoản được lấy chính xác từ regex ^(\\d+)\\., không đếm tuần tự enumerate.
    Giữ nguyên các dấu ngắt dòng \\n để không làm biến dạng văn bản.
    """
    # Loại phần mục lục
    muc_luc_match = re.search(r"(?im)^MỤC\s*LỤC", text)
    if muc_luc_match:
        text = text[: muc_luc_match.start()]

    # Tách thành các Điều
    parts = _DIEU_HEADER.split(text)
    chunks = []

    i = 1
    while i < len(parts) - 1:
        dieu_title = parts[i].strip()
        dieu_body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        i += 2

        # Tiêu đề Điều xuống dòng (Điều 22, 28): dòng đầu của thân bắt đầu bằng chữ thường
        # là phần còn lại của tiêu đề, không phải nội dung.
        first_line, _, rest = dieu_body.partition("\n")
        if first_line[:1].islower():
            dieu_title = f"{dieu_title} {first_line.strip()}"
            dieu_body = rest.strip()

        # Lấy số Điều
        dieu_num_match = re.search(r"\d+", dieu_title)
        dieu_num = dieu_num_match.group() if dieu_num_match else str(i)

        # Tách Khoản bên trong Điều bằng regex (?m)^(?=\d+\.\s)
        khoan_parts = re.split(r"(?m)^(?=\d+\.\s)", dieu_body)
        khoan_parts = [p.strip() for p in khoan_parts if p.strip()]

        if not khoan_parts:
            chunks.append((f"dieu{dieu_num}", dieu_title, f"{dieu_title}\n{dieu_body}"))
            continue

        for k_part in khoan_parts:
            m = re.match(r"^(\d+)\.\s", k_part)
            if m:
                khoan_num = m.group(1)
                chunk_id = f"dieu{dieu_num}_khoan{khoan_num}"
            else:
                # Đoạn dẫn đầu trước Khoản 1 (ví dụ Điều 10, 11, 22, 27, 28, 37, 39, 41)
                chunk_id = f"dieu{dieu_num}_intro"

            full_text = f"{dieu_title}\n{k_part}"
            words = full_text.split()
            has_diem = bool(re.search(r"(?m)^[a-zđ]\)\s", k_part))

            # Nếu Khoản dài > SLIDING_MAX_WORDS và có phân Điểm (a, b, c) → tách theo Điểm
            # và sliding_window tiếp cho Điểm dài (như dieu8_khoan1_diemb), đảm bảo MỌI cửa sổ đều có tiêu đề Điều
            if len(words) > SLIDING_MAX_WORDS and has_diem:
                diem_parts = re.split(r"(?m)^(?=[a-zđ]\)\s)", k_part)
                diem_intro = diem_parts[0].strip()
                for d_part in diem_parts[1:]:
                    dm = re.match(r"^([a-zđ])\)\s", d_part.strip())
                    if dm:
                        # Điểm đ có ID ASCII `diemdd` để tách khỏi Điểm d (`diemd`)
                        diem_letter = "dd" if dm.group(1) == "đ" else dm.group(1)
                        sub_id = f"{chunk_id}_diem{diem_letter}"
                        header = f"{dieu_title}\n{diem_intro}" if diem_intro else dieu_title
                        body = d_part.strip()
                    else:
                        sub_id = chunk_id
                        header = dieu_title
                        body = d_part.strip()
                    # Trượt cửa sổ trên thân Điểm và ghép header tiêu đề Điều vào MỌI cửa sổ
                    for cid, ctxt in sliding_window(body, sub_id, max_words=SLIDING_MAX_WORDS):
                        chunks.append((cid, dieu_title, f"{header}\n{ctxt}"))
            elif len(words) > SLIDING_MAX_WORDS:
                # Khoản dài không có Điểm (như dieu40_khoan2) → trượt cửa sổ trên k_part và ghép tiêu đề Điều vào MỌI cửa sổ
                for cid, ctxt in sliding_window(k_part, chunk_id, max_words=SLIDING_MAX_WORDS):
                    chunks.append((cid, dieu_title, f"{dieu_title}\n{ctxt}"))
            else:
                chunks.append((chunk_id, dieu_title, full_text))

    return chunks


def build_domain_index(log_wandb: bool = False):
    """Dựng index cho kho Quy chế ĐHQGHN (miền chính)."""
    try:
        import pdfplumber
    except ImportError:
        raise SystemExit("Cài pdfplumber trước: pip install pdfplumber")

    pdf_path = DOMAIN_DIR / "dhqghn_quy_che_dao_tao_dai_hoc_2014.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"{pdf_path} chưa tồn tại trong data/domain/")

    print("\n=== Kho Quy chế ĐHQGHN (miền chính) ===")
    with pdfplumber.open(pdf_path) as pdf:
        pages = pdf.pages[:-3]  # bỏ 3 trang cuối (mục lục)
        full_text = "\n".join(
            normalize_text(p.extract_text() or "") for p in pages
        )

    chunks = chunk_quy_che(full_text)
    print(f"  Số chunk sau khi cắt Điều → Khoản: {len(chunks)}")

    doc_ids = [c[0] for c in chunks]
    titles = {c[0]: c[1] for c in chunks}
    corpus_texts = [c[2] for c in chunks]

    save_bm25_index(doc_ids, corpus_texts, INDEX_DIR / "domain" / "bm25.pkl")
    encode_time = save_dense_index(doc_ids, corpus_texts, "domain_vn",
                                   INDEX_DIR / "domain" / "dense")
    if log_wandb:
        log_index_run("domain", len(doc_ids), encode_time)

    # Lưu metadata chunk
    meta_path = INDEX_DIR / "domain" / "chunks_meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            [{"id": did, "title": titles[did], "text": txt}
             for did, txt in zip(doc_ids, corpus_texts)],
            f, ensure_ascii=False, indent=2
        )
    print(f"  Metadata chunk → {meta_path}")


# ---------------------------------------------------------------------------
# Kho Samsung S24 (miền phụ, lệch ngôn ngữ)
# ---------------------------------------------------------------------------

def build_samsung_index(log_wandb: bool = False):
    """Dựng index cho 2 PDF Samsung Galaxy S24 (tiếng Anh)."""
    try:
        import pdfplumber
    except ImportError:
        raise SystemExit("Cài pdfplumber trước: pip install pdfplumber")

    pdfs = [
        DOMAIN_DIR / "samsung_galaxy_s24_128gb_device_specs.pdf",
        DOMAIN_DIR / "samsung_galaxy_s24_specifications.pdf",
    ]

    print("\n=== Kho Samsung S24 (miền phụ, lệch ngôn ngữ) ===")
    doc_ids, corpus_texts = [], []
    for pdf_path in pdfs:
        if not pdf_path.exists():
            print(f"  Bỏ qua (không tìm thấy): {pdf_path}")
            continue
        with pdfplumber.open(pdf_path) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                text = normalize_text(page.extract_text() or "").strip()
                if not text:
                    continue
                doc_id = f"s24_{pdf_path.stem}_p{page_idx + 1:03d}"
                # Cắt cửa sổ trượt nếu trang quá dài
                for cid, ctxt in sliding_window(text, doc_id):
                    doc_ids.append(cid)
                    corpus_texts.append(ctxt)

    print(f"  Số chunk từ Samsung S24 PDFs: {len(doc_ids)}")
    if not doc_ids:
        print("  (Không có chunk nào — bỏ qua index)")
        return

    save_bm25_index(doc_ids, corpus_texts, INDEX_DIR / "samsung" / "bm25.pkl")
    encode_time = save_dense_index(doc_ids, corpus_texts, "samsung_en",
                                   INDEX_DIR / "samsung" / "dense")
    if log_wandb:
        log_index_run("samsung", len(doc_ids), encode_time)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Dựng index BM25 và dense.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--corpus", choices=["viquad", "domain", "samsung"])
    group.add_argument("--all", action="store_true", help="Dựng tất cả các kho")
    parser.add_argument("--wandb", action="store_true",
                        help="Log số chunk và thời gian encode lên W&B (cần đăng nhập W&B)")
    args = parser.parse_args()

    if args.all or args.corpus == "viquad":
        build_viquad_index(log_wandb=args.wandb)
    if args.all or args.corpus == "domain":
        build_domain_index(log_wandb=args.wandb)
    if args.all or args.corpus == "samsung":
        build_samsung_index(log_wandb=args.wandb)

    print("\n✓ Dựng index hoàn tất.")


if __name__ == "__main__":
    main()
