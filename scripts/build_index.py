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
import unicodedata
from pathlib import Path
from typing import List, Dict, Tuple

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
DOMAIN_DIR = ROOT / "data" / "domain"
INDEX_DIR = ROOT / "data" / "index"  # gitignore

# Kích thước cửa sổ trượt cho Khoản quá dài (tính theo token BM25 xấp xỉ)
SLIDING_MAX_WORDS = 200
SLIDING_STRIDE = 100

VIQUAD_HF_NAME = "taidng/UIT-ViQuAD2.0"


# ---------------------------------------------------------------------------
# Tiện ích chung
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Chuẩn hóa Unicode NFC, bỏ ký tự điều khiển."""
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)


def sliding_window(text: str, chunk_id: str, max_words: int = SLIDING_MAX_WORDS,
                   stride: int = SLIDING_STRIDE) -> List[Tuple[str, str]]:
    """Cắt đoạn dài thành các cửa sổ trượt; trả về [(sub_id, sub_text), ...]."""
    words = text.split()
    if len(words) <= max_words:
        return [(chunk_id, text)]
    chunks = []
    start = 0
    part = 0
    while start < len(words):
        window = words[start: start + max_words]
        sub_id = f"{chunk_id}_w{part}"
        chunks.append((sub_id, " ".join(window)))
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
    """Encode bằng BGE-M3 và lưu numpy array; optionally upsert vào Qdrant."""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("BAAI/bge-m3")
    print(f"  Encoding {len(doc_ids)} docs với BGE-M3 ...")
    embeddings = model.encode(
        corpus_texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path) + ".npy", embeddings)
    with open(str(out_path) + "_ids.json", "w", encoding="utf-8") as f:
        json.dump(doc_ids, f, ensure_ascii=False)
    print(f"  Dense → {out_path}.npy + {out_path}_ids.json")

    # Upsert vào Qdrant nếu server đang chạy (bỏ qua nếu lỗi kết nối)
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct

        client = QdrantClient(host="localhost", port=6333)
        dim = embeddings.shape[1]
        if collection not in [c.name for c in client.get_collections().collections]:
            client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
        points = [
            PointStruct(id=i, vector=embeddings[i].tolist(),
                        payload={"doc_id": doc_ids[i], "text": corpus_texts[i]})
            for i in range(len(doc_ids))
        ]
        client.upsert(collection_name=collection, points=points)
        print(f"  Qdrant collection '{collection}': {len(points)} vectors upserted")
    except Exception as exc:
        print(f"  (Qdrant không khả dụng, bỏ qua: {exc})")


# ---------------------------------------------------------------------------
# Kho ViQuAD
# ---------------------------------------------------------------------------

def build_viquad_index():
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
    save_dense_index(doc_ids, corpus_texts, "viquad",
                     INDEX_DIR / "viquad" / "dense")

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
    Cắt văn bản Quy chế theo cấu trúc Điều → Khoản.
    Trả về [(chunk_id, title, chunk_text), ...].
    Loại bỏ phần mục lục (3 trang cuối = sau "MỤC LỤC" hoặc "Điều 47").
    """
    # Loại phần mục lục
    muc_luc_match = re.search(r"(?im)^MỤC\s*LỤC", text)
    if muc_luc_match:
        text = text[: muc_luc_match.start()]

    # Tách thành các Điều
    parts = _DIEU_HEADER.split(text)
    # parts[0] = phần trước Điều 1 (lời mở đầu), parts[1] = tiêu đề Điều 1, ...
    chunks = []

    i = 1
    while i < len(parts) - 1:
        dieu_title = parts[i].strip()
        dieu_body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        i += 2

        # Lấy số Điều
        dieu_num_match = re.search(r"\d+", dieu_title)
        dieu_num = dieu_num_match.group() if dieu_num_match else str(i)

        # Tách Khoản bên trong Điều
        khoan_parts = re.split(r"(?m)^(?=\d+\.\s)", dieu_body)
        khoan_parts = [p.strip() for p in khoan_parts if p.strip()]

        if len(khoan_parts) <= 1:
            # Điều không có Khoản hoặc chỉ 1 Khoản
            sub_chunks = sliding_window(
                dieu_title + "\n" + dieu_body,
                f"dieu{dieu_num}"
            )
            for cid, ctxt in sub_chunks:
                chunks.append((cid, dieu_title, ctxt))
        else:
            for khoan_idx, khoan_text in enumerate(khoan_parts, start=1):
                chunk_id = f"dieu{dieu_num}_khoan{khoan_idx}"
                full_text = f"{dieu_title}\n{khoan_text}"
                sub_chunks = sliding_window(full_text, chunk_id)
                for cid, ctxt in sub_chunks:
                    chunks.append((cid, dieu_title, ctxt))

    return chunks


def build_domain_index():
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
    save_dense_index(doc_ids, corpus_texts, "domain_vn",
                     INDEX_DIR / "domain" / "dense")

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

def build_samsung_index():
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
    save_dense_index(doc_ids, corpus_texts, "samsung_en",
                     INDEX_DIR / "samsung" / "dense")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Dựng index BM25 và dense.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--corpus", choices=["viquad", "domain", "samsung"])
    group.add_argument("--all", action="store_true", help="Dựng tất cả các kho")
    args = parser.parse_args()

    if args.all or args.corpus == "viquad":
        build_viquad_index()
    if args.all or args.corpus == "domain":
        build_domain_index()
    if args.all or args.corpus == "samsung":
        build_samsung_index()

    print("\n✓ Dựng index hoàn tất.")


if __name__ == "__main__":
    main()
