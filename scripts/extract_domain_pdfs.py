
from pathlib import Path
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / "data" / "domain"
OUTPUT_DIR = ROOT / "data" / "processed" / "domain"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for pdf_path in sorted(PDF_DIR.glob("*.pdf")):
    document = pymupdf.open(pdf_path)
    pages = []

    for page_number, page in enumerate(document, start=1):
        page_text = page.get_text("text").strip()
        pages.append(f"\n===== PAGE {page_number} =====\n{page_text}")

    full_text = "\n".join(pages)
    output_path = OUTPUT_DIR / f"{pdf_path.stem}.txt"
    output_path.write_text(full_text, encoding="utf-8")

    print(f"\nFile: {pdf_path.name}")
    print(f"So trang: {len(document)}")
    print(f"So ky tu trich xuat: {len(full_text)}")
    print(f"Da luu: {output_path}")

    print("\nTrich doan dau:")
    print(full_text[:700])

    document.close()
