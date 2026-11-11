from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union
import concurrent.futures

from langchain_core.documents import Document
import pdfplumber
from markitdown import MarkItDown
import fitz


class MarketingDocumentLoader:
    SUPPORTED_FORMATS = {
        ".txt",
        ".md",
        ".pdf",
        ".docx",
        ".pptx",
        ".xlsx",
        ".xls",
        ".csv",
        ".html",
        ".htm",
    }

    def __init__(
        self,
        markdown_dir: Optional[str] = None,
        *,
        verbose: bool = True,
    ) -> None:
        backend_dir = Path(__file__).resolve().parents[2]
        self.markdown_dir = (
            Path(markdown_dir) if markdown_dir else backend_dir / "data" / "markdown"
        )
        self.verbose = verbose

    def load(
        self,
        file_path: Union[str, List[str]],
        filename: Optional[str] = None,
        *,
        persist_markdown_to_disk: bool = False,
        max_workers: int = 4,
    ) -> List[Document]:
        paths = [file_path] if isinstance(file_path, str) else file_path

        if not paths:
            return []

        if len(paths) == 1:
            # Chạy đơn luồng
            return self._load_single(
                file_path=paths[0],
                filename=filename,
                persist_markdown_to_disk=persist_markdown_to_disk,
            )

        # Chạy đa luồng
        all_documents: List[Document] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {
                executor.submit(self._load_single, p, None, persist_markdown_to_disk): p
                for p in paths
            }
            for future in concurrent.futures.as_completed(future_to_path):
                path_str = future_to_path[future]
                try:
                    docs = future.result()
                    all_documents.extend(docs)
                except Exception as exc:
                    if self.verbose:
                        print(
                            f"[MarketingDocumentLoader] Lỗi load file trong batch ({path_str}): {exc}",
                            flush=True,
                        )

        return all_documents

    def _load_single(
        self,
        file_path: str,
        filename: Optional[str] = None,
        persist_markdown_to_disk: bool = False,
    ) -> List[Document]:
        src = Path(file_path)
        source_name = src.name

        if src.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Khong ho tro dinh dang `{src.suffix}` cho file: {source_name}. "
                f"SUPPORTED_FORMATS={sorted(self.SUPPORTED_FORMATS)}"
            )

        if persist_markdown_to_disk:
            self.markdown_dir.mkdir(parents=True, exist_ok=True)

        markdown_text = self._to_markdown(src)
        if not markdown_text.strip():
            return []

        # Tự động chèn Tên file thành Heading 1 (H1) ở đầu file nếu chưa có
        # Giúp các đoạn text/table bên dưới luôn kế thừa tên file thay vì rơi vào "General"
        display_name = filename or source_name
        display_name_clean = display_name.replace(".md", "").replace(".pdf", "")
        if not markdown_text.lstrip().startswith("# "):
            markdown_text = f"# {display_name_clean}\n\n{markdown_text.lstrip()}"

        if persist_markdown_to_disk:
            out_path = self.markdown_dir / f"{src.name}.md"
            out_path.write_text(markdown_text, encoding="utf-8", newline="\n")

        return [
            Document(
                page_content=markdown_text,
                metadata={
                    "filename": filename or source_name,
                    "file_type": src.suffix.lstrip("."),
                },
            )
        ]

    def _to_markdown(self, path: Path) -> str:
        ext = path.suffix.lower()

        if ext in {".txt", ".md"}:
            return path.read_text(encoding="utf-8", errors="replace")

        if ext == ".pdf":
            # Priority for PDF:
            # 1) MarkItDown (handles many office-export PDFs w/ text layer)
            # 2) PyMuPDF (fitz) text extraction
            # 3) pdfplumber (extract text fallback)
            text = self._pdf_via_markitdown(path)
            if text.strip():
                return text

            text = self._pdf_via_pymupdf(path)
            if text.strip():
                return text

            return self._pdf_via_pdfplumber(path)

        if ext == ".docx":
            text = self._docx_via_markitdown(path)
            if text.strip():
                return text
            return self._docx_via_python_docx(path)

        try:
            result = MarkItDown().convert(str(path))
            text = getattr(result, "text_content", None) or getattr(
                result, "markdown", None
            )
            return text if isinstance(text, str) else ""
        except Exception as exc:
            if self.verbose:
                print(f"[MarketingDocumentLoader] MarkItDown loi ({ext}): {exc}", flush=True)
            return ""

    def _pdf_via_markitdown(self, path: Path) -> str:
        """Trich text/markdown tu PDF bang MarkItDown (uu tien dau tien)."""

        try:
            result = MarkItDown().convert(str(path))
            text = getattr(result, "text_content", None) or getattr(
                result, "markdown", None
            )
            if isinstance(text, str) and text.strip():
                if self.verbose:
                    print(
                        "[MarketingDocumentLoader] PDF: dung MarkItDown.",
                        flush=True,
                    )
                return text
        except Exception as exc:
            if self.verbose:
                print(
                    f"[MarketingDocumentLoader] MarkItDown PDF loi: {exc}", flush=True
                )
        return ""

    def _pdf_via_pymupdf(self, path: Path) -> str:
        """Trich text tu PDF bang PyMuPDF (fitz). Khong lam OCR."""

        if fitz is None:
            if self.verbose:
                print(
                    "[MarketingDocumentLoader] PyMuPDF (fitz) khong import duoc.",
                    flush=True,
                )
            return ""

        doc = None
        try:
            doc = fitz.open(str(path))
            pages_text: list[str] = []
            for i in range(len(doc)):
                try:
                    text = doc[i].get_text() or ""
                except Exception:
                    text = ""
                if text.strip():
                    pages_text.append(f"## Trang {i + 1}\n\n{text.strip()}")
            out = "\n\n".join(pages_text)
            if out.strip() and self.verbose:
                print("[MarketingDocumentLoader] PDF: dung PyMuPDF.", flush=True)
            return out
        except Exception as exc:
            if self.verbose:
                print(f"[MarketingDocumentLoader] PyMuPDF PDF loi: {exc}", flush=True)
            return ""
        finally:
            if doc is not None:
                doc.close()

    def _docx_via_markitdown(self, path: Path) -> str:
        try:
            result = MarkItDown().convert(str(path))
            text = getattr(result, "text_content", None) or getattr(result, "markdown", None)
            if isinstance(text, str) and text.strip():
                if self.verbose:
                    print("[MarketingDocumentLoader] DOCX: dung MarkItDown.", flush=True)
                return text
        except Exception as exc:
            if self.verbose:
                print(f"[MarketingDocumentLoader] MarkItDown DOCX loi: {exc}", flush=True)
        return ""

    def _docx_via_python_docx(self, path: Path) -> str:
        try:
            import docx
            doc = docx.Document(str(path))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            text = "\n\n".join(paragraphs)
            if text.strip() and self.verbose:
                print("[MarketingDocumentLoader] DOCX: dung python-docx.", flush=True)
            return text
        except Exception as exc:
            if self.verbose:
                print(f"[MarketingDocumentLoader] python-docx loi: {exc}", flush=True)
            return ""

    def _pdf_via_pdfplumber(self, path: Path) -> str:
        """Trich text tu PDF bang pdfplumber (Khong lam OCR)."""
        try:
            pages_text: list[str] = []
            with pdfplumber.open(str(path)) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    if text.strip():
                        pages_text.append(f"## Trang {i + 1}\n\n{text.strip()}")
            out = "\n\n".join(pages_text)
            if out.strip() and self.verbose:
                print("[MarketingDocumentLoader] PDF: dung pdfplumber.", flush=True)
            return out
        except Exception as exc:
            if self.verbose:
                print(
                    f"[MarketingDocumentLoader] pdfplumber PDF loi: {exc}", flush=True
                )
            return ""
