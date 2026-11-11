"""Chia nho tai lieu thanh chunks de embedding."""

from typing import Iterable, List
from uuid import NAMESPACE_URL, uuid5

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from ..config import rag_settings


class MarketingTextSplitter:
    def __init__(
        self,
        chunk_size: int = rag_settings.CHUNK_SIZE,
        chunk_overlap: int = rag_settings.CHUNK_OVERLAP,
    ) -> None:
        self.headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on, strip_headers=False
        )
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",  # Đoạn văn
                "\n",  # Dòng
                ". ",  # Câu
                "! ",
                "? ",
                "; ",
                ": ",
                " ",  # Từ
                "",  # Ký tự
            ],
        )

    def split(self, documents: Iterable[Document]) -> List[Document]:
        """Split documents va gan metadata chunk on dinh."""
        all_chunks: List[Document] = []
        global_index = 0

        for doc in documents:
            # Bước 1: Chia theo cấu trúc Markdown (Heading)
            md_chunks = self.markdown_splitter.split_text(doc.page_content)

            # Bước 2: Chia nhỏ thêm nếu chunk vẫn còn quá dài
            final_chunks = self.fallback_splitter.split_documents(md_chunks)

            current_h1 = ""
            last_seen_pseudo_heading = ""

            for chunk in final_chunks:
                # Kế thừa metadata của file gốc (filename, file_type...)
                original_metadata = doc.metadata.copy()

                # Trích xuất Header từ bộ Markdown Splitter
                h1 = chunk.metadata.get("Header 1", "")
                h2 = chunk.metadata.get("Header 2", "")
                h3 = chunk.metadata.get("Header 3", "")

                headings = []
                if h1:
                    headings.append(h1)
                if h2:
                    headings.append(h2)
                if h3:
                    headings.append(h3)

                # Reset lại tiểu đề giả lập nếu đổi sang mục H1 mới (hoặc file mới)
                if h1 != current_h1:
                    current_h1 = h1
                    last_seen_pseudo_heading = ""

                # Nếu tài liệu chỉ có tiêu đề cấp 1 (thường là tên file) và không có
                # các tiêu đề con, ta sẽ tự động tạo tiêu đề con từ nội dung của chunk.
                # Điều này giúp các chunk trong cùng một file có heading khác nhau.
                is_flat_document = h1 and not h2 and not h3
                if is_flat_document:
                    lines = chunk.page_content.split("\n")
                    first_line = lines[0].strip()
                    clean_first_line = first_line.strip("*_ ")

                    # Mô phỏng kiểm tra độ nổi bật (thay thế cho font size) bằng các đặc trưng:
                    is_bold = first_line.startswith("**") and first_line.endswith("**")
                    is_uppercase = (
                        clean_first_line.isupper() and len(clean_first_line) > 3
                    )

                    # So sánh kích thước (độ dài) với dòng tiếp theo
                    second_line = lines[1].strip() if len(lines) > 1 else ""
                    is_shorter_than_next = (
                        len(second_line) > 20
                        and len(clean_first_line) < len(second_line) * 0.8
                    )

                    if (
                        clean_first_line
                        and len(clean_first_line) < 100
                        and not clean_first_line.endswith((".", "?", "!", ":", ","))
                        and clean_first_line != h1.strip("*_ ")
                        and (
                            is_bold
                            or is_uppercase
                            or is_shorter_than_next
                            or len(clean_first_line) < 40
                        )
                    ):
                        last_seen_pseudo_heading = clean_first_line

                    # Kế thừa tiểu đề giả lập cho tất cả các chunk thuộc phần này
                    if (
                        last_seen_pseudo_heading
                        and last_seen_pseudo_heading not in headings
                    ):
                        headings.append(last_seen_pseudo_heading)

                filename = original_metadata.get("filename", "unknown")

                # Tạo chuỗi đại diện từ headings
                heading_str = " > ".join(headings) if headings else "General"
                # Content preview 50 ký tự đầu tiên
                preview_text = chunk.page_content[:50].replace("\n", " ").strip()

                # ID dựa trên filename, index và nội dung
                raw_id = f"{filename}|{global_index}|{chunk.page_content[:120]}"
                chunk_id = str(uuid5(NAMESPACE_URL, raw_id))

                original_metadata.update(
                    {
                        "chunk_index": global_index,
                        "chunk_id": chunk_id,
                        "headings": headings,
                        "preview": f"{preview_text}...",
                    }
                )

                chunk.metadata = original_metadata
                all_chunks.append(chunk)
                global_index += 1

        return all_chunks
