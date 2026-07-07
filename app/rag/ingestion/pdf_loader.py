"""PDF document loader."""

from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.rag.ingestion.base import DocumentLoader
from app.rag.ingestion.exceptions import DocumentLoadError
from app.rag.ingestion.models import Document


class PDFLoader(DocumentLoader):
    """Load text content from PDF files."""

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        return (".pdf",)

    def load(self, source: str | Path) -> Document:
        path = self._resolve_path(source)
        try:
            reader = PdfReader(str(path))
            pages = [page.extract_text() or "" for page in reader.pages]
            page_texts = [page.strip() for page in pages]
            text = "\n".join(page for page in page_texts if page)
        except PdfReadError as exc:
            raise DocumentLoadError(f"Failed to read PDF: {path}") from exc
        except OSError as exc:
            raise DocumentLoadError(f"Failed to open PDF: {path}") from exc

        return Document(
            text=text,
            metadata=self._base_metadata(
                path,
                format="pdf",
                page_count=len(reader.pages),
                page_texts=page_texts,
            ),
        )
