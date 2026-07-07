"""Document loader registry and dispatch."""

from pathlib import Path

from app.rag.ingestion.base import DocumentLoader
from app.rag.ingestion.docx_loader import DocxLoader
from app.rag.ingestion.exceptions import UnsupportedDocumentError
from app.rag.ingestion.markdown_loader import MarkdownLoader
from app.rag.ingestion.models import Document
from app.rag.ingestion.pdf_loader import PDFLoader
from app.rag.ingestion.text_loader import TextLoader

DEFAULT_LOADERS: tuple[DocumentLoader, ...] = (
    PDFLoader(),
    DocxLoader(),
    MarkdownLoader(),
    TextLoader(),
)


def get_loader(
    source: str | Path,
    loaders: tuple[DocumentLoader, ...] = DEFAULT_LOADERS,
) -> DocumentLoader:
    """Return the first loader that supports the given source."""
    for loader in loaders:
        if loader.supports(source):
            return loader
    extension = Path(source).suffix.lower() or "<none>"
    raise UnsupportedDocumentError(f"No loader available for extension: {extension}")


def load_document(
    source: str | Path,
    loaders: tuple[DocumentLoader, ...] = DEFAULT_LOADERS,
) -> Document:
    """Load a document using the appropriate loader for its file type."""
    return get_loader(source, loaders).load(source)
