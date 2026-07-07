"""Document ingestion loaders and utilities."""

from app.rag.ingestion.base import DocumentLoader
from app.rag.ingestion.docx_loader import DocxLoader
from app.rag.ingestion.exceptions import (
    DocumentLoadError,
    IngestionError,
    UnsupportedDocumentError,
)
from app.rag.ingestion.markdown_loader import MarkdownLoader
from app.rag.ingestion.models import Document
from app.rag.ingestion.pdf_loader import PDFLoader
from app.rag.ingestion.registry import DEFAULT_LOADERS, get_loader, load_document
from app.rag.ingestion.text_loader import TextLoader

__all__ = [
    "DEFAULT_LOADERS",
    "DocxLoader",
    "Document",
    "DocumentLoadError",
    "DocumentLoader",
    "IngestionError",
    "MarkdownLoader",
    "PDFLoader",
    "TextLoader",
    "UnsupportedDocumentError",
    "get_loader",
    "load_document",
]
