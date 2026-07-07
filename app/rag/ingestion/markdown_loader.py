"""Markdown document loader."""

from pathlib import Path

from app.rag.ingestion.base import DocumentLoader
from app.rag.ingestion.exceptions import DocumentLoadError
from app.rag.ingestion.models import Document


class MarkdownLoader(DocumentLoader):
    """Load text content from Markdown files."""

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        return (".md", ".markdown")

    def load(self, source: str | Path) -> Document:
        path = self._resolve_path(source)
        try:
            text = path.read_text(encoding="utf-8").strip()
        except UnicodeDecodeError as exc:
            message = f"Markdown file is not valid UTF-8: {path}"
            raise DocumentLoadError(message) from exc
        except OSError as exc:
            raise DocumentLoadError(f"Failed to read Markdown file: {path}") from exc

        return Document(
            text=text,
            metadata=self._base_metadata(path, format="markdown"),
        )
