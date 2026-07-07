"""Document loader interface."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from app.rag.ingestion.exceptions import DocumentLoadError
from app.rag.ingestion.models import Document


class DocumentLoader(ABC):
    """Abstract base class for format-specific document loaders."""

    @property
    @abstractmethod
    def supported_extensions(self) -> tuple[str, ...]:
        """Return file extensions supported by this loader (including dot)."""

    def supports(self, source: str | Path) -> bool:
        """Return True if this loader can handle the given source path."""
        return Path(source).suffix.lower() in self.supported_extensions

    @abstractmethod
    def load(self, source: str | Path) -> Document:
        """Load a document from the given file path."""

    def _resolve_path(self, source: str | Path) -> Path:
        """Resolve and validate the source path."""
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise DocumentLoadError(f"File not found: {path}")
        return path

    def _base_metadata(self, path: Path, **extra: Any) -> dict[str, Any]:
        """Build common metadata shared across all loaders."""
        stat = path.stat()
        metadata: dict[str, Any] = {
            "source": str(path),
            "filename": path.name,
            "extension": path.suffix.lower(),
            "size_bytes": stat.st_size,
        }
        metadata.update(extra)
        return metadata
