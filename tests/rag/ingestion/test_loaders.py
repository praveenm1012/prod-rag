"""Ingestion loader tests."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pypdf.errors import PdfReadError

from app.rag.ingestion import (
    DocumentLoadError,
    DocxLoader,
    MarkdownLoader,
    PDFLoader,
    TextLoader,
    UnsupportedDocumentError,
    get_loader,
    load_document,
)


class TestTextLoader:
    def test_supports_txt_extension(self) -> None:
        loader = TextLoader()
        assert loader.supports("notes.txt")
        assert not loader.supports("notes.pdf")

    def test_loads_text_file(self, sample_txt: Path) -> None:
        document = TextLoader().load(sample_txt)

        assert document.text == "Hello from a text file.\nSecond line."
        assert document.metadata["format"] == "text"
        assert document.metadata["filename"] == "notes.txt"
        assert document.metadata["extension"] == ".txt"
        assert document.metadata["size_bytes"] == sample_txt.stat().st_size

    def test_raises_when_file_missing(self, tmp_path: Path) -> None:
        with pytest.raises(DocumentLoadError, match="File not found"):
            TextLoader().load(tmp_path / "missing.txt")


class TestMarkdownLoader:
    def test_supports_markdown_extensions(self) -> None:
        loader = MarkdownLoader()
        assert loader.supports("readme.md")
        assert loader.supports("guide.markdown")
        assert not loader.supports("readme.txt")

    def test_loads_markdown_file(self, sample_markdown: Path) -> None:
        document = MarkdownLoader().load(sample_markdown)

        assert document.text == "# Title\n\nMarkdown **content**."
        assert document.metadata["format"] == "markdown"
        assert document.metadata["filename"] == "readme.md"


class TestDocxLoader:
    def test_supports_docx_extension(self) -> None:
        loader = DocxLoader()
        assert loader.supports("report.docx")
        assert not loader.supports("report.doc")

    def test_loads_docx_file(self, sample_docx: Path) -> None:
        document = DocxLoader().load(sample_docx)

        assert document.text == "Hello from DOCX.\nSecond paragraph."
        assert document.metadata["format"] == "docx"
        assert document.metadata["paragraph_count"] == 2


class TestPDFLoader:
    def test_supports_pdf_extension(self) -> None:
        loader = PDFLoader()
        assert loader.supports("paper.pdf")
        assert not loader.supports("paper.docx")

    @patch("app.rag.ingestion.pdf_loader.PdfReader")
    def test_loads_pdf_file(self, mock_pdf_reader: MagicMock, tmp_path: Path) -> None:
        pdf_path = tmp_path / "paper.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        first_page = MagicMock()
        first_page.extract_text.return_value = "Page one"
        second_page = MagicMock()
        second_page.extract_text.return_value = "Page two"
        mock_pdf_reader.return_value.pages = [first_page, second_page]

        document = PDFLoader().load(pdf_path)

        assert document.text == "Page one\nPage two"
        assert document.metadata["format"] == "pdf"
        assert document.metadata["page_count"] == 2
        mock_pdf_reader.assert_called_once_with(str(pdf_path.resolve()))

    @patch("app.rag.ingestion.pdf_loader.PdfReader")
    def test_raises_on_invalid_pdf(
        self,
        mock_pdf_reader: MagicMock,
        tmp_path: Path,
    ) -> None:
        pdf_path = tmp_path / "broken.pdf"
        pdf_path.write_bytes(b"not-a-pdf")
        mock_pdf_reader.side_effect = PdfReadError("invalid pdf")

        with pytest.raises(DocumentLoadError, match="Failed to read PDF"):
            PDFLoader().load(pdf_path)


class TestRegistry:
    def test_get_loader_for_each_supported_format(
        self,
        sample_txt: Path,
        sample_markdown: Path,
        sample_docx: Path,
    ) -> None:
        assert isinstance(get_loader(sample_txt), TextLoader)
        assert isinstance(get_loader(sample_markdown), MarkdownLoader)
        assert isinstance(get_loader(sample_docx), DocxLoader)
        assert isinstance(get_loader("file.pdf"), PDFLoader)

    def test_load_document_dispatches_to_loader(self, sample_txt: Path) -> None:
        document = load_document(sample_txt)
        assert "Hello from a text file." in document.text

    def test_raises_for_unsupported_extension(self, tmp_path: Path) -> None:
        path = tmp_path / "data.csv"
        path.write_text("a,b,c", encoding="utf-8")

        with pytest.raises(UnsupportedDocumentError, match="No loader available"):
            get_loader(path)
