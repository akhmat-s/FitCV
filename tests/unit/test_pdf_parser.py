"""Unit tests for in-memory CV parsing (helprers/pdf_parser.py::extract_text).

Asserts that PDF bytes are parsed to text with no disk I/O.
"""

from __future__ import annotations

from io import BytesIO

import docx  # python-docx
import fitz  # PyMuPDF

from extract import build_extract
from helprers.pdf_parser import extract_text
from schemas import ErrorResponse, ErrorStage


def _make_pdf_bytes(text: str) -> bytes:
    """Build a single-page PDF carrying ``text`` entirely in memory."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    data: bytes = doc.tobytes()
    doc.close()
    return data


def _make_docx_bytes(text: str) -> bytes:
    """Build a one-paragraph DOCX carrying ``text`` entirely in memory."""
    document = docx.Document()
    document.add_paragraph(text)
    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def test_extract_text_returns_pdf_text() -> None:
    pdf_bytes = _make_pdf_bytes("Jane Doe Senior Engineer")

    result = extract_text(pdf_bytes, "cv.pdf")

    assert isinstance(result, str)
    assert "Jane Doe" in result
    assert "Senior Engineer" in result


def test_extract_text_does_no_disk_io(monkeypatch) -> None:
    pdf_bytes = _make_pdf_bytes("No files written")

    # Any Python-level filesystem access — reading a path or writing output — during an
    # in-memory parse is a regression. Patch every Python file door (not just
    # builtins.open) so a path-based read or an output write is caught, not just the
    # one access the test name implies.
    def _fail(*args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        raise AssertionError("extract_text must not touch the filesystem")

    monkeypatch.setattr("builtins.open", _fail)
    monkeypatch.setattr("pathlib.Path.open", _fail)
    monkeypatch.setattr("pathlib.Path.write_text", _fail)
    monkeypatch.setattr("pathlib.Path.write_bytes", _fail)

    result = extract_text(pdf_bytes, "cv.pdf")

    assert "No files written" in result


def test_extract_text_returns_docx_text() -> None:
    docx_bytes = _make_docx_bytes("Ada Lovelace Analyst")

    result = extract_text(docx_bytes, "cv.docx")

    assert isinstance(result, str)
    assert "Ada Lovelace" in result
    assert "Analyst" in result


def test_extract_text_returns_txt_text() -> None:
    txt_bytes = b"Grace Hopper Compiler Pioneer"

    result = extract_text(txt_bytes, "cv.txt")

    assert isinstance(result, str)
    assert "Grace Hopper" in result
    assert "Compiler Pioneer" in result


async def test_unsupported_format_returns_parse_error() -> None:
    rtf_bytes = b"{\\rtf1 not a supported format}"

    result = await build_extract(rtf_bytes, "cv.rtf", "Senior Python engineer")

    assert isinstance(result, ErrorResponse)
    assert result.stage is ErrorStage.PARSE
    assert result.error
