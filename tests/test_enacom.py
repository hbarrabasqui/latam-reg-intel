"""Smoke tests para EnacomScraper usando fixtures locales."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scrapers.argentina.enacom import EnacomScraper

FIXTURES = Path(__file__).parent / "fixtures"


def _mock_response(html_path: Path) -> MagicMock:
    mock = MagicMock()
    mock.content = html_path.read_bytes()
    mock.raise_for_status = MagicMock()
    return mock


# ------------------------------------------------------------------ #
# fetch_index / _parse_results                                         #
# ------------------------------------------------------------------ #


def test_parse_results_count():
    scraper = EnacomScraper(topics=["homologacion"], max_pages_per_topic=1)
    html = FIXTURES / "enacom_busqueda.html"

    with patch.object(scraper, "_post", return_value=_mock_response(html)):
        entries = scraper.fetch_index()

    assert len(entries) == 2


def test_parse_first_entry_metadata():
    scraper = EnacomScraper(topics=["homologacion"], max_pages_per_topic=1)
    html = FIXTURES / "enacom_busqueda.html"

    with patch.object(scraper, "_post", return_value=_mock_response(html)):
        entries = scraper.fetch_index()

    first = entries[0]
    assert first["title"] == "Resolución 1234 ENACOM/24"
    assert first["number"] == "1234"
    assert first["doc_type"] == "resolucion"
    assert first["date_published"] == date(2024, 3, 15)
    assert first["date_bo"] == date(2024, 3, 18)
    assert "HOMOLOGACION" in first["temas"]
    assert first["pdf_urls"] == ["https://www.enacom.gob.ar/multimedia/normativas/2024/res1234.pdf"]
    assert first["item_id"] == "normativa_12345"


def test_parse_second_entry_no_pdf():
    scraper = EnacomScraper(topics=["homologacion"], max_pages_per_topic=1)
    html = FIXTURES / "enacom_busqueda.html"

    with patch.object(scraper, "_post", return_value=_mock_response(html)):
        entries = scraper.fetch_index()

    second = entries[1]
    assert second["doc_type"] == "norma_tecnica"
    assert second["pdf_urls"] == []
    assert second["description"] == "Especificaciones técnicas para equipos de comunicación inalámbrica."


def test_total_parsed():
    scraper = EnacomScraper.__new__(EnacomScraper)
    from bs4 import BeautifulSoup
    html = (FIXTURES / "enacom_busqueda.html").read_bytes()
    soup = BeautifulSoup(html, "lxml")
    assert scraper._parse_total(soup) == 450


# ------------------------------------------------------------------ #
# fetch_document                                                        #
# ------------------------------------------------------------------ #


def test_fetch_document_uses_description_when_no_pdf():
    scraper = EnacomScraper(topics=["homologacion"])
    entry = {
        "title": "Norma Técnica ENACOM-Q2-01.01",
        "number": "01",
        "doc_type": "norma_tecnica",
        "date_published": date(2023, 6, 20),
        "date_bo": date(2023, 6, 22),
        "description": "Especificaciones técnicas para equipos inalámbricos.",
        "pdf_urls": [],
        "item_id": "normativa_12346",
        "temas": ["NORMAS TECNICAS", "EQUIPOS"],
        "organismo_empresa": "ENACOM",
        "publicacion": "BO. 34800 (22/06/2023)",
        "search_topic": "homologacion",
    }

    doc = scraper.fetch_document(entry)

    assert doc.country == "ARG"
    assert doc.organism == "ENACOM"
    assert doc.language == "es"
    assert doc.raw_text == "Especificaciones técnicas para equipos inalámbricos."
    assert doc.hash == doc.compute_hash(doc.raw_text)
    assert doc.date_published == date(2023, 6, 22)


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize(
    "title, expected",
    [
        ("Resolución 281 ENACOM/26", "resolucion"),
        ("Disposición 100 ENACOM/25", "disposicion"),
        ("Norma Técnica ENACOM-Q2-01.01", "norma_tecnica"),
        ("Decreto 1234/22", "decreto"),
        ("Algo sin tipo conocido", "normativa"),
    ],
)
def test_infer_doc_type(title: str, expected: str):
    scraper = EnacomScraper.__new__(EnacomScraper)
    assert scraper._infer_doc_type(title) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Firma: 15/03/2024", date(2024, 3, 15)),
        ("23/04/2026", date(2026, 4, 23)),
        ("sin fecha", None),
    ],
)
def test_parse_date(text: str, expected):
    scraper = EnacomScraper.__new__(EnacomScraper)
    assert scraper._parse_date(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("BO. 35100 (18/03/2024)", date(2024, 3, 18)),
        ("sin publicacion", None),
    ],
)
def test_parse_bo_date(text: str, expected):
    scraper = EnacomScraper.__new__(EnacomScraper)
    assert scraper._parse_bo_date(text) == expected
