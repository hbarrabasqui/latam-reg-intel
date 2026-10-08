"""Smoke tests para AnatelScraper usando fixtures locales."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scrapers.brasil.anatel import AnatelScraper

FIXTURES = Path(__file__).parent / "fixtures"


def _mock_response(html_path: Path) -> MagicMock:
    mock = MagicMock()
    mock.content = html_path.read_bytes()
    mock.raise_for_status = MagicMock()
    return mock


# ------------------------------------------------------------------ #
# fetch_index                                                          #
# ------------------------------------------------------------------ #


def test_fetch_index_parses_links():
    scraper = AnatelScraper(years=[2024], doc_type_paths={"resolucoes": "resolucao"})
    index_html = FIXTURES / "anatel_resolucoes_2024.html"

    with patch.object(scraper, "_get", return_value=_mock_response(index_html)):
        entries = scraper.fetch_index()

    # Debe encontrar 2 documentos y filtrar el link "Voltar ao índice"
    assert len(entries) == 2
    assert entries[0]["doc_type"] == "resolucao"
    assert entries[0]["year"] == 2024
    assert entries[0]["number"] == "748"
    assert "resolucao-748" in entries[0]["url"]


# ------------------------------------------------------------------ #
# fetch_document                                                        #
# ------------------------------------------------------------------ #


def test_fetch_document_extracts_metadata():
    scraper = AnatelScraper(years=[2024])
    detail_html = FIXTURES / "anatel_resolucao_748.html"

    entry = {
        "title": "Resolução nº 748, de 12 de março de 2024",
        "url": "https://informacoes.anatel.gov.br/legislacao/resolucoes/2024/resolucao-748",
        "doc_type": "resolucao",
        "year": 2024,
        "number": "748",
    }

    # Simula: primera llamada → detail HTML, segunda llamada → fallo de PDF
    detail_mock = _mock_response(detail_html)

    def side_effect(url: str, **kwargs):
        if url.endswith(".pdf"):
            raise Exception("PDF no disponible en fixture")
        return detail_mock

    with patch.object(scraper, "_get", side_effect=side_effect):
        doc = scraper.fetch_document(entry)

    assert doc.country == "BRA"
    assert doc.organism == "ANATEL"
    assert doc.doc_type == "resolucao"
    assert doc.number == "748"
    assert doc.language == "pt"
    assert doc.date_published == date(2024, 3, 12)
    assert len(doc.raw_text) > 0
    assert doc.hash == doc.compute_hash(doc.raw_text)


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Resolução nº 748", "748"),
        ("Resolução n.749 de", "749"),
        ("Portaria nº 1.234", "1"),   # solo captura primer grupo numérico
        ("Sin número", ""),
    ],
)
def test_extract_number(text: str, expected: str):
    scraper = AnatelScraper.__new__(AnatelScraper)
    assert scraper._extract_number(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("2024-03-12", date(2024, 3, 12)),
        ("12/03/2024", date(2024, 3, 12)),
        ("2024-03-12T00:00:00", date(2024, 3, 12)),
        ("no es fecha", None),
    ],
)
def test_parse_date(text: str, expected):
    scraper = AnatelScraper.__new__(AnatelScraper)
    assert scraper._parse_date(text) == expected
