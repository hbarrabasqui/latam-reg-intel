"""Scraper para Anatel (Brasil) — Portal de Legislação.

Fuente: https://informacoes.anatel.gov.br/legislacao/
Tecnología: Joomla, HTML estático — no requiere Playwright.
Estrategia: iterar por tipo de documento y año, extraer links a páginas de detalle,
            descargar PDFs y extraer texto con pdfplumber.
"""
from __future__ import annotations

import io
import logging
import re
import uuid
from datetime import date, datetime, timezone
from typing import Any

import pdfplumber
from bs4 import BeautifulSoup

from scrapers.base_scraper import BaseScraper
from storage.models import RegulatoryDocument

logger = logging.getLogger(__name__)

_NUMBER_RE = re.compile(r"n[oº°]?\s*\.?\s*(\d+)", re.IGNORECASE)
_DATE_SLASH_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b")

# "14 Novembro 2024" o "14 de novembro de 2024"
_PT_MONTH: dict[str, int] = {
    "janeiro": 1, "fevereiro": 2, "março": 3, "marco": 3,
    "abril": 4, "maio": 5, "junho": 6, "julho": 7,
    "agosto": 8, "setembro": 9, "outubro": 10,
    "novembro": 11, "dezembro": 12,
}
_DATE_PT_RE = re.compile(
    r"(\d{1,2})\s+(?:de\s+)?("
    + "|".join(_PT_MONTH)
    + r")\s+(?:de\s+)?(\d{4})",
    re.IGNORECASE,
)

# Rutas del portal → tipo normalizado de documento
DOC_TYPE_PATHS: dict[str, str] = {
    "resolucoes": "resolucao",
    "portarias-normativas": "portaria_normativa",
}


class AnatelScraper(BaseScraper):
    country = "BRA"
    organism = "ANATEL"
    base_url = "https://informacoes.anatel.gov.br/legislacao"

    def __init__(
        self,
        years: list[int] | None = None,
        doc_type_paths: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        current_year = datetime.now().year
        self.years = years or list(range(2020, current_year + 1))
        self.doc_type_paths = doc_type_paths or DOC_TYPE_PATHS

    # ------------------------------------------------------------------ #
    # Índice                                                               #
    # ------------------------------------------------------------------ #

    def fetch_index(self) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for path, doc_type in self.doc_type_paths.items():
            for year in self.years:
                url = f"{self.base_url}/{path}/{year}"
                try:
                    page_entries = self._fetch_index_page(url, doc_type, year, path)
                    entries.extend(page_entries)
                    logger.info("%s/%d -> %d documentos", path, year, len(page_entries))
                except Exception as exc:
                    logger.warning("Error al obtener indice %s: %s", url, exc)
        return entries

    def _fetch_index_page(
        self, url: str, doc_type: str, year: int, path: str
    ) -> list[dict[str, Any]]:
        response = self._get(url)
        soup = BeautifulSoup(response.content, "lxml")
        entries: list[dict[str, Any]] = []
        seen: set[str] = set()

        # Solo aceptamos links que estén estrictamente bajo /legislacao/{path}/{year}/
        # para descartar links de navegación (home, idioma, otros años, etc.)
        required_prefix = f"/legislacao/{path}/{year}/"

        for tag in soup.select(f"a[href*='{required_prefix}']"):
            href: str = tag.get("href", "")
            title = tag.get_text(" ", strip=True)

            if not href or not title or href in seen:
                continue
            if any(skip in href for skip in ["#", "javascript:", "mailto:"]):
                continue

            seen.add(href)
            full_url = (
                href
                if href.startswith("http")
                else f"https://informacoes.anatel.gov.br{href}"
            )
            entries.append(
                {
                    "title": title,
                    "url": full_url,
                    "doc_type": doc_type,
                    "year": year,
                    "number": self._extract_number(title),
                }
            )

        return entries

    # ------------------------------------------------------------------ #
    # Documento individual                                                 #
    # ------------------------------------------------------------------ #

    def fetch_document(self, entry: dict[str, Any]) -> RegulatoryDocument:
        response = self._get(entry["url"])
        soup = BeautifulSoup(response.content, "lxml")

        pdf_url = self._find_pdf_url(soup)
        raw_text = ""
        source_url = entry["url"]

        if pdf_url:
            source_url = pdf_url
            try:
                raw_text = self._download_and_extract_pdf(pdf_url)
            except Exception as exc:
                logger.warning("No se pudo extraer PDF %s: %s", pdf_url, exc)

        if not raw_text:
            # Fallback: texto visible del contenido HTML principal
            container = soup.select_one(
                "div#jsn-mainbody, div.item-page, article.item, div.art-content"
            )
            if container:
                raw_text = container.get_text(" ", strip=True)

        if not raw_text:
            logger.warning("Sin texto extraído para %s", entry["url"])

        pub_date = self._extract_date(soup)
        number = entry.get("number") or self._extract_number(entry.get("title", ""))

        return RegulatoryDocument(
            id=str(uuid.uuid4()),
            country=self.country,
            organism=self.organism,
            doc_type=entry["doc_type"],
            number=number,
            title=entry["title"],
            date_published=pub_date,
            date_scraped=datetime.now(timezone.utc),
            source_url=source_url,
            raw_text=raw_text,
            language="pt",
            status="vigente",
            version=1,
            hash=RegulatoryDocument.compute_hash(raw_text),
            metadata={"index_url": entry["url"]},
        )

    # ------------------------------------------------------------------ #
    # Helpers privados                                                     #
    # ------------------------------------------------------------------ #

    def _find_pdf_url(self, soup: BeautifulSoup) -> str | None:
        for tag in soup.find_all("a", href=True):
            href: str = tag["href"]
            if href.lower().endswith(".pdf"):
                return (
                    href
                    if href.startswith("http")
                    else f"https://informacoes.anatel.gov.br{href}"
                )
        return None

    def _download_and_extract_pdf(self, pdf_url: str) -> str:
        response = self._get(pdf_url)
        with pdfplumber.open(io.BytesIO(response.content)) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        text = "\n".join(pages).strip()
        if not text:
            logger.warning("pdfplumber no extrajo texto de %s — puede ser PDF escaneado", pdf_url)
        return text

    def _extract_number(self, text: str) -> str:
        m = _NUMBER_RE.search(text)
        return m.group(1) if m else ""

    def _extract_date(self, soup: BeautifulSoup) -> date:
        # Anatel Joomla: "Publicado: Quinta, 14 Novembro 2024 07:18"
        pub_span = soup.select_one("span.documentPublished")
        if pub_span:
            parsed = self._parse_date(pub_span.get_text(" ", strip=True))
            if parsed:
                return parsed

        # Fallback: buscar en el título del documento (h1)
        h1 = soup.select_one("h1.documentFirstHeading, h1")
        if h1:
            parsed = self._parse_date(h1.get_text(" ", strip=True))
            if parsed:
                return parsed

        # Fallback genérico: primeros selectores Joomla estándar
        for selector in ["time[datetime]", "dd.create", "span.published"]:
            tag = soup.select_one(selector)
            if not tag:
                continue
            raw = tag.get("datetime") or tag.get_text(strip=True)
            parsed = self._parse_date(raw)
            if parsed:
                return parsed

        return date.today()

    def _parse_date(self, text: str) -> date | None:
        text = text.strip()
        # Normalizar ISO con hora: "2024-03-12T00:00:00" → "2024-03-12"
        if "T" in text:
            text = text.split("T")[0]

        # Formatos numéricos directos
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
            try:
                return datetime.strptime(text, fmt).date()
            except ValueError:
                continue

        # Fecha en portugués: "14 Novembro 2024" o "14 de novembro de 2024"
        m = _DATE_PT_RE.search(text)
        if m:
            month_num = _PT_MONTH.get(m.group(2).lower())
            if month_num:
                try:
                    return date(int(m.group(3)), month_num, int(m.group(1)))
                except ValueError:
                    pass

        # Último recurso: dd/mm/aaaa dentro de un string más largo
        m2 = _DATE_SLASH_RE.search(text)
        if m2:
            d, mo, y = m2.group(1), m2.group(2), m2.group(3)
            if len(y) == 2:
                y = "20" + y
            try:
                return date(int(y), int(mo), int(d))
            except ValueError:
                pass

        return None
