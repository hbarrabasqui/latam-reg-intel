"""Scraper para ENACOM (Argentina) — Portal de Normativas.

Fuente: https://www.enacom.gob.ar/normativas
Tecnología: HTML dinámico (AJAX POST), no requiere Playwright.
Estrategia: POST al endpoint de búsqueda filtrando por temas relevantes,
            paginar de a 30 resultados, descargar PDFs con pdfplumber.

Temas por defecto (relevantes para certificación/homologación de equipos):
  homologacion, normas tecnicas, espectro radioelectrico, certificacion
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

_DATE_RE = re.compile(r"(\d{1,2})/(\d{1,2})/(\d{4})")
_BO_DATE_RE = re.compile(r"\((\d{1,2}/\d{1,2}/\d{4})\)")
_NUMBER_RE = re.compile(r"(\d+)")
_TOTAL_RE = re.compile(r"DE\s+([\d\.]+)", re.IGNORECASE)

# Temas relevantes para homologación y certificación de equipos de telecomunicaciones
DEFAULT_TOPICS: list[str] = [
    "homologacion",
    "normas tecnicas",
    "espectro radioelectrico",
    "certificacion equipos",
]

DOC_TYPE_MAP: dict[str, str] = {
    "resolución": "resolucion",
    "resolucion": "resolucion",
    "disposición": "disposicion",
    "disposicion": "disposicion",
    "decreto": "decreto",
    "ley": "ley",
    "norma técnica": "norma_tecnica",
    "norma tecnica": "norma_tecnica",
    "directiva general": "directiva",
}


class EnacomScraper(BaseScraper):
    country = "ARG"
    organism = "ENACOM"
    base_url = "https://www.enacom.gob.ar"
    search_url = "https://www.enacom.gob.ar/cit_normativas_lst.php"

    def __init__(
        self,
        topics: list[str] | None = None,
        max_pages_per_topic: int = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.topics = topics or DEFAULT_TOPICS
        self.max_pages_per_topic = max_pages_per_topic

    # ------------------------------------------------------------------ #
    # Índice                                                               #
    # ------------------------------------------------------------------ #

    def fetch_index(self) -> list[dict[str, Any]]:
        seen_ids: set[str] = set()
        entries: list[dict[str, Any]] = []

        for topic in self.topics:
            topic_entries = self._search_topic(topic, seen_ids)
            entries.extend(topic_entries)
            logger.info("Tema '%s' -> %d documentos nuevos", topic, len(topic_entries))

        return entries

    def _search_topic(
        self, topic: str, seen_ids: set[str]
    ) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []

        for page in range(1, self.max_pages_per_topic + 1):
            page_entries, total = self._fetch_search_page(topic, page)

            for entry in page_entries:
                item_id = entry.get("item_id", "")
                if item_id and item_id not in seen_ids:
                    seen_ids.add(item_id)
                    entries.append(entry)

            logger.debug("  Pagina %d/%d: %d items (total=%d)", page, self.max_pages_per_topic, len(page_entries), total)

            if not page_entries:
                break
            # Si ya trajimos todos los resultados disponibles
            if len(entries) >= total:
                break

        return entries

    def _fetch_search_page(
        self, topic: str, page: int
    ) -> tuple[list[dict[str, Any]], int]:
        payload = {
            "citnormativatipo": "",
            "citnormativanumero": "",
            "citnormativatemas": topic,
            "fechainferior": "",
            "fechasuperior": "",
            "pagina": str(page),
            "tipolistado": "0",
            "orden": "fecha_desc",
        }
        response = self._post(self.search_url, data=payload)
        soup = BeautifulSoup(response.content, "lxml")

        total = self._parse_total(soup)
        entries = self._parse_results(soup, topic)
        return entries, total

    def _parse_total(self, soup: BeautifulSoup) -> int:
        totales = soup.select_one(".totales")
        if not totales:
            return 0
        m = _TOTAL_RE.search(totales.get_text())
        if m:
            return int(m.group(1).replace(".", ""))
        return 0

    def _parse_results(
        self, soup: BeautifulSoup, topic: str
    ) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []

        for item in soup.select(".resultadoNorm"):
            try:
                entry = self._parse_item(item, topic)
                if entry:
                    entries.append(entry)
            except Exception as exc:
                logger.warning("Error parseando item: %s", exc)

        return entries

    def _parse_item(
        self, item: BeautifulSoup, topic: str
    ) -> dict[str, Any] | None:
        # Título y fecha de firma están en el header del accordion
        title_div = item.select_one(".normativa .col-sm-7.texto")
        date_div = item.select_one(".normativa .col-sm-5.texto")
        detail_div = item.select_one(".detallenormativa")

        if not title_div:
            return None

        raw_title = title_div.get_text(strip=True)
        raw_date = date_div.get_text(strip=True) if date_div else ""

        # ID único del item (ej: "normativa_95935")
        item_id = detail_div.get("id", "") if detail_div else ""

        # Campos del detalle (etiqueta → valor)
        fields = self._parse_detail_fields(detail_div) if detail_div else {}

        # URL del PDF (puede haber más de uno)
        pdf_urls = self._find_pdf_urls(detail_div) if detail_div else []

        # Descripción (primer .normdescription sin etiqueta strong)
        description = self._parse_description(detail_div) if detail_div else ""

        # Temas como lista
        temas_raw = fields.get("Temas", "")
        temas = [t.strip() for t in temas_raw.split(" - ") if t.strip()]

        return {
            "title": raw_title,
            "raw_date": raw_date,
            "date_published": self._parse_date(raw_date),
            "date_bo": self._parse_bo_date(fields.get("Publicación", "")),
            "description": description,
            "organismo_empresa": fields.get("Organismo / empresa", ""),
            "publicacion": fields.get("Publicación", ""),
            "temas": temas,
            "pdf_urls": pdf_urls,
            "item_id": item_id,
            "doc_type": self._infer_doc_type(raw_title),
            "number": self._extract_number(raw_title),
            "search_topic": topic,
        }

    def _parse_detail_fields(self, detail: BeautifulSoup) -> dict[str, str]:
        fields: dict[str, str] = {}
        # El sitio agrupa múltiples campos dentro de un mismo .normdescription.
        # Emparejamos cada col-sm-4 (label) con su col-sm-8 (valor) correspondiente.
        for block in detail.select(".normdescription"):
            labels = block.select(".col-sm-4 strong")
            values = block.select(".col-sm-8")
            for label_tag, value_tag in zip(labels, values):
                key = label_tag.get_text(strip=True).rstrip(":")
                val = value_tag.get_text(" ", strip=True)
                fields[key] = val
        return fields

    def _parse_description(self, detail: BeautifulSoup) -> str:
        for block in detail.select(".normdescription"):
            if not block.select_one("strong"):
                text = block.get_text(" ", strip=True)
                if text:
                    return text
        return ""

    def _find_pdf_urls(self, detail: BeautifulSoup) -> list[str]:
        urls = []
        for tag in detail.select("a[href*='/multimedia/']"):
            href = tag["href"]
            if not href.startswith("http"):
                href = f"{self.base_url}{href}"
            urls.append(href)
        return urls

    # ------------------------------------------------------------------ #
    # Documento individual                                                 #
    # ------------------------------------------------------------------ #

    def fetch_document(self, entry: dict[str, Any]) -> RegulatoryDocument:
        pdf_urls: list[str] = entry.get("pdf_urls", [])
        raw_text = ""
        source_url = ""

        for pdf_url in pdf_urls:
            try:
                raw_text = self._download_pdf(pdf_url)
                source_url = pdf_url
                if raw_text:
                    break
            except Exception as exc:
                logger.warning("No se pudo descargar PDF %s: %s", pdf_url, exc)

        if not raw_text:
            # Sin PDF: usar la descripción del índice como texto base
            raw_text = entry.get("description", "")
            source_url = f"{self.base_url}/normativas"

        pub_date = entry.get("date_bo") or entry.get("date_published") or date.today()

        return RegulatoryDocument(
            id=str(uuid.uuid4()),
            country=self.country,
            organism=self.organism,
            doc_type=entry["doc_type"],
            number=entry["number"],
            title=entry["title"],
            date_published=pub_date,
            date_scraped=datetime.now(timezone.utc),
            source_url=source_url,
            raw_text=raw_text,
            language="es",
            status="vigente",
            version=1,
            hash=RegulatoryDocument.compute_hash(raw_text),
            metadata={
                "item_id": entry.get("item_id", ""),
                "temas": entry.get("temas", []),
                "organismo_empresa": entry.get("organismo_empresa", ""),
                "publicacion": entry.get("publicacion", ""),
                "search_topic": entry.get("search_topic", ""),
            },
        )

    # ------------------------------------------------------------------ #
    # Helpers privados                                                     #
    # ------------------------------------------------------------------ #

    def _download_pdf(self, pdf_url: str) -> str:
        response = self._get(pdf_url)
        with pdfplumber.open(io.BytesIO(response.content)) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        text = "\n".join(pages).strip()
        if not text:
            logger.warning("PDF sin texto extraible (posiblemente escaneado): %s", pdf_url)
        return text

    def _parse_date(self, text: str) -> date | None:
        # "Firma: 23/04/2026" o directamente "23/04/2026"
        m = _DATE_RE.search(text)
        if m:
            try:
                return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
            except ValueError:
                pass
        return None

    def _parse_bo_date(self, publicacion: str) -> date | None:
        # "BO. 35890 (20/04/2026)"
        m = _BO_DATE_RE.search(publicacion)
        if m:
            return self._parse_date(m.group(1))
        return None

    def _extract_number(self, title: str) -> str:
        # "Resolución 281 ENACOM/26" → "281"
        m = _NUMBER_RE.search(title)
        return m.group(1) if m else ""

    def _infer_doc_type(self, title: str) -> str:
        lower = title.lower()
        for keyword, doc_type in DOC_TYPE_MAP.items():
            if keyword in lower:
                return doc_type
        return "normativa"
