"""Scraper para el portal de Homologaciones de ENACOM.

Fuentes que cubre:
  - /normas-tecnicas_p474          → normas técnicas vigentes + notas de aplicación (PDFs)
  - /novedades_p1673               → novedades recientes: nuevas normas y resoluciones (PDFs)
  - /preguntas-frecuentes          → resoluciones base RAMATEL + texto FAQ (HTML + PDFs)
  - /identificacion-reglamentaria  → reglamento de marcado de equipos (PDFs)
  - /laboratorios-acreditados_p349 → datos de laboratorios acreditados (HTML estático)

Estrategia:
  1. Recorre cada sección y recolecta URLs únicas de PDF + páginas HTML de texto.
  2. Para PDFs: extrae texto con pdfplumber, crea RegulatoryDocument.
  3. Para páginas HTML informativas: extrae texto limpio, crea RegulatoryDocument.
  4. Deduplica por source_url antes de descargar.
"""
from __future__ import annotations

import hashlib
import io
import logging
import re
import uuid
from datetime import date, datetime, timezone
from typing import Any
from urllib.parse import urljoin

import pdfplumber
from bs4 import BeautifulSoup

from scrapers.base_scraper import BaseScraper
from storage.models import RegulatoryDocument

logger = logging.getLogger(__name__)

BASE_URL = "https://www.enacom.gob.ar"

# Secciones a scrapear: (nombre_seccion, url, tipo_doc_default)
SECTIONS: list[tuple[str, str, str]] = [
    ("normas_tecnicas",          f"{BASE_URL}/normas-tecnicas_p474",                              "norma_tecnica"),
    ("novedades",                f"{BASE_URL}/novedades_p1673",                                    "norma_tecnica"),
    ("preguntas_frecuentes",     f"{BASE_URL}/homologacion-de-equipos_p347/preguntas-frecuentes",  "documento_informativo"),
    ("identificacion_reglam",    f"{BASE_URL}/identificacion-reglamentaria_p3420",                 "reglamento"),
    ("laboratorios_acreditados", f"{BASE_URL}/laboratorios-acreditados_p349",                      "documento_informativo"),
]

# Páginas HTML que se indexan como texto (sin PDF)
HTML_TEXT_SECTIONS: set[str] = {"laboratorios_acreditados", "preguntas_frecuentes"}

_NORM_NUMBER_RE = re.compile(
    r"(ENACOM|CNC|SC|CNT)-[A-Z0-9]+-[\d.]+(?:\s+V[\d.]+)?", re.IGNORECASE
)
_DATE_FROM_PATH_RE = re.compile(r"/(\d{4})(\d{2})/")


def _resolve_url(href: str, page_url: str) -> str:
    """Resuelve href relativo (incluyendo '../...') a URL absoluta."""
    return urljoin(page_url, href)


def _extract_norm_number(text: str) -> str:
    """Intenta extraer número de norma del texto (ej: ENACOM-Q2-60.14 V21.1)."""
    m = _NORM_NUMBER_RE.search(text)
    return m.group(0).strip() if m else ""


def _date_from_url(url: str) -> date:
    """Extrae fecha aproximada de la URL del archivo (ej: /202308/ → 2023-08-01)."""
    m = _DATE_FROM_PATH_RE.search(url)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), 1)
        except ValueError:
            pass
    return date(2000, 1, 1)


def _clean_html_text(soup: BeautifulSoup) -> str:
    """Extrae texto limpio de una página HTML eliminando nav/footer/scripts."""
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    # Conservar solo el contenido principal
    main = (
        soup.find("div", class_="contenido")
        or soup.find("main")
        or soup.find("article")
        or soup.find("div", id="content")
        or soup.find("body")
    )
    return (main or soup).get_text(separator="\n", strip=True)


class ENACOMHomologacionesScraper(BaseScraper):
    """Scraper del portal de Homologaciones de ENACOM."""

    country = "ARG"
    organism = "ENACOM"
    base_url = BASE_URL

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(verify_ssl=False, delay_seconds=2.0, **kwargs)

    # ─── INDEX ────────────────────────────────────────────────────────────────

    def fetch_index(self) -> list[dict[str, Any]]:
        """
        Recorre todas las secciones y devuelve una lista de entradas únicas
        (por source_url) a descargar.
        """
        entries: dict[str, dict[str, Any]] = {}  # url → entry

        for section_id, section_url, doc_type in SECTIONS:
            logger.info("Escaneando sección: %s (%s)", section_id, section_url)

            if section_id in HTML_TEXT_SECTIONS:
                # Indexar la página HTML completa como documento de texto
                key = section_url
                if key not in entries:
                    entries[key] = {
                        "type": "html",
                        "section": section_id,
                        "url": section_url,
                        "doc_type": doc_type,
                        "link_text": section_id.replace("_", " ").title(),
                    }

            try:
                resp = self._get(section_url)
            except Exception as exc:
                logger.warning("Error al acceder a %s: %s", section_url, exc)
                continue

            soup = BeautifulSoup(resp.content, "html.parser")

            for a in soup.find_all("a", href=True):
                href: str = a["href"]
                if ".pdf" not in href.lower():
                    continue
                abs_url = _resolve_url(href, section_url)
                if abs_url in entries:
                    continue  # ya registrado desde otra sección
                link_text = a.get_text(strip=True)
                entries[abs_url] = {
                    "type": "pdf",
                    "section": section_id,
                    "url": abs_url,
                    "doc_type": doc_type,
                    "link_text": link_text,
                }

        logger.info("Total entradas únicas a procesar: %d", len(entries))
        return list(entries.values())

    # ─── DOCUMENT ─────────────────────────────────────────────────────────────

    def fetch_document(self, entry: dict[str, Any]) -> RegulatoryDocument:
        if entry["type"] == "html":
            return self._fetch_html_doc(entry)
        return self._fetch_pdf_doc(entry)

    def _fetch_pdf_doc(self, entry: dict[str, Any]) -> RegulatoryDocument:
        url = entry["url"]
        link_text = entry["link_text"]

        resp = self._get(url)
        raw_text = self._extract_pdf_text(resp.content)

        norm_number = _extract_norm_number(link_text) or _extract_norm_number(raw_text[:800])
        doc_type = self._infer_doc_type(link_text, raw_text, entry["doc_type"])
        pub_date = _date_from_url(url)
        title = self._build_title(link_text, norm_number, doc_type, raw_text)

        return RegulatoryDocument(
            id=str(uuid.uuid4()),
            country=self.country,
            organism=self.organism,
            doc_type=doc_type,
            number=norm_number or self._slug_from_url(url),
            title=title,
            date_published=pub_date,
            date_scraped=datetime.now(timezone.utc),
            source_url=url,
            raw_text=raw_text,
            language="es",
            status="vigente",
            version=1,
            hash=hashlib.sha256(raw_text.encode()).hexdigest(),
            metadata={
                "section": entry["section"],
                "link_text": link_text,
                "fuente": "enacom_homologaciones",
            },
        )

    def _fetch_html_doc(self, entry: dict[str, Any]) -> RegulatoryDocument:
        url = entry["url"]
        resp = self._get(url)
        soup = BeautifulSoup(resp.content, "html.parser")
        raw_text = _clean_html_text(soup)
        section_id = entry["section"]
        titles = {
            "laboratorios_acreditados": "Laboratorios Acreditados ENACOM - Información completa",
            "preguntas_frecuentes": "ENACOM Homologaciones - Preguntas Frecuentes y guía del proceso",
        }
        title = titles.get(section_id, section_id.replace("_", " ").title())

        return RegulatoryDocument(
            id=str(uuid.uuid4()),
            country=self.country,
            organism=self.organism,
            doc_type=entry["doc_type"],
            number=section_id,
            title=title,
            date_published=date.today(),
            date_scraped=datetime.now(timezone.utc),
            source_url=url,
            raw_text=raw_text,
            language="es",
            status="vigente",
            version=1,
            hash=hashlib.sha256(raw_text.encode()).hexdigest(),
            metadata={"section": section_id, "fuente": "enacom_homologaciones"},
        )

    # ─── HELPERS ──────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_pdf_text(content: bytes) -> str:
        try:
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                return "\n".join(p.extract_text() or "" for p in pdf.pages)
        except Exception as exc:
            logger.warning("Error extrayendo texto del PDF: %s", exc)
            return ""

    @staticmethod
    def _infer_doc_type(link_text: str, raw_text: str, default: str) -> str:
        combined = (link_text + " " + raw_text[:300]).lower()
        if "nota de aplicación" in combined or "nota de aplicacion" in combined:
            return "nota_aplicacion"
        if "nota" in combined and re.search(r"nota\s+\d+/\d{4}", combined):
            return "nota_aplicacion"
        if "resolución" in combined or "resolucion" in combined or "resol-" in combined:
            return "resolucion"
        if "protocolo" in combined:
            return "protocolo_medicion"
        return default

    @staticmethod
    def _build_title(link_text: str, norm_number: str, doc_type: str, raw_text: str = "") -> str:
        if link_text and len(link_text) > 5 and not link_text.startswith("archivo_"):
            return link_text[:200]
        if norm_number:
            return f"Norma Técnica {norm_number}"
        # Intentar extraer título de la primera línea del PDF
        first_lines = [l.strip() for l in raw_text[:400].split("\n") if len(l.strip()) > 10]
        if first_lines:
            return first_lines[0][:200]
        return f"{doc_type.replace('_', ' ').title()} ENACOM"

    @staticmethod
    def _slug_from_url(url: str) -> str:
        """Genera un identificador legible desde la URL del archivo."""
        filename = url.rstrip("/").split("/")[-1]
        return re.sub(r"[^a-zA-Z0-9_.-]", "_", filename)[:50]
