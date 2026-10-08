"""Scraper para InfoLeg (Argentina) — Repositorio Legal Oficial.

Fuente: https://servicios.infoleg.gob.ar/infolegInternet
Tecnología: HTML estático, encoding iso-8859-1. No requiere Playwright.
Estrategia: búsqueda por palabras clave, paginación vía POST al form,
            texto completo vía /anexos/{rango}/{id}/norma.htm (HTML limpio,
            sin necesidad de OCR).

Complementa a enacom.py: InfoLeg tiene texto en HTML y permite navegar
vínculos entre normas (qué modifica / qué la modifica).
"""
from __future__ import annotations

import logging
import re
import uuid
from datetime import date, datetime, timezone
from typing import Any

from bs4 import BeautifulSoup, NavigableString

from scrapers.base_scraper import BaseScraper
from storage.models import RegulatoryDocument

logger = logging.getLogger(__name__)

_HOST = "https://servicios.infoleg.gob.ar"
_BASE = f"{_HOST}/infolegInternet"

# InfoLeg usa fechas en dos formatos:
#   DD/MM/YYYY  → dentro del texto de norma.htm
#   DD-mon-YYYY → en la página de resultados y verNorma.do
_DATE_SLASH_RE = re.compile(r"(\d{1,2})/(\d{1,2})/(\d{4})")
_DATE_DASH_RE = re.compile(r"(\d{1,2})-([a-z]{3})-(\d{4})", re.IGNORECASE)
_ES_MONTHS = {
    "ene": 1, "feb": 2, "mar": 3, "abr": 4, "may": 5, "jun": 6,
    "jul": 7, "ago": 8, "sep": 9, "oct": 10, "nov": 11, "dic": 12,
}

_NUMBER_YEAR_RE = re.compile(r"(\d+)\s*/\s*(\d{4})")
_ID_RE = re.compile(r"[?&]id=(\d+)")

# Bug real encontrado (3/8/2026, auditoría de corpus): no todas las normas de
# InfoLeg tienen un /anexos/{rango}/{id}/norma.htm escaneado — las "Resoluciones
# Sintetizadas" (sin texto completo, solo resumen) no lo tienen, y ese endpoint
# responde con una página de error HTML en vez de un 404 real. El código anterior
# no distinguía esto de una respuesta válida y guardaba el texto de la página de
# error como si fuera el contenido de la norma. Encontrados 102 documentos así en
# el corpus (todos con el mismo texto de 232 caracteres). Ver scripts/fix_broken_infoleg_text.py
# para el backfill de los ya afectados.
_ERROR_PAGE_SIGNALS = (
    "no se pudo acceder al archivo solicitado",
    "archivo no encontrado",
)


def _looks_like_error_page(text: str) -> bool:
    lower = text.lower()
    return any(sig in lower for sig in _ERROR_PAGE_SIGNALS)

_DOC_TYPE_MAP: dict[str, str] = {
    "resolución": "resolucion",
    "resolucion": "resolucion",
    "disposición": "disposicion",
    "disposicion": "disposicion",
    "decreto": "decreto",
    "ley": "ley",
    "nota": "nota",
    "providencia": "providencia",
}

# Nombres bajo los que aparece ENACOM en InfoLeg
_ENACOM_NAMES = {
    "ente nacional de comunicaciones",
    "enacom",
    "ministerio de comunicaciones",  # denominación anterior
}

DEFAULT_KEYWORDS: list[str] = [
    "telecomunicaciones",
    "RAMATEL",
    "homologación",
    "certificación",
    "ENACOM-Q2",
]

SEED_IDS: list[int] = [
    394789,  # Res 2097/2023
    312629,  # Res 4709/2018
    305478,  # Res E5762/2017
]

_PAGE_SIZE = 20  # resultados por página en InfoLeg


class InfolegScraper(BaseScraper):
    """Scraper para InfoLeg — repositorio legal del Estado argentino.

    Descarga normativa ENACOM con texto completo en HTML (sin OCR).
    """

    country = "ARG"
    organism = "ENACOM"
    base_url = _BASE

    def __init__(
        self,
        keywords: list[str] | None = None,
        seed_ids: list[int] | None = None,
        max_results_per_keyword: int = 200,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.keywords = keywords or DEFAULT_KEYWORDS
        self.seed_ids = seed_ids if seed_ids is not None else SEED_IDS
        self.max_results_per_keyword = max_results_per_keyword

    # ------------------------------------------------------------------
    # Índice
    # ------------------------------------------------------------------

    def fetch_index(self) -> list[dict[str, Any]]:
        seen_ids: set[int] = set()
        entries: list[dict[str, Any]] = []

        for doc_id in self.seed_ids:
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                entries.append({"infoleg_id": doc_id})
        logger.info("Seeds cargados: %d", len(self.seed_ids))

        for keyword in self.keywords:
            new_entries = self._search_keyword(keyword, seen_ids)
            entries.extend(new_entries)
            logger.info("Keyword '%s' → %d documentos nuevos", keyword, len(new_entries))

        logger.info("Total entradas en índice: %d", len(entries))
        return entries

    def _search_keyword(
        self, keyword: str, seen_ids: set[int]
    ) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []

        # Primera página: GET con los parámetros de búsqueda
        try:
            response = self._get(
                f"{_BASE}/buscarNormas.do",
                params={"texto": keyword, "organismo": "ENACOM"},
            )
            soup = BeautifulSoup(response.content, "lxml", from_encoding="iso-8859-1")
        except Exception as exc:
            logger.warning("Error buscando '%s': %s", keyword, exc)
            return entries

        page_entries = _parse_search_results(soup)
        _add_new(page_entries, entries, seen_ids)

        # URL del form para paginación (contiene jsessionid)
        form_action = _get_form_action(soup)

        # Páginas siguientes: POST al form con desplazamiento + irAPagina
        page_num = 2
        while (
            form_action
            and len(page_entries) >= _PAGE_SIZE
            and len(entries) < self.max_results_per_keyword
        ):
            try:
                response = self._post(
                    f"{_HOST}{form_action}",
                    data={"desplazamiento": "AP", "irAPagina": str(page_num)},
                )
                soup = BeautifulSoup(response.content, "lxml", from_encoding="iso-8859-1")
            except Exception as exc:
                logger.warning("Error paginando '%s' página %d: %s", keyword, page_num, exc)
                break

            page_entries = _parse_search_results(soup)
            if not page_entries:
                break
            _add_new(page_entries, entries, seen_ids)
            logger.debug("  página %d: %d resultados", page_num, len(page_entries))
            page_num += 1

        return entries

    # ------------------------------------------------------------------
    # Documento individual
    # ------------------------------------------------------------------

    def fetch_document(self, entry: dict[str, Any]) -> RegulatoryDocument:
        doc_id: int = entry["infoleg_id"]

        # Texto completo + metadatos extraídos de la cabecera del HTML
        raw_text, text_url, text_meta = self._fetch_full_text(doc_id)

        # verNorma.do como fuente de resumen y estado (sin full text)
        norm_meta = self._fetch_norm_meta(doc_id)

        # Prioridad: texto de norma.htm > búsqueda > verNorma.do > fallback
        doc_type = text_meta.get("doc_type") or entry.get("doc_type") or norm_meta.get("doc_type") or "normativa"
        number = text_meta.get("number") or entry.get("number") or norm_meta.get("number") or str(doc_id)
        title = norm_meta.get("title") or entry.get("title") or f"Norma InfoLeg {doc_id}"
        date_pub = text_meta.get("date_published") or entry.get("date_published") or norm_meta.get("date_published") or date.today()
        status = norm_meta.get("status", "vigente")

        infoleg_url = f"{_BASE}/verNorma.do?id={doc_id}"
        source_url = text_url or infoleg_url

        if not raw_text.strip():
            # Sin texto completo (norma.htm no existe para esta norma — ej. una
            # "Resolución Sintetizada"). Usar el resumen de verNorma.do como
            # contenido mínimo real, en vez de dejar el documento vacío.
            raw_text = _build_fallback_text(self.organism, doc_type, number, date_pub, title)

        # Vínculos normativos (metadata, no afecta al RAG)
        links = self._fetch_links(doc_id)

        return RegulatoryDocument(
            id=str(uuid.uuid4()),
            country=self.country,
            organism=self.organism,
            doc_type=doc_type,
            number=number,
            title=title,
            date_published=date_pub,
            date_scraped=datetime.now(timezone.utc),
            source_url=source_url,
            raw_text=raw_text,
            language="es",
            status=status,
            version=1,
            hash=RegulatoryDocument.compute_hash(raw_text),
            metadata={
                "infoleg_id": doc_id,
                "infoleg_url": infoleg_url,
                "modifica": links.get("modifica", []),
                "modificado_por": links.get("modificado_por", []),
            },
        )

    # ------------------------------------------------------------------
    # Fetchers internos
    # ------------------------------------------------------------------

    def _fetch_full_text(self, doc_id: int) -> tuple[str, str, dict[str, Any]]:
        """
        Descarga /anexos/{rango}/{id}/norma.htm y extrae texto + metadatos.

        El rango es floor(id/5000)*5000 → +4999.
        Ejemplo: id=394789 → /anexos/390000-394999/394789/norma.htm
        """
        rango_start = (doc_id // 5000) * 5000
        rango = f"{rango_start}-{rango_start + 4999}"
        url = f"{_BASE}/anexos/{rango}/{doc_id}/norma.htm"

        try:
            response = self._get(url)
            soup = BeautifulSoup(response.content, "lxml", from_encoding="iso-8859-1")
            for tag in soup(["script", "style", "nav", "header", "footer"]):
                tag.decompose()
            text = _clean_text(soup.get_text(separator="\n", strip=True))
            if _looks_like_error_page(text):
                # No es un 404 HTTP real (InfoLeg responde 200 con una página de
                # error) — pasa silenciosamente el try/except de abajo si no se
                # chequea esto. Típico de "Resoluciones Sintetizadas" sin texto
                # completo escaneado. fetch_document() cae al resumen de
                # verNorma.do como contenido mínimo en vez de guardar esto.
                logger.info(
                    "norma.htm id=%d es una página de error (norma sin texto completo "
                    "escaneado) — se usará el resumen de verNorma.do como fallback.",
                    doc_id,
                )
                return "", "", {}
            meta = _extract_meta_from_norma_text(text)
            logger.debug("norma.htm id=%d: %d chars", doc_id, len(text))
            return text, url, meta
        except Exception as exc:
            logger.warning("Sin norma.htm para id=%d: %s", doc_id, exc)
            return "", "", {}

    def _fetch_norm_meta(self, doc_id: int) -> dict[str, Any]:
        """Extrae título, estado y fecha de publicación en BO desde verNorma.do."""
        url = f"{_BASE}/verNorma.do?id={doc_id}"
        try:
            response = self._get(url)
            soup = BeautifulSoup(response.content, "lxml", from_encoding="iso-8859-1")
            return _parse_vernorma_page(soup)
        except Exception as exc:
            logger.debug("Sin metadatos verNorma.do id=%d: %s", doc_id, exc)
            return {}

    def _fetch_links(self, doc_id: int) -> dict[str, list[int]]:
        """
        modo=1 → normas que ESTE documento modifica/deroga
        modo=2 → normas que MODIFICAN a este documento
        """
        links: dict[str, list[int]] = {"modifica": [], "modificado_por": []}
        for modo, key in ((1, "modifica"), (2, "modificado_por")):
            url = f"{_BASE}/verVinculos.do?modo={modo}&id={doc_id}"
            try:
                response = self._get(url)
                soup = BeautifulSoup(response.content, "lxml", from_encoding="iso-8859-1")
                ids = [
                    int(m.group(1))
                    for a in soup.select("a[href*='verNorma.do']")
                    if (m := _ID_RE.search(a.get("href", "")))
                    and "resaltar" not in a.get("href", "")
                ]
                links[key] = ids
            except Exception as exc:
                logger.debug("Links modo=%d id=%d: %s", modo, doc_id, exc)
        return links


# ------------------------------------------------------------------
# Parsers de página
# ------------------------------------------------------------------

def _parse_search_results(soup: BeautifulSoup) -> list[dict[str, Any]]:
    """
    Parsea la tabla de resultados de buscarNormas.do.

    Estructura real de InfoLeg:
    - Cada resultado es una <tr> con 3 <td>: Número/Dependencia | Fecha | Descripción
    - Primera celda: <a href="verNorma.do;jsessionid=...?id=X"> Tipo Número/Año </a>
                     texto con el nombre del organismo (fuera del <a>)
    - Segunda celda: fecha "DD-mon-YYYY"
    - Tercera celda: descripción/título

    El organismo=ENACOM en la URL no filtra — se filtra en código.
    """
    entries = []
    seen_in_page: set[int] = set()

    for link in soup.select("a[href*='verNorma.do']"):
        href = link.get("href", "")
        if "resaltar" in href:
            continue

        m = _ID_RE.search(href)
        if not m:
            continue
        doc_id = int(m.group(1))
        if doc_id in seen_in_page:
            continue
        seen_in_page.add(doc_id)

        # Celda que contiene el link
        cell = link.find_parent("td")
        if not cell:
            entries.append({"infoleg_id": doc_id, "doc_type": "normativa",
                            "number": "", "date_published": None, "title": ""})
            continue

        # Texto del link: "Resolución GENERAL 5864 / 2026"
        link_text = link.get_text(" ", strip=True)
        doc_type, number = _parse_link_text(link_text)

        # Organismo: texto directo en la celda (no dentro de <a>)
        organism = _extract_direct_text(cell)

        # Solo incluir si es ENACOM
        if not _is_enacom(organism):
            logger.debug("Descartado (no ENACOM): %r — %r", doc_type, organism)
            continue

        # Celdas hermanas para fecha y descripción
        row = cell.find_parent("tr")
        cells = row.select("td") if row else []
        date_pub = _parse_date(cells[1].get_text(strip=True)) if len(cells) > 1 else None
        title = cells[2].get_text(" ", strip=True)[:200] if len(cells) > 2 else ""

        entries.append({
            "infoleg_id": doc_id,
            "doc_type": doc_type,
            "number": number,
            "date_published": date_pub,
            "title": title,
        })

    return entries


def _parse_vernorma_page(soup: BeautifulSoup) -> dict[str, Any]:
    """
    Extrae metadatos de verNorma.do?id=X.

    La página tiene un layout en tabla sin labels explícitos;
    el texto fluye como: Tipo / Número / Organismo / Fecha / Título / Resumen
    """
    for tag in soup(["script", "style"]):
        tag.decompose()
    lines = [
        l.strip()
        for l in soup.get_text(separator="\n").splitlines()
        if l.strip() and l.strip() not in ("InfoLeg - Información Legislativa",)
    ]

    meta: dict[str, Any] = {}

    # Buscar resumen (línea que sigue a "Resumen:")
    for i, line in enumerate(lines):
        if line.lower().startswith("resumen:"):
            # El resumen puede estar en la misma línea o la siguiente
            inline = line[8:].strip()
            meta["title"] = inline or (lines[i + 1] if i + 1 < len(lines) else "")
            break

    # Estado: buscar líneas que indiquen derogación
    full_text = "\n".join(lines).lower()
    meta["status"] = "derogado" if "derogad" in full_text else "vigente"

    # Fecha de publicación en BO (segunda fecha que aparece en la página)
    dates_found = []
    for line in lines:
        d = _parse_date(line)
        if d:
            dates_found.append(d)
    if len(dates_found) >= 2:
        meta["date_bo"] = dates_found[1]
    elif dates_found:
        meta["date_published"] = dates_found[0]

    return meta


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _get_form_action(soup: BeautifulSoup) -> str | None:
    """Extrae la action del form de paginación (incluye jsessionid)."""
    form = soup.select_one("form[action*='buscarNormas']")
    return form.get("action") if form else None


def _add_new(
    page_entries: list[dict[str, Any]],
    entries: list[dict[str, Any]],
    seen_ids: set[int],
) -> None:
    for entry in page_entries:
        doc_id = entry.get("infoleg_id")
        if doc_id and doc_id not in seen_ids:
            seen_ids.add(doc_id)
            entries.append(entry)


def _parse_link_text(text: str) -> tuple[str, str]:
    """
    Extrae doc_type y number del texto del link en la fila de resultados.
    Ej: "Resolución GENERAL 5864 / 2026" → ("resolucion", "5864/2026")
        "Disposición 157 / 2026"         → ("disposicion", "157/2026")
    """
    m = _NUMBER_YEAR_RE.search(text)
    if m:
        number = f"{m.group(1)}/{m.group(2)}"
        type_part = text[: text.index(m.group(1))].strip()
    else:
        number = ""
        type_part = text

    doc_type = _normalize_doc_type(type_part)
    return doc_type, number


def _extract_direct_text(tag: Any) -> str:
    """Texto directo de un tag (excluyendo el texto dentro de <a> hijos)."""
    parts = []
    for child in tag.children:
        if isinstance(child, NavigableString):
            t = str(child).strip()
            if t:
                parts.append(t)
    return " ".join(parts).strip()


def _extract_meta_from_norma_text(text: str) -> dict[str, Any]:
    """
    Extrae metadatos de las primeras líneas de norma.htm.

    Estructura típica:
      InfoLEG - Ministerio de Justicia y Derechos Humanos - Argentina
      ENTE NACIONAL DE COMUNICACIONES
      Resolución 2097/2023
      RESOL-2023-2097-APN-ENACOM#JGM
      Ciudad de Buenos Aires, 05/12/2023
    """
    meta: dict[str, Any] = {}
    lines = [l.strip() for l in text.splitlines() if l.strip()][:15]

    for line in lines:
        # Tipo y número: "Resolución 2097/2023"
        if not meta.get("doc_type"):
            for keyword, doc_type in _DOC_TYPE_MAP.items():
                if keyword in line.lower():
                    meta["doc_type"] = doc_type
                    m = re.search(r"(\d+/\d{4}|\d+)", line)
                    if m:
                        meta["number"] = m.group(1)
                    break

        # Fecha en el texto de la norma
        if not meta.get("date_published"):
            d = _parse_date(line)
            if d:
                meta["date_published"] = d

    return meta


def _build_fallback_text(
    organism: str, doc_type: str, number: str, date_pub: date, title: str
) -> str:
    """Texto mínimo para normas sin full text escaneado en InfoLeg (ej.
    Resoluciones Sintetizadas) — arma un texto real y útil a partir del resumen
    ya obtenido de verNorma.do, en vez de dejar el documento vacío."""
    parts = [
        organism,
        f"{doc_type.capitalize()} {number}".strip(),
        f"Fecha: {date_pub.isoformat()}" if date_pub else "",
        title,
    ]
    return "\n".join(p for p in parts if p)


def _is_enacom(organism_text: str) -> bool:
    lower = organism_text.lower()
    return any(name in lower for name in _ENACOM_NAMES)


def _parse_date(text: str) -> date | None:
    # Formato DD/MM/YYYY
    m = _DATE_SLASH_RE.search(text)
    if m:
        try:
            return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        except ValueError:
            pass

    # Formato DD-mon-YYYY (ej: "05-dic-2023")
    m = _DATE_DASH_RE.search(text)
    if m:
        month = _ES_MONTHS.get(m.group(2).lower())
        if month:
            try:
                return date(int(m.group(3)), month, int(m.group(1)))
            except ValueError:
                pass

    return None


def _normalize_doc_type(raw: str) -> str:
    lower = raw.lower().strip()
    for keyword, doc_type in _DOC_TYPE_MAP.items():
        if keyword in lower:
            return doc_type
    return "normativa"


def _clean_text(text: str) -> str:
    """Colapsa líneas en blanco múltiples a una sola."""
    lines = [l.strip() for l in text.splitlines()]
    result: list[str] = []
    prev_blank = False
    for line in lines:
        if not line:
            if not prev_blank:
                result.append("")
            prev_blank = True
        else:
            result.append(line)
            prev_blank = False
    return "\n".join(result).strip()
