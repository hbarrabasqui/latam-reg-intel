"""Extracción de texto de PDFs y HTML."""
from __future__ import annotations

import io
import logging
import re

import pdfplumber
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extrae texto de un PDF combinando texto corrido y tablas serializadas.

    Las tablas en PDFs técnicos (normas, reglamentos) contienen datos críticos
    (límites de potencia, bandas de frecuencia, etc.) que no se recuperan bien
    con extracción de texto corrido porque pierden estructura y contexto semántico.

    Esta función detecta tablas por página con pdfplumber y las serializa a texto
    descriptivo antes de unirlas al texto corrido, mejorando la calidad del embedding.
    """
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            page_texts = []
            for page in pdf.pages:
                # 1. Texto corrido de la página
                plain = page.extract_text() or ""

                # 2. Tablas de la página serializadas a texto
                tables = page.extract_tables() or []
                table_texts = [_serialize_table(t) for t in tables if _table_has_data(t)]

                if table_texts:
                    page_texts.append(plain + "\n\n" + "\n\n".join(table_texts))
                else:
                    page_texts.append(plain)

        text = "\n".join(page_texts).strip()
        if not text:
            logger.warning("pdfplumber no encontró texto — el PDF puede ser escaneado (imagen)")
        return text
    except Exception as exc:
        logger.error("Error extrayendo texto de PDF: %s", exc)
        raise


def extract_html_text(html: str | bytes, selector: str = "body") -> str:
    """Extrae texto visible de un HTML usando un selector CSS."""
    soup = BeautifulSoup(html, "lxml")
    container = soup.select_one(selector) or soup
    return container.get_text(" ", strip=True)


# ─── HELPERS DE TABLAS ────────────────────────────────────────────────────────

def _table_has_data(table: list[list]) -> bool:
    """Devuelve True si la tabla tiene al menos una celda con contenido real."""
    for row in table:
        for cell in row:
            if cell and str(cell).strip():
                return True
    return False


def _serialize_table(table: list[list]) -> str:
    """Convierte una tabla de pdfplumber en texto descriptivo para embedding.

    Estrategia:
    - La primera fila con contenido se trata como encabezado.
    - Cada fila de datos se serializa como "col1: val1 | col2: val2 | ...".
    - Las celdas None o vacías se saltan para no añadir ruido.
    - Las tablas muy pequeñas (1 fila o 1 col) se serializan como lista simple.
    """
    if not table:
        return ""

    # Limpiar: reemplazar None por "" y normalizar espacios
    cleaned = []
    for row in table:
        cleaned.append([re.sub(r"\s+", " ", str(cell).strip()) if cell else "" for cell in row])

    # Separar encabezado del resto
    header_row, data_rows = _extract_header(cleaned)

    if not header_row:
        # Sin encabezado claro: serializar como lista de filas
        lines = []
        for row in cleaned:
            values = [c for c in row if c]
            if values:
                lines.append(" | ".join(values))
        return "\n".join(lines)

    # Con encabezado: serializar cada fila como "header: valor"
    lines = []
    for row in data_rows:
        pairs = []
        for header, value in zip(header_row, row):
            if value and header:
                pairs.append(f"{header}: {value}")
            elif value:
                pairs.append(value)
        if pairs:
            lines.append(" | ".join(pairs))

    return "\n".join(lines)


def _extract_header(table: list[list[str]]) -> tuple[list[str], list[list[str]]]:
    """Identifica la fila de encabezado.

    En PDFs con celdas fusionadas, el encabezado puede estar distribuido en
    varias filas con muchas celdas vacías. Buscamos la primera fila que tenga
    al menos la mitad de sus celdas con contenido.
    """
    n_cols = max(len(row) for row in table) if table else 0
    if n_cols == 0:
        return [], table

    for i, row in enumerate(table):
        filled = sum(1 for c in row if c)
        if filled >= max(2, n_cols // 2):
            return row, table[i + 1:]

    return [], table
