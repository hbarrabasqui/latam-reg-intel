"""Descarga y limpia los 2 documentos consolidados (texto refundido) de la
normativa Multibanda/SAE de Chile (homologación de celulares/equipos móviles).

Contexto: a diferencia de la carga anterior de Chile (Resolución 1.985/2017 +
737/2025, copiadas a mano de PDFs que Horacio compartió), estos 2 documentos
se bajan directo de fuentes oficiales:
  - Página de referencia: https://multibanda.cl/normativa/ (micrositio oficial
    de SUBTEL, confirmado: lo referencia subtel.gob.cl como su propio sitio de
    Multibanda/SAE, y el pie de página trae la dirección oficial de SUBTEL).
  - Los PDFs en sí son generados por la Biblioteca del Congreso Nacional
    (LeyChile) — el equivalente chileno a InfoLeg — y son el "texto
    refundido" (versión consolidada, con todas las modificaciones históricas
    ya incorporadas y anotadas al margen), no el texto original de 2016 sin
    actualizar.

Por qué solo 2 documentos y no las 10 resoluciones de la cadena completa
(1.463/2016 base + 1.474/2016 SAE + 8 modificatorias): el texto refundido de
cada una YA incorpora sus propias modificaciones históricas como anotaciones
inline (ej. "NOTA: El artículo único de la Resolución 634... modifica la
presente norma..."), así que cargar las 10 por separado sería redundante —
el estado vigente completo, con su historial de cambios, ya está en estos 2
documentos. Decisión tomada con Horacio el 13/8/2026.

Qué hace este script:
  1. Descarga los 2 PDFs desde multibanda.cl.
  2. Extrae el texto con pypdf.
  3. Limpia el header/footer repetido de la Biblioteca del Congreso que se
     repite en cada página ("Biblioteca del Congreso Nacional de Chile -
     www.leychile.cl - documento generado el ... página N de 40").
  4. Guarda el resultado en data/seed_chile/multibanda/*.txt — el siguiente
     paso (script de seed, todavía no escrito) carga esos .txt al corpus,
     igual que se hizo con seed_chile_subtel.py.

Uso:
    pip install pypdf   # si no está instalado
    python -m scripts.fetch_chile_multibanda
"""
from __future__ import annotations

import io
import logging
import re
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_OUT_DIR = Path("data") / "seed_chile" / "multibanda"

_DOCS = {
    "SUBTEL-1463-2016-refundido": (
        "https://multibanda.cl/wp-content/uploads/2018/10/"
        "Resolucion_1463_EXENTA_16_JUN_2016_03102023.pdf"
    ),
    "SUBTEL-1474-2016-SAE-refundido": (
        "https://multibanda.cl/wp-content/uploads/2018/10/"
        "RESOLUCION_1474_EXENTA_2016_06_22.pdf"
    ),
}

# NOTA (13/8/2026): el ruido repetido de paginación que `pypdf` extrae de
# estos PDFs de LeyChile NO es el que originalmente supuse acá ("Biblioteca
# del Congreso Nacional de Chile - www.leychile.cl...") — esa variante
# aparecía en la extracción que hizo `web_fetch` al leer el PDF por otra
# vía, no en la de `pypdf`. Con `pypdf` cada documento tiene su propio
# patrón de encabezado repetido por página, hay que armarlo por documento
# (ver `_HEADER_PATTERNS` abajo, uno por cada entrada de `_DOCS`). Esto se
# corrigió a mano después de que Code corriera la primera versión del
# script y se revisara el resultado — quedó documentado en ESTADO_SESION.md.
_HEADER_PATTERNS = {
    "SUBTEL-1463-2016-refundido": re.compile(
        r"Resolucion 1463 EXENTA, TRANSPORTES \(2016\)\s*\n\s*"
        r"\d{1,2}-[A-Za-z]{3}-\d{4}\s*\n\s*página \d+ de \d+\s*\n?"
    ),
    # Este documento no repite un bloque de metadata por página, solo la
    # fecha de generación sola como separador de página ("10-Jul-2017"). Se
    # limpia distinto: se deja la primera aparición (es parte del header
    # real con la metadata) y se sacan las siguientes.
    "SUBTEL-1474-2016-SAE-refundido": re.compile(r"^10-Jul-2017$", re.MULTILINE),
}


def clean_text(name: str, raw: str) -> str:
    pattern = _HEADER_PATTERNS.get(name)
    if pattern is None:
        logger.warning(
            "Sin patrón de limpieza conocido para %s — guardando el texto crudo, "
            "revisar a mano antes de usarlo.",
            name,
        )
        text = raw
    elif name == "SUBTEL-1474-2016-SAE-refundido":
        # Caso especial: sacar todas las repeticiones MENOS la primera.
        matches = list(pattern.finditer(raw))
        if matches:
            keep_end = matches[0].end()
            text = raw[:keep_end] + pattern.sub("", raw[keep_end:])
        else:
            text = raw
    else:
        text = pattern.sub("", raw)
    # Colapsar líneas en blanco repetidas que quedan al sacar el ruido.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fetch_and_clean(name: str, url: str) -> str:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(resp.content))
    pages_text = [page.extract_text() or "" for page in reader.pages]
    raw = "\n".join(pages_text)
    return clean_text(name, raw)


def main() -> None:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in _DOCS.items():
        logger.info("Descargando y limpiando %s ...", name)
        try:
            cleaned = fetch_and_clean(name, url)
        except Exception as exc:
            logger.error("Falló %s: %s", name, exc)
            continue
        out_path = _OUT_DIR / f"{name}.txt"
        out_path.write_text(cleaned, encoding="utf-8")
        logger.info("  Guardado: %s (%d caracteres)", out_path, len(cleaned))

    logger.info("=" * 60)
    logger.info("Listo. Revisar los .txt en %s antes de armar el script de seed.", _OUT_DIR)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
