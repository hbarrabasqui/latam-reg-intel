"""Audita y limpia los documentos "Norma relacionada con X" — un título genérico
que dejó la migración de InfoLeg en 399 documentos del corpus (46 de ellos con
review_status="approved", o sea indexados y citables por el bot hoy).

Encontrado el 24/8/2026 probando `scripts/plot_normativa_graph.py`: algunos de
esos 46 son basura real (una línea suelta, sin fecha real), pero otros son
copias completas y duplicadas de normas importantes (ej. INFOLEG-319964 es en
realidad el texto completo de la Resolución ENACOM 793/2019, con un título que
no le sirve a nadie si el bot lo cita como fuente).

Clasificación (mirando el texto real, no el título):
  A) JUNK — texto muy corto (< 400 caracteres) sin contenido sustantivo real.
     Acción: review_status="rejected" (deja de indexarse/citarse). No se borra
     el documento, solo se excluye del RAG — reversible.
  B) DUPLICADO — texto sustancial, pero ya existe otro documento approved,
     mismo organismo y mismo número de norma extraído del propio texto, con
     un título real (no genérico). Acción: review_status="rejected" — se
     prefiere el documento con título real, se descarta el duplicado.
  C) RETITULAR — texto sustancial, no se encontró ningún duplicado con título
     real ya en el corpus. Acción: NO se rechaza (es contenido real y único),
     se le pone un título extraído del propio texto (regex sobre las primeras
     líneas) para que las citas del bot tengan sentido.
  D) SIN CLASIFICAR — no se pudo extraír un número de norma confiable ni
     decidir automáticamente. Queda listado aparte para revisión manual —
     NO se toca.

Uso (dos pasos, revisar el reporte antes de aplicar):
    python -m scripts.clean_norma_relacionada --dry-run
        → genera data/norma_relacionada_audit.md con la clasificación
          propuesta para los 46 documentos approved, SIN tocar la base.
    python -m scripts.clean_norma_relacionada --apply
        → aplica los casos A, B y C. Los casos D quedan sin tocar y se
          listan al final para decidir a mano.

Después de aplicar, correr `python -m scripts.reindex` (los rejected salen
del índice; los retitulados necesitan reindexarse para que el metadata
`title` del chunk en Chroma se actualice — el reindex compara por hash de
contenido, así que puede hacer falta `--reset` si el título no se refleja
solo con un reindex incremental; confirmar mirando el resultado).
"""
from __future__ import annotations

import argparse
import logging
import re
import sqlite3
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_DB_PATH = Path("data") / "latam_reg.db"
_REPORT_PATH = Path("data") / "norma_relacionada_audit.md"

_JUNK_MIN_CHARS = 400

# Extrae "Resolución <número>[/año]" en variantes comunes del texto scrapeado,
# sin importar si dice "Resolución", "RESOLUCION" o el patrón GEDO
# "RESOL-YYYY-NUM-APN-...".
_PATTERNS = [
    re.compile(r"RESOL-(\d{4})-(\d+)-APN", re.IGNORECASE),
    re.compile(r"Resoluci[oó]n\s+(?:ENACOM\s+)?(?:N[°º]?\s*)?(\d[\d\.]{0,6})\s*[/\-]\s*(\d{2,4})", re.IGNORECASE),
    re.compile(r"Resoluci[oó]n\s+(?:CNC|SC|ENACOM)?\s*(?:N[°º]?\s*)?(\d[\d\.]{0,6})\b", re.IGNORECASE),
]

# Patrón de "stub" genérico de InfoLeg: cabecera "ENACOM\nResolucion <su-propio-id>\n
# Fecha: 1900-01-01\n..." — la "fecha" es un relleno falso y el número que sigue a
# "Resolucion" es el ID interno del scrape, NO un número de norma real. Si el texto
# empieza así, no hay que confiar en la extracción de número — es indicio de stub.
_STUB_HEADER = re.compile(r"^ENACOM\s*\nResolucion\s+\d+\s*\nFecha:\s*1900-01-01", re.IGNORECASE)


def _extract_number_and_title(raw_text: str) -> tuple[str | None, str | None]:
    """Intenta sacar un número de norma y armar un título legible desde las
    primeras ~600 letras del texto. Devuelve (numero, titulo) o (None, None)
    si no encontró nada confiable."""
    head = raw_text[:600]

    m = _PATTERNS[0].search(head)  # RESOL-YYYY-NUM-APN
    if m:
        anio, numero = m.group(1), m.group(2)
        return f"{numero}/{anio}", f"Resolución ENACOM {numero}/{anio}"

    m = _PATTERNS[1].search(head)  # Resolución NNNN/YYYY
    if m:
        numero, anio = m.group(1), m.group(2)
        return f"{numero}/{anio}", f"Resolución {numero}/{anio}"

    m = _PATTERNS[2].search(head)  # Resolución NNNN (sin año)
    if m:
        numero = m.group(1)
        return numero, f"Resolución {numero}"

    return None, None


def _classify(cur: sqlite3.Cursor, doc_id: str, title: str, number: str, raw_text: str):
    stripped = raw_text.strip()

    if len(stripped) < _JUNK_MIN_CHARS:
        return ("JUNK", None, "texto muy corto, sin contenido sustantivo real")

    if _STUB_HEADER.match(stripped):
        return ("JUNK", None, "stub genérico de InfoLeg (cabecera 'Resolucion <id-propio> / Fecha: 1900-01-01') — no es contenido real")

    extracted_number, extracted_title = _extract_number_and_title(raw_text)
    if not extracted_number:
        return ("SIN_CLASIFICAR", None, "no se pudo extraer un número de norma confiable del texto")

    # ¿Ya existe otro documento approved, con título real (no genérico), mismo
    # número extraído?
    cur.execute(
        """
        SELECT id, title FROM documents
        WHERE review_status='approved'
          AND id != ?
          AND title NOT LIKE 'Norma relacionada con%'
          AND (number = ? OR title LIKE ?)
        """,
        (doc_id, extracted_number, f"%{extracted_number}%"),
    )
    dup = cur.fetchone()
    if dup:
        return ("DUPLICADO", dup, f"ya existe {dup[0]} ('{dup[1][:70]}') con el mismo número")

    return ("RETITULAR", extracted_title, f"contenido único, se propone título '{extracted_title}'")


def run(dry_run: bool) -> None:
    conn = sqlite3.connect(_DB_PATH)
    cur = conn.cursor()

    cur.execute(
        "SELECT id, title, number, raw_text FROM documents "
        "WHERE review_status='approved' AND title LIKE 'Norma relacionada con%' "
        "ORDER BY id"
    )
    rows = cur.fetchall()
    logger.info("Encontrados %d documentos approved con título genérico.", len(rows))

    results = []
    for doc_id, title, number, raw_text in rows:
        categoria, detalle, motivo = _classify(cur, doc_id, title, number, raw_text)
        results.append([doc_id, title, categoria, detalle, motivo, len(raw_text.strip())])

    # Chequeo cruzado: si dos candidatos a RETITULAR proponen el MISMO título
    # (probable bug de extracción o ambigüedad real — dos normas distintas no
    # deberían terminar con el mismo número), bajarlos a SIN_CLASIFICAR en vez
    # de aplicar un título posiblemente incorrecto a ciegas.
    from collections import Counter

    titulos_propuestos = Counter(
        r[3] for r in results if r[2] == "RETITULAR" and r[3]
    )
    for r in results:
        if r[2] == "RETITULAR" and titulos_propuestos[r[3]] > 1:
            r[2] = "SIN_CLASIFICAR"
            r[4] = f"CHOQUE: otro candidato también propuso el título '{r[3]}' — revisar a mano"
            r[3] = None

    counts = {}
    for _, _, categoria, _, _, _ in results:
        counts[categoria] = counts.get(categoria, 0) + 1

    lines = [
        "# Auditoría de documentos \"Norma relacionada con X\" (24/8/2026)",
        "",
        f"Total analizados: {len(results)}",
        "",
        "Resumen: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())),
        "",
    ]

    for categoria in ("JUNK", "DUPLICADO", "RETITULAR", "SIN_CLASIFICAR"):
        subset = [r for r in results if r[2] == categoria]
        if not subset:
            continue
        lines.append(f"## {categoria} ({len(subset)})")
        lines.append("")
        for doc_id, title, _, detalle, motivo, largo in subset:
            lines.append(f"- **{doc_id}** ({largo} caracteres) — {motivo}")
            if categoria == "DUPLICADO" and detalle:
                lines.append(f"  - Se propone: `review_status = rejected` (queda {detalle[0]} como canónico)")
            elif categoria == "RETITULAR" and detalle:
                lines.append(f"  - Título propuesto: \"{detalle}\"")
            elif categoria == "JUNK":
                lines.append("  - Se propone: `review_status = rejected`")
        lines.append("")

    _REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Reporte escrito en %s", _REPORT_PATH)
    logger.info("Resumen: %s", counts)

    if dry_run:
        logger.info("(dry-run: no se tocó la base. Revisá el reporte y corré con --apply)")
        return

    applied = {"JUNK": 0, "DUPLICADO": 0, "RETITULAR": 0}
    for doc_id, title, categoria, detalle, motivo, _ in results:
        if categoria in ("JUNK", "DUPLICADO"):
            cur.execute(
                "UPDATE documents SET review_status='rejected', "
                "metadata=json_set(coalesce(metadata,'{}'), '$.cleanup_note', ?) WHERE id=?",
                (f"rechazado 24/8/2026 por scripts/clean_norma_relacionada.py: {motivo}", doc_id),
            )
            applied[categoria] += 1
        elif categoria == "RETITULAR" and detalle:
            cur.execute(
                "UPDATE documents SET title=?, "
                "metadata=json_set(coalesce(metadata,'{}'), '$.cleanup_note', ?) WHERE id=?",
                (detalle, f"retitulado 24/8/2026 desde 'Norma relacionada con X' por scripts/clean_norma_relacionada.py", doc_id),
            )
            applied[categoria] += 1

    conn.commit()
    conn.close()
    logger.info("Aplicado: %s", applied)
    logger.info("Siguiente paso: python -m scripts.reindex --reset")


def main() -> None:
    parser = argparse.ArgumentParser(description="Limpieza de documentos 'Norma relacionada con X'")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true", help="solo genera el reporte, no toca la base")
    group.add_argument("--apply", action="store_true", help="aplica los cambios (JUNK/DUPLICADO/RETITULAR)")
    args = parser.parse_args()
    run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
