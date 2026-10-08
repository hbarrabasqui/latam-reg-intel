"""Migra el corpus histórico scrapeado por el proyecto standalone `infoleg_telecom`
hacia el pipeline de latam-reg-intel (SQLite + raw_texts).

Contexto: `infoleg_telecom` fue un scraper puntual (no un proyecto activo) que en
junio/2026 bajó 685 normas de InfoLeg buscando por palabra clave "telecomunicaciones",
cubriendo un rango histórico amplio (incluye normas de cuando el organismo se llamaba
CNC / AFTIC, antes de ser ENACOM). Ese trabajo no hay que repetirlo: latam-reg-intel ya
tiene su propio scraper de InfoLeg (scrapers/argentina/infoleg.py) para ingesta
incremental hacia adelante, pero el historial ya bajado se importa una sola vez con
este script.

Qué hace este script:
  1. Lee normas_enacom.xlsx (metadata) + los .txt de textos_normas/ (texto completo).
  2. Arma un RegulatoryDocument por fila y lo guarda en SQLite vía DocumentStore.
  3. NO genera embeddings ni toca ChromaDB — eso lo hace `scripts/reindex.py`,
     que además aplica el DomainClassifier para descartar ruido (se comprobó que
     el scraping por keyword trajo normas de otros organismos sin relación con
     telecom, ej. resoluciones de Agricultura).
  4. Deduplica por hash de contenido: si un documento ya existe (por ejemplo,
     porque el scraper propio de latam-reg-intel ya lo había traído), se saltea.

Uso:
    python -m scripts.migrate_infoleg_seed                  # migra todo
    python -m scripts.migrate_infoleg_seed --dry-run         # solo reporta, no escribe
    python -m scripts.migrate_infoleg_seed --infoleg-dir /ruta/a/infoleg_telecom

Después de correr esto:
    python -m scripts.reindex          # genera embeddings e indexa en ChromaDB
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_ES_MONTHS = {
    "ene": 1, "feb": 2, "mar": 3, "abr": 4, "may": 5, "jun": 6,
    "jul": 7, "ago": 8, "sep": 9, "oct": 10, "nov": 11, "dic": 12,
}

_TIPO_MAP = {
    "resolución": "resolucion",
    "resolucion": "resolucion",
    "disposición": "disposicion",
    "disposicion": "disposicion",
    "decreto": "decreto",
    "decisión": "decision",
    "decision": "decision",
    "ley": "ley",
}

# Mojibake típico de InfoLeg: comillas tipográficas mal decodificadas (cp1252 -> utf-8)
_MOJIBAKE_FIXES = {
    "\x93": "“",  # “
    "\x94": "”",  # ”
    "\x91": "‘",  # '
    "\x92": "’",  # '
}

_DEFAULT_INFOLEG_DIR = Path(__file__).resolve().parents[2] / "infoleg_telecom"


def _clean_text(text: str) -> str:
    for bad, good in _MOJIBAKE_FIXES.items():
        text = text.replace(bad, good)
    return text.strip()


def _parse_fecha(raw: str) -> tuple[date, bool]:
    """Parsea 'DD-mon-YYYY' (ej: '31-may-2021') a date.

    Devuelve (fecha, fecha_desconocida). Si el origen no tenía fecha
    (bastante común en el scrape de infoleg_telecom para algunas normas
    viejas), se marca fecha_desconocida=True en vez de fallar.
    """
    if not raw.strip():
        return date(1900, 1, 1), True
    try:
        day_s, mon_s, year_s = raw.strip().split("-")
        month = _ES_MONTHS[mon_s.lower()[:3]]
        return date(int(year_s), month, int(day_s)), False
    except Exception:
        logger.warning("No pude parsear fecha %r, uso fecha desconocida.", raw)
        return date(1900, 1, 1), True


def _normalize_tipo(tipo: str) -> str:
    key = (tipo or "").strip().lower()
    return _TIPO_MAP.get(key, key or "desconocido")


def _load_rows(xlsx_path: Path) -> list[dict[str, Any]]:
    import openpyxl

    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    headers = rows[0]
    out = []
    for r in rows[1:]:
        out.append(dict(zip(headers, r)))
    return out


def migrate(infoleg_dir: Path, dry_run: bool = False) -> None:
    from pipeline.chunker import chunk_document
    from storage.classifier import DomainClassifier
    from storage.models import RegulatoryDocument
    from storage.relational import DocumentStore

    xlsx_path = infoleg_dir / "normas_enacom.xlsx"
    textos_dir = infoleg_dir / "textos_normas"

    if not xlsx_path.exists():
        logger.error("No encontré %s", xlsx_path)
        sys.exit(1)

    rows = _load_rows(xlsx_path)
    logger.info("Filas leídas de %s: %d", xlsx_path.name, len(rows))

    store = None if dry_run else DocumentStore()
    classifier = DomainClassifier()

    now = datetime.now(timezone.utc)
    stats = {
        "total": len(rows),
        "sin_texto": 0,
        "nuevos": 0,
        "duplicados": 0,
        "preview_telecom": 0,
        "preview_descartado": 0,
    }

    raw_text_dir = infoleg_dir.parents[0] / "latam-reg-intel" / "data" / "raw_texts"
    # No siempre coincide el layout exacto; fallback al store si existe.
    if store is not None:
        raw_text_dir = store.raw_text_dir

    for row in rows:
        infoleg_id = str(row["ID InfoLeg"])
        txt_path = textos_dir / f"{infoleg_id}.txt"
        if not txt_path.exists():
            stats["sin_texto"] += 1
            continue

        # Fast-path de resumen: si ya escribimos el raw_text para este id en una
        # corrida anterior, no hace falta volver a tocar la DB (el mount de red
        # hace que cada conexión SQLite sea cara; esto evita reprocesar de cero).
        if store is not None and (raw_text_dir / f"INFOLEG-{infoleg_id}.txt").exists():
            stats["duplicados"] += 1
            continue

        raw_text = _clean_text(txt_path.read_text(encoding="utf-8", errors="replace"))
        title = _clean_text(str(row.get("Descripción") or ""))
        fecha_pub, fecha_desconocida = _parse_fecha(str(row.get("Fecha Pub.") or ""))

        doc = RegulatoryDocument(
            id=f"INFOLEG-{infoleg_id}",
            country="ARG",
            organism=str(row.get("Organismo") or "ENACOM"),
            doc_type=_normalize_tipo(str(row.get("Tipo") or "")),
            number=str(row.get("Número") or ""),
            title=title,
            date_published=fecha_pub,
            date_scraped=now,
            source_url=str(row.get("URL Norma") or ""),
            raw_text=raw_text,
            language="es",
            status=str(row.get("Estado") or "vigente").lower(),
            version=1,
            hash=RegulatoryDocument.compute_hash(raw_text),
            metadata={
                "seed_source": "infoleg_telecom",
                "seed_migrated_at": now.isoformat(),
                "original_scrape_date": "2026-06-17",
                "infoleg_id": infoleg_id,
                "url_texto": str(row.get("URL Texto") or ""),
                "palabra_clave_origen": str(row.get("Palabra clave origen") or ""),
                "modifica_a": row.get("Modifica a (IDs)"),
                "modificada_por": row.get("Modificada por (IDs)"),
                "fecha_desconocida": fecha_desconocida,
            },
            # BUG REAL encontrado y corregido el 5/8/2026 (ver
            # scripts/fix_seed_migration_review_status.py para la corrección
            # retroactiva sobre los datos ya migrados): acá antes decía
            # review_status="approved" fijo para TODA fila migrada, sin
            # importar el resultado de `classifier.classify(doc)` de abajo —
            # el comentario original decía "el filtro real ocurre en
            # reindex.py al momento de indexar", pero reindex.py solo aplica
            # ese filtro a los que YA están approved, no decide el
            # review_status en sí. Resultado real: 508 documentos de este
            # seed (normas laborales, de otros ministerios, etc. — muchos
            # traídos por InfoLeg como "norma relacionada" de una norma real
            # de ENACOM, sin ninguna relación temática) quedaron
            # "approved" sin haber sido revisados nunca, ni por el
            # clasificador de keywords ni por el juez Haiku de
            # semantic_cleanup.py (que solo revisa documentos que SÍ pasan
            # el filtro de keywords, buscando falsos positivos — nunca miró
            # estos, que directamente fallan el filtro). Se colaron al bot
            # (ej. Ley de Contrato de Trabajo respondiendo una pregunta de
            # derecho laboral, citada como si fuera "ENACOM 401266").
            #
            # Fix: acá honramos el resultado real del clasificador — sigue
            # siendo una heurística de keywords, no una revisión real, pero
            # al menos ya no se auto-aprueba nada a ciegas. Los que fallan el
            # filtro quedan 'pending_review' para pasar por el juez Haiku
            # (scripts/semantic_cleanup.py --review-pending) antes de poder
            # entrar al índice.
            review_status="approved" if classifier.classify(doc) else "pending_review",
        )

        # Preview de clasificación (ya no es solo preview: ver arriba, ahora
        # decide el review_status real de la fila)
        domain = classifier.classify(doc)
        if domain:
            stats["preview_telecom"] += 1
        else:
            stats["preview_descartado"] += 1

        if dry_run:
            continue

        is_new = store.save(doc)
        if is_new:
            stats["nuevos"] += 1
        else:
            stats["duplicados"] += 1

    logger.info("=" * 60)
    logger.info("RESUMEN MIGRACIÓN")
    logger.info("Total filas en xlsx:        %d", stats["total"])
    logger.info("Sin archivo de texto:       %d", stats["sin_texto"])
    if dry_run:
        logger.info("(dry-run: no se escribió nada en la base)")
    else:
        logger.info("Guardados como nuevos:      %d", stats["nuevos"])
        logger.info("Ya existían (duplicados):   %d", stats["duplicados"])
    logger.info("-" * 60)
    logger.info("Preview de clasificación de dominio (la aplica scripts/reindex.py):")
    logger.info("  Clasificarían como 'telecom':  %d", stats["preview_telecom"])
    logger.info("  Se descartarían (ruido):       %d", stats["preview_descartado"])
    logger.info("=" * 60)
    if not dry_run:
        logger.info("Siguiente paso: correr `python -m scripts.reindex` para generar")
        logger.info("los embeddings e indexar en ChromaDB.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Migra el corpus histórico de infoleg_telecom")
    parser.add_argument(
        "--infoleg-dir",
        type=Path,
        default=_DEFAULT_INFOLEG_DIR,
        help=f"Carpeta del proyecto infoleg_telecom (default: {_DEFAULT_INFOLEG_DIR})",
    )
    parser.add_argument("--dry-run", action="store_true", help="Solo reporta, no escribe nada")
    args = parser.parse_args()

    migrate(args.infoleg_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
