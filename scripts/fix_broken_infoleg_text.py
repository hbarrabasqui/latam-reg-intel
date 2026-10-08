"""Backfill: corrige documentos cuyo raw_text es en realidad la página de error
de InfoLeg, no el texto de la norma.

Contexto (auditoría de corpus, 3/8/2026): se encontraron 102 documentos donde
`raw_text` es literalmente "InfoLeg - Información Legislativa No se pudo acceder
al archivo solicitado..." — un bug real en scrapers/argentina/infoleg.py, ya
corregido ahí para scrapes futuros (ver _looks_like_error_page en ese archivo).
Este script repara los que YA quedaron mal guardados en la base.

Causa raíz: no todas las normas de InfoLeg tienen un /anexos/{rango}/{id}/norma.htm
con texto completo escaneado — las "Resoluciones Sintetizadas" (solo resumen, sin
texto completo) no lo tienen, y ese endpoint responde 200 OK con una página HTML de
error en vez de un 404 real. El scraper guardaba ese texto de error como si fuera
el contenido de la norma.

Qué hace:
  1. Busca en la base todos los documentos cuyo raw_text tiene la firma de la
     página de error de InfoLeg.
  2. Para cada uno, reintenta el texto completo (por si esta vez sí existe) y si
     no, usa el resumen de verNorma.do como contenido mínimo real (misma lógica
     que el fix del scraper).
  3. Actualiza el documento en la base (mismo id, se reemplaza raw_text y hash).

Requiere acceso de red real a servicios.infoleg.gob.ar — no correr en el sandbox
de Cowork (bloqueado), correr localmente.

Uso:
    python -m scripts.fix_broken_infoleg_text                # aplica los cambios
    python -m scripts.fix_broken_infoleg_text --dry-run       # solo reporta
"""
from __future__ import annotations

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _find_broken_docs(store):
    from scrapers.argentina.infoleg import _looks_like_error_page
    docs = [d for d in store.get_approved()]
    # También revisar rechazados, por las dudas de que alguno haya quedado
    # marcado rejected por el juez semántico basándose en el texto de error.
    from storage.relational import _connect, _row_to_doc
    with _connect(store.db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM documents WHERE review_status = 'rejected'"
        ).fetchall()
    docs += [_row_to_doc(r) for r in rows]
    return [d for d in docs if _looks_like_error_page(d.raw_text)]


def _extract_infoleg_id(doc) -> int | None:
    raw = doc.metadata.get("infoleg_id") if doc.metadata else None
    if raw is not None:
        try:
            return int(raw)
        except (TypeError, ValueError):
            pass
    # Fallback: extraer del source_url
    import re
    m = re.search(r"[?&]id=(\d+)", doc.source_url or "")
    if m:
        return int(m.group(1))
    m = re.search(r"/anexos/\d+-\d+/(\d+)/norma\.htm", doc.source_url or "")
    if m:
        return int(m.group(1))
    return None


def run(dry_run: bool = False) -> None:
    from storage.models import RegulatoryDocument
    from storage.relational import DocumentStore
    from scrapers.argentina.infoleg import InfolegScraper, _build_fallback_text

    store = DocumentStore()
    scraper = InfolegScraper()

    broken = _find_broken_docs(store)
    logger.info("Documentos con texto de error encontrados: %d", len(broken))

    arreglados = sin_infoleg_id = errores = 0

    for i, doc in enumerate(broken, 1):
        infoleg_id = _extract_infoleg_id(doc)
        if infoleg_id is None:
            logger.warning("[%d/%d] %s %s — no se pudo determinar infoleg_id, salteado.",
                            i, len(broken), doc.organism, doc.number)
            sin_infoleg_id += 1
            continue

        try:
            # Reintentar texto completo primero (por si esta vez sí existe).
            full_text, full_url, _ = scraper._fetch_full_text(infoleg_id)
            norm_meta = scraper._fetch_norm_meta(infoleg_id)

            if full_text.strip():
                new_text = full_text
                fuente = "norma.htm (texto completo)"
            else:
                title = norm_meta.get("title") or doc.title
                new_text = _build_fallback_text(
                    doc.organism, doc.doc_type, doc.number, doc.date_published, title
                )
                fuente = "resumen de verNorma.do (sin texto completo disponible)"

            logger.info(
                "[%d/%d] %s %s → %d chars (%s)",
                i, len(broken), doc.organism, doc.number, len(new_text), fuente,
            )

            if not dry_run:
                doc.raw_text = new_text
                doc.hash = RegulatoryDocument.compute_hash(new_text)
                store.save(doc)
            arreglados += 1

        except Exception as exc:
            logger.warning("[%d/%d] Error con infoleg_id=%d: %s", i, len(broken), infoleg_id, exc)
            errores += 1

    logger.info("=" * 60)
    logger.info("RESUMEN BACKFILL TEXTO ROTO")
    logger.info("Documentos encontrados:     %d", len(broken))
    logger.info("Arreglados:                 %d", arreglados)
    logger.info("Sin infoleg_id (salteados): %d", sin_infoleg_id)
    logger.info("Errores:                    %d", errores)
    if dry_run:
        logger.info("(dry-run: no se guardó nada en la base)")
    logger.info("=" * 60)
    logger.info("Después de esto: correr `python -m scripts.reindex` para que los")
    logger.info("documentos que ahora sí tengan contenido relevante entren al índice.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Repara documentos con texto de error de InfoLeg")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
