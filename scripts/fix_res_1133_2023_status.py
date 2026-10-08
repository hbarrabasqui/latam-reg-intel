"""Verifica y corrige el estado de la Resolución ENACOM 1133/2023 (Reglamento
de Permisos de Uso Experimental y Temporario de Bandas de Frecuencias) en el
corpus.

Contexto (2/9/2026): al armar la nota interpretativa sobre la banda 5.850-
5.925 MHz y C-ITS/V2X, se encontró que el texto completo de esta resolución
(incluido el Anexo I con el reglamento entero) YA está en el repo, en
`data/raw_texts/INFOLEG-388492.txt` — 388492 es el ID real de InfoLeg para
esta norma. Esto casi seguro significa que el documento ya pasó por
`DocumentStore.save()` en algún momento (ese archivo lo escribe el propio
store como efecto secundario de guardar, no es un caché manual) — es decir,
lo más probable es que YA esté cargado en `data/latam_reg.db`, probablemente
vía `scrapers/argentina/infoleg.py` o la migración del seed histórico.

Lo que este script NO sabe de antemano (no se pudo confirmar sin poder correr
código en la sesión de Cowork donde se escribió esto): si su `review_status`
actual es `approved`, `pending_review` o `rejected`, y si su `doc_type` está
seteado como `resolucion`. Este script lo reporta y, si hace falta, lo deja
en `approved` con `doc_type="resolucion"` — es una norma oficial real, con
fuente verificable, no necesita pasar por revisión de dominio como una nota
interpretativa.

Uso:
    python -m scripts.fix_res_1133_2023_status --dry-run
    python -m scripts.fix_res_1133_2023_status
"""
from __future__ import annotations

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_DOC_ID = "INFOLEG-388492"
_SOURCE_URL = "https://www.argentina.gob.ar/normativa/nacional/resoluci%C3%B3n-1133-2023-388492/texto"


def run(dry_run: bool = False) -> None:
    from storage.relational import DocumentStore

    store = DocumentStore()
    doc = store.get_by_id(_DOC_ID)

    if doc is None:
        logger.error(
            "No encontré %s en la base. No lo voy a crear desde acá para no "
            "duplicar la Resolución 1133/2023 si ya existe con otro id — "
            "revisar manualmente con scripts/review_queue.py --list antes de "
            "decidir cómo cargarla.",
            _DOC_ID,
        )
        return

    logger.info("Encontrado %s:", _DOC_ID)
    logger.info("  organism=%s  number=%s  doc_type=%s", doc.organism, doc.number, doc.doc_type)
    logger.info("  review_status=%s  status=%s", doc.review_status, doc.status)
    logger.info("  title=%s", doc.title[:100])
    logger.info("  source_url=%s", doc.source_url or "(vacío)")
    logger.info("  len(raw_text)=%d caracteres", len(doc.raw_text))
    logger.info("  ¿incluye el Anexo I completo?: %s", "sí" if "ALCANCE" in doc.raw_text and "ANEXO" in doc.raw_text else "NO — solo tendría la resolución sintetizada, sin el reglamento")

    needs_fix = doc.review_status != "approved" or doc.doc_type != "resolucion" or not doc.source_url

    if not needs_fix:
        logger.info("Ya está approved, doc_type=resolucion y con source_url. Nada que hacer.")
        return

    logger.info("-" * 60)
    if doc.review_status != "approved":
        logger.info("%s review_status: %s -> approved", "[dry-run] " if dry_run else "", doc.review_status)
    if doc.doc_type != "resolucion":
        logger.info("%s doc_type: %s -> resolucion (no se aplica automático, ver nota abajo)", "[dry-run] " if dry_run else "", doc.doc_type)
    if not doc.source_url:
        logger.info("%s source_url vacío -> se completaría con %s (no se aplica automático, ver nota abajo)", "[dry-run] " if dry_run else "", _SOURCE_URL)

    if dry_run:
        logger.info("(dry-run: no se cambió nada. Repetir sin --dry-run para aplicar)")
        return

    if doc.review_status != "approved":
        store.set_review_status(_DOC_ID, "approved")
        logger.info("review_status actualizado a approved.")

    if doc.doc_type != "resolucion" or not doc.source_url:
        logger.info(
            "doc_type y source_url no tienen un método de update directo en "
            "DocumentStore — si hace falta corregirlos, releer el doc con "
            "get_by_id, modificar los campos y volver a pasarlo por save() "
            "(cuidado: save() solo pisa si el hash cambia, así que forzar el "
            "reguardado requiere tocar raw_text o aceptar que quede como está)."
        )

    logger.info("Siguiente paso: python -m scripts.reindex")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verifica/corrige el estado de la Res. ENACOM 1133/2023 en el corpus")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
