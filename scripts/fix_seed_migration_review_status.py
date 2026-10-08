"""Corrección retroactiva de un bug real en `scripts/migrate_infoleg_seed.py`
(encontrado el 5/8/2026): esa migración marcaba `review_status="approved"`
para TODA fila migrada del seed histórico de `infoleg_telecom`, sin importar
si pasaba o no el clasificador de keywords — nunca hubo una revisión real.

Consecuencia real detectada: documentos totalmente ajenos a telecom (Ley de
Contrato de Trabajo, la "Ley de Bases" 27742, resoluciones de AFIP/ARCA,
ministerios varios) entraron al índice del bot etiquetados como "ENACOM
NNNNNN", porque InfoLeg los trae como "norma relacionada" de una norma real
de ENACOM (mención en un VISTO, por ejemplo) sin que eso signifique que el
documento en sí sea de telecom.

Qué hace este script: para cada documento con `metadata.seed_source ==
"infoleg_telecom"` y `review_status == "approved"`, vuelve a evaluar
`DomainClassifier.classify()` (la misma heurística de keywords que ya se usa
en todo el resto del pipeline). Si falla el filtro, lo pasa a
`review_status = "pending_review"` — no a "rejected" directamente, porque el
clasificador de keywords también tiene falsos negativos (ver
`semantic_cleanup.py`) y merece la misma segunda oportunidad con el juez
Haiku que el resto del corpus, no un rechazo automático.

Documentos NO migrados por el seed (los que trajo el scraper propio de
latam-reg-intel) no se tocan — esos ya pasaron por el pipeline real de
curación en algún momento y su `review_status` actual sí refleja una
decisión real, no un bug de migración.

Uso:
    python -m scripts.fix_seed_migration_review_status --dry-run
    python -m scripts.fix_seed_migration_review_status

Después de correr esto sin --dry-run:
    python -m scripts.semantic_cleanup --review-pending
"""
from __future__ import annotations

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run(dry_run: bool = False) -> None:
    from storage.classifier import DomainClassifier
    from storage.relational import DocumentStore

    store = DocumentStore()
    classifier = DomainClassifier()

    docs = store.get_approved()
    seed_docs = [d for d in docs if (d.metadata or {}).get("seed_source") == "infoleg_telecom"]
    logger.info("Documentos 'approved' totales: %d", len(docs))
    logger.info("De ellos, migrados del seed infoleg_telecom: %d", len(seed_docs))

    to_flip = [d for d in seed_docs if not classifier.classify(d)]
    logger.info(
        "Fallan el clasificador de keywords (nunca fueron revisados de verdad): %d",
        len(to_flip),
    )

    for doc in to_flip:
        verbo = "PASARÍA A pending_review" if dry_run else "PASADO A pending_review"
        logger.info("%s: %s %s — %s", verbo, doc.organism, doc.number, doc.title[:80])
        if not dry_run:
            store.set_review_status(doc.id, "pending_review")

    logger.info("=" * 60)
    logger.info("RESUMEN")
    logger.info("Documentos corregidos: %d", len(to_flip))
    if dry_run:
        logger.info("(dry-run: no se cambió nada. Repetir sin --dry-run para aplicar)")
    else:
        logger.info("Siguiente paso: python -m scripts.semantic_cleanup --review-pending")
    logger.info("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Corrige review_status de la migración seed infoleg_telecom")
    parser.add_argument("--dry-run", action="store_true", help="Solo reporta, no escribe nada")
    args = parser.parse_args()
    run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
