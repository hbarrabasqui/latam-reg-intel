"""Script principal para correr el pipeline piloto manualmente.

Uso básico:
    python -m scripts.run_pilot                              # todos los países
    python -m scripts.run_pilot --country ARG                # ENACOM portal normativas
    python -m scripts.run_pilot --country BRA                # Anatel
    python -m scripts.run_pilot --country INFOLEG            # InfoLeg (ENACOM vía repositorio legal)
    python -m scripts.run_pilot --country HOMOLOGACIONES     # Portal de homologaciones ENACOM
    python -m scripts.run_pilot --country ARG --limit 5      # prueba rápida
    python -m scripts.run_pilot --no-index                   # solo scraping, sin indexar en ChromaDB
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _run_scraper(
    scraper: Any,
    store: Any,
    limit: int | None,
    vector_store: Any | None,
    embedder: Any | None,
) -> dict[str, int]:
    """Corre un scraper completo, persiste y opcionalmente indexa en ChromaDB."""
    from pipeline.chunker import chunk_document

    stats = {"fetched": 0, "saved": 0, "skipped": 0, "errors": 0, "chunks": 0}

    logger.info("=== %s (%s) — obteniendo índice ===", scraper.organism, scraper.country)
    try:
        entries = scraper.fetch_index()
    except Exception as exc:
        logger.error("Error al obtener índice: %s", exc)
        return stats

    if limit:
        entries = entries[:limit]

    logger.info("Procesando %d documentos...", len(entries))

    for i, entry in enumerate(entries, 1):
        try:
            doc = scraper.fetch_document(entry)
            stats["fetched"] += 1

            saved = store.save(doc)
            if saved:
                stats["saved"] += 1
                logger.info("[%d/%d] Guardado: %s %s", i, len(entries), doc.organism, doc.number or doc.title[:60])

                # Clasificar dominio y indexar en ChromaDB solo si pasa el filtro
                if vector_store and embedder and doc.review_status == "approved" and doc.raw_text:
                    from storage.classifier import DomainClassifier
                    domain = DomainClassifier().classify(doc)
                    if domain:
                        chunks = chunk_document(doc)
                        if chunks:
                            chunk_vectors = embedder.embed_chunks(chunks)
                            vector_store.delete_by_doc_id(doc.id)
                            n = vector_store.index_chunks(chunk_vectors)
                            stats["chunks"] += n
                            logger.info("  → %d chunks indexados en ChromaDB (dominio=%s)", n, domain)
                    else:
                        logger.info("  → Descartado por clasificador (fuera del dominio telecom)")
            else:
                stats["skipped"] += 1
                logger.debug("[%d/%d] Sin cambios (hash igual), saltado.", i, len(entries))

        except Exception as exc:
            stats["errors"] += 1
            logger.warning("[%d/%d] Error procesando entrada: %s", i, len(entries), exc)

    return stats


def _print_summary(country: str, stats: dict[str, int]) -> None:
    logger.info(
        "--- %s: %d procesados | %d guardados | %d saltados | %d errores ---",
        country,
        stats["fetched"],
        stats["saved"],
        stats["skipped"],
        stats["errors"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline piloto LATAM Reg Intel")
    parser.add_argument(
        "--country",
        choices=["ARG", "BRA", "INFOLEG", "HOMOLOGACIONES"],
        help="Fuente a procesar: ARG=ENACOM normativas, BRA=Anatel, INFOLEG=InfoLeg, HOMOLOGACIONES=portal homologaciones ENACOM (omitir para correr todos)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Limitar a los primeros N documentos por scraper (útil para pruebas)",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=None,
        metavar="AÑO",
        help="Años a procesar para Anatel (ej: --years 2024 2025)",
    )
    parser.add_argument(
        "--no-index",
        action="store_true",
        help="Solo scraping y SQLite, sin indexar en ChromaDB",
    )
    args = parser.parse_args()

    from scrapers.argentina.enacom import EnacomScraper
    from scrapers.argentina.enacom_homologaciones import ENACOMHomologacionesScraper
    from scrapers.argentina.infoleg import InfolegScraper
    from scrapers.brasil.anatel import AnatelScraper
    from storage.relational import DocumentStore

    store = DocumentStore()
    logger.info("Base de datos: %s", store.db_path.resolve())
    logger.info("Documentos existentes: %d", store.count())

    vector_store = None
    embedder = None
    if not args.no_index:
        from pipeline.embedder import Embedder
        from storage.vector_store import VectorStore
        embedder = Embedder()
        vector_store = VectorStore(domain="telecom")

    run_arg = (args.country or "ARG,BRA,INFOLEG,HOMOLOGACIONES").split(",")
    overall: dict[str, int] = {"fetched": 0, "saved": 0, "skipped": 0, "errors": 0, "chunks": 0}

    if "ARG" in run_arg:
        scraper = EnacomScraper(verify_ssl=False)
        stats = _run_scraper(scraper, store, args.limit, vector_store, embedder)
        _print_summary("ARG/ENACOM", stats)
        for k in overall:
            overall[k] += stats[k]

    if "INFOLEG" in run_arg:
        scraper = InfolegScraper()
        stats = _run_scraper(scraper, store, args.limit, vector_store, embedder)
        _print_summary("ARG/INFOLEG", stats)
        for k in overall:
            overall[k] += stats[k]

    if "HOMOLOGACIONES" in run_arg:
        scraper = ENACOMHomologacionesScraper()
        stats = _run_scraper(scraper, store, args.limit, vector_store, embedder)
        _print_summary("ARG/HOMOLOGACIONES", stats)
        for k in overall:
            overall[k] += stats[k]

    if "BRA" in run_arg:
        kwargs: dict[str, Any] = {}
        if args.years:
            kwargs["years"] = args.years
        scraper = AnatelScraper(**kwargs)
        stats = _run_scraper(scraper, store, args.limit, vector_store, embedder)
        _print_summary("BRA/ANATEL", stats)
        for k in overall:
            overall[k] += stats[k]

    logger.info(
        "=== TOTAL: %d procesados | %d guardados | %d saltados | %d errores | %d chunks ===",
        overall["fetched"],
        overall["saved"],
        overall["skipped"],
        overall["errors"],
        overall["chunks"],
    )
    logger.info("Documentos en SQLite: %d", store.count())
    if vector_store:
        logger.info("Chunks en ChromaDB:  %d", vector_store.count())

    if overall["errors"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
