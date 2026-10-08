"""Re-indexa en ChromaDB todos los documentos aprobados que ya están en SQLite.

Útil cuando hay documentos en SQLite que nunca se indexaron (ej: corridos con --no-index)
o cuando se quiere reconstruir el índice vectorial desde cero.

Uso:
    python -m scripts.reindex                  # todos los aprobados
    python -m scripts.reindex --country ARG    # solo un país
    python -m scripts.reindex --reset          # borra el índice y lo reconstruye
"""
from __future__ import annotations

import argparse
import logging
import sys

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-indexar documentos en ChromaDB")
    parser.add_argument("--country", metavar="ISO3", help="Filtrar por país")
    parser.add_argument("--reset", action="store_true", help="Borrar índice y reconstruir desde cero")
    args = parser.parse_args()

    from pipeline.chunker import chunk_document
    from pipeline.embedder import Embedder
    from storage.classifier import DomainClassifier
    from storage.relational import DocumentStore
    from storage.vector_store import VectorStore

    store = DocumentStore()
    embedder = Embedder()
    vector_store = VectorStore(domain="telecom")
    classifier = DomainClassifier()

    if args.reset:
        logger.warning("Borrando colección completa en ChromaDB (dominio='telecom')...")
        vector_store.reset()
        logger.info("Colección recreada vacía.")

    docs = store.get_approved()
    if args.country:
        docs = [d for d in docs if d.country == args.country]

    logger.info("%d documentos aprobados para indexar.", len(docs))

    indexed = skipped = errors = total_chunks = 0

    for i, doc in enumerate(docs, 1):
        if not doc.raw_text.strip():
            logger.warning("[%d/%d] %s %s sin texto, saltado.", i, len(docs), doc.organism, doc.number)
            skipped += 1
            continue

        # NOTA (5/8/2026): antes acá se usaba `classifier.classify(doc)` para
        # DESCARTAR documentos con puntaje insuficiente de palabras clave —
        # pero eso duplicaba el filtro de dominio que ya se aplicó al hacer
        # scraping/curación, y en la práctica pisaba las decisiones de
        # `semantic_cleanup.py` (el juez Haiku que rescata documentos que el
        # clasificador de palabras clave califica mal, ej: laboratorios
        # acreditados, inhibidores de señal). Resultado real detectado hoy:
        # la Resolución 25/2026 (ya aprobada y rescatada) volvía a quedar
        # afuera del índice vectorial en cada reindex --reset, porque el
        # clasificador de palabras clave la sigue puntuando bajo (telecom=3)
        # aunque Haiku y el propio Horacio confirmaron que es relevante.
        # `review_status == 'approved'` (ya filtrado en `store.get_approved()`)
        # es la autoridad real acá; el clasificador solo se usa para etiquetar
        # el dominio en los logs, nunca para excluir.
        domain = classifier.classify(doc) or "telecom"

        try:
            chunks = chunk_document(doc)
            if not chunks:
                skipped += 1
                continue
            chunk_vectors = embedder.embed_chunks(chunks)
            vector_store.delete_by_doc_id(doc.id)
            n = vector_store.index_chunks(chunk_vectors)
            total_chunks += n
            indexed += 1
            logger.info("[%d/%d] %s %s → %d chunks (dominio=%s)", i, len(docs), doc.organism, doc.number, n, domain)
        except Exception as exc:
            errors += 1
            logger.warning("[%d/%d] Error: %s", i, len(docs), exc)

    logger.info(
        "=== Listo: %d indexados | %d saltados | %d errores | %d chunks totales ===",
        indexed, skipped, errors, total_chunks,
    )
    logger.info("Chunks en ChromaDB: %d", vector_store.count())

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
