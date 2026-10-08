"""Diagnóstico puntual de retrieval: para una pregunta dada, muestra en qué
puesto (si es que aparece) queda cada buscador (BM25, denso) para un doc_id
esperado, y los scores/tokens involucrados.

No es una herramienta de uso normal del bot — es para diagnosticar por qué
una pregunta puntual no recupera el documento correcto, cuando no alcanza con
mirar el eval_report.md (que solo dice si acertó o no, no por qué).

Uso:
    python -m scripts.diag_bm25 --query "¿De qué trata la Resolución 25/26 de ENACOM?" --doc-id 323b0a40-cc19-4690-933e-1332f3ed42ca
"""
from __future__ import annotations

import argparse

from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnóstico de retrieval BM25 + denso")
    parser.add_argument("--query", required=True, help="Texto de la pregunta")
    parser.add_argument("--doc-id", required=True, help="doc_id esperado (prefijo alcanza)")
    parser.add_argument("--top", type=int, default=15, help="Cuántos primeros puestos mostrar")
    args = parser.parse_args()

    from pipeline.embedder import Embedder
    from storage.vector_store import VectorStore, _tokenize

    vs = VectorStore(domain="telecom")
    query_tokens = _tokenize(args.query)
    print(f"Tokens de la pregunta: {query_tokens}\n")

    # --- BM25 ---
    if vs._bm25 is None:
        print("BM25 no está disponible (rank_bm25 no instalado o índice vacío).")
    else:
        scores = vs._bm25.get_scores(query_tokens)
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        print(f"=== BM25: top {args.top} ===")
        found_at = None
        for rank, i in enumerate(ranked_idx[: args.top], start=1):
            meta = vs._bm25_metadatas[i]
            marker = " <-- ESPERADO" if meta.get("doc_id", "").startswith(args.doc_id) else ""
            print(f"  #{rank} score={scores[i]:.3f} doc_id={meta.get('doc_id')} {meta.get('organism')} {meta.get('number')}{marker}")
        for i, meta in enumerate(vs._bm25_metadatas):
            if meta.get("doc_id", "").startswith(args.doc_id):
                rank = sorted(scores, reverse=True).index(scores[i]) + 1
                print(f"\n  Doc esperado: score={scores[i]:.3f}, puesto aproximado #{rank} de {len(scores)} chunks totales")
                found_at = rank
                break
        if found_at is None:
            print("\n  Doc esperado: NO ENCONTRADO en el índice BM25 (metadata sin ese doc_id).")

    # --- Denso ---
    print(f"\n=== Denso (ChromaDB): top {args.top} ===")
    embedder = Embedder()
    query_vector = embedder.embed_query(args.query)
    dense_results = vs._collection.query(
        query_embeddings=[query_vector],
        n_results=min(200, vs._collection.count()),
        include=["metadatas", "distances"],
    )
    metas = dense_results["metadatas"][0]
    distances = dense_results["distances"][0]
    for rank, (meta, dist) in enumerate(zip(metas[: args.top], distances[: args.top]), start=1):
        marker = " <-- ESPERADO" if meta.get("doc_id", "").startswith(args.doc_id) else ""
        print(f"  #{rank} score={1 - dist:.3f} doc_id={meta.get('doc_id')} {meta.get('organism')} {meta.get('number')}{marker}")
    for rank, meta in enumerate(metas, start=1):
        if meta.get("doc_id", "").startswith(args.doc_id):
            print(f"\n  Doc esperado: puesto #{rank} de {len(metas)} candidatos densos, score={1 - distances[rank - 1]:.3f}")
            break
    else:
        print(f"\n  Doc esperado: NO apareció entre los {len(metas)} candidatos densos devueltos.")


if __name__ == "__main__":
    main()
