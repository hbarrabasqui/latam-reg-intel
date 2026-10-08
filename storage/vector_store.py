from __future__ import annotations

import logging
import re
import uuid
from typing import Any

import chromadb

from pipeline.chunker import Chunk

logger = logging.getLogger(__name__)

# Cada dominio tiene su propia colección en ChromaDB.
# Dominios disponibles: "telecom", "electrico", "seguridad", etc.
_PERSIST_DIR = "./data/chroma"

_RRF_K = 60  # constante estándar de Reciprocal Rank Fusion (Cormack et al.)
_CANDIDATE_POOL_MULTIPLIER = 4  # candidatos que trae cada buscador antes de fusionar


_COMPOUND_RE = re.compile(r"[a-záéíóúñü0-9]+(?:[.\-/][a-záéíóúñü0-9]+)*")
_COMPOUND_SEP_RE = re.compile(r"[.\-/]")


def _tokenize(text: str) -> list[str]:
    """Tokenización simple para BM25.

    Minúsculas + regex que preserva códigos alfanuméricos compuestos (ej.
    "ENACOM-Q2-60.21", "1268/2024") como un solo token, en vez de partirlos
    en pedazos sueltos ("enacom", "q2", "60", "21"). Esto es clave: el
    problema real que motivó agregar BM25 es que esos identificadores
    exactos se diluyen en el espacio de embeddings (hay miles de números
    parecidos) y la búsqueda semántica no los ancla bien. BM25 los encuentra
    por coincidencia léxica literal — pero solo si el tokenizador no los
    destruye antes.

    Además de cada token compuesto, se agregan también sus partes sueltas
    (encontrado en evals del 5/8/2026: la pregunta "Resolución 25/26" generaba
    el token compuesto "25/26", pero el documento real solo tiene "25" suelto
    en el título — "ENACOM 25 — Resolución 25 ENACOM/26", un artefacto del
    formato de InfoLeg donde el organismo y el año quedan pegados con "/".
    Ni "25/26" ni "enacom" sueltos hacían match contra "25" y "enacom/26" del
    documento, y esa norma quedaba con score BM25 nulo pese a estar bien
    indexada). Emitir ambas formas no le quita precisión al match exacto —
    BM25 sigue premiando más el solapamiento de tokens compuestos, que son
    más raros (mayor IDF) — pero permite que un match parcial (mismo número,
    distinto formato de separador) todavía sume puntos."""
    text = text.lower()
    tokens: list[str] = []
    for m in _COMPOUND_RE.finditer(text):
        compound = m.group(0)
        tokens.append(compound)
        if _COMPOUND_SEP_RE.search(compound):
            tokens.extend(p for p in _COMPOUND_SEP_RE.split(compound) if p)
    return tokens


def _reciprocal_rank_fusion(rankings: list[list[str]], k: int = _RRF_K) -> dict[str, float]:
    """Combina varias listas rankeadas (mejor primero) en un único score por id.

    RRF en vez de sumar/promediar scores directamente porque las escalas no son
    comparables (similitud coseno ~[0,1] vs. score BM25 sin cota superior) y
    normalizar a mano requeriría calibrar pesos arbitrarios. RRF solo usa la
    posición en cada ranking, así que funciona sin ese ajuste."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


class VectorStore:
    """
    Base de datos vectorial usando ChromaDB (persistente en disco) + un índice
    léxico BM25 en memoria sobre el mismo contenido. La búsqueda combina ambos
    con Reciprocal Rank Fusion.

    Por qué el híbrido: los embeddings densos son buenos para similitud
    semántica (temas, conceptos relacionados) pero fallan en anclar
    identificadores exactos y cortos — números de resolución, códigos de
    norma tipo "ENACOM-Q2-60.21". Se diluyen entre miles de otros números en
    el espacio vectorial. BM25 los encuentra por coincidencia léxica literal.
    Ninguno reemplaza al otro: por eso se combinan en vez de elegir uno.

    Una instancia = un dominio = una colección ChromaDB.
    Esto permite tener RAGs completamente separados por rubro
    sin que se mezcle normativa de telecomunicaciones con eléctrica, etc.

    Uso:
        store = VectorStore(domain="telecom")
        store_elec = VectorStore(domain="electrico")

    Nota sobre el índice BM25: se construye una sola vez al crear la
    instancia (leyendo todo lo que ya está en ChromaDB) y no se actualiza
    solo. Si se indexan chunks nuevos en la misma instancia de proceso
    (ej. dentro de scripts/reindex.py) y después se necesita buscar sobre
    ellos en ese mismo proceso, hay que llamar a `refresh_bm25_index()`
    explícitamente. En el uso normal del proyecto esto no es un problema:
    reindex.py (escribe) y chat/eval (leen) son invocaciones de proceso
    separadas, así que cada lectura arranca con una instancia nueva que ya
    ve todo lo escrito.
    """

    def __init__(self, domain: str = "telecom", persist_dir: str = _PERSIST_DIR) -> None:
        self._domain = domain
        # PersistentClient guarda todo en disco automáticamente
        self._client = chromadb.PersistentClient(path=persist_dir)
        # get_or_create_collection: crea la colección si no existe, la reutiliza si ya existe
        self._collection = self._client.get_or_create_collection(
            name=domain,
            metadata={"hnsw:space": "cosine"},  # similitud coseno (igual que antes)
        )
        logger.info(
            "ChromaDB listo. Dominio='%s' | Chunks indexados: %d",
            domain,
            self._collection.count(),
        )

        self._bm25 = None
        self._bm25_ids: list[str] = []
        self._bm25_docs: list[str] = []
        self._bm25_metadatas: list[dict[str, Any]] = []
        self._bm25_id_to_idx: dict[str, int] = {}
        self._build_bm25_index()

    # ------------------------------------------------------------------
    # Escritura
    # ------------------------------------------------------------------

    def index_chunks(self, chunk_vectors: list[tuple[Chunk, list[float]]]) -> int:
        """
        Indexa una lista de (chunk, vector) en ChromaDB.
        Devuelve la cantidad de chunks indexados.

        ChromaDB espera tres listas paralelas:
          - ids: identificador único por chunk
          - embeddings: los vectores numéricos
          - documents: el texto (para poder recuperarlo luego)
          - metadatas: diccionario con campos filtrables
        """
        if not chunk_vectors:
            return 0

        ids, embeddings, documents, metadatas = [], [], [], []

        for chunk, vector in chunk_vectors:
            ids.append(str(uuid.uuid4()))
            embeddings.append(vector)
            documents.append(chunk.text)
            metadatas.append({
                "doc_id": chunk.doc_id,
                "chunk_index": chunk.chunk_index,
                "country": chunk.country,
                "organism": chunk.organism,
                "doc_type": chunk.doc_type,
                "number": chunk.number,
                "title": chunk.title,
                "source_url": chunk.source_url,
                "language": chunk.language,
            })

        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info("Indexados %d chunks en ChromaDB (dominio='%s').", len(ids), self._domain)
        return len(ids)

    def delete_by_doc_id(self, doc_id: str) -> None:
        """Elimina todos los chunks de un documento dado su doc_id."""
        self._collection.delete(where={"doc_id": doc_id})

    # ------------------------------------------------------------------
    # Índice léxico (BM25)
    # ------------------------------------------------------------------

    def _build_bm25_index(self) -> None:
        """(Re)construye el índice BM25 en memoria a partir de todo lo que
        hay ahora mismo en la colección de ChromaDB."""
        if self._collection.count() == 0:
            logger.info("Colección vacía — el índice BM25 se construye recién cuando haya chunks.")
            return
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning(
                "rank_bm25 no está instalado — el retrieval corre solo con embeddings "
                "densos, sin el complemento léxico. Instalar con `pip install rank-bm25`."
            )
            return

        data = self._collection.get(include=["documents", "metadatas"])
        self._bm25_ids = data["ids"]
        self._bm25_docs = data["documents"]
        self._bm25_metadatas = data["metadatas"]
        self._bm25_id_to_idx = {doc_id: i for i, doc_id in enumerate(self._bm25_ids)}
        tokenized_corpus = [_tokenize(doc) for doc in self._bm25_docs]
        self._bm25 = BM25Okapi(tokenized_corpus)
        logger.info("Índice BM25 construido sobre %d chunks (dominio='%s').", len(self._bm25_ids), self._domain)

    def refresh_bm25_index(self) -> None:
        """Reconstruye el índice BM25 a mano. Ver nota en el docstring de la
        clase sobre cuándo hace falta (procesos largos que escriben y leen a
        la vez sobre la misma instancia)."""
        self._build_bm25_index()

    def _bm25_candidates(
        self,
        query_text: str,
        limit: int,
        country: str | None,
        organism: str | None,
    ) -> list[str]:
        """Devuelve ids de chunk rankeados por BM25 (mejor primero), ya
        filtrados por country/organism (mismo criterio AND que el filtro
        `where` de ChromaDB, para que ambos buscadores compitan sobre el
        mismo subconjunto)."""
        if self._bm25 is None or not self._bm25_ids:
            return []

        scores = self._bm25.get_scores(_tokenize(query_text))
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results: list[str] = []
        for i in ranked_idx:
            if scores[i] <= 0:
                break  # a partir de acá no comparten ningún término — cortar
            meta = self._bm25_metadatas[i]
            if country and meta.get("country") != country:
                continue
            if organism and meta.get("organism") != organism:
                continue
            results.append(self._bm25_ids[i])
            if len(results) >= limit:
                break
        return results

    # ------------------------------------------------------------------
    # Búsqueda
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: list[float],
        limit: int = 5,
        country: str | None = None,
        organism: str | None = None,
        query_text: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Búsqueda híbrida: similitud semántica (embeddings, ChromaDB) + coincidencia
        léxica (BM25), combinadas con Reciprocal Rank Fusion.

        `query_text` es opcional por compatibilidad hacia atrás: si no se pasa (o
        no hay índice BM25 disponible), el comportamiento es el de siempre, puramente
        denso. Para aprovechar el híbrido hay que pasar el texto de la pregunta.

        Parámetros opcionales de filtro (se combinan con AND):
          - country: "ARG", "BRA", "MEX"...
          - organism: "ENACOM", "ANATEL"...

        Devuelve lista de dicts con el texto, metadata y score.
        """
        where = _build_where_filter(country=country, organism=organism)
        pool = max(limit * _CANDIDATE_POOL_MULTIPLIER, 20)
        n_results = min(pool, self._collection.count()) or 1

        dense_results = self._collection.query(
            query_embeddings=[query_vector],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        dense_ids = dense_results["ids"][0]

        # Lookup de texto/metadata por id — se arma con lo que trae cada buscador,
        # para poder reconstruir el resultado final sin importar de cuál vino.
        lookup: dict[str, dict[str, Any]] = {}
        for doc_id, doc, meta, distance in zip(
            dense_ids,
            dense_results["documents"][0],
            dense_results["metadatas"][0],
            dense_results["distances"][0],
        ):
            lookup[doc_id] = {"text": doc, "meta": meta, "dense_score": 1 - distance}

        bm25_ids: list[str] = []
        if query_text:
            bm25_ids = self._bm25_candidates(query_text, pool, country, organism)
            for doc_id in bm25_ids:
                if doc_id not in lookup:
                    idx = self._bm25_id_to_idx[doc_id]
                    lookup[doc_id] = {
                        "text": self._bm25_docs[idx],
                        "meta": self._bm25_metadatas[idx],
                        "dense_score": None,
                    }

        if not bm25_ids:
            # Sin BM25 disponible: comportamiento anterior, puramente denso.
            fused_ids = dense_ids[:limit]
            fused_scores: dict[str, float] = {}
        else:
            fused_scores = _reciprocal_rank_fusion([dense_ids, bm25_ids])
            fused_ids = sorted(fused_scores, key=lambda i: fused_scores[i], reverse=True)[:limit]

        hits = []
        for doc_id in fused_ids:
            entry = lookup[doc_id]
            # Para mostrar un score interpretable (0-1, estilo similitud) se prioriza
            # el score denso cuando existe. Los hits que BM25 encontró pero el buscador
            # denso no trajo en su pool no tienen score denso comparable — no afecta el
            # orden (ya decidido por RRF), solo el número que se muestra en el contexto.
            score = entry["dense_score"] if entry["dense_score"] is not None else fused_scores.get(doc_id, 0.0)
            hits.append({"score": score, "text": entry["text"], **entry["meta"]})

        return hits

    def count(self) -> int:
        """Devuelve la cantidad total de chunks indexados en este dominio."""
        return self._collection.count()

    def reset(self) -> None:
        """Elimina y recrea la colección (para reindexar desde cero)."""
        self._client.delete_collection(self._domain)
        self._collection = self._client.get_or_create_collection(
            name=self._domain,
            metadata={"hnsw:space": "cosine"},
        )
        self._bm25 = None
        self._bm25_ids = []
        self._bm25_docs = []
        self._bm25_metadatas = []
        self._bm25_id_to_idx = {}
        logger.warning("Colección '%s' eliminada y recreada vacía.", self._domain)


# ------------------------------------------------------------------
# Helper interno
# ------------------------------------------------------------------

def _build_where_filter(
    country: str | None,
    organism: str | None,
) -> dict | None:
    """
    Construye el filtro de metadata para ChromaDB.

    ChromaDB requiere:
      - Un solo filtro: {"campo": "valor"}
      - Múltiples filtros: {"$and": [{"campo1": "val1"}, {"campo2": "val2"}]}
    """
    conditions = []
    if country:
        conditions.append({"country": country})
    if organism:
        conditions.append({"organism": organism})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}
