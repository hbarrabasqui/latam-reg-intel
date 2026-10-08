from __future__ import annotations

import logging
from typing import Any

from pipeline.chunker import Chunk
from pipeline.excerpt import build_salient_excerpt

logger = logging.getLogger(__name__)

# Modelo multilingüe liviano, soporta español y portugués sin configuración extra
_DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# El modelo trunca internamente a ~128 tokens (~600-700 caracteres) por texto
# que se le pasa a encode() — ver nota larga en pipeline/chunker.py sobre esto.
# Encontrado en evals (5/8/2026): con el chunking arreglado, un chunk de 5168
# caracteres (Resolución 25/2026) se indexaba como un solo chunk, pero el
# embedding denso solo "veía" los primeros ~700 caracteres — puro preámbulo
# VISTO/CONSIDERANDO — y nunca llegaba al contenido real ("inscripción en el
# Registro de Laboratorios Acreditados"), a pesar de que BM25 sí la encontraba
# bien (posición #6 de 8014). Resultado: cero señal semántica, el doc no
# aparecía ni entre 200 candidatos densos.
#
# Fix: para chunks más largos que este umbral, en vez de embeddear el texto
# completo (que igual se trunca sin criterio) se embeddea un excerpt armado
# con la misma lógica que ya usa `scripts/semantic_cleanup.py` para elegir
# qué le muestra a Haiku: preámbulo + ventana alrededor de la primera mención
# de un término fuerte de telecom en todo el documento. Esto no cambia qué
# texto se guarda/devuelve como resultado de búsqueda (`chunk.text` completo
# sigue yendo a BM25 y al LLM) — solo cambia qué texto ve el encoder de
# embeddings para calcular el vector denso.
_EMBED_EXCERPT_THRESHOLD = 700


class Embedder:
    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        # Importación diferida: sentence-transformers tarda en cargar torch
        from sentence_transformers import SentenceTransformer
        logger.info("Cargando modelo de embeddings: %s", model_name)
        self._model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.dimension = self._model.get_sentence_embedding_dimension()
        logger.info("Modelo listo. Dimensión de embeddings: %d", self.dimension)

    def embed_chunks(self, chunks: list[Chunk]) -> list[tuple[Chunk, list[float]]]:
        """Genera embeddings para una lista de chunks. Devuelve (chunk, vector) pares.

        El vector se calcula sobre `_text_for_embedding(chunk)`, que puede ser
        un excerpt más corto que `chunk.text` — pero el chunk devuelto (y lo
        que se termina guardando/mostrando) siempre tiene el texto completo."""
        if not chunks:
            return []
        texts = [self._text_for_embedding(c) for c in chunks]
        vectors = self._model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return [(chunk, vec.tolist()) for chunk, vec in zip(chunks, vectors)]

    @staticmethod
    def _text_for_embedding(chunk: Chunk) -> str:
        if len(chunk.text) <= _EMBED_EXCERPT_THRESHOLD:
            return chunk.text
        from storage.classifier import _TELECOM_STRONG_SIGNALS
        return build_salient_excerpt(chunk.text, _TELECOM_STRONG_SIGNALS)

    def embed_query(self, query: str) -> list[float]:
        """Genera embedding para una query de búsqueda."""
        vector = self._model.encode(query, convert_to_numpy=True)
        return vector.tolist()
