from __future__ import annotations

import re
from dataclasses import dataclass

from storage.models import RegulatoryDocument

# Detecta inicio de artículo/sección en español y portugués.
# Cubre dos estilos de numeración:
#   - Formal: "Artículo 5", "Capítulo II", "Sección 3", "Anexo A"
#   - Decimal: "1.", "5.5", "5.5.1", "5.5.1.2" (común en normas técnicas ENACOM-Q2)
_ARTICLE_RE = re.compile(
    r"(?m)^("
    r"art(?:ículo|iculo|igo)?\.?\s*\d+|"   # Artículo 5 / Art. 5 / Artigo 5
    r"cap[íi]tulo\s+\w+|"                   # Capítulo III
    r"secci[oó]n\s+\d+|"                    # Sección 3
    r"anexo\s+\w+|"                          # Anexo A / Anexo I
    r"disposici[oó]n\s+\w+|"                # Disposición transitoria
    r"\d+(?:\.\d+){0,3}\.\s+[A-ZÁÉÍÓÚÑ]"   # 1. Objetivo / 5.5 Parámetros / 5.5.1 Potencia
    r")\b",
    re.IGNORECASE,
)

_MIN_CHUNK_CHARS = 100
_MAX_CHUNK_CHARS = 2000

# Documentos hasta este largo se indexan como UN SOLO chunk, sin partir por
# artículos. Encontrado en evals (4/8/2026): una resolución de 5168 caracteres
# partida en 6 chunks tenía uno de apenas 127 caracteres (solo la referencia a
# un anexo) que terminó ganando el ranking de búsqueda por sobre el chunk con
# el contenido sustantivo real — fragmentar un documento corto resta precisión,
# no suma. Con este umbral (cubre ~40% del corpus, mediana de largo total:
# 7452 caracteres) esa resolución entra entera en un solo chunk y no hay forma
# de que la búsqueda traiga solo un pedazo desinformativo de ella.
#
# Nota aparte (no es algo que este cambio empeore, ya pasaba con los chunks por
# artículo también): el modelo de embeddings (`paraphrase-multilingual-
# MiniLM-L12-v2`) trunca internamente a ~128 tokens (~600-700 caracteres) por
# texto que se le pasa a encode(). Un chunk de varios miles de caracteres no se
# "ve" completo en el embedding denso — pero sí se indexa completo en BM25
# (que no trunca), y sí llega completo al LLM una vez recuperado. O sea: este
# fix mejora BM25 y el contenido final que ve el modelo generador, pero no
# resuelve la limitación de truncamiento del embedding denso en sí — eso
# quedaría para más adelante (subir max_seq_length si el modelo lo soporta, o
# evaluar otro modelo de embeddings).
_SINGLE_CHUNK_THRESHOLD = 6000


@dataclass
class Chunk:
    doc_id: str
    chunk_index: int
    text: str
    # Metadatos para el vector store
    country: str
    organism: str
    doc_type: str
    number: str
    title: str
    source_url: str
    language: str


def chunk_document(doc: RegulatoryDocument) -> list[Chunk]:
    """Divide un documento en chunks para indexar en el vector store.

    Estrategia:
    1. Intentar dividir por artículos/secciones detectados con regex.
    2. Si el documento no tiene estructura (ej: descripción corta sin artículos),
       dividir por párrafos o devolver el texto completo como un solo chunk.
    3. Chunks muy largos se subdividen; chunks muy cortos se fusionan con el siguiente.
    """
    text = doc.raw_text.strip()
    if not text:
        return []

    if len(text) <= _SINGLE_CHUNK_THRESHOLD:
        segments = [text]
    else:
        segments = _split_by_articles(text)
        if len(segments) <= 1:
            segments = _split_by_paragraphs(text)

        segments = _merge_short_and_split_long(segments)

    # Encabezado que se antepone a cada chunk para que la búsqueda por
    # número/organismo funcione aunque el chunk no lo mencione explícitamente
    header = f"{doc.organism} {doc.number} — {doc.title}\n"

    return [
        Chunk(
            doc_id=doc.id,
            chunk_index=i,
            text=header + seg.strip(),
            country=doc.country,
            organism=doc.organism,
            doc_type=doc.doc_type,
            number=doc.number,
            title=doc.title,
            source_url=doc.source_url,
            language=doc.language,
        )
        for i, seg in enumerate(segments)
        if seg.strip()
    ]


def _split_by_articles(text: str) -> list[str]:
    boundaries = [m.start() for m in _ARTICLE_RE.finditer(text)]
    if not boundaries:
        return [text]
    segments: list[str] = []
    if boundaries[0] > 0:
        segments.append(text[: boundaries[0]])
    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(text)
        segments.append(text[start:end])
    return segments


def _split_by_paragraphs(text: str) -> list[str]:
    # Dividir por líneas en blanco dobles
    parts = re.split(r"\n{2,}", text)
    return [p.strip() for p in parts if p.strip()]


def _merge_short_and_split_long(segments: list[str]) -> list[str]:
    result: list[str] = []
    buffer = ""
    for seg in segments:
        if len(buffer) + len(seg) < _MIN_CHUNK_CHARS:
            buffer = (buffer + " " + seg).strip()
        else:
            if buffer:
                result.extend(_split_long(buffer))
            buffer = seg
    if buffer:
        result.extend(_split_long(buffer))
    return result


def _split_long(text: str) -> list[str]:
    if len(text) <= _MAX_CHUNK_CHARS:
        return [text]
    # Dividir en oraciones aproximadas
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) > _MAX_CHUNK_CHARS and current:
            chunks.append(current.strip())
            current = sent
        else:
            current = (current + " " + sent).strip()
    if current:
        chunks.append(current.strip())
    return chunks
