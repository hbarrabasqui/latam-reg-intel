# latam-reg-intel — Motor RAG sobre normativa regulatoria LATAM

Motor de *Retrieval-Augmented Generation* para consultar en lenguaje natural la
normativa de telecomunicaciones de Latinoamérica (homologación de equipos,
espectro radioeléctrico, normas técnicas), citando la fuente.

Está pensado para un dominio donde **una respuesta inventada es un riesgo real**:
un bot que alucina una norma o un número de resolución no es un error cosmético,
es información que alguien podría usar para presentar un trámite mal. Por eso el
diseño prioriza *recuperación correcta y verificable* por encima de todo.

> **Nota sobre los datos:** este repo contiene el **motor**, no el corpus. Los
> textos normativos provienen de fuentes públicas (ENACOM, Anatel, IFT) y se
> regeneran con los scrapers incluidos. La base vectorial, la base relacional y
> las claves de API quedan fuera del control de versiones.

## Por qué está hecho así (decisiones de arquitectura)

Estas son las decisiones que diferencian a este motor de un RAG de tutorial:

### 1. Recuperación híbrida (densa + léxica), no solo embeddings
La búsqueda combina **semántica (ChromaDB sobre embeddings multilingües)** con
**léxica (BM25)**, y fusiona ambos rankings con **Reciprocal Rank Fusion (RRF)**.

El motivo es concreto del dominio: las consultas regulatorias están llenas de
**códigos exactos** —`ENACOM-Q2-60.21`, `Res. 57/2026`, `1268/2024`—. Un
embedding denso "entiende" el sentido pero es malo con el match literal de un
código; BM25 es exactamente lo contrario. Solos, cada uno falla en casos reales
(encontrados en evals). Juntos se cubren.

El tokenizador (`storage/vector_store.py`) está hecho a mano para **preservar
los códigos compuestos como un solo token** en vez de partir `ENACOM-Q2-60.21`
en pedazos inservibles.

### 2. Chunking por artículo/sección, no por longitud fija
`pipeline/chunker.py` corta el texto respetando la estructura de la norma
(artículos, anexos), no cada N caracteres. Una resolución partida a ciegas por
longitud mezcla el final de un artículo con el principio de otro y ensucia el
embedding. El chunking estructural mantiene cada unidad normativa entera.

### 3. Clasificador de dominio antes de indexar
`storage/classifier.py` filtra qué documentos son realmente de telecom antes de
gastar embeddings en ellos. Las fuentes oficiales mezclan todo; filtrar por
keywords sueltas ("certificación", "habilitación") dejaba pasar ruido de otros
rubros (ENARGAS, código penal). El clasificador exige además **señales fuertes
específicas de telecom**, y un segundo paso opcional usa un LLM como juez
semántico en lotes para depurar los casos límite.

### 4. Versionado por hash de contenido
`storage/relational.py` guarda cada documento con un hash de su contenido y un
`review_status`. Eso permite **detectar cuándo una norma cambió** (no solo que
apareció una nueva) y sostener una cola de revisión humana antes de que algo
entre al índice — clave para no corromper el corpus con falsos positivos.

### 5. Evaluación como parte del diseño, no como adorno
`scripts/eval_rag.py` es un harness de evals con *code grader* (¿recuperó los
chunks correctos?) y *model grader* (¿la respuesta es fiel al contexto?). Casi
todas las decisiones de arriba salieron de correr evals y leer dónde fallaba el
retrieval, no de suponer. Ver `eval_report.md` para una corrida real.

## Estructura

```
pipeline/     extractor, chunker (por artículo), embedder, excerpt
storage/      vector_store (Chroma + BM25), relational (SQLite + hash), classifier
chat/         rag_engine (recuperación híbrida + RRF + generación), normativa_links
scrapers/     argentina (ENACOM, InfoLeg), brasil (Anatel), mexico (IFT)
scripts/      reindex, eval_rag, run_pilot, seeds, diagnósticos
tests/        tests de scrapers y capa relacional
```

## Stack

Python · ChromaDB · sentence-transformers (embeddings multilingües) ·
rank-bm25 · API de Anthropic (Claude) con *prompt caching* · pytest ·
BeautifulSoup / requests para scraping.

## Cómo correrlo

```bash
python -m venv .venv && .venv\Scripts\activate   # Windows (source .venv/bin/activate en Unix)
pip install -r requirements.txt
cp .env.example .env        # completar ANTHROPIC_API_KEY
python -m scripts.run_pilot # scraping + indexado de un cuerpo normativo
python -m scripts.chat      # consultar
```

## Estado

Proyecto en desarrollo activo. El motor (recuperación híbrida, chunking,
clasificación, evals) está funcionando; la expansión multi-país (Brasil, México)
está en forma de scrapers base. Reporte de evals incluido en `eval_report.md`.
