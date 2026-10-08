"""Motor de RAG (Retrieval-Augmented Generation) sobre normativa regulatoria LATAM.

Flujo:
  pregunta del usuario
    → embedding de la query
    → búsqueda híbrida (semántica en ChromaDB + léxica BM25, fusionadas con RRF)
    → construcción del contexto con los chunks relevantes
    → llamada a Claude API con el contexto
    → respuesta con citas al documento fuente
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

import anthropic

logger = logging.getLogger(__name__)

_MODEL = "claude-sonnet-4-6"
_MAX_TOKENS = 3000
_TOP_K = 12  # chunks a recuperar por query

_SYSTEM_PROMPT = """\
Sos un experto en regulación de telecomunicaciones en Latinoamérica, especializado en \
homologación y certificación de equipos. Respondés preguntas basándote exclusivamente en \
los documentos normativos que se te proveen en el contexto.

Reglas:
- Si la información está en el contexto, respondé con precisión y citá la fuente \
  (organismo, número de norma, artículo cuando sea posible).
- Si la información NO está en el contexto, tu respuesta completa debe ser: \
  "No encontré información sobre eso en los documentos disponibles." — y nada más.
  NO agregues después de esa frase ningún número de ley, decreto, resolución o \
  artículo, ni el nombre de una norma, aunque la conozcas y sea información real y \
  correcta. Mencionar una norma que no está en el CONTEXTO NORMATIVO de arriba es una \
  falla grave del sistema, sin importar si el dato que agregaste es correcto o no —lo \
  que importa es que no se puede verificar contra los documentos provistos. Ejemplo de \
  lo que NO hay que hacer: si te preguntan por una ley laboral y no está en el \
  contexto, no digas "eso lo regula la Ley N° 20.744" aunque sea cierto — simplemente \
  decí que no está en los documentos disponibles.
- No inventés normas ni artículos que no aparezcan en el contexto, tampoco como dato \
  complementario o aclaración "de contexto general".
- Importante — esto NO es lo mismo que lo anterior: si el contexto no tiene el \
  documento EXACTO que te piden (por ejemplo, te preguntan por la Resolución 25 y no \
  está, pero sí tenés la Resolución 57 sobre un tema relacionado), no rechaces de \
  plano. Usá la información que sí tenés disponible, aclarando explícitamente que no es \
  el documento exacto solicitado sino uno relacionado, y citando de qué documento sale \
  cada dato. Reservá la frase de rechazo total únicamente para cuando el contexto no \
  tenga NADA relacionado con el tema de la pregunta — no la uses solo porque falta el \
  número de norma puntual que te pidieron.
- Variante del punto anterior, distinta pero igual de importante: si el documento \
  correcto SÍ está en el contexto pero no menciona el DATO PUNTUAL que te piden (ej. \
  preguntan cuánto cuesta un trámite y la norma solo describe los requisitos técnicos, \
  sin mencionar costos ni aranceles), tampoco apliques la frase de rechazo total acá. \
  Contestá con lo que el documento sí dice (de qué trata, qué exige, a quién aplica) y \
  aclará explícitamente que el dato puntual pedido no está mencionado en ese documento \
  — no digas simplemente "no encontré información" cuando en realidad sí encontraste \
  el documento, solo que no responde esa pregunta específica.
- Cuando el documento esté en portugués, podés responder en español de todas formas.
- Sé conciso pero completo. Si hay diferencias entre países, destacalas.
- Si la pregunta es amplia o básica (el tipo de pregunta que hace alguien que no \
  conoce el tema — ej. "¿cómo registro un equipo en el ENACOM?", sin pedir un \
  paso puntual), NO vuelques toda la información disponible de una sola vez. Dala \
  en pasos o puntos breves, cada uno de UNA sola línea, sin entrar en el detalle de \
  cada uno (documentación exacta, aranceles, organismos extranjeros específicos, \
  etc.), y cerrá preguntando sobre cuál de esos puntos quiere que te expandas. La \
  idea es que la respuesta inicial nunca abrume a alguien que recién está \
  entendiendo el trámite — mejor que la persona vaya pidiendo profundidad de a \
  poco, a que reciba tanto de entrada que tenga que pedirte que se lo achiques. \
  Esta regla no aplica si la pregunta ya es puntual o técnica (ahí respondé \
  directo con el nivel de detalle que corresponda).
- Si un fragmento del contexto tiene una línea que empieza con "⚠", es una nota \
  automática sobre su vigencia (indica si esa norma fue modificada o derogada por \
  otra norma que también está en este contexto). Prestale atención: NO atribuyas un \
  dato (banda de frecuencia, plazo, requisito, etc.) a una norma marcada como \
  derogada o modificada si el mismo contexto tiene la norma que la reemplaza — \
  preferí siempre el dato de la norma más nueva/vigente, y si mencionás la norma \
  anterior, aclará explícitamente que fue derogada/modificada y por cuál.
- No agregues palabras, adjetivos o calificadores que no estén literalmente en el \
  contexto, aunque te parezcan una paráfrasis razonable. Ejemplo real de lo que NO \
  hay que hacer: el contexto dice que la homologación "permite comercializar el \
  equipo en el país" y la respuesta agrega "permite la comercialización masiva" — \
  la palabra "masiva" no está en ningún lado y es una alucinación, aunque suene \
  natural.
- Si vas a citar algo entre comillas o com bloque de cita textual, tiene que ser \
  una transcripción EXACTA del contexto, carácter por carácter. Si estás \
  resumiendo o parafraseando una idea, no la pongas entre comillas ni la \
  presentes como cita textual — decila en tu propia redacción.
- No inventes estructura que no está en el documento: si el contexto no organiza \
  la información en una tabla, no armes una tabla vos; si no tiene "N secciones" \
  numeradas, no le pongas ese formato. Podés reorganizar para que se entienda \
  mejor, pero sin sugerir que esa estructura viene del documento original.
- No agregues prefijos o formatos a un número de norma que no aparecen así en el \
  contexto (ej. si el contexto dice "Resolución 270/2002" sin la sigla del \
  organismo pegada al número, no escribas "Resolución CNC N° 270/2002" como si \
  ese formato exacto estuviera en el texto).
- Si la pregunta tiene varias partes (ej. "¿hay normas de la época CNC o AFTIC?"), \
  respondé TODAS las partes explícitamente. Si el contexto tiene información sobre \
  una parte pero no sobre la otra, decilo — no ignores silenciosamente la parte que \
  no encontraste.
- Un fragmento del contexto puede venir de una "nota interpretativa" (vas a ver \
  "doc_type: nota_interpretativa" o el organismo diciendo lo mismo en el texto). No es \
  una norma oficial publicada por el organismo — es un análisis propio que cruza varias \
  resoluciones para responder una pregunta que ninguna resolución sola responde \
  completa, ya validado por un experto humano. Si usás una nota de este tipo, decilo \
  explícitamente (ej. "según un análisis interno que cruza las Resoluciones X, Y y Z...") \
  y no la presentes como si fuera texto de una resolución. Si la nota menciona números \
  de norma concretos, esos números sí podés citarlos como tal.
- Cada fragmento del contexto puede tener una línea "Fuente: <url>". Si te piden el \
  link, la fuente, o cómo descargar una norma, compartí exactamente esa URL (nunca \
  inventes ni modifiques una) y aclará que es un enlace a la fuente oficial (InfoLeg \
  o ENACOM), no una copia certificada del documento. Si el fragmento relevante no \
  tiene línea "Fuente:", decí que no tenés un link para esa norma en particular — no \
  inventes uno ni ofrezcas un link genérico al sitio del organismo."""

# Patrón de rechazo — si la respuesta lo contiene, se le aplica la guardia de citas
# no respaldadas (ver _find_unsupported_citation). Debe ser consistente con el patrón
# equivalente en scripts/eval_rag.py.
_REFUSAL_PATTERN = re.compile(
    r"no encontr[eé]|no tengo información|no está en los documentos", re.IGNORECASE
)
_CLEAN_REFUSAL = "No encontré información sobre eso en los documentos disponibles."

# Patrones típicos de cita legal/normativa en español rioplatense.
_CITATION_RE = re.compile(
    r"(?:ley|decreto|resoluci[oó]n|disposici[oó]n)\s*n?[°ºo]?\.?\s*[\d][\d./-]*"
    r"|art[íi]culo\s*\d+(?:\s*bis)?",
    re.IGNORECASE,
)

# Si una cita aparece cerca de una negación ("la Resolución 25 NO está en los
# documentos"), no es una fabricación — es el modelo siendo honesto sobre lo que
# el usuario pidió y no está disponible. Sin este chequeo, la guardia castigaba
# por error justamente el tipo de respuesta matizada que se busca fomentar
# (mencionar qué falta y ofrecer lo relacionado que sí hay).
_NEGATION_NEARBY_RE = re.compile(
    r"no (?:est[aá]|aparece|encontr[eé]|tengo|figura|hay (?:referencia|informaci[oó]n))",
    re.IGNORECASE,
)
_NEGATION_WINDOW = 60  # caracteres a mirar antes/después de la cita


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def _find_unsupported_citation(answer: str, context: str) -> str | None:
    """Guardia de código (determinística, sin IA) contra un patrón real detectado en
    evals: el modelo a veces declina correctamente ("no encontré información...") pero
    igual agrega una cita legal específica que no viene del contexto recuperado —una
    ley/decreto/resolución/artículo real, pero no verificable contra los documentos
    provistos. La instrucción del system prompt no alcanzó sola como mitigación (se
    reprodujo en dos corridas de eval distintas), así que esto actúa como red de
    seguridad después de generar la respuesta.

    Ignora menciones que aparecen calificadas como ausentes (ej. "la Resolución 25 no
    está en los documentos") — eso es honestidad, no fabricación.

    Devuelve la cita sospechosa encontrada, o None si todas las citas mencionadas están
    respaldadas por el contexto, están negadas explícitamente, o no hay ninguna."""
    context_digits = _digits(context)
    for match in _CITATION_RE.finditer(answer):
        cite = match.group(0)
        cite_digits = _digits(cite)
        if len(cite_digits) < 2:
            continue  # muy corto/genérico (ej. "artículo 1") como para ser una alerta confiable
        if cite_digits in context_digits:
            continue

        start = max(0, match.start() - _NEGATION_WINDOW)
        end = match.end() + _NEGATION_WINDOW
        vecindad = answer[start:end]
        if _NEGATION_NEARBY_RE.search(vecindad):
            continue  # ej. "la Resolución 25 no está en los documentos" — honesto, no fabricado

        return cite
    return None


@dataclass
class RAGResponse:
    answer: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    chunks_used: int = 0
    model: str = _MODEL
    # Texto completo de los chunks recuperados, tal cual se le mandó al LLM.
    # No se muestra al usuario final (por eso no está en `sources`), pero hace
    # falta para poder auditar/evaluar si la respuesta es fiel al contexto real
    # (ver scripts/eval_rag.py — sin esto el model grader no puede verificar
    # nada, solo ve metadata).
    context: str = ""


class RAGEngine:
    def __init__(
        self,
        embedder: Any | None = None,
        vector_store: Any | None = None,
        top_k: int = _TOP_K,
    ) -> None:
        self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self._top_k = top_k

        # Lazy init si no se pasan desde afuera
        if embedder is None:
            from pipeline.embedder import Embedder
            embedder = Embedder()
        if vector_store is None:
            from storage.vector_store import VectorStore
            vector_store = VectorStore(domain="telecom")

        self._embedder = embedder
        self._vector_store = vector_store

        # Resuelve relaciones de modificación/derogación entre los documentos
        # que terminan juntos en un mismo contexto (ver chat/normativa_links.py
        # — mitiga el patrón real de mezcla de contenido entre una norma
        # vigente y la que reemplaza, encontrado en los evals).
        from storage.relational import DocumentStore
        from chat.normativa_links import NormativaLinkResolver
        self._link_resolver = NormativaLinkResolver(DocumentStore())

    def ask(
        self,
        question: str,
        country: str | None = None,
        organism: str | None = None,
    ) -> RAGResponse:
        """Responde una pregunta usando RAG sobre los documentos indexados."""

        # 1. Embedding de la query
        query_vector = self._embedder.embed_query(question)

        # 2. Búsqueda híbrida: semántica (embeddings) + léxica (BM25).
        # query_text habilita el complemento BM25 — clave para anclar códigos
        # y números de norma exactos que la búsqueda puramente semántica
        # tiende a diluir entre miles de números parecidos.
        hits = self._vector_store.search(
            query_vector,
            limit=self._top_k,
            country=country,
            organism=organism,
            query_text=question,
        )

        if not hits:
            return RAGResponse(
                answer="No encontré documentos indexados para responder esta pregunta. "
                       "Asegurate de haber corrido el pipeline de scraping e indexado.",
                chunks_used=0,
            )

        # 3. Construir contexto — con notas de vigencia/derogación cuando dos
        # documentos relacionados (uno modifica/deroga al otro) caen juntos en
        # el mismo resultado de búsqueda.
        doc_ids = [hit.get("doc_id") for hit in hits if hit.get("doc_id")]
        link_notes = self._link_resolver.resolve(doc_ids)
        context = _build_context(hits, link_notes)

        # 4. Llamada a Claude con prompt caching en el system prompt
        message = self._client.messages.create(
            model=_MODEL,
            max_tokens=_MAX_TOKENS,
            system=[
                {
                    "type": "text",
                    "text": _SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": f"CONTEXTO NORMATIVO:\n\n{context}\n\n---\n\nPREGUNTA: {question}",
                }
            ],
        )

        answer = message.content[0].text

        # Guardia post-generación: si el modelo declina pero igual coló una cita
        # legal no respaldada por el contexto, se reemplaza por un rechazo limpio.
        if _REFUSAL_PATTERN.search(answer):
            bad_cite = _find_unsupported_citation(answer, context)
            if bad_cite:
                logger.warning(
                    "Cita no respaldada detectada en respuesta de rechazo y removida: %r",
                    bad_cite,
                )
                answer = _CLEAN_REFUSAL

        sources = _extract_sources(hits)

        logger.info(
            "RAG: %d chunks usados, %d tokens entrada, %d tokens salida",
            len(hits),
            message.usage.input_tokens,
            message.usage.output_tokens,
        )

        return RAGResponse(
            answer=answer,
            sources=sources,
            chunks_used=len(hits),
            model=_MODEL,
            context=context,
        )


def _build_context(hits: list[dict[str, Any]], link_notes: dict[str, list[str]] | None = None) -> str:
    link_notes = link_notes or {}
    parts = []
    for i, hit in enumerate(hits, 1):
        header = (
            f"[{i}] {hit.get('organism', '')} {hit.get('number', '')} "
            f"({hit.get('country', '')}) — {hit.get('title', '')[:80]}"
        )
        text = hit.get("text", "").strip()
        score = hit.get("score", 0)
        notes = link_notes.get(hit.get("doc_id", ""), [])
        notas_str = "".join(f"\n⚠ {n}" for n in notes)
        # Se agrega acá (5/8/2026) para que el modelo pueda compartir el link
        # cuando se lo pidan en la misma conversación — antes solo lo mostraba
        # `scripts/chat.py` por afuera, en la lista de fuentes al final, así
        # que si el usuario preguntaba "pasame el link" en el medio del chat,
        # el modelo no tenía ese dato disponible para contestar.
        source_url = hit.get("source_url") or ""
        url_str = f"\nFuente: {source_url}" if source_url else ""
        parts.append(f"{header}\nRelevancia: {score:.2f}{notas_str}{url_str}\n{text}")
    return "\n\n---\n\n".join(parts)


def _extract_sources(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    sources = []
    for hit in hits:
        doc_id = hit.get("doc_id", "")
        if doc_id not in seen:
            seen.add(doc_id)
            sources.append(
                {
                    "organism": hit.get("organism"),
                    "number": hit.get("number"),
                    "country": hit.get("country"),
                    "title": hit.get("title"),
                    "source_url": hit.get("source_url"),
                    "score": round(hit.get("score", 0), 3),
                }
            )
    return sources
