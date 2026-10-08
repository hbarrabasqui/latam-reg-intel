"""Harness de evals para el chat RAG (chat/rag_engine.py), siguiendo el
framework del curso de Anthropic sobre evaluación de prompts:

    dataset de casos de prueba → correr el prompt bajo prueba → calificar
    (code grader + model grader) → promediar → reporte

Separación clave (la del curso): generar el dataset con un modelo BARATO
(Haiku) y distinto del que se evalúa; correr el prompt bajo prueba (acá,
el RAG completo); calificar con dos jueces independientes — uno de código
(determinístico, verifica hechos objetivos) y uno de modelo (evalúa fidelidad
y calidad, pidiendo razonamiento antes del puntaje).

Uso:
    python -m scripts.eval_rag                    # dataset default + eval completo
    python -m scripts.eval_rag --generate 10       # además genera 10 casos con Haiku
    python -m scripts.eval_rag --report reporte.md # guarda el reporte en markdown
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_DATASET_MODEL = "claude-haiku-4-5"   # barato y distinto del modelo evaluado
_GRADER_MODEL = "claude-haiku-4-5"    # juez de modelo — no hace falta el modelo caro para esto

_GENERATOR_INSTRUCTIONS = (
    "Te doy el texto de una norma regulatoria de telecomunicaciones de Argentina. "
    "Generá UNA pregunta en español que un usuario real le haría a un chat sobre esta "
    "norma, respondible solo con este texto. Respondé EXCLUSIVAMENTE con JSON: "
    '{"question": "..."}'
)


# ─── DATASET ───────────────────────────────────────────────────────────────

@dataclass
class EvalCase:
    id: str
    question: str
    kind: str  # "in_corpus" | "out_of_corpus"
    country: str | None = None
    organism: str | None = None
    expected_number: str | None = None   # si in_corpus: número de norma que debería citar
    source: str = "manual"               # "manual" | "generated"


# Casos semilla escritos a mano, sobre contenido que sabemos que está indexado
# (ver muestra revisada en la conversación). Cubren: pregunta genérica, norma
# puntual verificable, pregunta fuera de corpus (otro país) y pregunta
# claramente no-telecom para probar el rechazo correcto.
_SEED_CASES: list[EvalCase] = [
    EvalCase(
        id="generica_homologacion",
        question="¿Qué es la homologación de equipos de telecomunicaciones en Argentina?",
        kind="in_corpus",
        country="ARG",
    ),
    EvalCase(
        id="norma_puntual_q2_60_21",
        question="¿Qué establece la norma técnica ENACOM-Q2-60.21 sobre equipos transmisores y receptores?",
        kind="in_corpus",
        country="ARG",
        expected_number="1268/2024",  # resolución que aprueba esa norma técnica
    ),
    EvalCase(
        id="norma_puntual_resolucion_25_26",
        question="¿De qué trata la Resolución 25/26 de ENACOM?",
        kind="in_corpus",
        country="ARG",
        expected_number="25",
    ),
    EvalCase(
        # Actualizado 6/8/2026: dejó de ser "fuera de corpus" — se cargó
        # contenido real de Chile/SUBTEL (scripts/seed_chile_subtel.py), y la
        # Resolución 737 es justamente uno de los dos documentos cargados.
        # OJO: la pregunta puntual (costo del trámite) NO está en ninguno de
        # los dos documentos (son sobre requisitos técnicos, no aranceles) —
        # lo correcto ahora no es declinar de plano ni inventar un costo, sino
        # citar la norma real y aclarar que el dato del costo no está
        # disponible. El code grader de abajo solo chequea recuperación +
        # cita del número; el matiz de "no inventar el costo" lo evalúa mejor
        # el model grader.
        id="fuera_de_corpus_chile",
        question="¿Cuánto cobra SUBTEL en Chile por el trámite de homologación bajo la Resolución 737?",
        kind="in_corpus",
        country="CHL",
        organism="SUBTEL",
        expected_number="737",
    ),
    EvalCase(
        # Caso nuevo agregado 6/8/2026 (a pedido de Code): el de arriba
        # depende de que el bot conteste bien "no lo dice" — útil, pero no
        # alcanza para confirmar que el contenido de Chile se recupera y se
        # usa bien cuando la respuesta SÍ está en el documento. Este caso
        # tiene una respuesta verificable real dentro de la Resolución 737.
        id="norma_puntual_subtel_qr",
        question="Según la Resolución 737 de SUBTEL, ¿qué debe llevar el empaque de la mayoría de los "
                  "equipos de alcance reducido en vez de la certificación individual?",
        kind="in_corpus",
        country="CHL",
        organism="SUBTEL",
        expected_number="737",
    ),
    EvalCase(
        # Agregado 13/8/2026: valida la nota interpretativa nueva
        # (data/notas_interpretativas/ARG-fcc-terminales-moviles.txt). El
        # objetivo no es solo "recupera el chunk" sino que la respuesta sea
        # BREVE (Horacio lo pidió explícito: "para el usuario final, sino no
        # entiende") y no repita el enredo completo de las 5 resoluciones a
        # menos que se lo pidan en detalle.
        id="interpretativa_fcc_terminales",
        question="¿Por qué se sigue aceptando un certificado FCC para homologar celulares "
                  "en Argentina si ya hay laboratorios argentinos acreditados?",
        kind="in_corpus",
        country="ARG",
        organism="ENACOM",
    ),
    EvalCase(
        id="fuera_de_dominio_no_telecom",
        question="¿Qué requisitos exige la ley de contrato de trabajo para el período de prueba?",
        kind="out_of_corpus",  # no es telecom, no debería estar en el corpus ni responderse
    ),
    EvalCase(
        id="historico_aftic_cnc",
        question="¿Hay alguna norma de la época en que el organismo regulador se llamaba CNC o AFTIC "
                  "(antes de ENACOM) sobre radiocomunicaciones?",
        kind="in_corpus",
        country="ARG",
    ),
    EvalCase(
        # Agregado 24/8/2026: valida una corrección real y grave — el bot (en una
        # simulación manual, no en el pipeline real, pero el riesgo es el mismo)
        # afirmó que WiFi y Bluetooth se regulan por ENACOM-Q2-60.14 "Dispositivos
        # de Baja Potencia", lo cual es falso. La norma real es Q2-63.02 ("Sistemas
        # de Modulación Digital de Banda Ancha", cubre WiFi y Bluetooth LE) y
        # Q2-63.03 ("Sistemas de Salto de Frecuencia", cubre Bluetooth clásico).
        # Ver data/tramites_empresas/ARG-mapa-normas-corto-alcance.txt.
        id="norma_tecnica_wifi_bluetooth",
        question="¿Qué norma técnica de ENACOM regula los equipos WiFi y Bluetooth?",
        kind="in_corpus",
        country="ARG",
        organism="ENACOM",
    ),
]


def _load_seed_cases() -> list[EvalCase]:
    return list(_SEED_CASES)


def generate_cases_with_haiku(client: Any, n: int) -> list[EvalCase]:
    """Genera casos de prueba adicionales con Haiku a partir de una muestra
    real de documentos indexados. Modelo barato y distinto del que se evalúa,
    para no sesgar el examen a favor del propio RAG (ver curso, capítulo 3)."""
    from storage.classifier import DomainClassifier
    from storage.relational import DocumentStore

    store = DocumentStore()
    classifier = DomainClassifier()
    docs = store.get_approved()
    telecom_docs = [d for d in docs if classifier.classify(d) == "telecom"]

    # Algunos documentos del scraping original tienen en el campo "número" un
    # nombre de archivo (ej. "archivo_20220309120332_8089.pdf") en vez de un
    # número de norma real, cuando el scraper no pudo parsear el número. Usar
    # eso como expected_number no es una prueba justa — ningún RAG razonable
    # va a citar un nombre de archivo como si fuera un número de resolución.
    def _numero_valido(numero: str) -> bool:
        n = (numero or "").strip().lower()
        if not n or ".pdf" in n or n.startswith("archivo_") or len(n) > 25:
            return False
        return True

    telecom_docs = [d for d in telecom_docs if _numero_valido(d.number)]

    if not telecom_docs:
        logger.warning("No hay documentos telecom aprobados para generar casos.")
        return []

    random.seed(42)
    sample = random.sample(telecom_docs, min(n, len(telecom_docs)))

    cases: list[EvalCase] = []
    for i, doc in enumerate(sample):
        excerpt = doc.raw_text[:1200]
        try:
            msg = client.messages.create(
                model=_DATASET_MODEL,
                max_tokens=300,
                # Instrucciones estáticas separadas del texto variable de cada
                # documento (ver nota de prompt caching junto a _GRADER_SYSTEM
                # más abajo — acá el bloque estático mide ~100 tokens, también
                # por debajo del mínimo cacheable de Haiku, así que separar en
                # `system` es por prolijidad/futuro, no genera ahorro hoy).
                system=[
                    {
                        "type": "text",
                        "text": _GENERATOR_INSTRUCTIONS,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[
                    {
                        "role": "user",
                        "content": f"Organismo: {doc.organism} {doc.number}\nTexto:\n{excerpt}",
                    }
                ],
            )
            raw = msg.content[0].text.strip().strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()
            data = json.loads(raw)
            cases.append(
                EvalCase(
                    id=f"generado_{i}_{doc.id[:8]}",
                    question=data["question"],
                    kind="in_corpus",
                    country="ARG",
                    expected_number=doc.number,
                    source="generated",
                )
            )
        except Exception as exc:
            logger.warning("No pude generar caso para %s: %s", doc.id, exc)

    return cases


# ─── EJECUCIÓN ──────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    case: EvalCase
    answer: str
    sources: list[dict[str, Any]]
    chunks_used: int
    context: str = ""  # texto real recuperado, para que el model grader pueda verificar de verdad
    code_score: float = 0.0
    code_notes: list[str] = field(default_factory=list)
    model_score: float | None = None
    model_reasoning: str = ""


def run_case(engine: Any, case: EvalCase) -> EvalResult:
    response = engine.ask(case.question, country=case.country, organism=case.organism)
    return EvalResult(
        case=case,
        answer=response.answer,
        sources=response.sources,
        chunks_used=response.chunks_used,
        context=response.context,
    )


# ─── CODE GRADER (determinístico, sin IA) ──────────────────────────────────

_REFUSAL_PATTERN = re.compile(r"no encontr[eé]|no tengo información|no está en los documentos", re.IGNORECASE)


def _numbers_match(expected: str, actual: str) -> bool:
    """Igualdad exacta (normalizada) para campos estructurados como
    source['number']. Antes se usaba `in` (substring), lo que daba falsos
    positivos: "25" matcheaba dentro de "2540" o "8925" sin ninguna relación
    real. Acá exigimos que sea el mismo valor, no una coincidencia parcial."""
    return expected.strip().casefold() == (actual or "").strip().casefold()


def _appears_as_token(expected: str, text: str) -> bool:
    """Busca `expected` dentro de texto libre (la respuesta del chat), pero
    evitando que un número corto matchee como substring de uno más largo
    (mismo bug que _numbers_match, pero para texto libre en vez de un campo
    exacto: acá no podemos exigir igualdad total porque `expected` es solo
    una parte de una oración)."""
    expected = expected.strip()
    if not expected:
        return False
    pattern = re.compile(r"(?<!\d)" + re.escape(expected) + r"(?!\d)")
    return bool(pattern.search(text))


def code_grade(result: EvalResult) -> None:
    """Chequeos objetivos, sin IA. Escribe code_score (0-10) y code_notes en el result."""
    notes = []
    case = result.case

    if case.kind == "out_of_corpus":
        # Éxito = declina responder en vez de inventar
        if _REFUSAL_PATTERN.search(result.answer) or result.chunks_used == 0:
            notes.append("OK: declinó correctamente en vez de inventar.")
            result.code_score = 10.0
        else:
            notes.append("FALLO: debería haber declinado (pregunta fuera de corpus) y no lo hizo.")
            result.code_score = 0.0
        result.code_notes = notes
        return

    # in_corpus
    if result.chunks_used == 0:
        notes.append("FALLO: no recuperó ningún chunk para una pregunta que debería estar cubierta.")
        result.code_score = 0.0
        result.code_notes = notes
        return

    score = 5.0  # base por haber recuperado contexto
    notes.append(f"Recuperó {result.chunks_used} chunks.")

    if case.expected_number:
        found_in_sources = any(
            _numbers_match(str(case.expected_number), str(s.get("number", ""))) for s in result.sources
        )
        found_in_answer = _appears_as_token(str(case.expected_number), result.answer)
        if found_in_sources:
            score += 3.0
            notes.append(f"OK: el número esperado ({case.expected_number}) está entre las fuentes recuperadas.")
        else:
            notes.append(f"ALERTA: el número esperado ({case.expected_number}) NO está entre las fuentes recuperadas.")
        if found_in_answer:
            score += 2.0
            notes.append("OK: la respuesta cita el número de norma.")
        else:
            notes.append("ALERTA: la respuesta no menciona el número de norma esperado.")
    else:
        score += 5.0  # sin número puntual para chequear, no penalizamos

    result.code_score = min(score, 10.0)
    result.code_notes = notes


# ─── MODEL GRADER (Claude como juez, razonamiento antes del puntaje) ───────

# Antes esto era un solo template con las instrucciones estáticas ANTES y
# DESPUÉS del contenido variable (pregunta/contexto/respuesta en el medio) —
# eso ni siquiera dejaba plantear caching bien: el prefijo cacheable tiene
# que ser 100% idéntico entre llamadas, y acá cambiaba en cada caso desde el
# primer bloque variable. Separado en instrucciones estáticas (_GRADER_SYSTEM,
# van al `system`) + contenido variable (va al `user`). Mismo caveat que en
# semantic_cleanup.py: el bloque estático mide ~280 tokens, por debajo del
# mínimo cacheable de Haiku (4096) — no genera cache hit hoy, pero deja la
# estructura correcta. El costo real de este grader no viene de instrucciones
# repetidas (son chicas) sino del CONTEXTO variable de cada caso (los chunks
# recuperados, que pueden ser miles de tokens) — eso no es cacheable porque
# es distinto en cada pregunta, es el costo real e irreducible de evaluar con
# el contexto completo en vez de solo metadata.
_GRADER_SYSTEM = """\
Sos un evaluador estricto de un sistema de RAG sobre normativa regulatoria. Te doy la \
pregunta de un usuario, el CONTEXTO REAL (el texto completo que el sistema recuperó y le \
pasó al modelo, no solo metadata), y la respuesta que generó. Evaluá la respuesta \
comparándola palabra por palabra contra ese contexto.

Primero escribí, en 2-4 líneas, fortalezas y debilidades de la respuesta: ¿todo dato \
concreto (números, fechas, valores técnicos) que menciona la respuesta aparece \
literalmente en el contexto de arriba, o los inventó? ¿cita la fuente correctamente? \
¿es clara y completa? Si el contexto no alcanza para responder algo y la respuesta lo \
inventa igual, marcalo como alucinación explícita, citando qué dato exacto no está \
respaldado.

Después, en una línea aparte, escribí el puntaje final del 1 al 10 en el formato exacto:
PUNTAJE: <número>
"""

_GRADER_USER_TEMPLATE = """\
<pregunta>
{question}
</pregunta>

<contexto_real_recuperado>
{context}
</contexto_real_recuperado>

<respuesta>
{answer}
</respuesta>
"""


_PUNTAJE_RE = re.compile(r"PUNTAJE\**:?\**\s*(\d+(?:\.\d+)?)", re.IGNORECASE)


def model_grade(client: Any, result: EvalResult) -> None:
    context_text = result.context.strip() or "(no se recuperó ningún contexto — el sistema no encontró documentos)"

    user_content = _GRADER_USER_TEMPLATE.format(
        question=result.case.question,
        context=context_text,
        answer=result.answer,
    )
    try:
        msg = client.messages.create(
            model=_GRADER_MODEL,
            max_tokens=1500,  # antes 500: con "razonamiento antes del puntaje" en casos
                              # ricos en contexto, el texto se cortaba ANTES de llegar a
                              # la línea "PUNTAJE: N" — eso hacía fallar el parseo en
                              # silencio (ver nota más abajo, esto se detecta con stop_reason)
            system=[
                {
                    "type": "text",
                    "text": _GRADER_SYSTEM,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_content}],
        )
        text = msg.content[0].text.strip()
        match = _PUNTAJE_RE.search(text)
        result.model_score = float(match.group(1)) if match else None
        result.model_reasoning = text
        if match is None:
            truncado = getattr(msg, "stop_reason", None) == "max_tokens"
            motivo = "se cortó por max_tokens antes de llegar al puntaje" if truncado else "no se encontró la línea PUNTAJE en la respuesta"
            logger.warning("Model grader sin puntaje parseable para %s (%s).", result.case.id, motivo)
            result.model_reasoning = f"[FALLO DE PARSEO: {motivo}]\n\n{text}"
    except Exception as exc:
        logger.warning("Error en model grader para %s: %s", result.case.id, exc)
        result.model_score = None
        result.model_reasoning = f"[ERROR DE API: {exc}]"


# ─── REPORTE ────────────────────────────────────────────────────────────────

def build_report(results: list[EvalResult]) -> str:
    lines = ["# Reporte de evals — chat RAG latam-reg-intel", ""]
    code_scores = [r.code_score for r in results]
    model_scores = [r.model_score for r in results if r.model_score is not None]
    fallidos = [r for r in results if r.model_score is None]

    lines.append(f"Casos evaluados: {len(results)}")
    lines.append(f"Promedio code grader: {sum(code_scores)/len(code_scores):.2f} / 10")
    if model_scores:
        lines.append(
            f"Promedio model grader: {sum(model_scores)/len(model_scores):.2f} / 10 "
            f"(calculado sobre {len(model_scores)}/{len(results)} casos — "
            f"{len(fallidos)} sin puntaje válido, ver detalle abajo)"
        )
    if fallidos:
        lines.append(
            f"⚠️ {len(fallidos)} caso(s) sin puntaje del model grader: "
            + ", ".join(r.case.id for r in fallidos)
        )
    lines.append("")

    for r in results:
        lines.append(f"## {r.case.id} ({r.case.kind}, fuente={r.case.source})")
        lines.append(f"**Pregunta:** {r.case.question}")
        lines.append("")
        lines.append(f"**Respuesta:** {r.answer[:500]}")
        lines.append("")
        lines.append(f"**Code grader:** {r.code_score}/10 — {' '.join(r.code_notes)}")
        if r.model_score is not None:
            lines.append(f"**Model grader:** {r.model_score}/10")
            lines.append(f"> {r.model_reasoning}")
        else:
            lines.append("**Model grader:** ⚠️ FALLÓ — sin puntaje válido (no promediado arriba)")
            lines.append(f"> {r.model_reasoning}")
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


# ─── MAIN ───────────────────────────────────────────────────────────────────

def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="Evals del chat RAG")
    parser.add_argument("--generate", type=int, default=0, help="Cantidad de casos adicionales a generar con Haiku")
    parser.add_argument("--report", type=Path, default=Path("eval_report.md"))
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: falta ANTHROPIC_API_KEY en el .env")
        sys.exit(1)

    import anthropic
    from chat.rag_engine import RAGEngine

    client = anthropic.Anthropic(api_key=api_key)

    cases = _load_seed_cases()
    if args.generate:
        print(f"Generando {args.generate} casos adicionales con Haiku...")
        cases += generate_cases_with_haiku(client, args.generate)

    print(f"Total de casos: {len(cases)}")
    print("Cargando motor RAG...")
    engine = RAGEngine()

    results = []
    for case in cases:
        print(f"  Corriendo: {case.id} ...")
        result = run_case(engine, case)
        code_grade(result)
        model_grade(client, result)
        results.append(result)

    report = build_report(results)
    args.report.write_text(report, encoding="utf-8")

    code_avg = sum(r.code_score for r in results) / len(results)
    model_scores = [r.model_score for r in results if r.model_score is not None]
    model_avg = sum(model_scores) / len(model_scores) if model_scores else None

    print("=" * 60)
    print(f"Promedio code grader:  {code_avg:.2f} / 10")
    if model_avg is not None:
        print(f"Promedio model grader: {model_avg:.2f} / 10")
    print(f"Reporte completo guardado en: {args.report}")
    print("=" * 60)


if __name__ == "__main__":
    main()
