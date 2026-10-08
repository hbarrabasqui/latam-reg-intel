"""Segunda pasada de limpieza semántica sobre los documentos clasificados como
'telecom' por el clasificador de keywords (storage/classifier.py).

Por qué existe: el clasificador por keywords tiene un techo. Documentos de
otros rubros (tránsito, AFIP, penal) que mencionan de pasada "homologación"
(de un radar) o tienen un área interna llamada "telecomunicaciones" pasan el
puntaje sin ser realmente normativa de telecom. Un filtro de keywords no puede
distinguir eso — hace falta entender el contexto, y para eso el enfoque
correcto es un modelo, no más reglas (mismo concepto de "model based grading"
del curso de evals, aplicado acá a curar el corpus en vez de calificar
respuestas del chat).

Qué hace:
  1. Toma todos los documentos con review_status='approved' que el clasificador
     de keywords marca como 'telecom'.
  2. Los manda en lotes a Haiku (barato) pidiendo un juicio YES/NO sobre si el
     tema PRINCIPAL del documento es telecomunicaciones/radiocomunicaciones.
  3. Los que Haiku marca NO se pasan a review_status='rejected' — no hace
     falta tocar el clasificador de nuevo ni borrar nada, reindex.py ya
     respeta review_status y no los va a indexar.

Uso:
    python -m scripts.semantic_cleanup                    # aplica los cambios
    python -m scripts.semantic_cleanup --dry-run           # solo reporta
    python -m scripts.semantic_cleanup --recheck-rejected  # re-evalúa solo los que
                                                            # este mismo script rechazó
                                                            # antes (ej. después de
                                                            # ajustar el prompt/criterio)
    python -m scripts.semantic_cleanup --review-pending    # revisa documentos en
                                                            # review_status='pending_review'
                                                            # (ej. los que
                                                            # fix_seed_migration_review_status.py
                                                            # marcó como nunca revisados de
                                                            # verdad) — decide approved/rejected
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_MODEL = "claude-haiku-4-5"
# Subido de 12 a 30 (6/8/2026, corrección de prompt caching): cada llamada
# repite las mismas ~780 tokens de instrucciones de clasificación sin
# importar el tamaño del lote — duplicar el lote es la forma real de
# reducir ese gasto repetido (ver nota de cache_control más abajo sobre por
# qué el caching en sí no ayuda acá).
_BATCH_SIZE = 30
_CHECKPOINT_PATH = Path("data") / "semantic_cleanup_checkpoint.json"
_TIME_BUDGET_SECONDS = 35  # se corta solo antes de que lo mate un timeout externo


def _load_checkpoint() -> dict[str, bool]:
    if _CHECKPOINT_PATH.exists():
        return json.loads(_CHECKPOINT_PATH.read_text(encoding="utf-8"))
    return {}


def _save_checkpoint(data: dict[str, bool]) -> None:
    _CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CHECKPOINT_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

# NOTA sobre prompt caching (6/8/2026): estas instrucciones son idénticas en
# TODAS las llamadas de este script (una por lote), así que en teoría son un
# candidato perfecto para cache_control. En la práctica hoy NO generan cache
# hit: Haiku 4.5 exige un mínimo de 4096 tokens para cachear algo, y este
# bloque mide ~780 tokens — por debajo del piso, la API simplemente lo
# procesa como texto normal sin cachear (no rompe nada, tampoco cobra de
# más, pero no ahorra). Igual se separó en `system` (en vez de quedar
# mezclado con el bloque de documentos variable de cada lote) porque: (a) es
# la estructura correcta si este bloque crece más adelante y cruza el
# umbral, y (b) el ahorro real hoy no viene de acá sino de subir
# `_BATCH_SIZE` (ver arriba) — repetir estas ~780 tokens una vez cada 30
# documentos en vez de una vez cada 12 ya es la mayor parte del ahorro
# posible en este script.
_SYSTEM_INSTRUCTIONS = """\
Sos un clasificador experto en normativa de telecomunicaciones de Argentina (ENACOM \
y organismos predecesores: CNC, AFTIC). Te paso una lista de documentos legales. \
Para cada uno, decidí si corresponde a un acto regulatorio real de telecomunicaciones/ \
radiocomunicaciones.

Contá como SÍ (telecom) cualquiera de estos casos, SIN IMPORTAR qué industria vaya a \
usar el equipo o servicio final (medios, seguridad vial, transporte, salud, etc.) — lo \
que importa es que el acto regulatorio lo emite el área de homologaciones/espectro de \
ENACOM (o sus antecesores CNC/AFTIC), no la industria de destino:
- Homologación de un equipo para comercializarlo (el caso más común, >99% de los trámites
  reales de homologación).
- Autorización de uso propio de un equipo que emite en espectro radioeléctrico — este es
  un trámite DISTINTO de la homologación: lo pide quien va a usar el equipo para sí mismo,
  no para venderlo (ejemplo real: una empresa importa un transmisor de AM para su propia
  planta transmisora, no para comercializarlo — eso es "autorización", no "homologación",
  pero es igual de telecom). Este tipo de autorización de uso propio es distinta de la
  "autorización para prestar servicios radioeléctricos" (licencias de servicio), que es
  otro trámite aparte — ambas cuentan como SÍ si aparecen, pero no las confundas entre sí
  al razonar.
- Normas técnicas de equipos, espectro radioeléctrico, redes, licencias de
  telecomunicaciones.
- Acreditación/inscripción de un laboratorio en el Registro de Laboratorios
  Acreditados para realizar las mediciones/ensayos que se usan para homologar
  equipos de telecomunicaciones — es un trámite del área de Homologaciones de
  ENACOM, aunque el documento no homologue un equipo puntual sino que habilite
  a quien va a hacer las mediciones (ejemplo real: inscripción de un instituto
  de ensayos para medir según normas técnicas ENACOM-Q2-xx.xx).
- Regulación sobre inhibidores/bloqueadores de señal (dispositivos que
  interfieren la interconexión radioeléctrica): aclaración de dominio (fuente:
  Horacio, ex-ENACOM) — estos dispositivos NO se consideran "equipos de
  comunicaciones" y por lo tanto NO se homologan; su comercialización está
  prohibida en general, salvo excepciones puntuales para fuerzas de seguridad o
  defensa. Aun así, TODO documento sobre esta prohibición general, su
  fundamento, o una excepción/autorización puntual otorgada (ej. a una fuerza
  de seguridad para instalar bloqueadores) cuenta como SÍ — es regulación real
  del espectro radioeléctrico coordinada con ENACOM, aunque no sea un trámite
  de homologación.

Contá como NO solo cuando la mención de telecom es INCIDENTAL — no hay ningún acto real \
de homologación/autorización/espectro de por medio. Ejemplo real: una resolución de AFIP \
que en su organigrama menciona una "dirección de sistemas y telecomunicaciones" — ahí no \
hay ningún trámite de telecom, es solo el nombre de un área interna. Lo mismo con normas \
de tránsito, impositivas, penales o laborales que no tengan ningún pedido de \
homologación/autorización de equipo de por medio.

Respondé EXCLUSIVAMENTE con un JSON array, sin texto adicional, con este formato:
[{"idx": 0, "es_telecom": true}, {"idx": 1, "es_telecom": false}, ...]
"""


def _build_excerpt(raw_text: str) -> str:
    """Arma el excerpt que ve Haiku. Ver `pipeline/excerpt.py` para el
    detalle de por qué hace falta esto (preámbulos largos de VISTO/
    CONSIDERANDO que esconden el contenido sustantivo). Este script solo le
    pone los términos fuertes de telecom como señal de dónde buscar."""
    from pipeline.excerpt import build_salient_excerpt
    from storage.classifier import _TELECOM_STRONG_SIGNALS

    return build_salient_excerpt(raw_text, _TELECOM_STRONG_SIGNALS)


def _format_docs_block(batch: list[tuple[int, "object"]]) -> str:
    parts = []
    for idx, doc in batch:
        excerpt = _build_excerpt(doc.raw_text)
        parts.append(
            f"[{idx}] {doc.organism} {doc.number} ({doc.doc_type}) — {doc.title[:150]}\n"
            f"    Texto: {excerpt}"
        )
    return "\n\n".join(parts)


def _get_rejected_docs(store) -> list:
    """Documentos con review_status='rejected', sin importar quién los rechazó
    (este script o una revisión manual con scripts/review_queue.py)."""
    from storage.relational import _connect, _row_to_doc
    with _connect(store.db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM documents WHERE review_status = 'rejected' ORDER BY date_scraped DESC"
        ).fetchall()
    return [_row_to_doc(r) for r in rows]


def _get_pending_docs(store) -> list:
    """Documentos con review_status='pending_review'. Se procesan igual que
    los candidatos normales (mismo prompt, mismo juez), pero NO se filtran
    primero por `classifier.classify(d) == "telecom"` — por definición ya
    fallaron ese filtro (así llegaron a pending_review, ver
    `scripts/fix_seed_migration_review_status.py`), y es justamente lo que
    hay que resolver con el juez semántico en vez de descartarlos a ciegas."""
    return store.get_pending_review()


def run(dry_run: bool = False, recheck_rejected: bool = False, review_pending: bool = False) -> None:
    from dotenv import load_dotenv
    load_dotenv()

    import anthropic
    from storage.classifier import DomainClassifier
    from storage.relational import DocumentStore

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("Falta ANTHROPIC_API_KEY en el .env")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    store = DocumentStore()
    classifier = DomainClassifier()

    checkpoint = _load_checkpoint()

    if recheck_rejected:
        rejected_docs = _get_rejected_docs(store)
        # Solo los que ESTE script rechazó antes (checkpoint[id] == False) — no
        # tocar rechazos manuales hechos con scripts/review_queue.py, esos son
        # decisión humana y no deben revertirse solos por un cambio de prompt.
        candidates = [d for d in rejected_docs if checkpoint.get(d.id) is False]
        logger.info(
            "Recheck: %d documentos rechazados en total, %d de ellos por este script "
            "(se re-evalúan con el criterio actualizado).",
            len(rejected_docs), len(candidates),
        )
        # Sacarlos del checkpoint para forzar que se reprocesen con el prompt nuevo.
        for d in candidates:
            checkpoint.pop(d.id, None)
    elif review_pending:
        candidates = _get_pending_docs(store)
        logger.info(
            "Revisión de pendientes: %d documentos en review_status='pending_review' "
            "(fallan el filtro de keywords, nunca fueron revisados de verdad).",
            len(candidates),
        )
    else:
        docs = store.get_approved()
        candidates = [d for d in docs if classifier.classify(d) == "telecom"]
        logger.info("Candidatos a revisar (pasaron el filtro de keywords): %d", len(candidates))

    pending = [d for d in candidates if d.id not in checkpoint]
    logger.info("Ya resueltos en corridas anteriores: %d | Pendientes: %d", len(candidates) - len(pending), len(pending))

    # Los contadores se escalan solo a los ids de ESTE run (candidate_ids), no a todo
    # el checkpoint acumulado histórico — si no, --recheck-rejected (que comparte el
    # mismo archivo de checkpoint con corridas normales muchos más grandes) mostraría
    # "confirmados"/"rechazados" contando cosas ajenas a esta tanda.
    candidate_ids = {d.id for d in candidates}
    confirmados = sum(1 for cid, v in checkpoint.items() if cid in candidate_ids and v)
    rechazados = sum(1 for cid, v in checkpoint.items() if cid in candidate_ids and not v)
    errores = 0
    start = time.monotonic()

    for i in range(0, len(pending), _BATCH_SIZE):
        if time.monotonic() - start > _TIME_BUDGET_SECONDS:
            logger.info("Presupuesto de tiempo agotado para esta corrida, cortando acá (se puede resumir).")
            break

        batch = list(enumerate(pending[i:i + _BATCH_SIZE]))
        docs_block = _format_docs_block(batch)

        try:
            message = client.messages.create(
                model=_MODEL,
                max_tokens=2000,  # subido de 1000: lote más grande (30 en vez de 12) => más JSON de salida
                system=[
                    {
                        "type": "text",
                        "text": _SYSTEM_INSTRUCTIONS,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": f"Documentos:\n{docs_block}"}],
            )
            raw = message.content[0].text.strip()
            # Por si Haiku envuelve el JSON en ```json ... ```
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()
            results = json.loads(raw)
        except Exception as exc:
            logger.warning("Error en lote %d-%d: %s", i, i + len(batch), exc)
            errores += len(batch)
            continue

        results_by_idx = {r["idx"]: r["es_telecom"] for r in results}
        for idx, doc in batch:
            es_telecom = results_by_idx.get(idx)
            if es_telecom is None:
                logger.warning("Sin resultado para idx %d (%s %s)", idx, doc.organism, doc.number)
                errores += 1
                continue
            if es_telecom:
                confirmados += 1
                if recheck_rejected or review_pending:
                    verbo = "RESCATARÍA" if dry_run else "RESCATADO"
                    logger.info(
                        "%s por juez semántico (criterio actualizado): %s %s — %s",
                        verbo, doc.organism, doc.number, doc.title[:80],
                    )
                    if not dry_run:
                        store.set_review_status(doc.id, "approved")
            else:
                rechazados += 1
                verbo = "RECHAZARÍA" if dry_run else "RECHAZADO"
                logger.info(
                    "%s por juez semántico: %s %s — %s",
                    verbo, doc.organism, doc.number, doc.title[:80],
                )
                if not dry_run:
                    store.set_review_status(doc.id, "rejected")

            # Importante: en dry-run NO se toca el checkpoint. Si se guardara acá iría
            # a parar al archivo aunque no se haya cambiado nada en la base, y una
            # corrida real posterior (sin --dry-run) ya no vería estos documentos como
            # pendientes — con --recheck-rejected eso significa que quedarían
            # rechazados para siempre sin ninguna corrida real que los rescate. Bug
            # real encontrado antes de correrlo, gracias a la revisión de Code.
            if not dry_run:
                checkpoint[doc.id] = bool(es_telecom)

        if not dry_run:
            _save_checkpoint(checkpoint)
        logger.info(
            "Progreso: %d/%d revisados (confirmados=%d, rechazados=%d, errores=%d)",
            confirmados + rechazados + errores, len(candidates),
            confirmados, rechazados, errores,
        )

    total_revisados = confirmados + rechazados + errores
    logger.info("=" * 60)
    logger.info("RESUMEN LIMPIEZA SEMÁNTICA (corrida actual)")
    logger.info("Candidatos totales:    %d", len(candidates))
    logger.info("Revisados esta corrida: %d", total_revisados)
    logger.info("Confirmados telecom:   %d", confirmados)
    logger.info("Rechazados (ruido):    %d", rechazados)
    logger.info("Errores:               %d", errores)
    if total_revisados < len(candidates):
        logger.info("Faltan %d — volver a correr el mismo comando para continuar.", len(candidates) - total_revisados)
    if dry_run:
        logger.info("(dry-run: no se cambió review_status de nada NI el checkpoint — se puede correr sin --dry-run después con los mismos candidatos)")
    logger.info("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Limpieza semántica del corpus telecom con Haiku")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--recheck-rejected",
        action="store_true",
        help="Re-evalúa solo los documentos que este script rechazó antes, con el criterio actual del prompt.",
    )
    parser.add_argument(
        "--review-pending",
        action="store_true",
        help="Revisa documentos en review_status='pending_review' (ej. marcados por "
             "fix_seed_migration_review_status.py) y decide approved/rejected.",
    )
    args = parser.parse_args()
    if args.recheck_rejected and args.review_pending:
        parser.error("--recheck-rejected y --review-pending son mutuamente excluyentes.")
    run(dry_run=args.dry_run, recheck_rejected=args.recheck_rejected, review_pending=args.review_pending)


if __name__ == "__main__":
    main()
