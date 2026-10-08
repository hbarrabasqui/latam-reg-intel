"""Detecta relaciones de modificación/derogación entre los documentos que
aparecen juntos en el contexto de una respuesta del RAG.

Por qué existe: en los evals se encontraron casos reales donde el modelo
generador mezcla contenido entre una norma vigente y la norma anterior que esa
misma resolución deroga/modifica y reemplaza (ej. Resolución 863/2020 vs
4.479/2017) — porque ambas son temáticamente parecidas, ambas terminan en el
contexto recuperado, y no hay ninguna señal de cuál es la vigente. El dato para
resolver esto YA EXISTE en el corpus (campo `status` de cada documento —
vigente/modificada/derogada — y las listas `modifica`/`modificado_por` en
metadata, sacadas del propio sistema de vínculos de InfoLeg) pero hasta ahora
no se usaba en tiempo de generación. Este módulo cruza esos datos SOLO entre
los documentos que efectivamente están en el contexto actual — no tiene
sentido anotar una relación hacia una norma que el modelo ni siquiera va a ver.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _as_int_list(value: Any) -> list[int]:
    if not value:
        return []
    result = []
    for v in value:
        try:
            result.append(int(v))
        except (TypeError, ValueError):
            continue
    return result


def build_infoleg_index(store: Any) -> dict[int, list[str]]:
    """Mapea infoleg_id -> lista de ids internos de documento.

    Uno-a-MUCHOS a propósito: el corpus tiene normas duplicadas (mismo
    infoleg_id scrapeado más de una vez por caminos distintos). Compartido por
    `NormativaLinkResolver` (anotaciones en el contexto del chat) y
    `NormativaGraphBuilder` (diagramas — ver scripts/plot_normativa_graph.py)."""
    index: dict[int, list[str]] = {}
    for doc in store.get_approved():
        infoleg_id = (doc.metadata or {}).get("infoleg_id")
        if infoleg_id is None:
            continue
        try:
            key = int(infoleg_id)
        except (TypeError, ValueError):
            continue
        index.setdefault(key, []).append(doc.id)
    logger.debug("Índice infoleg_id -> doc_ids construido: %d claves", len(index))
    return index


class NormativaLinkResolver:
    """Resuelve notas de vigencia/derogación para un conjunto de documentos
    recuperados en una misma búsqueda.

    Uso:
        resolver = NormativaLinkResolver(document_store)
        notas = resolver.resolve(["doc_id_1", "doc_id_2", ...])
        # notas["doc_id_1"] -> lista de strings, ej.:
        #   ["Este documento fue modificado por ENACOM 863/2020 (presente en este contexto)."]
    """

    def __init__(self, store: Any) -> None:
        self._store = store
        self._infoleg_id_to_doc_ids: dict[int, list[str]] | None = None

    def resolve(self, doc_ids: list[str]) -> dict[str, list[str]]:
        if self._infoleg_id_to_doc_ids is None:
            self._infoleg_id_to_doc_ids = build_infoleg_index(self._store)

        unique_ids = list(dict.fromkeys(doc_ids))  # preserva orden, sin duplicados
        docs = {d: self._store.get_by_id(d) for d in unique_ids}
        docs = {d: doc for d, doc in docs.items() if doc is not None}

        notes: dict[str, list[str]] = {d: [] for d in unique_ids}
        seen_pairs: set[tuple[str, str]] = set()

        for doc_id, doc in docs.items():
            meta = doc.metadata or {}
            # "modifica"/"modifica_a" según venga del scraper en vivo o de la
            # migración histórica — mismo significado, nombre distinto.
            targets = _as_int_list(meta.get("modifica")) + _as_int_list(meta.get("modifica_a"))

            for target_infoleg_id in targets:
                # Puede haber más de una copia del documento afectado en el
                # corpus — probar TODAS las que coincidan con este infoleg_id
                # y anotar la(s) que efectivamente estén en este contexto.
                candidate_doc_ids = self._infoleg_id_to_doc_ids.get(target_infoleg_id, [])
                target_doc_ids_in_context = [c for c in candidate_doc_ids if c in docs and c != doc_id]

                for target_doc_id in target_doc_ids_in_context:
                    pair = (doc_id, target_doc_id)
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)

                    target = docs[target_doc_id]
                    is_derogada = "derogad" in (target.status or "").lower()
                    verbo_pasivo = "derogada" if is_derogada else "modificada"
                    verbo_activo = "derogó" if is_derogada else "modificó"

                    notes[doc_id].append(
                        f"Esta norma {verbo_activo} a {target.organism} {target.number} "
                        f"(también presente en este contexto)."
                    )
                    notes[target_doc_id].append(
                        f"Esta norma fue {verbo_pasivo} por {doc.organism} {doc.number} "
                        f"(también presente en este contexto)."
                    )

        # Aviso más genérico (sin identificar la norma específica) para cuando
        # el documento tiene status != vigente pero su reemplazo/modificación
        # no está en este contexto puntual — igual vale la pena avisar que no
        # es necesariamente la versión vigente.
        for doc_id, doc in docs.items():
            if notes[doc_id]:
                continue  # ya tiene una nota específica, no hace falta la genérica
            status = (doc.status or "vigente").lower()
            if status != "vigente":
                notes[doc_id].append(
                    f"Atención: el estado registrado de esta norma es '{doc.status}', "
                    f"no necesariamente vigente en su totalidad — no se identificó en este "
                    f"contexto puntual qué norma la modificó/derogó."
                )

        return notes


class NormativaGraphBuilder:
    """Arma el grafo de modificaciones/derogaciones alrededor de una norma,
    recorriendo modifica/modificado_por más allá de un único resultado de
    búsqueda (a diferencia de `NormativaLinkResolver`, que solo cruza vínculos
    DENTRO de un conjunto de documentos ya dado). Pensado para generar
    diagramas — ver scripts/plot_normativa_graph.py — no se usa en el chat.

    Uso:
        builder = NormativaGraphBuilder(document_store)
        nodos, aristas = builder.build(seed_doc_id, max_hops=2)
        # nodos: dict doc_id -> RegulatoryDocument
        # aristas: list[(origen_doc_id, destino_doc_id, tipo)]
        #          origen modifica/deroga a destino; tipo es "deroga" o "modifica"
    """

    def __init__(self, store: Any) -> None:
        self._store = store
        self._infoleg_index: dict[int, list[str]] | None = None

    def _index(self) -> dict[int, list[str]]:
        if self._infoleg_index is None:
            self._infoleg_index = build_infoleg_index(self._store)
        return self._infoleg_index

    def build(
        self, seed_doc_id: str, max_hops: int = 2, max_nodes: int = 30
    ) -> tuple[dict[str, Any], list[tuple[str, str, str]]]:
        index = self._index()

        seed = self._store.get_by_id(seed_doc_id)
        if seed is None:
            raise ValueError(f"No existe un documento con id={seed_doc_id!r}")

        nodes: dict[str, Any] = {seed_doc_id: seed}
        edges: set[tuple[str, str, str]] = set()
        explored: set[str] = set()
        frontier: set[str] = {seed_doc_id}

        for _hop in range(max_hops):
            if not frontier or len(nodes) >= max_nodes:
                break
            new_frontier: set[str] = set()

            for doc_id in frontier:
                if doc_id in explored:
                    continue
                explored.add(doc_id)
                doc = nodes[doc_id]
                meta = doc.metadata or {}

                # doc_id -> target : doc_id modifica/deroga a target
                modifica_ids = _as_int_list(meta.get("modifica")) + _as_int_list(meta.get("modifica_a"))
                for target_infoleg_id in modifica_ids:
                    for target_doc_id in index.get(target_infoleg_id, []):
                        if target_doc_id == doc_id:
                            continue
                        target_doc = nodes.get(target_doc_id) or self._store.get_by_id(target_doc_id)
                        if target_doc is None:
                            continue
                        nodes[target_doc_id] = target_doc
                        tipo = "deroga" if "derogad" in (target_doc.status or "").lower() else "modifica"
                        edges.add((doc_id, target_doc_id, tipo))
                        if target_doc_id not in explored:
                            new_frontier.add(target_doc_id)

                # source -> doc_id : source modifica/deroga a doc_id
                modificado_por_ids = _as_int_list(meta.get("modificado_por")) + _as_int_list(meta.get("modificada_por"))
                for source_infoleg_id in modificado_por_ids:
                    for source_doc_id in index.get(source_infoleg_id, []):
                        if source_doc_id == doc_id:
                            continue
                        source_doc = nodes.get(source_doc_id) or self._store.get_by_id(source_doc_id)
                        if source_doc is None:
                            continue
                        nodes[source_doc_id] = source_doc
                        tipo = "deroga" if "derogad" in (doc.status or "").lower() else "modifica"
                        edges.add((source_doc_id, doc_id, tipo))
                        if source_doc_id not in explored:
                            new_frontier.add(source_doc_id)

                if len(nodes) >= max_nodes:
                    break

            frontier = new_frontier

        return nodes, sorted(edges)
