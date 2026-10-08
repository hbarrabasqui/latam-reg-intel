"""Genera un diagrama Mermaid de la cadena de modificaciones/derogaciones
alrededor de una norma.

Usa los mismos datos que `chat/normativa_links.py` (metadata `modifica` /
`modificado_por`, ya scrapeados del sistema de vínculos de InfoLeg) — no hace
falta scrapear nada nuevo. Puro Python + SQLite, no requiere red ni la API de
Anthropic.

Uso:
    python -m scripts.plot_normativa_graph --numero 863/2020
    python -m scripts.plot_normativa_graph --id INFOLEG-275641
    python -m scripts.plot_normativa_graph --numero 863/2020 --hops 3 --out grafo.md
"""
from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _find_seed(store, numero: str | None, doc_id: str | None):
    if doc_id:
        doc = store.get_by_id(doc_id)
        if not doc:
            print(f"ERROR: no existe un documento con id={doc_id!r}")
            sys.exit(1)
        return doc

    # Búsqueda por número — puede haber más de un documento con el mismo
    # número (normas duplicadas en el corpus, ver ESTADO_SESION.md). Si hay
    # varios, se listan para que el usuario elija por id.
    matches = [d for d in store.get_approved() if (d.number or "").strip() == numero.strip()]
    if not matches:
        print(f"No se encontró ningún documento aprobado con número={numero!r}")
        sys.exit(1)
    if len(matches) > 1:
        print(f"Hay {len(matches)} documentos con número={numero!r} — elegí uno por --id:")
        for d in matches:
            print(f"  {d.id}  |  status={d.status}  |  {d.title[:80]}")
        sys.exit(1)
    return matches[0]


def _node_label(doc) -> str:
    status = (doc.status or "vigente").lower()
    marca = {"vigente": "", "derogada": " ⛔ derogada", "derogado": " ⛔ derogada", "modificada": " ✏ modificada"}.get(status, f" ({doc.status})")
    titulo = (doc.title or "")[:60].replace('"', "'")
    return f"{doc.organism} {doc.number}{marca}<br/>{titulo}"


def _mermaid_id(doc_id: str) -> str:
    """Mermaid no acepta guiones/espacios en los ids de nodo — se sanitiza."""
    return "n" + "".join(c if c.isalnum() else "_" for c in doc_id)


def build_mermaid(nodes: dict, edges: list[tuple[str, str, str]]) -> str:
    lines = ["graph TD"]
    for doc_id, doc in nodes.items():
        nid = _mermaid_id(doc_id)
        label = _node_label(doc).replace('"', "'")
        lines.append(f'    {nid}["{label}"]')

    for origen, destino, tipo in edges:
        estilo = "-->|deroga|" if tipo == "deroga" else "-.->|modifica|"
        lines.append(f"    {_mermaid_id(origen)} {estilo} {_mermaid_id(destino)}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagrama de vínculos modifica/deroga de una norma")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--numero", help='Número de norma, ej. "863/2020"')
    group.add_argument("--id", dest="doc_id", help="Id interno del documento (ej. INFOLEG-275641)")
    parser.add_argument("--hops", type=int, default=2, help="Saltos a explorar en cada dirección (default: 2)")
    parser.add_argument("--out", default=None, help="Archivo de salida .md (default: imprime a stdout)")
    args = parser.parse_args()

    from storage.relational import DocumentStore
    from chat.normativa_links import NormativaGraphBuilder

    store = DocumentStore()
    seed = _find_seed(store, args.numero, args.doc_id)

    builder = NormativaGraphBuilder(store)
    nodes, edges = builder.build(seed.id, max_hops=args.hops)

    mermaid = build_mermaid(nodes, edges)
    output = (
        f"# Cadena de modificaciones/derogaciones — {seed.organism} {seed.number}\n\n"
        f"{len(nodes)} norma(s), {len(edges)} vínculo(s) — línea sólida = deroga, "
        f"línea punteada = modifica.\n\n"
        f"```mermaid\n{mermaid}\n```\n"
    )

    if args.out:
        from pathlib import Path
        Path(args.out).write_text(output, encoding="utf-8")
        print(f"Guardado en {args.out}")
    else:
        print(output)


if __name__ == "__main__":
    main()
