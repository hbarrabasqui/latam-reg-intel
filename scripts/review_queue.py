"""CLI para revisar documentos en cola de revisión.

Uso:
    python -m scripts.review_queue --list
    python -m scripts.review_queue --list --country ARG
    python -m scripts.review_queue --approve <id>
    python -m scripts.review_queue --reject  <id>
    python -m scripts.review_queue --interactive
"""
from __future__ import annotations

import argparse
import logging
import sys
import textwrap

from storage.relational import DocumentStore
from storage.models import RegulatoryDocument

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


# ── Presentación ────────────────────────────────────────────────────────────

def _fmt_doc(doc: RegulatoryDocument, index: int | None = None) -> str:
    prefix = f"[{index}] " if index is not None else ""
    date_str = doc.date_published.isoformat()
    text_preview = textwrap.shorten(doc.raw_text.replace("\n", " "), width=120, placeholder="…")
    return (
        f"{prefix}{doc.organism} {doc.number or '—'}  |  {date_str}  |  [{doc.country}]\n"
        f"    Título : {doc.title}\n"
        f"    URL    : {doc.source_url}\n"
        f"    ID     : {doc.id}\n"
        f"    Texto  : {text_preview}\n"
    )


def _print_list(docs: list[RegulatoryDocument], review_status: str) -> None:
    if not docs:
        print(f"No hay documentos con estado '{review_status}'.")
        return
    print(f"\n{'─'*70}")
    print(f"  {len(docs)} documento(s) con estado '{review_status}'")
    print(f"{'─'*70}\n")
    for i, doc in enumerate(docs, 1):
        print(_fmt_doc(doc, index=i))


# ── Modos ────────────────────────────────────────────────────────────────────

def cmd_list(store: DocumentStore, country: str | None, review_status: str) -> None:
    if review_status == "pending_review":
        docs = store.get_pending_review()
    elif review_status == "approved":
        docs = store.get_approved()
    else:
        # rejected: query directa
        from storage.relational import _connect, _row_to_doc
        with _connect(store.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM documents WHERE review_status = 'rejected' ORDER BY date_scraped DESC"
            ).fetchall()
        docs = [_row_to_doc(r) for r in rows]

    if country:
        docs = [d for d in docs if d.country == country]
    _print_list(docs, review_status)

    # Resumen general siempre visible
    print(
        f"  Total en base: {store.count()}  |  "
        f"pending_review: {store.count(review_status='pending_review')}  |  "
        f"approved: {store.count(review_status='approved')}  |  "
        f"rejected: {store.count(review_status='rejected')}\n"
    )


def cmd_approve(store: DocumentStore, doc_id: str) -> None:
    doc = store.get_by_id(doc_id)
    if not doc:
        print(f"ERROR: no se encontró documento con id={doc_id!r}")
        sys.exit(1)
    store.set_review_status(doc_id, "approved")
    print(f"✓ Aprobado: {doc.organism} {doc.number} — {doc.title}")


def cmd_reject(store: DocumentStore, doc_id: str) -> None:
    doc = store.get_by_id(doc_id)
    if not doc:
        print(f"ERROR: no se encontró documento con id={doc_id!r}")
        sys.exit(1)
    store.set_review_status(doc_id, "rejected")
    print(f"✗ Rechazado: {doc.organism} {doc.number} — {doc.title}")


def cmd_interactive(store: DocumentStore) -> None:
    """Revisión interactiva: muestra cada documento pendiente y pide decisión."""
    docs = store.get_pending_review()
    if not docs:
        print("No hay documentos pendientes de revisión.")
        return

    print(f"\n{'═'*70}")
    print(f"  REVISIÓN INTERACTIVA — {len(docs)} documento(s) pendiente(s)")
    print(f"  Comandos: [a] aprobar  [r] rechazar  [s] saltear  [q] salir")
    print(f"{'═'*70}\n")

    approved = rejected = skipped = 0

    for doc in docs:
        print(_fmt_doc(doc))
        while True:
            choice = input("  > ").strip().lower()
            if choice in ("a", "aprobar"):
                store.set_review_status(doc.id, "approved")
                print("  ✓ Aprobado\n")
                approved += 1
                break
            elif choice in ("r", "rechazar"):
                store.set_review_status(doc.id, "rejected")
                print("  ✗ Rechazado\n")
                rejected += 1
                break
            elif choice in ("s", "saltear", ""):
                print("  → Salteado\n")
                skipped += 1
                break
            elif choice in ("q", "salir"):
                print(f"\nSesión terminada. Aprobados: {approved} | Rechazados: {rejected} | Salteados: {skipped}")
                return
            else:
                print("  Opción inválida. Usá: a / r / s / q")

    print(f"\nListo. Aprobados: {approved} | Rechazados: {rejected} | Salteados: {skipped}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Cola de revisión de documentos regulatorios")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="Listar documentos")
    group.add_argument("--approve", metavar="ID", help="Aprobar documento por ID")
    group.add_argument("--reject",  metavar="ID", help="Rechazar documento por ID")
    group.add_argument("--interactive", action="store_true", help="Revisión interactiva de pendientes")

    parser.add_argument("--country", metavar="ISO3", help="Filtrar por país (ARG, BRA…)")
    parser.add_argument(
        "--status",
        default="pending_review",
        choices=["pending_review", "approved", "rejected"],
        help="Estado a listar (default: pending_review)",
    )

    args = parser.parse_args()
    store = DocumentStore()

    if args.list:
        cmd_list(store, args.country, args.status)
    elif args.approve:
        cmd_approve(store, args.approve)
    elif args.reject:
        cmd_reject(store, args.reject)
    elif args.interactive:
        cmd_interactive(store)


if __name__ == "__main__":
    main()
