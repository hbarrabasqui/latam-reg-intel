"""CLI de chat con RAG sobre normativa regulatoria LATAM.

Uso:
    python -m scripts.chat                        # modo interactivo
    python -m scripts.chat --country ARG          # filtrar por país
    python -m scripts.chat --organism ANATEL      # filtrar por organismo
    python -m scripts.chat -q "¿Qué es la homologación?"  # pregunta directa
"""
from __future__ import annotations

import argparse
import io
import logging
import os
import sys

from dotenv import load_dotenv

# Forzar UTF-8 en la consola de Windows para soportar caracteres especiales
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

load_dotenv()

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


def _print_response(response: "RAGResponse", show_sources: bool = True) -> None:  # noqa: F821
    print(f"\n{response.answer}\n")
    if show_sources and response.sources:
        print("─" * 60)
        print("Fuentes consultadas:")
        for s in response.sources:
            print(
                f"  • {s['organism']} {s['number']} [{s['country']}]"
                f"  (relevancia: {s['score']})"
            )
            if s.get("source_url"):
                print(f"    {s['source_url']}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat RAG sobre normativa regulatoria LATAM")
    parser.add_argument("-q", "--query", help="Pregunta directa (sin modo interactivo)")
    parser.add_argument("--country", metavar="ISO3", help="Filtrar por país (ARG, BRA…)")
    parser.add_argument("--organism", help="Filtrar por organismo (ENACOM, ANATEL…)")
    parser.add_argument("--no-sources", action="store_true", help="No mostrar fuentes")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: falta ANTHROPIC_API_KEY en el archivo .env")
        sys.exit(1)

    from chat.rag_engine import RAGEngine

    print("Cargando motor RAG...")
    engine = RAGEngine()
    print("Listo.\n")

    if args.query:
        response = engine.ask(args.query, country=args.country, organism=args.organism)
        _print_response(response, show_sources=not args.no_sources)
        return

    # Modo interactivo
    print("=" * 60)
    print("  Chat RAG — Normativa Regulatoria LATAM")
    if args.country:
        print(f"  Filtro: país = {args.country}")
    if args.organism:
        print(f"  Filtro: organismo = {args.organism}")
    print("  Escribí tu pregunta. Comandos: /salir, /fuentes on|off")
    print("=" * 60)

    show_sources = True

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nHasta luego.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("/salir", "/exit", "/quit"):
            print("Hasta luego.")
            break
        if user_input.lower() == "/fuentes on":
            show_sources = True
            print("Fuentes activadas.")
            continue
        if user_input.lower() == "/fuentes off":
            show_sources = False
            print("Fuentes desactivadas.")
            continue

        try:
            response = engine.ask(user_input, country=args.country, organism=args.organism)
            _print_response(response, show_sources=show_sources)
        except Exception as exc:
            print(f"Error: {exc}")


if __name__ == "__main__":
    main()
