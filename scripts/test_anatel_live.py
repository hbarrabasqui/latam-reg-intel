"""Prueba rápida del AnatelScraper contra el sitio real.
Limita a resoluciones 2024 para no sobrecargar el servidor.
"""
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)

from scrapers.brasil.anatel import AnatelScraper

scraper = AnatelScraper(
    years=[2024],
    doc_type_paths={"resolucoes": "resolucao"},
)

print("\n--- FETCH INDEX ---")
entries = scraper.fetch_index()
print(f"\nTotal entradas encontradas: {len(entries)}")

if entries:
    print("\nPrimeras 5:")
    for e in entries[:5]:
        print(f"  [{e['doc_type']}] #{e['number']:>5}  {e['title'][:70]}")
        print(f"           {e['url']}")

    print("\n--- FETCH DOCUMENT (primero) ---")
    doc = scraper.fetch_document(entries[0])
    print(f"  País:       {doc.country}")
    print(f"  Organismo:  {doc.organism}")
    print(f"  Tipo:       {doc.doc_type}")
    print(f"  Número:     {doc.number}")
    print(f"  Fecha pub:  {doc.date_published}")
    print(f"  Idioma:     {doc.language}")
    print(f"  URL fuente: {doc.source_url}")
    print(f"  Hash:       {doc.hash[:16]}...")
    print(f"  Texto ({len(doc.raw_text)} chars):")
    print(f"  {doc.raw_text[:300]}")
else:
    print("Sin entradas — revisar selectores CSS")
