"""Prueba rápida del EnacomScraper contra el sitio real.
Solo busca 1 tema, 1 página, y descarga el primer documento.
"""
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)

from scrapers.argentina.enacom import EnacomScraper

scraper = EnacomScraper(
    topics=["homologacion"],
    max_pages_per_topic=1,
    verify_ssl=False,  # enacom.gob.ar usa CA argentina no incluida en certifi
)

print("\n--- FETCH INDEX (1 pagina, tema: homologacion) ---")
entries = scraper.fetch_index()
print(f"\nTotal entradas encontradas: {len(entries)}")

if entries:
    print("\nPrimeras 5:")
    for e in entries[:5]:
        print(f"  [{e['doc_type']:15}] #{e['number']:>6}  {e['title'][:60]}")
        print(f"    Fecha firma: {e['date_published']}  |  BO: {e['date_bo']}")
        print(f"    Temas: {', '.join(e['temas'][:3])}")
        print(f"    PDFs: {len(e['pdf_urls'])}")

    # Buscar el primer entry con PDF para probarlo
    with_pdf = next((e for e in entries if e["pdf_urls"]), None)
    if with_pdf:
        print(f"\n--- FETCH DOCUMENT (primer doc con PDF) ---")
        print(f"  Descargando: {with_pdf['pdf_urls'][0]}")
        doc = scraper.fetch_document(with_pdf)
        print(f"  Pais:       {doc.country}")
        print(f"  Tipo:       {doc.doc_type}")
        print(f"  Numero:     {doc.number}")
        print(f"  Fecha pub:  {doc.date_published}")
        print(f"  Idioma:     {doc.language}")
        print(f"  Hash:       {doc.hash[:16]}...")
        print(f"  Texto ({len(doc.raw_text)} chars):")
        print(f"  {doc.raw_text[:400]}")
    else:
        print("\nNinguna entrada tiene PDF en esta pagina.")
        first = entries[0]
        doc = scraper.fetch_document(first)
        print(f"  Texto base (descripcion): {doc.raw_text[:200]}")
else:
    print("Sin entradas. Posiblemente el endpoint POST no funciona como se esperaba.")
    print("Revisar si el sitio usa AJAX puro o acepta POST directo.")
