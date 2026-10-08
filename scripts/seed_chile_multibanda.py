"""Carga los 2 documentos consolidados (texto refundido) de la normativa
Multibanda/SAE de Chile al corpus.

Contexto completo: ver `pipeline` hermano `scripts/fetch_chile_multibanda.py`
(que baja y limpia los .txt) y la sección "Plan preparado para Multibanda/SAE"
+ su seguimiento en ESTADO_SESION.md. Resumen: en vez de cargar las 10
resoluciones de la cadena (1.463/2016 base + 1.474/2016 SAE + 8
modificatorias) por separado, se cargan solo 2 — el "texto refundido"
(versión consolidada, con el historial de modificaciones ya incorporado
como notas al margen) de cada una, tal como los publica la Biblioteca del
Congreso Nacional (LeyChile) a través del micrositio oficial de SUBTEL
(multibanda.cl/normativa).

Dos nuncias importantes de vigencia, encontradas al revisar el texto real
(no asumidas, confirmadas leyendo el propio documento):

1. **1.463/2016**: el texto refundido bajado tiene como "Última Versión"
   el 18-DIC-2021 (Resolución 2298 EXENTA) — una modificación que NO
   aparece en la lista de 10 resoluciones de multibanda.cl (esa página
   dejó de actualizarse en 2018). O sea, el texto SÍ está actualizado
   (viene directo de LeyChile), pero la lista de resoluciones que veníamos
   usando como referencia estaba incompleta. No se encontró evidencia de
   modificaciones posteriores a diciembre 2021 al buscar, pero tampoco se
   pudo confirmar con 100% de certeza que no exista ninguna más reciente.

2. **1.474/2016 (SAE)**: el texto refundido bajado quedó "generado" el
   10-Jul-2017, reflejando modificaciones hasta la Resolución 1.179/2017
   inclusive. A diferencia del caso anterior, ACÁ SÍ se confirmó que está
   completo para su alcance: de las resoluciones modificatorias restantes
   de la cadena (2.350/2017, 947/2018, 1.372/2018, 2.672/2018, 17/2019),
   ninguna modifica la 1.474 — todas esas modifican únicammente la 1.463.
   Entonces este texto, aunque su fecha de generación es más vieja, está
   vigente y completo para el ámbito SAE específicamente.

Uso:
    python -m scripts.seed_chile_multibanda
    python -m scripts.seed_chile_multibanda --dry-run
"""
from __future__ import annotations

import argparse
import logging
from datetime import date, datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_SEED_DIR = Path("data") / "seed_chile" / "multibanda"

_DOCS = [
    {
        "id": "SUBTEL-1463-2016-multibanda",
        "file": "SUBTEL-1463-2016-refundido.txt",
        "doc_type": "resolucion_exenta",
        "number": "1463",
        "title": (
            "Resolución 1.463 Exenta — Fija Norma Técnica que Regula las "
            "Especificaciones Técnicas Mínimas que Deberán Cumplir los Equipos "
            "Terminales Utilizados en las Redes Móviles (texto refundido, "
            "última versión conocida al 18-12-2021)"
        ),
        "date_published": date(2016, 6, 16),
        "source_url": "https://bcn.cl/2ulpw",
        "status": "vigente",
        "seed_note": (
            "Texto refundido (LeyChile/BCN) con última modificación conocida "
            "18-12-2021 (Res. 2.298 Exenta) — no confirmada al 100% como la "
            "más reciente posible, ver docstring de este script."
        ),
    },
    {
        "id": "SUBTEL-1474-2016-SAE-multibanda",
        "file": "SUBTEL-1474-2016-SAE-refundido.txt",
        "doc_type": "resolucion_exenta",
        "number": "1474",
        "title": (
            "Resolución 1.474 Exenta — Modifica Resolución 3.261 Exenta de 2012 "
            "(Norma Técnica para el Sistema de Alerta de Emergencias — SAE — "
            "sobre las Redes de Servicio Público de Telefonía Móvil), texto "
            "refundido"
        ),
        "date_published": date(2016, 6, 22),
        "source_url": "https://www.leychile.cl/N?i=1091854&f=2017-09-23&p=",
        "status": "vigente",
        "seed_note": (
            "Texto refundido generado 10-07-2017, con modificaciones hasta la "
            "Res. 1.179/2017 inclusive. Confirmado que ninguna resolución "
            "posterior de la cadena Multibanda/SAE modifica específicamente "
            "esta norma (solo modifican la 1.463) — está completo para su "
            "alcance (SAE) pese a la fecha de generación más antigua."
        ),
    },
]


def run(dry_run: bool = False) -> None:
    from storage.classifier import DomainClassifier
    from storage.models import RegulatoryDocument
    from storage.relational import DocumentStore

    store = None if dry_run else DocumentStore()
    classifier = DomainClassifier()
    now = datetime.now(timezone.utc)

    for spec in _DOCS:
        txt_path = _SEED_DIR / spec["file"]
        if not txt_path.exists():
            logger.error("No encontré %s — abortando.", txt_path)
            return
        raw_text = txt_path.read_text(encoding="utf-8").strip()

        doc = RegulatoryDocument(
            id=spec["id"],
            country="CHL",
            organism="SUBTEL",
            doc_type=spec["doc_type"],
            number=spec["number"],
            title=spec["title"],
            date_published=spec["date_published"],
            date_scraped=now,
            source_url=spec["source_url"],
            raw_text=raw_text,
            language="es",
            status=spec["status"],
            version=1,
            hash=RegulatoryDocument.compute_hash(raw_text),
            metadata={
                "seed_source": "manual_chile_multibanda_2026-08-13",
                "seed_note": spec["seed_note"],
            },
            review_status="approved",
        )

        domain = classifier.classify(doc)
        logger.info(
            "%s %s → clasificador dice domain=%s (esperado: 'telecom')",
            doc.organism, doc.number, domain,
        )

        if dry_run:
            logger.info("[dry-run] no se guarda: %s (%d caracteres)", doc.id, len(raw_text))
            continue

        is_new = store.save(doc)
        logger.info("Guardado %s: %s", doc.id, "nuevo" if is_new else "ya existía (actualizado)")

    logger.info("=" * 60)
    if dry_run:
        logger.info("(dry-run: no se escribió nada. Repetir sin --dry-run para aplicar)")
    else:
        logger.info("Listo. Siguiente paso: python -m scripts.reindex")
    logger.info("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Semilla manual de Chile/Multibanda-SAE")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
