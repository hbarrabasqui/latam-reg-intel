"""Carga "notas interpretativas" al corpus — un tipo de documento nuevo, distinto
de una norma oficial.

Por qué existen: hay preguntas cuya respuesta correcta no está en un solo
documento, sino que requiere cruzar varias resoluciones y entender qué
derogó/sustituyó/dejó vigente cada una — el tipo de análisis que en esta
misma sesión llevó varias idas y vueltas (y un error real de mi parte,
mezclando el Capítulo V/STeFI con la Banda 5 de 3GPP) para llegar a la
respuesta correcta. Pedirle al modelo que rearme ese cruce en vivo, cada
vez que alguien pregunta, es propenso a error. Una nota interpretativa es
un documento donde ese análisis ya se hizo una vez, se validó con Horacio
(dominio real, ex-ENACOM), y queda disponible para recuperar directo.

Diferencia importante con una norma oficial: esto NO es un documento
publicado por ENACOM/SUBTEL — es un análisis propio. El campo
`doc_type="nota_interpretativa"` lo marca así, y el system prompt de
`chat/rag_engine.py` tiene una regla para que el bot, al citar una nota de
este tipo, aclare que es una interpretación validada por Horacio y no una
publicación oficial, y siga citando los números de norma reales en los que
se basa (eso si están en el propio texto de la nota).

Uso:
    python -m scripts.seed_notas_interpretativas
    python -m scripts.seed_notas_interpretativas --dry-run
"""
from __future__ import annotations

import argparse
import logging
from datetime import date, datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_NOTAS_DIR = Path("data") / "notas_interpretativas"

_NOTAS = [
    {
        "id": "NOTA-ARG-fcc-terminales-moviles",
        "file": "ARG-fcc-terminales-moviles.txt",
        "country": "ARG",
        "organism": "ENACOM",
        "title": (
            "Nota interpretativa — por qué se acepta ensayo de laboratorio extranjero "
            "(FCC u organismo similar) para homologar terminales móviles (2G/3G/4G) "
            "aunque existan laboratorios argentinos acreditados"
        ),
        "date_published": date(2026, 8, 13),  # fecha de esta nota, no de una norma
    },
    {
        "id": "NOTA-ARG-cabfra-banda-5900-mhz-its-v2x",
        "file": "ARG-cabfra-banda-5900-mhz-its-v2x.txt",
        "country": "ARG",
        "organism": "ENACOM",
        "title": (
            "Nota interpretativa — Sistemas de Transporte Inteligentes (STI/ITS) en "
            "Argentina: radares STI (24GHz/76-77GHz, con norma técnica ENACOM-Q2-64.01) "
            "vs. comunicaciones C-ITS/V2X (5,9GHz, sin atribución en el CABFRA — vía "
            "disponible: permiso experimental Res. ENACOM 1133/2023)"
        ),
        "date_published": date(2026, 9, 2),  # fecha de esta nota, no de una norma
        # Validada por Horacio el 2/9/2026 (ver cierre del propio texto de la nota):
        # confirma que el CABFRA no tiene atribución específica para C-ITS/V2X, y
        # aporta contexto de dominio (el CABFRA es un instrumento viejo, ENACOM no
        # lo actualiza con frecuencia). Ampliada el mismo día con la Norma Técnica
        # ENACOM-Q2-64.01 (radares STI), encontrada en el corpus ya cargado.
        # review_status default = "approved".
    },
]


def run(dry_run: bool = False) -> None:
    from storage.models import RegulatoryDocument
    from storage.relational import DocumentStore

    store = None if dry_run else DocumentStore()
    now = datetime.now(timezone.utc)

    for spec in _NOTAS:
        txt_path = _NOTAS_DIR / spec["file"]
        if not txt_path.exists():
            logger.error("No encontré %s — abortando.", txt_path)
            return
        raw_text = txt_path.read_text(encoding="utf-8").strip()

        doc = RegulatoryDocument(
            id=spec["id"],
            country=spec["country"],
            organism=spec["organism"],
            doc_type="nota_interpretativa",
            number="",  # no es una norma, no tiene número de resolución propio
            title=spec["title"],
            date_published=spec["date_published"],
            date_scraped=now,
            source_url="",  # no tiene URL propia — las fuentes reales están citadas dentro del texto
            raw_text=raw_text,
            language="es",
            status="vigente",
            version=1,
            hash=RegulatoryDocument.compute_hash(raw_text),
            metadata={
                "seed_source": f"nota_interpretativa_manual_{spec['date_published'].isoformat()}",
                "seed_note": (
                    "Análisis propio (Claude + Horacio, ex-ENACOM) cruzando varias "
                    "resoluciones — no es una norma oficial. Ver detalle de origen al "
                    "final del propio texto de la nota."
                ),
            },
            review_status=spec.get("review_status", "approved"),
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
    parser = argparse.ArgumentParser(description="Carga de notas interpretativas al corpus")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
