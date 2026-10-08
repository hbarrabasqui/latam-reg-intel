"""Carga la Resolución ENACOM 843/2026 — prorroga la entrada en vigencia del
RAMATEL (Reglamento aprobado por Resolución 57/2026) del 1° de septiembre al
1° de diciembre de 2026, y con ella también la derogación de las resoluciones
del régimen anterior (Res. SC 729/80 y las demás listadas en su Art. 16).

Subida directamente por Horacio el 27/8/2026 (PDF, texto pre-publicación vía
GDE) — no viene de un scraper, se carga a mano por su relevancia ("sumamente
importante", dicho por Horacio).

Importante para el corpus existente: la Resolución 57/2026 ya está cargada
dos veces (duplicado conocido, ver ESTADO_SESION.md) — una copia scrapeada
del sitio de ENACOM (id c069f0da-..., sin infoleg_id) y otra migrada de
InfoLeg (id INFOLEG-423392, con infoleg_id=423392 real). Esta segunda copia
YA tiene metadata `modifica_a` poblada — así que para que el sistema de
vínculos (`chat/normativa_links.py`) detecte automáticamente que esta nueva
resolución modifica a la 57/2026 (y muestre el aviso "⚠" de vigencia cuando
ambas aparezcan juntas en una búsqueda), alcanza con declarar en la metadata
de este documento nuevo `modifica_a: "423392"` — no hace falta tocar ningún
documento existente.

Uso:
    python -m scripts.seed_res_843_2026 --dry-run
    python -m scripts.seed_res_843_2026
"""
from __future__ import annotations

import argparse
import logging
from datetime import date, datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_TXT_PATH = Path("data") / "tramites_empresas" / "ARG-resolucion-843-2026-prorroga-ramatel.txt"


def run(dry_run: bool = False) -> None:
    from storage.models import RegulatoryDocument
    from storage.relational import DocumentStore

    if not _TXT_PATH.exists():
        logger.error("No encontré %s — abortando.", _TXT_PATH)
        return
    raw_text = _TXT_PATH.read_text(encoding="utf-8").strip()

    doc = RegulatoryDocument(
        id="ARG-resolucion-843-2026-prorroga-ramatel",
        country="ARG",
        organism="ENACOM",
        doc_type="resolucion",
        number="843/2026",
        title=(
            "Resolución ENACOM 843/2026 — Prorroga la entrada en vigencia del "
            "RAMATEL (Res. 57/2026) del 1° de septiembre al 1° de diciembre de 2026"
        ),
        date_published=date(2026, 8, 26),
        date_scraped=datetime.now(timezone.utc),
        source_url="",  # cargada a mano desde el PDF que subió Horacio, no scrapeada
        raw_text=raw_text,
        language="es",
        status="vigente",
        version=1,
        hash=RegulatoryDocument.compute_hash(raw_text),
        metadata={
            "seed_source": "manual_upload_horacio_2026-08-27",
            "seed_note": (
                "PDF subido directamente por Horacio, texto pre-publicación vía GDE. "
                "Prorroga los Arts. 13 y 16 de la Resolución ENACOM 57/2026."
            ),
            # infoleg_id sintético (negativo, no colisiona con IDs reales de InfoLeg,
            # todos positivos) — permite que otro documento futuro pueda referenciar
            # a este vía modifica_a/modificado_por si hiciera falta.
            "infoleg_id": -843,
            # Apunta al infoleg_id REAL de la copia migrada de la Resolución 57/2026
            # (INFOLEG-423392) — así el resolver de vínculos los conecta sin tocar
            # ningún documento existente. OJO: usar LISTA de int, no string — el
            # parser `_as_int_list` de chat/normativa_links.py itera el valor
            # carácter por carácter si es string, así que "423392" (string) se
            # rompe en dígitos sueltos en vez de leerse como un solo número. Bug
            # preexistente del proyecto, no arreglado todavía (ver ESTADO_SESION.md
            # del 27/8/2026) — acá lo esquivamos usando el formato lista, que sí
            # funciona bien.
            "modifica": [423392],
        },
        review_status="approved",  # ya revisado, fuente directa del organismo
    )

    if dry_run:
        logger.info("[dry-run] no se guarda: %s (%d caracteres)", doc.id, len(raw_text))
        logger.info("(dry-run: no se escribió nada. Repetir sin --dry-run para aplicar)")
        return

    store = DocumentStore()
    is_new = store.save(doc)
    logger.info("Guardado %s: %s", doc.id, "nuevo" if is_new else "ya existía (actualizado)")
    logger.info("Siguiente paso: python -m scripts.reindex")


def main() -> None:
    parser = argparse.ArgumentParser(description="Carga de la Resolución ENACOM 843/2026")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
