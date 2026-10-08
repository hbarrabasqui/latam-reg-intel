"""Carga al corpus los trámites oficiales de ENACOM para inscribir/modificar una
empresa en el RAMATEL (Registro de Actividades), más un documento de síntesis del
flujo completo (RAMATEL por TAD + representantes en HERTZ).

Por qué existen estos documentos: hasta el 24/8/2026 el corpus no tenía el detalle
de estos 4 trámites (solo tenía normativa y una FAQ general), y hubo dos errores
reales en base a eso: (1) describir el Subregistro de Actividades como si tuviera
tres subregistros paralelos (Fabricante/Importador/Representante Local) cuando en
realidad son dos (Fabricación y Comercialización, con Representante Local como una
modalidad DENTRO de Comercialización — ver corrección en ESTADO_SESION.md,
24/8/2026); y (2) no saber que el trámite de inscripción de la EMPRESA se hace por
TAD, no directamente en HERTZ (en HERTZ solo se registran los representantes y
luego los equipos). Ambos puntos, corregidos por Horacio, quedaron reflejados en
estos documentos.

doc_type="documento_informativo" para los 4 trámites (son transcripciones fieles
de páginas oficiales de ENACOM, con su fuente citada) y también para el documento
de síntesis del flujo (organiza información oficial dispersa en dos páginas
distintas, sin agregar interpretación legal propia — por eso no es
"nota_interpretativa").

Uso:
    python -m scripts.seed_tramites_empresas
    python -m scripts.seed_tramites_empresas --dry-run
"""
from __future__ import annotations

import argparse
import logging
from datetime import date, datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_DIR = Path("data") / "tramites_empresas"

_DOCS = [
    {
        "id": "ARG-tramite-t123-fabricacion-y-comercializacion",
        "file": "ARG-tramite-fabricacion-y-comercializacion.txt",
        "number": "t123",
        "title": "Inscripción o Renovación del Registro de Actividades para la Fabricación y Comercialización de Equipos",
        "source_url": "https://www.enacom.gob.ar/tramites/inscripcion-renovacion-registro-actividades-fabricacion-y-comercializacion-equipos_t123",
    },
    {
        "id": "ARG-tramite-t124-fabricacion",
        "file": "ARG-tramite-fabricacion.txt",
        "number": "t124",
        "title": "Inscripción o Renovación del Registro de Actividades para la Fabricación de Equipos",
        "source_url": "https://www.enacom.gob.ar/tramites/inscripcion-renovacion-registro-actividades-fabricacion-equipos_t124",
    },
    {
        "id": "ARG-tramite-t125-comercializacion",
        "file": "ARG-tramite-comercializacion.txt",
        "number": "t125",
        "title": "Inscripción o Renovación del Registro de Actividades para la Comercialización de Equipos",
        "source_url": "https://www.enacom.gob.ar/tramites/inscripcion-renovacion-registro-actividades-comercializacion-equipos_t125",
    },
    {
        "id": "ARG-tramite-t126-modificacion",
        "file": "ARG-tramite-modificacion.txt",
        "number": "t126",
        "title": "Modificación del Registro de Actividades para la Comercialización / Fabricación de Equipos",
        "source_url": "https://www.enacom.gob.ar/tramites/modificacion-registro-actividades-comercializacion-fabricacion-equipos_t126",
    },
    {
        "id": "ARG-flujo-registro-empresa-y-representantes",
        "file": "ARG-flujo-registro-empresa-y-representantes.txt",
        "number": "",
        "title": "Flujo completo de registro de una empresa y sus representantes (RAMATEL + HERTZ)",
        "source_url": "",  # síntesis de varias páginas, sin URL propia
    },
    {
        "id": "ARG-mapa-normas-corto-alcance",
        "file": "ARG-mapa-normas-corto-alcance.txt",
        "number": "",
        "title": "Mapa de normas técnicas ENACOM para WiFi, Bluetooth, Zigbee y dispositivos de corto alcance",
        "source_url": "",  # síntesis verificada contra el texto de cada norma técnica (Q2-63.02, Q2-63.03, Q2-60.14)
    },
]


def run(dry_run: bool = False) -> None:
    from storage.models import RegulatoryDocument
    from storage.relational import DocumentStore

    store = None if dry_run else DocumentStore()
    now = datetime.now(timezone.utc)

    for spec in _DOCS:
        txt_path = _DIR / spec["file"]
        if not txt_path.exists():
            logger.error("No encontré %s — abortando.", txt_path)
            return
        raw_text = txt_path.read_text(encoding="utf-8").strip()

        doc = RegulatoryDocument(
            id=spec["id"],
            country="ARG",
            organism="ENACOM",
            doc_type="documento_informativo",
            number=spec["number"],
            title=spec["title"],
            date_published=date(2026, 8, 24),
            date_scraped=now,
            source_url=spec["source_url"],
            raw_text=raw_text,
            language="es",
            status="vigente",
            version=1,
            hash=RegulatoryDocument.compute_hash(raw_text),
            metadata={
                "seed_source": "tramites_empresas_manual_2026-08-24",
                "seed_note": (
                    "Transcripción/síntesis de páginas oficiales de ENACOM (trámites "
                    "de Inscripción de Empresas + Hertz), cargada manualmente el "
                    "24/8/2026 tras corregir dos errores del bot sobre el tema "
                    "(estructura del Subregistro de Actividades y canal TAD vs Hertz)."
                ),
            },
            review_status="approved",
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
    parser = argparse.ArgumentParser(description="Carga de trámites de inscripción de empresas al corpus")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
