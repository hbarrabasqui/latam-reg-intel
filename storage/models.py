from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any


@dataclass
class RegulatoryDocument:
    id: str
    country: str           # ISO 3166-1 alpha-3 (ARG, BRA, MEX)
    organism: str          # ENACOM, ANATEL, IFT
    doc_type: str          # resolucion, norma_tecnica, acuerdo, ley
    number: str
    title: str
    date_published: date
    date_scraped: datetime
    source_url: str
    raw_text: str
    language: str          # es, pt
    status: str            # vigente, derogado, modificado
    version: int           # para versionado de cambios
    hash: str              # SHA256 del contenido para detectar cambios
    metadata: dict[str, Any] = field(default_factory=dict)
    # Workflow de revisión humana — independiente del estado regulatorio
    # pending_review: esperando que el usuario lo valide
    # approved: validado, elegible para ser indexado en el RAG
    # rejected: descartado, no se indexa
    review_status: str = "approved"

    @classmethod
    def compute_hash(cls, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def has_changed(self, other: RegulatoryDocument) -> bool:
        return self.hash != other.hash
