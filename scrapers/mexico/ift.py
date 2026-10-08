"""Scraper para IFT (México) — HTML + DOF.
NOTA: El IFT está en proceso de extinción como órgano autónomo (decreto dic 2024).
Monitorear quién asume sus funciones y hacia qué URLs migra la normativa.
"""
from __future__ import annotations

import logging
from typing import Any

from scrapers.base_scraper import BaseScraper
from storage.models import RegulatoryDocument

logger = logging.getLogger(__name__)


class IftScraper(BaseScraper):
    country = "MEX"
    organism = "IFT"
    base_url = "https://www.ift.org.mx/industria/registro-de-regulaciones"

    # TODO: implementar fetch_index y fetch_document

    def fetch_index(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    def fetch_document(self, entry: dict[str, Any]) -> RegulatoryDocument:
        raise NotImplementedError
