from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

import requests
from requests import Response

from storage.models import RegulatoryDocument

logger = logging.getLogger(__name__)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "es-AR,es;q=0.9,en;q=0.8,pt;q=0.7",
}


class BaseScraper(ABC):
    country: str
    organism: str
    base_url: str

    def __init__(
        self,
        delay_seconds: float = 2.0,
        timeout: int = 30,
        headers: dict[str, str] | None = None,
        verify_ssl: bool = True,
    ) -> None:
        self.delay_seconds = delay_seconds
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self.session = requests.Session()
        self.session.headers.update(headers or DEFAULT_HEADERS)
        self._last_request_time: float = 0.0
        if not verify_ssl:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            logger.warning("SSL verification DESACTIVADA para este scraper")

    def _get(self, url: str, **kwargs: Any) -> Response:
        self._respect_rate_limit()
        logger.debug("GET %s", url)
        response = self.session.get(url, timeout=self.timeout, verify=self.verify_ssl, **kwargs)
        response.raise_for_status()
        self._last_request_time = time.monotonic()
        return response

    def _post(self, url: str, data: dict[str, Any], **kwargs: Any) -> Response:
        self._respect_rate_limit()
        logger.debug("POST %s", url)
        response = self.session.post(url, data=data, timeout=self.timeout, verify=self.verify_ssl, **kwargs)
        response.raise_for_status()
        self._last_request_time = time.monotonic()
        return response

    def _respect_rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        wait = self.delay_seconds - elapsed
        if wait > 0:
            logger.debug("Rate limit: waiting %.2fs", wait)
            time.sleep(wait)

    @abstractmethod
    def fetch_index(self) -> list[dict[str, Any]]:
        """Fetch the list of available documents (metadata only)."""

    @abstractmethod
    def fetch_document(self, entry: dict[str, Any]) -> RegulatoryDocument:
        """Download and parse a single document."""

    def run(self) -> list[RegulatoryDocument]:
        logger.info("Starting scraper: %s / %s", self.country, self.organism)
        entries = self.fetch_index()
        logger.info("Found %d entries", len(entries))
        documents: list[RegulatoryDocument] = []
        for entry in entries:
            try:
                doc = self.fetch_document(entry)
                documents.append(doc)
                logger.info("Scraped: %s %s", doc.doc_type, doc.number)
            except requests.HTTPError as exc:
                logger.warning("HTTP error scraping %s: %s", entry, exc)
            except Exception as exc:
                logger.error("Unexpected error scraping %s: %s", entry, exc)
                raise
        logger.info("Finished: %d documents scraped", len(documents))
        return documents
