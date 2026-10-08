from __future__ import annotations

import json
import logging
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from storage.models import RegulatoryDocument

logger = logging.getLogger(__name__)

_DB_PATH = Path("data") / "latam_reg.db"
_RAW_TEXT_DIR = Path("data") / "raw_texts"

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS documents (
    id             TEXT PRIMARY KEY,
    country        TEXT NOT NULL,
    organism       TEXT NOT NULL,
    doc_type       TEXT NOT NULL,
    number         TEXT NOT NULL,
    title          TEXT NOT NULL,
    date_published TEXT NOT NULL,
    date_scraped   TEXT NOT NULL,
    source_url     TEXT NOT NULL,
    raw_text       TEXT NOT NULL,
    language       TEXT NOT NULL,
    status         TEXT NOT NULL,
    version        INTEGER NOT NULL DEFAULT 1,
    hash           TEXT NOT NULL,
    metadata       TEXT NOT NULL DEFAULT '{}',
    review_status  TEXT NOT NULL DEFAULT 'approved'
)
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_country       ON documents (country)",
    "CREATE INDEX IF NOT EXISTS idx_organism      ON documents (organism)",
    "CREATE INDEX IF NOT EXISTS idx_hash          ON documents (hash)",
    "CREATE INDEX IF NOT EXISTS idx_review_status ON documents (review_status)",
]

_VALID_REVIEW_STATUSES = {"pending_review", "approved", "rejected"}


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(_CREATE_TABLE)
    # Migración: agregar review_status si la tabla ya existía sin él
    cols = {row[1] for row in conn.execute("PRAGMA table_info(documents)")}
    if "review_status" not in cols:
        conn.execute("ALTER TABLE documents ADD COLUMN review_status TEXT NOT NULL DEFAULT 'approved'")
        logger.info("Migración: columna review_status agregada a documents.")
    for idx in _CREATE_INDEXES:
        conn.execute(idx)
    conn.commit()


def _row_to_doc(row: sqlite3.Row) -> RegulatoryDocument:
    return RegulatoryDocument(
        id=row["id"],
        country=row["country"],
        organism=row["organism"],
        doc_type=row["doc_type"],
        number=row["number"],
        title=row["title"],
        date_published=date.fromisoformat(row["date_published"]),
        date_scraped=datetime.fromisoformat(row["date_scraped"]),
        source_url=row["source_url"],
        raw_text=row["raw_text"],
        language=row["language"],
        status=row["status"],
        version=row["version"],
        hash=row["hash"],
        metadata=json.loads(row["metadata"]),
        review_status=row["review_status"],
    )


class DocumentStore:
    def __init__(self, db_path: Path = _DB_PATH, raw_text_dir: Path = _RAW_TEXT_DIR) -> None:
        self.db_path = db_path
        self.raw_text_dir = raw_text_dir
        self.raw_text_dir.mkdir(parents=True, exist_ok=True)
        with _connect(self.db_path) as conn:
            _ensure_schema(conn)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save(self, doc: RegulatoryDocument) -> bool:
        """Insert or replace a document. Returns True if it was new/changed."""
        if self.exists_by_hash(doc.hash):
            logger.debug("Document %s unchanged (hash match), skipping.", doc.id)
            return False

        with _connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO documents
                    (id, country, organism, doc_type, number, title,
                     date_published, date_scraped, source_url, raw_text,
                     language, status, version, hash, metadata, review_status)
                VALUES
                    (:id, :country, :organism, :doc_type, :number, :title,
                     :date_published, :date_scraped, :source_url, :raw_text,
                     :language, :status, :version, :hash, :metadata, :review_status)
                """,
                {
                    "id": doc.id,
                    "country": doc.country,
                    "organism": doc.organism,
                    "doc_type": doc.doc_type,
                    "number": doc.number,
                    "title": doc.title,
                    "date_published": doc.date_published.isoformat(),
                    "date_scraped": doc.date_scraped.isoformat(),
                    "source_url": doc.source_url,
                    "raw_text": doc.raw_text,
                    "language": doc.language,
                    "status": doc.status,
                    "version": doc.version,
                    "hash": doc.hash,
                    "metadata": json.dumps(doc.metadata, ensure_ascii=False),
                    "review_status": doc.review_status,
                },
            )
            conn.commit()

        self._write_raw_text(doc)
        logger.info(
            "Saved document %s (%s %s) [review=%s].",
            doc.id, doc.organism, doc.number, doc.review_status,
        )
        return True

    def set_review_status(self, doc_id: str, review_status: str) -> bool:
        """Cambia el review_status de un documento. Devuelve True si existía."""
        if review_status not in _VALID_REVIEW_STATUSES:
            raise ValueError(f"review_status inválido: {review_status!r}. Válidos: {_VALID_REVIEW_STATUSES}")
        with _connect(self.db_path) as conn:
            cursor = conn.execute(
                "UPDATE documents SET review_status = ? WHERE id = ?",
                (review_status, doc_id),
            )
            conn.commit()
        return cursor.rowcount > 0

    def _write_raw_text(self, doc: RegulatoryDocument) -> None:
        path = self.raw_text_dir / f"{doc.id}.txt"
        path.write_text(doc.raw_text, encoding="utf-8")

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_by_id(self, doc_id: str) -> Optional[RegulatoryDocument]:
        with _connect(self.db_path) as conn:
            row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        return _row_to_doc(row) if row else None

    def get_by_country(self, country: str) -> list[RegulatoryDocument]:
        with _connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM documents WHERE country = ? ORDER BY date_published DESC",
                (country,),
            ).fetchall()
        return [_row_to_doc(r) for r in rows]

    def get_by_organism(self, organism: str) -> list[RegulatoryDocument]:
        with _connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM documents WHERE organism = ? ORDER BY date_published DESC",
                (organism,),
            ).fetchall()
        return [_row_to_doc(r) for r in rows]

    def get_pending_review(self) -> list[RegulatoryDocument]:
        with _connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM documents WHERE review_status = 'pending_review' ORDER BY date_scraped DESC",
            ).fetchall()
        return [_row_to_doc(r) for r in rows]

    def get_approved(self) -> list[RegulatoryDocument]:
        with _connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM documents WHERE review_status = 'approved' ORDER BY date_published DESC",
            ).fetchall()
        return [_row_to_doc(r) for r in rows]

    def exists_by_hash(self, hash_value: str) -> bool:
        with _connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT 1 FROM documents WHERE hash = ? LIMIT 1", (hash_value,)
            ).fetchone()
        return row is not None

    def count(self, country: Optional[str] = None, review_status: Optional[str] = None) -> int:
        conditions = []
        params: list[str] = []
        if country:
            conditions.append("country = ?")
            params.append(country)
        if review_status:
            conditions.append("review_status = ?")
            params.append(review_status)
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        with _connect(self.db_path) as conn:
            row = conn.execute(f"SELECT COUNT(*) FROM documents {where}", params).fetchone()
        return row[0]
