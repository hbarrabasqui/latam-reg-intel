from __future__ import annotations

import tempfile
from datetime import date, datetime
from pathlib import Path

import pytest

from storage.models import RegulatoryDocument
from storage.relational import DocumentStore


def _make_doc(number: str = "001", text: str = "texto de prueba") -> RegulatoryDocument:
    return RegulatoryDocument(
        id=f"test-{number}",
        country="ARG",
        organism="ENACOM",
        doc_type="resolucion",
        number=number,
        title=f"Resolución {number}",
        date_published=date(2024, 1, 15),
        date_scraped=datetime(2024, 1, 20, 10, 0, 0),
        source_url=f"https://enacom.gob.ar/res/{number}",
        raw_text=text,
        language="es",
        status="vigente",
        version=1,
        hash=RegulatoryDocument.compute_hash(text),
        metadata={"categoria": "homologacion"},
    )


@pytest.fixture
def store(tmp_path: Path) -> DocumentStore:
    return DocumentStore(
        db_path=tmp_path / "test.db",
        raw_text_dir=tmp_path / "raw_texts",
    )


def test_save_new_document(store: DocumentStore) -> None:
    doc = _make_doc("100")
    saved = store.save(doc)
    assert saved is True


def test_save_duplicate_by_hash_is_skipped(store: DocumentStore) -> None:
    doc = _make_doc("100")
    store.save(doc)
    saved_again = store.save(doc)
    assert saved_again is False


def test_get_by_id_returns_document(store: DocumentStore) -> None:
    doc = _make_doc("200")
    store.save(doc)
    retrieved = store.get_by_id("test-200")
    assert retrieved is not None
    assert retrieved.number == "200"
    assert retrieved.country == "ARG"


def test_get_by_id_missing_returns_none(store: DocumentStore) -> None:
    assert store.get_by_id("no-existe") is None


def test_get_by_country(store: DocumentStore) -> None:
    store.save(_make_doc("301", "texto arg"))
    store.save(_make_doc("302", "texto arg 2"))
    results = store.get_by_country("ARG")
    assert len(results) == 2


def test_get_by_organism(store: DocumentStore) -> None:
    store.save(_make_doc("401"))
    results = store.get_by_organism("ENACOM")
    assert len(results) == 1
    assert results[0].organism == "ENACOM"


def test_exists_by_hash(store: DocumentStore) -> None:
    doc = _make_doc("500")
    assert store.exists_by_hash(doc.hash) is False
    store.save(doc)
    assert store.exists_by_hash(doc.hash) is True


def test_count(store: DocumentStore) -> None:
    assert store.count() == 0
    store.save(_make_doc("601", "a"))
    store.save(_make_doc("602", "b"))
    assert store.count() == 2
    assert store.count(country="ARG") == 2
    assert store.count(country="BRA") == 0


def test_raw_text_file_created(store: DocumentStore, tmp_path: Path) -> None:
    doc = _make_doc("700", "contenido del documento")
    store.save(doc)
    raw_file = tmp_path / "raw_texts" / "test-700.txt"
    assert raw_file.exists()
    assert raw_file.read_text(encoding="utf-8") == "contenido del documento"


def test_metadata_roundtrip(store: DocumentStore) -> None:
    doc = _make_doc("800")
    store.save(doc)
    retrieved = store.get_by_id("test-800")
    assert retrieved is not None
    assert retrieved.metadata == {"categoria": "homologacion"}


# ── Review queue ─────────────────────────────────────────────────────────────

def test_default_review_status_is_approved(store: DocumentStore) -> None:
    doc = _make_doc("900")
    store.save(doc)
    retrieved = store.get_by_id("test-900")
    assert retrieved is not None
    assert retrieved.review_status == "approved"


def test_save_with_pending_review(store: DocumentStore) -> None:
    doc = _make_doc("901")
    doc.review_status = "pending_review"
    store.save(doc)
    retrieved = store.get_by_id("test-901")
    assert retrieved is not None
    assert retrieved.review_status == "pending_review"


def test_set_review_status_approve(store: DocumentStore) -> None:
    doc = _make_doc("902")
    doc.review_status = "pending_review"
    store.save(doc)
    store.set_review_status("test-902", "approved")
    retrieved = store.get_by_id("test-902")
    assert retrieved is not None
    assert retrieved.review_status == "approved"


def test_set_review_status_reject(store: DocumentStore) -> None:
    doc = _make_doc("903")
    doc.review_status = "pending_review"
    store.save(doc)
    store.set_review_status("test-903", "rejected")
    retrieved = store.get_by_id("test-903")
    assert retrieved is not None
    assert retrieved.review_status == "rejected"


def test_set_review_status_invalid_raises(store: DocumentStore) -> None:
    doc = _make_doc("904")
    store.save(doc)
    with pytest.raises(ValueError):
        store.set_review_status("test-904", "publicado")


def test_get_pending_review(store: DocumentStore) -> None:
    pending = _make_doc("910", "pendiente")
    pending.review_status = "pending_review"
    approved = _make_doc("911", "aprobado")
    approved.review_status = "approved"
    store.save(pending)
    store.save(approved)
    results = store.get_pending_review()
    assert len(results) == 1
    assert results[0].id == "test-910"


def test_get_approved(store: DocumentStore) -> None:
    doc = _make_doc("920")
    doc.review_status = "approved"
    store.save(doc)
    results = store.get_approved()
    assert len(results) == 1
    assert results[0].id == "test-920"


def test_count_by_review_status(store: DocumentStore) -> None:
    pending = _make_doc("930", "p")
    pending.review_status = "pending_review"
    approved = _make_doc("931", "a")
    approved.review_status = "approved"
    store.save(pending)
    store.save(approved)
    assert store.count(review_status="pending_review") == 1
    assert store.count(review_status="approved") == 1


def test_migration_adds_review_status_column(tmp_path: Path) -> None:
    """Una DB creada sin review_status debe migrarse automáticamente."""
    import sqlite3
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.execute("""
        CREATE TABLE documents (
            id TEXT PRIMARY KEY, country TEXT, organism TEXT, doc_type TEXT,
            number TEXT, title TEXT, date_published TEXT, date_scraped TEXT,
            source_url TEXT, raw_text TEXT, language TEXT, status TEXT,
            version INTEGER, hash TEXT, metadata TEXT DEFAULT '{}'
        )
    """)
    conn.commit()
    conn.close()
    # Al instanciar DocumentStore debe migrar sin error
    store2 = DocumentStore(db_path=db, raw_text_dir=tmp_path / "raw")
    doc = _make_doc("999")
    store2.save(doc)
    retrieved = store2.get_by_id("test-999")
    assert retrieved is not None
    assert retrieved.review_status == "approved"
