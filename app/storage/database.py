"""
SQLite storage layer for the Evidence-Grounded Fact Knowledge Layer.

This module owns all direct access to the SQLite database. Every other
part of the project (pipeline, UI, tests) goes through the functions
defined here instead of writing raw SQL elsewhere.
"""

import hashlib
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Dict, List, Optional

from app.config import DATABASE_PATH

logger = logging.getLogger(__name__)

_connection: Optional[sqlite3.Connection] = None


def get_connection() -> sqlite3.Connection:
    """
    Return a single, shared SQLite connection for the whole application.

    We reuse one connection instead of opening/closing a new one for
    every tiny read or write. check_same_thread=False is needed because
    Streamlit can call into this module from different internal threads.
    """
    global _connection
    if _connection is None:
        _connection = sqlite3.connect(str(DATABASE_PATH), check_same_thread=False)
        _connection.row_factory = sqlite3.Row
        _connection.execute("PRAGMA foreign_keys = ON")
    return _connection


@contextmanager
def transaction():
    """Commits on success, rolls back and re-raises on any error."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        logger.exception("Database transaction failed; rolled back.")
        raise


def init_db() -> None:
    """Create tables and indexes if they do not already exist."""
    with transaction() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                path TEXT NOT NULL,
                file_hash TEXT NOT NULL UNIQUE,
                total_pages INTEGER DEFAULT 0,
                status TEXT DEFAULT 'processing',
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                subject TEXT,
                predicate TEXT,
                value TEXT,
                unit TEXT,
                time_period TEXT,
                scope TEXT,
                location TEXT,
                confidence REAL,
                llm_confidence REAL,
                evidence TEXT,
                page INTEGER,
                FOREIGN KEY (document_id) REFERENCES documents (id)
            )
            """
        )

        # Migration for existing databases created before llm_confidence existed.
        existing_cols = {row["name"] for row in conn.execute("PRAGMA table_info(facts)")}
        if "llm_confidence" not in existing_cols:
            conn.execute("ALTER TABLE facts ADD COLUMN llm_confidence REAL")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact_a INTEGER NOT NULL,
                fact_b INTEGER NOT NULL,
                relationship TEXT NOT NULL,
                similarity REAL,
                explanation TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (fact_a) REFERENCES facts (id),
                FOREIGN KEY (fact_b) REFERENCES facts (id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS extraction_failures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                page INTEGER,
                reason TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (document_id) REFERENCES documents (id)
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_document ON facts (document_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts (subject)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rel_fact_a ON relationships (fact_a)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rel_fact_b ON relationships (fact_b)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_failures_document ON extraction_failures (document_id)"
        )
    logger.info("Database initialized at %s", DATABASE_PATH)


def compute_file_hash(file_path: str) -> str:
    """SHA-256 hash of a file's contents, used to detect duplicate uploads."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_document_by_hash(file_hash: str) -> Optional[Dict]:
    conn = get_connection()
    row = conn.execute("SELECT * FROM documents WHERE file_hash = ?", (file_hash,)).fetchone()
    return dict(row) if row else None


def insert_document(name: str, path: str, file_hash: str, total_pages: int) -> int:
    now = datetime.now(timezone.utc).isoformat()
    with transaction() as conn:
        cursor = conn.execute(
            """
            INSERT INTO documents (name, path, file_hash, total_pages, status, created_at)
            VALUES (?, ?, ?, ?, 'processing', ?)
            """,
            (name, path, file_hash, total_pages, now),
        )
        return cursor.lastrowid


def update_document_status(document_id: int, status: str) -> None:
    with transaction() as conn:
        conn.execute("UPDATE documents SET status = ? WHERE id = ?", (status, document_id))


def get_documents() -> List[Dict]:
    """All documents with a fact count and failure count for each, for the 'Documents' tab."""
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT
            d.id, d.name, d.path, d.file_hash, d.total_pages, d.status, d.created_at,
            (SELECT COUNT(*) FROM facts f WHERE f.document_id = d.id) AS fact_count,
            (SELECT COUNT(*) FROM extraction_failures ef WHERE ef.document_id = d.id) AS failure_count
        FROM documents d
        ORDER BY d.created_at DESC
        """
    ).fetchall()
    return [dict(row) for row in rows]
def delete_document(document_id: int) -> None:
    """
    Delete a document and everything derived from it: its facts, any
    relationships involving those facts, and its extraction failures.
    Order matters — children must go before the parent row.
    """
    with transaction() as conn:
        fact_ids = [
            row["id"] for row in conn.execute(
                "SELECT id FROM facts WHERE document_id = ?", (document_id,)
            ).fetchall()
        ]

        if fact_ids:
            placeholders = ", ".join("?" for _ in fact_ids)
            conn.execute(
                f"DELETE FROM relationships WHERE fact_a IN ({placeholders}) OR fact_b IN ({placeholders})",
                fact_ids + fact_ids,
            )

        conn.execute("DELETE FROM facts WHERE document_id = ?", (document_id,))
        conn.execute("DELETE FROM extraction_failures WHERE document_id = ?", (document_id,))
        conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))

    logger.info("Deleted document id=%s and all associated facts/relationships.", document_id)

def insert_fact(document_id: int, fact: Dict) -> int:
    with transaction() as conn:
        cursor = conn.execute(
            """
            INSERT INTO facts (
                document_id, subject, predicate, value, unit,
                time_period, scope, location, confidence, llm_confidence, evidence, page
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document_id,
                fact.get("subject", ""),
                fact.get("predicate", ""),
                str(fact.get("value", "")),
                fact.get("unit", ""),
                fact.get("time_period", ""),
                fact.get("scope", ""),
                fact.get("location", ""),
                fact.get("confidence", 0.0),
                fact.get("llm_confidence"),
                fact.get("evidence", ""),
                fact.get("page", 0),
            ),
        )
        return cursor.lastrowid


def get_facts(document_id: Optional[int] = None) -> List[Dict]:
    conn = get_connection()
    if document_id is not None:
        rows = conn.execute(
            """
            SELECT f.*, d.name AS document_name
            FROM facts f JOIN documents d ON f.document_id = d.id
            WHERE f.document_id = ?
            """,
            (document_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT f.*, d.name AS document_name
            FROM facts f JOIN documents d ON f.document_id = d.id
            """
        ).fetchall()
    return [dict(row) for row in rows]


def get_fact_by_id(fact_id: int) -> Optional[Dict]:
    conn = get_connection()
    row = conn.execute(
        """
        SELECT f.*, d.name AS document_name
        FROM facts f JOIN documents d ON f.document_id = d.id
        WHERE f.id = ?
        """,
        (fact_id,),
    ).fetchone()
    return dict(row) if row else None


def insert_failure(document_id: int, page: Optional[int], reason: str) -> None:
    """Record a page that failed extraction (different from a skipped page)."""
    now = datetime.now(timezone.utc).isoformat()
    with transaction() as conn:
        conn.execute(
            """
            INSERT INTO extraction_failures (document_id, page, reason, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (document_id, page, reason, now),
        )


def get_failures() -> List[Dict]:
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT ef.*, d.name AS document_name
        FROM extraction_failures ef JOIN documents d ON ef.document_id = d.id
        ORDER BY ef.created_at DESC
        """
    ).fetchall()
    return [dict(row) for row in rows]


def relationship_exists(fact_a: int, fact_b: int) -> bool:
    """Check whether a relationship between these two facts already exists (either order)."""
    conn = get_connection()
    row = conn.execute(
        """
        SELECT 1 FROM relationships
        WHERE (fact_a = ? AND fact_b = ?) OR (fact_a = ? AND fact_b = ?)
        LIMIT 1
        """,
        (fact_a, fact_b, fact_b, fact_a),
    ).fetchone()
    return row is not None


def insert_relationship(fact_a: int, fact_b: int, relationship: str, similarity: float, explanation: str) -> int:
    now = datetime.now(timezone.utc).isoformat()
    with transaction() as conn:
        cursor = conn.execute(
            """
            INSERT INTO relationships (fact_a, fact_b, relationship, similarity, explanation, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (fact_a, fact_b, relationship, similarity, explanation, now),
        )
        return cursor.lastrowid


def get_relationships() -> List[Dict]:
    """All relationships, joined with both facts' full details for evidence display."""
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT
            r.id, r.relationship, r.similarity, r.explanation, r.created_at,
            fa.id AS fact_a_id, fa.subject AS a_subject, fa.predicate AS a_predicate,
            fa.value AS a_value, fa.unit AS a_unit, fa.time_period AS a_time_period,
            fa.scope AS a_scope, fa.location AS a_location, fa.evidence AS a_evidence,
            fa.page AS a_page, da.name AS a_document,
            fb.id AS fact_b_id, fb.subject AS b_subject, fb.predicate AS b_predicate,
            fb.value AS b_value, fb.unit AS b_unit, fb.time_period AS b_time_period,
            fb.scope AS b_scope, fb.location AS b_location, fb.evidence AS b_evidence,
            fb.page AS b_page, db.name AS b_document
        FROM relationships r
        JOIN facts fa ON r.fact_a = fa.id
        JOIN documents da ON fa.document_id = da.id
        JOIN facts fb ON r.fact_b = fb.id
        JOIN documents db ON fb.document_id = db.id
        ORDER BY r.created_at DESC
        """
    ).fetchall()
    return [dict(row) for row in rows]