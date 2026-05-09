"""SQLite helpers for library.sqlite (bullets + embeddings) and history.sqlite (decisions)."""
from __future__ import annotations
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np


def _conn(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _lib():
    from .config import settings
    return settings.library_db


def _hist():
    from .config import settings
    return settings.history_db


def init_dbs() -> None:
    """Create all tables in both DBs if they don't exist."""
    with _conn(_lib()) as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS bullets (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                text        TEXT    NOT NULL UNIQUE,
                section     TEXT    NOT NULL DEFAULT '',
                company     TEXT    NOT NULL DEFAULT '',
                role        TEXT    NOT NULL DEFAULT '',
                source_file TEXT    NOT NULL DEFAULT '',
                action_verb TEXT    NOT NULL DEFAULT '',
                has_metric  INTEGER NOT NULL DEFAULT 0,
                word_count  INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS bullet_embedding (
                bullet_id   INTEGER PRIMARY KEY
                                REFERENCES bullets(id) ON DELETE CASCADE,
                embedding   BLOB    NOT NULL
            );
        """)

    with _conn(_hist()) as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                jd_slug     TEXT    NOT NULL DEFAULT '',
                created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS decisions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  INTEGER NOT NULL REFERENCES sessions(id),
                bullet_id   INTEGER NOT NULL,
                decision    TEXT    NOT NULL,
                edited_text TEXT    NOT NULL DEFAULT '',
                created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS keyword_outcomes (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                keyword     TEXT    NOT NULL DEFAULT '',
                jd_slug     TEXT    NOT NULL DEFAULT '',
                accepted    INTEGER NOT NULL DEFAULT 0,
                rejected    INTEGER NOT NULL DEFAULT 0,
                UNIQUE(keyword, jd_slug)
            );
        """)


def clear_library() -> None:
    """Delete all bullets and embeddings (does not touch history)."""
    with _conn(_lib()) as db:
        db.execute("DELETE FROM bullet_embedding")
        db.execute("DELETE FROM bullets")


def upsert_bullet(bullet) -> int:
    """Insert or update a Bullet dataclass. Returns the row id."""
    with _conn(_lib()) as db:
        cur = db.execute(
            """
            INSERT INTO bullets
                (text, section, company, role, source_file, action_verb, has_metric, word_count)
            VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(text) DO UPDATE SET
                section     = excluded.section,
                company     = excluded.company,
                role        = excluded.role,
                source_file = excluded.source_file,
                action_verb = excluded.action_verb,
                has_metric  = excluded.has_metric,
                word_count  = excluded.word_count
            """,
            (
                bullet.text, bullet.section, bullet.company, bullet.role,
                bullet.source_file, bullet.action_verb,
                1 if bullet.has_metric else 0, bullet.word_count,
            ),
        )
        if cur.lastrowid:
            return cur.lastrowid
        row = db.execute("SELECT id FROM bullets WHERE text = ?", (bullet.text,)).fetchone()
        return row["id"] if row else 0


def count_bullets() -> int:
    with _conn(_lib()) as db:
        return db.execute("SELECT COUNT(*) FROM bullets").fetchone()[0]


def get_all_bullets() -> list[dict]:
    with _conn(_lib()) as db:
        return [dict(r) for r in db.execute("SELECT * FROM bullets ORDER BY id").fetchall()]


def list_sources() -> list[dict]:
    with _conn(_lib()) as db:
        rows = db.execute(
            "SELECT source_file, COUNT(*) as count FROM bullets"
            " GROUP BY source_file ORDER BY source_file"
        ).fetchall()
        return [dict(r) for r in rows]


def save_embedding(bullet_id: int, embedding: np.ndarray) -> None:
    blob = embedding.astype(np.float32).tobytes()
    with _conn(_lib()) as db:
        db.execute(
            """
            INSERT INTO bullet_embedding (bullet_id, embedding)
            VALUES (?,?)
            ON CONFLICT(bullet_id) DO UPDATE SET embedding = excluded.embedding
            """,
            (bullet_id, blob),
        )


def get_embedding(bullet_id: int) -> Optional[np.ndarray]:
    with _conn(_lib()) as db:
        row = db.execute(
            "SELECT embedding FROM bullet_embedding WHERE bullet_id = ?", (bullet_id,)
        ).fetchone()
    return np.frombuffer(row["embedding"], dtype=np.float32) if row else None


def get_bullets_without_embeddings() -> list[dict]:
    with _conn(_lib()) as db:
        rows = db.execute(
            """
            SELECT b.* FROM bullets b
            LEFT JOIN bullet_embedding e ON b.id = e.bullet_id
            WHERE e.bullet_id IS NULL
            """
        ).fetchall()
        return [dict(r) for r in rows]


def get_bullets_for_company(company: str, section: str = "EXPERIENCE", limit: int = 10) -> list[dict]:
    """Return bullets for a company, best ones first (has_metric desc, word_count desc)."""
    with _conn(_lib()) as db:
        rows = db.execute(
            """
            SELECT * FROM bullets
            WHERE company = ? AND section = ?
            ORDER BY has_metric DESC, word_count DESC
            LIMIT ?
            """,
            (company, section, limit),
        ).fetchall()
        return [dict(r) for r in rows]
