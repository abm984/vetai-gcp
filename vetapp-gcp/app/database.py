"""
database.py — PostgreSQL connection pool via psycopg2.

Cloud Run connects to Cloud SQL through a Unix socket automatically
injected at /cloudsql/<PROJECT>:<REGION>:<INSTANCE>.  Set DATABASE_URL
to use the socket form:

    postgresql://USER:PASS@/DBNAME?host=/cloudsql/PROJECT:REGION:INSTANCE

For local development use a normal TCP URL:

    postgresql://vetapp:vetapp@localhost:5432/vetapp
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

import psycopg2
from psycopg2 import pool as pg_pool

from app.config import DATABASE_URL

_pool: pg_pool.ThreadedConnectionPool | None = None


def init_pool() -> None:
    """Create the connection pool.  Call once at application startup."""
    global _pool
    _pool = pg_pool.ThreadedConnectionPool(
        minconn=2,
        maxconn=20,
        dsn=DATABASE_URL,
    )
    print("[DB] Connection pool initialised")


@contextmanager
def get_conn() -> Generator[psycopg2.extensions.connection, None, None]:
    """Yield a connection from the pool, auto-commit on success or rollback on error."""
    assert _pool is not None, "DB pool not initialised — call init_pool() first"
    conn = _pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        _pool.putconn(conn)


def init_tables() -> None:
    """Create all tables if they don't exist.  Idempotent."""
    with get_conn() as conn:
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS seen_hashes (
                hash     TEXT PRIMARY KEY,
                added_at TEXT NOT NULL
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS preclean (
                id           SERIAL PRIMARY KEY,
                filepath     TEXT UNIQUE NOT NULL,
                species      TEXT,
                caption      TEXT,
                sender_id    TEXT,
                sender_name  TEXT,
                wa_timestamp TEXT,
                queued_at    TEXT,
                vet_species  TEXT,
                vet_label    TEXT,
                vet_id       TEXT,
                reviewed_at  TEXT,
                notes        TEXT,
                status       TEXT DEFAULT 'pending'
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS vet_queue (
                id           SERIAL PRIMARY KEY,
                filepath     TEXT UNIQUE NOT NULL,
                species      TEXT,
                pred_label   TEXT,
                confidence   REAL,
                sender_id    TEXT,
                sender_name  TEXT,
                wa_timestamp TEXT,
                queued_at    TEXT,
                vet_label    TEXT,
                vet_id       TEXT,
                reviewed_at  TEXT,
                status       TEXT DEFAULT 'pending'
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS custom_classes (
                id         SERIAL PRIMARY KEY,
                species    TEXT NOT NULL,
                class_name TEXT NOT NULL,
                added_at   TEXT,
                UNIQUE (species, class_name)
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS dataset_log (
                id           SERIAL PRIMARY KEY,
                filename     TEXT,
                species      TEXT,
                label        TEXT,
                sender_id    TEXT,
                sender_name  TEXT,
                wa_timestamp TEXT,
                processed_at TEXT,
                status       TEXT,
                reason       TEXT
            )
        """)

        # Lightweight counters to avoid expensive GCS listing for split balancing.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS dataset_counts (
                species    TEXT NOT NULL,
                label      TEXT NOT NULL,
                split      TEXT NOT NULL,
                count      INTEGER DEFAULT 0,
                PRIMARY KEY (species, label, split)
            )
        """)

    print("[DB] Tables verified / created")


# ── Helpers used by the pipeline ──────────────────────────────────────────────

def is_duplicate(md5_hash: str) -> bool:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM seen_hashes WHERE hash = %s", (md5_hash,))
        return cur.fetchone() is not None


def record_hash(md5_hash: str, added_at: str) -> None:
    with get_conn() as conn:
        conn.cursor().execute(
            "INSERT INTO seen_hashes (hash, added_at) VALUES (%s, %s) ON CONFLICT DO NOTHING",
            (md5_hash, added_at),
        )


def get_split_counts(species: str, label: str) -> dict:
    """Return {split: count} for a given species/label combination."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT split, count FROM dataset_counts WHERE species = %s AND label = %s",
            (species, label),
        )
        rows = cur.fetchall()
    return {r[0]: r[1] for r in rows}


def increment_split_count(species: str, label: str, split: str) -> None:
    with get_conn() as conn:
        conn.cursor().execute(
            """
            INSERT INTO dataset_counts (species, label, split, count) VALUES (%s, %s, %s, 1)
            ON CONFLICT (species, label, split) DO UPDATE SET count = dataset_counts.count + 1
            """,
            (species, label, split),
        )


def log_dataset_entry(
    filename: str, species: str, label: str,
    sender_id: str, sender_name: str,
    wa_timestamp: str, processed_at: str,
    status: str, reason: str,
) -> None:
    with get_conn() as conn:
        conn.cursor().execute(
            """
            INSERT INTO dataset_log
              (filename, species, label, sender_id, sender_name,
               wa_timestamp, processed_at, status, reason)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (filename, species, label, sender_id, sender_name,
             wa_timestamp, processed_at, status, reason),
        )


def get_all_classes() -> dict:
    """Config classes merged with any custom classes stored in the DB."""
    from app.config import CLASSES
    result = {sp: list(cls_list) for sp, cls_list in CLASSES.items()}
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT species, class_name FROM custom_classes ORDER BY added_at")
        for sp, cls in cur.fetchall():
            if sp in result and cls not in result[sp]:
                result[sp].append(cls)
    return result
