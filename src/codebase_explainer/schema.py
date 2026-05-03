"""SQLite schema for the symbol/call graph index.

The index stores everything the agent's tools query against:
- files: every source file we've parsed
- symbols: functions, classes, methods (with parent links for nesting)
- imports: import statements per file
- calls: caller -> callee edges, both textual and resolved

Designed to be tiny, queryable from cold without ORM, and easy to drop and
rebuild incrementally as the indexer evolves.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 1

DDL = [
    """
    CREATE TABLE IF NOT EXISTS schema_meta (
        version INTEGER PRIMARY KEY,
        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS files (
        id INTEGER PRIMARY KEY,
        path TEXT NOT NULL UNIQUE,
        language TEXT NOT NULL,
        content_hash TEXT,
        indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS symbols (
        id INTEGER PRIMARY KEY,
        file_id INTEGER NOT NULL,
        kind TEXT NOT NULL,
        name TEXT NOT NULL,
        qualified_name TEXT NOT NULL,
        parent_id INTEGER,
        start_line INTEGER NOT NULL,
        end_line INTEGER NOT NULL,
        signature TEXT,
        docstring TEXT,
        FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE,
        FOREIGN KEY (parent_id) REFERENCES symbols(id) ON DELETE SET NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name)",
    "CREATE INDEX IF NOT EXISTS idx_symbols_qualified_name ON symbols(qualified_name)",
    "CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(file_id)",
    """
    CREATE TABLE IF NOT EXISTS imports (
        id INTEGER PRIMARY KEY,
        file_id INTEGER NOT NULL,
        module TEXT NOT NULL,
        name TEXT,
        alias TEXT,
        line INTEGER NOT NULL,
        FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS calls (
        id INTEGER PRIMARY KEY,
        caller_id INTEGER NOT NULL,
        callee_name TEXT NOT NULL,
        callee_id INTEGER,
        line INTEGER NOT NULL,
        FOREIGN KEY (caller_id) REFERENCES symbols(id) ON DELETE CASCADE,
        FOREIGN KEY (callee_id) REFERENCES symbols(id) ON DELETE SET NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_calls_callee_name ON calls(callee_name)",
    "CREATE INDEX IF NOT EXISTS idx_calls_caller ON calls(caller_id)",
    "CREATE INDEX IF NOT EXISTS idx_calls_callee_id ON calls(callee_id)",
]


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open a connection with row_factory=Row and foreign keys enabled."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: str | Path) -> None:
    """Create all tables and indexes if they don't already exist.

    Idempotent: safe to call repeatedly. Records the schema version once.
    """
    with connect(db_path) as conn:
        for ddl in DDL:
            conn.execute(ddl)
        conn.execute(
            "INSERT OR IGNORE INTO schema_meta (version) VALUES (?)",
            (SCHEMA_VERSION,),
        )
        conn.commit()
