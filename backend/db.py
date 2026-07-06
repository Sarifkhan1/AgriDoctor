"""
AgriDoctor AI - Database layer (SQLite).

Extracted from the original monolithic main.py. Provides a connection factory,
a transactional context manager, and schema initialization (idempotent).
"""

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
DB_PATH = Path(os.getenv("DATABASE_PATH", str(DATA_DIR / "agridoctor.db")))

# Ensure the upload tree exists (side effect kept minimal and explicit).
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")  # better concurrency for read/write mix
    return conn


@contextmanager
def get_db():
    """Transactional connection: commits on success, rolls back on error."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    full_name TEXT,
    password_hash TEXT NOT NULL,
    role TEXT DEFAULT 'user',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS cases (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    category TEXT CHECK (category IN ('crop', 'livestock')),
    crop_name TEXT,
    animal_type TEXT,
    status TEXT DEFAULT 'created',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS media (
    id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL,
    type TEXT CHECK (type IN ('image', 'speech', 'audio', 'video')),
    uri TEXT NOT NULL,
    content_type TEXT,
    duration_sec REAL,
    quality_score REAL,
    transcript TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (case_id) REFERENCES cases(id)
);

CREATE TABLE IF NOT EXISTS metadata (
    case_id TEXT PRIMARY KEY,
    onset_days INTEGER,
    spread_speed TEXT,
    weather_json TEXT,
    treatments_json TEXT,
    notes_text TEXT,
    language TEXT,
    FOREIGN KEY (case_id) REFERENCES cases(id)
);

CREATE TABLE IF NOT EXISTS predictions (
    id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL,
    model_version TEXT,
    provider TEXT,
    model_id TEXT,
    kind TEXT,
    detected_crop TEXT,
    crop_supported INTEGER,
    is_leaf INTEGER,
    primary_label TEXT,
    disease_name TEXT,
    secondary_labels_json TEXT,
    severity_score REAL,
    urgency_level TEXT,
    confidence REAL,
    visual_evidence TEXT,
    advice_json TEXT,
    raw_response TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (case_id) REFERENCES cases(id)
);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL,
    status TEXT DEFAULT 'queued',
    progress INTEGER DEFAULT 0,
    error TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (case_id) REFERENCES cases(id)
);

CREATE INDEX IF NOT EXISTS idx_cases_user ON cases(user_id);
CREATE INDEX IF NOT EXISTS idx_media_case ON media(case_id);
CREATE INDEX IF NOT EXISTS idx_predictions_case ON predictions(case_id);
"""


def init_db() -> None:
    """Create tables and indexes if they do not exist (idempotent)."""
    with get_db() as conn:
        conn.executescript(_SCHEMA)
