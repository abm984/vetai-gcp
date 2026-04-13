-- ──────────────────────────────────────────────────────────────────────────────
-- schema.sql — PostgreSQL schema for VetApp
--
-- This file is for reference / manual bootstrapping only.
-- The application creates all tables automatically via database.init_tables()
-- on startup.  Running this script directly is optional.
--
-- Usage:
--   psql $DATABASE_URL < infra/schema.sql
-- ──────────────────────────────────────────────────────────────────────────────

-- MD5 hashes of every image processed (deduplication)
CREATE TABLE IF NOT EXISTS seen_hashes (
    hash     TEXT PRIMARY KEY,
    added_at TEXT NOT NULL
);

-- Images with no caption → vet must label from scratch
CREATE TABLE IF NOT EXISTS preclean (
    id           SERIAL PRIMARY KEY,
    filepath     TEXT UNIQUE NOT NULL,   -- GCS path: preclean/<filename>
    species      TEXT,                   -- extracted from caption (may be null)
    caption      TEXT,
    sender_id    TEXT,
    sender_name  TEXT,
    wa_timestamp TEXT,
    queued_at    TEXT,
    vet_species  TEXT,                   -- filled by vet
    vet_label    TEXT,                   -- filled by vet
    vet_id       TEXT,
    reviewed_at  TEXT,
    notes        TEXT,
    status       TEXT DEFAULT 'pending'  -- pending | approved | rejected
);

-- Low-confidence predictions → vet approves / corrects / rejects
CREATE TABLE IF NOT EXISTS vet_queue (
    id           SERIAL PRIMARY KEY,
    filepath     TEXT UNIQUE NOT NULL,   -- GCS path: vet_queue/<filename>
    species      TEXT,
    pred_label   TEXT,                   -- model's top-1 prediction
    confidence   REAL,
    sender_id    TEXT,
    sender_name  TEXT,
    wa_timestamp TEXT,
    queued_at    TEXT,
    vet_label    TEXT,                   -- filled by vet
    vet_id       TEXT,
    reviewed_at  TEXT,
    status       TEXT DEFAULT 'pending'  -- pending | approved | corrected | rejected
);

-- User-defined disease classes added via the dashboard
CREATE TABLE IF NOT EXISTS custom_classes (
    id         SERIAL PRIMARY KEY,
    species    TEXT NOT NULL,
    class_name TEXT NOT NULL,
    added_at   TEXT,
    UNIQUE (species, class_name)
);

-- Full audit log of every processed image
CREATE TABLE IF NOT EXISTS dataset_log (
    id           SERIAL PRIMARY KEY,
    filename     TEXT,
    species      TEXT,
    label        TEXT,
    sender_id    TEXT,
    sender_name  TEXT,
    wa_timestamp TEXT,
    processed_at TEXT,
    status       TEXT,   -- accepted | vet_queue | preclean | rejected
    reason       TEXT
);

-- Cached split counts to avoid expensive GCS listing during split balancing
CREATE TABLE IF NOT EXISTS dataset_counts (
    species TEXT NOT NULL,
    label   TEXT NOT NULL,
    split   TEXT NOT NULL,   -- train | valid | test
    count   INTEGER DEFAULT 0,
    PRIMARY KEY (species, label, split)
);

-- Useful indexes
CREATE INDEX IF NOT EXISTS idx_preclean_status  ON preclean  (status);
CREATE INDEX IF NOT EXISTS idx_vet_queue_status ON vet_queue (status);
CREATE INDEX IF NOT EXISTS idx_log_status       ON dataset_log (status);
CREATE INDEX IF NOT EXISTS idx_log_processed_at ON dataset_log (processed_at DESC);
