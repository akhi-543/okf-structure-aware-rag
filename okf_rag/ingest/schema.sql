CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
  doc_id BIGSERIAL PRIMARY KEY,
  bundle TEXT NOT NULL,
  path TEXT NOT NULL,
  okf_type TEXT NOT NULL,
  title TEXT NOT NULL,
  description TEXT,
  resource TEXT,
  status TEXT NOT NULL DEFAULT 'stable',
  frontmatter JSONB NOT NULL DEFAULT '{}',
  body_md TEXT NOT NULL,
  breadcrumb TEXT[] NOT NULL DEFAULT '{}',
  depth INT NOT NULL DEFAULT 0,
  git_last_modified TIMESTAMPTZ,
  char_len INT NOT NULL,
  UNIQUE (bundle, path)
);

CREATE TABLE IF NOT EXISTS chunks (
  chunk_id BIGSERIAL PRIMARY KEY,
  doc_id BIGINT NOT NULL REFERENCES documents(doc_id),
  policy TEXT NOT NULL CHECK (policy IN ('flat','struct')),
  ord INT NOT NULL,
  heading_path TEXT[] NOT NULL DEFAULT '{}',
  text TEXT NOT NULL,
  n_tokens INT NOT NULL,
  embedding vector(768)
);

CREATE TABLE IF NOT EXISTS edges (
  src_doc BIGINT NOT NULL REFERENCES documents(doc_id),
  dst_doc BIGINT NOT NULL REFERENCES documents(doc_id),
  edge_kind TEXT NOT NULL
    CONSTRAINT edges_edge_kind_check
    CHECK (edge_kind IN ('child','curated','related','prose','sibling')),
  weight REAL NOT NULL,
  provenance TEXT NOT NULL CHECK (provenance IN ('authored','extracted'))
);

-- ---------------------------------------------------------------------------
-- Constraint policy for this file: ADDITIVE, CATALOG-GUARDED SYNC.
-- Not drop-and-recreate, and deliberately not a migration tool.
--
-- Every constraint above lives inside a `CREATE TABLE IF NOT EXISTS`, so a
-- database created before the constraint was written to this file never
-- acquires it -- the statement is a silent no-op on the table that already
-- exists. That is exactly what happened to `edges_edge_kind_check` (final
-- review, finding 5): the integration test passed only because its target
-- database happened to be created after that commit, while any older
-- database still accepted an arbitrary `edge_kind` string.
--
-- The policy, for this and any constraint added to an existing table later:
-- add it explicitly below, guarded by a `pg_constraint` catalog lookup so
-- the statement is idempotent, and `NOT VALID` so a pre-existing database's
-- historical rows are not re-checked (they were written before the rule
-- existed; every future INSERT and UPDATE is checked either way). On a
-- fresh database the CREATE TABLE above has already created the identically
-- named constraint, so the guard finds it and does nothing.
-- ---------------------------------------------------------------------------
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conrelid = 'edges'::regclass AND conname = 'edges_edge_kind_check'
  ) THEN
    ALTER TABLE edges ADD CONSTRAINT edges_edge_kind_check
      CHECK (edge_kind IN ('child','curated','related','prose','sibling')) NOT VALID;
  END IF;
END $$;

CREATE TABLE IF NOT EXISTS queries (
  query_id TEXT PRIMARY KEY,
  corpus TEXT NOT NULL,
  stratum TEXT NOT NULL,
  text TEXT NOT NULL,
  origin TEXT NOT NULL,
  reference_claim TEXT,
  reference_answer TEXT,
  source_paths TEXT[] NOT NULL DEFAULT '{}',
  benchmark_fingerprint TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS qrels (
  query_id TEXT NOT NULL REFERENCES queries(query_id),
  doc_id BIGINT NOT NULL REFERENCES documents(doc_id),
  relevance INT NOT NULL,
  CONSTRAINT qrels_pk PRIMARY KEY (query_id, doc_id)
);

-- ---------------------------------------------------------------------------
-- Guard the qrels primary key constraint (idempotent on pre-existing tables).
-- On a fresh database, CREATE TABLE above has already created the constraint.
-- On a pre-existing database that needs the table added later, this guard
-- adds it using the same catalog-lookup pattern as edges_edge_kind_check.
-- ---------------------------------------------------------------------------
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conrelid = 'qrels'::regclass AND conname = 'qrels_pk'
  ) THEN
    ALTER TABLE qrels ADD CONSTRAINT qrels_pk
      PRIMARY KEY (query_id, doc_id);
  END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_fm ON documents USING GIN (frontmatter);
CREATE INDEX IF NOT EXISTS idx_edges_src ON edges (src_doc);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges (dst_doc);
-- HNSW index created AFTER bulk embedding load (faster build):
-- CREATE INDEX idx_emb ON chunks USING hnsw (embedding vector_cosine_ops);
