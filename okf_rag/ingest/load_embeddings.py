"""Load a GPU-produced embeddings parquet into `chunks.embedding`, and build
the HNSW index once the vectors are in (spec §7, A2).

`okf_rag/jobs/embed.py` runs the actual embedding model (locally or on
Kaggle) and writes a parquet with exactly three columns -- `chunk_id`
(int64), `doc_id` (int64) and `vector` (list_(float32)) -- but never touches
Postgres (`okf_rag/jobs/*` never imports Postgres, by project rule). This
module is the other half of that round trip: it reads that parquet back and
writes the vectors into the `chunks` rows the ingest side already created.

A5 stale-artifact guard: `load_pg.load_chunks` is delete-then-insert per
(doc, policy), so a re-ingest reassigns every `chunk_id` (fresh BIGSERIAL
values). An embeddings parquet produced before that re-ingest still carries
the *old* chunk_ids. Silently updating nothing (or updating the wrong rows,
if ids happened to get reused) would be a hard-to-notice correctness bug, so
`load_embeddings` checks every chunk_id against the table up front and
raises before writing anything if any are unknown.

Why `doc_id` rides along (finding 5 of the final review): the existence
check above is sufficient for the *normal* re-ingest, because BIGSERIAL
never reuses a value -- the old ids simply vanish and the guard fires. It is
not sufficient for a database that was dropped and recreated: the sequence
restarts at 1, so the old ids all exist again but now belong to entirely
different chunks. Existence passes, every vector lands on the wrong row, and
nothing anywhere says so -- the silent, confidently-wrong failure this guard
exists to prevent. Carrying `doc_id` (the one column a chunk cannot change
without being a different chunk) and checking the `(chunk_id, doc_id)` pairs
against the table turns that into a loud refusal. Three int64/float32
columns are still a small file next to the vectors themselves.

`pyarrow` is imported lazily, inside `load_embeddings`, not at module scope:
`scripts/ingest_corpus.py` imports this module at module scope, and that
script already relies on `pyarrow` NOT being imported before it lazily loads
`transformers`/`torch` in its own tokenizer step (see that module's
docstring for the Windows DLL-ordering conflict this avoids). Keeping the
import here function-local preserves that ordering regardless of which mode
of the script runs first.
"""
from collections import Counter
from pathlib import Path

_REQUIRED_COLUMNS = ("chunk_id", "doc_id", "vector")


def _sample(values: list) -> str:
    """Render at most 20 offending ids into an error message."""
    return f"{values[:20]}{' ...' if len(values) > 20 else ''}"


def load_embeddings(conn, parquet_path: Path) -> int:
    """Write `chunk_id, doc_id, vector` rows from `parquet_path` into
    `chunks.embedding`. Returns the number of rows updated.

    Every check below runs before a single UPDATE is issued -- and the
    parquet-only ones before the connection is touched at all -- so a bad
    parquet can never partially write. Raises `ValueError` if:

    - a required column is missing (in particular `doc_id`: a parquet
      written by an older, slimmer `embed.py` cannot be pair-checked and is
      refused rather than trusted);
    - the same `chunk_id` appears twice (one of the two vectors would win
      by row order and the other would be lost silently);
    - a `vector` is null (it would be written through as
      `embedding = NULL`: a chunk that counts as loaded but is invisible to
      every vector query);
    - a `chunk_id` is not present in `chunks` (the A5 stale-artifact guard);
    - a `(chunk_id, doc_id)` pair disagrees with the table (the recreated
      database case -- ids that exist but mean something else now).

    Both module-docstring failure modes are covered by that last pair of
    checks: ids that vanished, and ids that came back meaning something
    different.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    missing_columns = [c for c in _REQUIRED_COLUMNS if c not in table.column_names]
    if missing_columns:
        raise ValueError(
            f"load_embeddings: {parquet_path} is missing required column(s) "
            f"{missing_columns}; expected {list(_REQUIRED_COLUMNS)}. A parquet without "
            "doc_id cannot be checked against the table it is being loaded into "
            "(see this module's docstring) and is refused."
        )

    chunk_ids = table.column("chunk_id").to_pylist()
    doc_ids = table.column("doc_id").to_pylist()
    vectors = table.column("vector").to_pylist()

    duplicates = sorted(c for c, n in Counter(chunk_ids).items() if n > 1)
    if duplicates:
        raise ValueError(
            f"load_embeddings: {parquet_path} contains {len(duplicates)} duplicate "
            f"chunk_id(s): {_sample(duplicates)}"
        )

    null_vectors = sorted(c for c, v in zip(chunk_ids, vectors) if v is None)
    if null_vectors:
        raise ValueError(
            f"load_embeddings: {parquet_path} has a null vector for "
            f"{len(null_vectors)} chunk_id(s): {_sample(null_vectors)}"
        )

    with conn.cursor() as cur:
        cur.execute(
            "SELECT chunk_id, doc_id FROM chunks WHERE chunk_id = ANY(%s)", (chunk_ids,)
        )
        stored_doc_id = dict(cur.fetchall())
        missing = sorted(set(chunk_ids) - set(stored_doc_id))
        if missing:
            raise ValueError(
                f"load_embeddings: {parquet_path} references {len(missing)} chunk_id(s) "
                f"not present in chunks (stale artifact from before a re-ingest?): "
                f"{_sample(missing)}"
            )
        mismatched = sorted(
            c for c, d in zip(chunk_ids, doc_ids) if stored_doc_id[c] != d
        )
        if mismatched:
            raise ValueError(
                f"load_embeddings: {parquet_path} disagrees with chunks on doc_id for "
                f"{len(mismatched)} chunk_id(s): {_sample(mismatched)}. Those chunk_ids "
                "exist but belong to different documents now -- the parquet was built "
                "against a different database (dropped and recreated, so BIGSERIAL "
                "restarted and reused the ids). Re-run embed against the current chunks."
            )
        cur.executemany(
            "UPDATE chunks SET embedding = %s WHERE chunk_id = %s",
            list(zip(vectors, chunk_ids)),
        )
    return len(chunk_ids)


def create_hnsw_index(conn) -> None:
    """Build the HNSW index on `chunks.embedding`, then commit.

    Deliberately not part of `init_schema` (see the comment left in
    `schema.sql`): an HNSW build is much faster once the vectors it indexes
    already exist, so this is called once, after `load_embeddings` has
    populated the column, not at schema-creation time when the table is
    still empty.
    """
    with conn.cursor() as cur:
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_emb ON chunks USING hnsw (embedding vector_cosine_ops)"
        )
    conn.commit()
