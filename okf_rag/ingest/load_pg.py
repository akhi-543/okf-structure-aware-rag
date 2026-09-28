"""Postgres storage layer (spec §7): documents, chunks, edges.

Single Postgres instance carries all three retrieval signals -- pgvector for
dense embeddings, JSONB with a GIN index for authored frontmatter, and an
edges table for the authored link graph -- so that later phases comparing
retrieval arms on latency are comparing apples to apples (one storage
engine, not several).

`psycopg` / `pgvector` are imported lazily, inside the two functions that
actually need the driver (`get_conn`, `init_schema`), rather than at module
scope (I10). `tests/ingest/test_load_pg.py` imports this module at
collection time, before pytest's `-m 'not integration'` deselection ever
applies -- a module-level `import psycopg` would make the whole suite fail
to collect on a `pip install -e .[dev]` machine that never installed the
optional `db` extra. The remaining functions here (`load_documents`,
`load_edges`, `load_chunks`) only use the generic DB-API `conn.cursor()`
surface and need no driver import at all.
"""
from pathlib import Path
import json

from okf_rag.transcode.model import ConceptDoc, Edge


def get_conn(dsn: str = "postgresql://okf:okf@localhost:5434/okf"):
    """Open a connection with the pgvector type adapter registered.

    Without this, a `vector` column reads back as a plain string instead of
    a numeric array, which surfaces as a confusing type error far from its
    cause (Task 10+ do vector arithmetic on rows loaded through this
    function). We register eagerly here so every caller gets a correctly
    typed connection without having to remember.

    Registration requires the `vector` extension to already exist in the
    database, and `get_conn` may be the very first call against a brand new
    database (before `init_schema` has ever run). We tolerate that one
    specific condition: pgvector's `register_vector` raises a
    `psycopg.ProgrammingError` locally (not from a server round-trip) when
    the extension's `vector` type isn't in `pg_type` yet, and such a
    locally-raised error carries no SQLSTATE -- `e.sqlstate is None` -- so
    that is what we key on, re-raising anything with a real SQLSTATE (e.g.
    an insufficient-privilege or other catalog-access failure raised by the
    server) so it isn't swallowed silently and doesn't surface as a
    confusing type error later.
    `init_schema` creates the extension and calls `register_vector` again
    once it exists, so the adapter is registered by the time any caller
    could plausibly touch a `vector` column. Re-registering on a connection
    that already has it (the common case: extension already exists) is a
    harmless no-op.

    Caveat: a connection returned here while the extension was still absent
    stays unregistered for its whole lifetime unless someone later calls
    `init_schema` on that same connection object -- `get_conn` itself never
    retries. No current call site obtains a connection and skips
    `init_schema`, so this isn't reachable in practice, but it's worth
    knowing if that ever changes.
    """
    import psycopg
    from pgvector.psycopg import register_vector

    conn = psycopg.connect(dsn)
    try:
        register_vector(conn)
    except psycopg.ProgrammingError as e:
        # A locally-raised error (pgvector's own signal, not a server
        # response) carries no SQLSTATE -- e.sqlstate is None -- which is a
        # more precise discriminant than sniffing the message text for
        # "vector type not found" (fragile: a real server-raised error
        # could coincidentally contain the same words and get swallowed).
        if e.sqlstate is not None:
            raise
        # Fresh database, `vector` extension not created yet. init_schema
        # will create it and register the adapter on this connection.
    return conn


def init_schema(conn) -> None:
    from pgvector.psycopg import register_vector

    sql = (Path(__file__).parent / "schema.sql").read_text(encoding="utf-8")
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()
    # The extension now certainly exists (created above); make sure this
    # connection has the vector adapter registered even if get_conn ran
    # before the extension existed. Idempotent if already registered.
    register_vector(conn)


def load_documents(conn, bundle: str, docs: list[ConceptDoc]) -> dict[str, int]:
    ids: dict[str, int] = {}
    with conn.cursor() as cur:
        for d in docs:
            breadcrumb = d.path.split("/")[:-1]
            cur.execute(
                """INSERT INTO documents (bundle, path, okf_type, title, description, resource,
                       status, frontmatter, body_md, breadcrumb, depth, char_len)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (bundle, path) DO UPDATE SET
                       okf_type = EXCLUDED.okf_type,
                       title = EXCLUDED.title,
                       description = EXCLUDED.description,
                       resource = EXCLUDED.resource,
                       status = EXCLUDED.status,
                       frontmatter = EXCLUDED.frontmatter,
                       body_md = EXCLUDED.body_md,
                       breadcrumb = EXCLUDED.breadcrumb,
                       depth = EXCLUDED.depth,
                       char_len = EXCLUDED.char_len
                   RETURNING doc_id""",
                (bundle, d.path, d.okf_type, d.title, d.description, d.resource,
                 d.status, json.dumps(d.x_source, default=str), d.body,
                 breadcrumb, len(breadcrumb), len(d.body)),
            )
            ids[d.path] = cur.fetchone()[0]
    return ids


def load_edges(conn, edges: list[Edge], provenance: str, path_ids: dict[str, int]) -> None:
    """Load edges for the given provenance, scoped to the documents in
    `path_ids` (the full set of documents this ingestion run just loaded).

    Idempotent (C2): re-running ingestion on an unchanged or updated corpus
    must not double the graph. Delete-then-insert, scoped to
    `src_doc IN path_ids.values() AND provenance = provenance`, is used
    instead of a unique constraint because it also correctly drops an edge
    that *disappeared* between runs (a source document's link removed
    upstream) -- a unique constraint alone would silently leave that stale
    edge behind since nothing would violate it.
    """
    rows = [(path_ids[e.src], path_ids[e.dst], e.kind, e.weight, provenance)
            for e in edges if e.src in path_ids and e.dst in path_ids]
    src_ids = sorted(set(path_ids.values()))
    with conn.cursor() as cur:
        if src_ids:
            cur.execute(
                "DELETE FROM edges WHERE src_doc = ANY(%s) AND provenance = %s",
                (src_ids, provenance),
            )
        cur.executemany(
            "INSERT INTO edges (src_doc, dst_doc, edge_kind, weight, provenance) VALUES (%s,%s,%s,%s,%s)",
            rows,
        )


def load_chunks(conn, doc_id: int, policy: str, chunks: list[tuple[int, list[str], str, int]]) -> None:
    """Load chunks for one (doc_id, policy). Idempotent (C2): delete any
    chunks already stored for this doc_id+policy before inserting the fresh
    set, so a re-run doesn't append a second full copy (see `load_edges`
    docstring for why delete-then-insert beats a unique constraint here).
    """
    with conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE doc_id = %s AND policy = %s", (doc_id, policy))
        cur.executemany(
            "INSERT INTO chunks (doc_id, policy, ord, heading_path, text, n_tokens) VALUES (%s,%s,%s,%s,%s,%s)",
            [(doc_id, policy, o, hp, t, n) for (o, hp, t, n) in chunks],
        )
