"""Read-only Postgres access for the web bench.

Two retrieval paths, matching the pilot's evaluated arms:

- ``flat``: dense bge cosine over the bundle's flat chunks (the baseline).
- ``structured``: metadata-aware BM25 over each document's normalized
  frontmatter, built with the same helpers the evaluation run uses
  (``metadata_doc_dict`` -> ``metadata_pseudo_docs_v3`` -> ``structured_rank``).
  The ranked documents' structured chunks are delivered as evidence, lead
  sections of all ranked documents first (see ``order_structured_evidence``).

Every SQL statement here is a SELECT; the web bench never writes.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any

import numpy as np

from okf_rag.retrieve.brightmart_pilot_arms_v3 import (
    metadata_doc_dict,
    metadata_pseudo_docs_v3,
    structured_rank,
)
from okf_rag.retrieve.dense import DenseRetriever
from okf_web_bench.okf_web_bench_compare import ARM_FLAT, ARM_STRUCTURED
from okf_web_bench.okf_web_bench_corpus_map import (
    identical_chunk_pair,
    map_document,
    overlap_ids,
)
from okf_web_bench.okf_web_bench_models import BgeQueryEncoder

_DOCUMENTS_SQL = """
SELECT d.path, d.title,
       COUNT(*) FILTER (WHERE c.policy = 'flat') AS n_flat,
       COUNT(*) FILTER (WHERE c.policy = 'struct') AS n_struct
FROM documents d
JOIN chunks c ON c.doc_id = d.doc_id
WHERE d.bundle = %s
GROUP BY d.doc_id, d.path, d.title
ORDER BY d.path
"""

_CHUNKS_SQL = """
SELECT c.chunk_id, c.doc_id, c.embedding
FROM chunks c
JOIN documents d ON d.doc_id = c.doc_id
WHERE d.bundle = %s AND c.policy = %s AND c.embedding IS NOT NULL
ORDER BY c.chunk_id
"""

_DOC_ID_SQL = """
SELECT doc_id, title FROM documents WHERE bundle = %s AND path = %s
"""

_PATH_CHUNKS_SQL = """
SELECT c.chunk_id, c.ord, c.heading_path, c.text, c.n_tokens
FROM chunks c
WHERE c.doc_id = %s AND c.policy = %s
ORDER BY c.ord
"""

_EDGES_SQL = """
SELECT ds.path AS src_path, dd.path AS dst_path, e.edge_kind
FROM edges e
JOIN documents ds ON ds.doc_id = e.src_doc
JOIN documents dd ON dd.doc_id = e.dst_doc
WHERE (e.src_doc = %s OR e.dst_doc = %s) AND e.provenance = 'authored'
"""

_IDENTICAL_TEXTS_SQL = """
SELECT c.text
FROM chunks c
JOIN documents d ON d.doc_id = c.doc_id
WHERE d.bundle = %s AND d.path = %s AND c.policy = %s
ORDER BY c.chunk_id
"""

_CHUNK_DETAILS_SQL = """
SELECT c.chunk_id, c.doc_id, d.path, c.heading_path, c.text
FROM chunks c
JOIN documents d ON d.doc_id = c.doc_id
WHERE c.chunk_id = ANY(%s)
"""

_METADATA_DOCS_SQL = """
SELECT path, title, description, frontmatter, okf_type, status
FROM documents
WHERE bundle = %s
ORDER BY path
"""

_STRUCT_CHUNKS_FOR_DOCS_SQL = """
SELECT c.chunk_id, c.doc_id, d.path, c.heading_path, c.text, c.ord
FROM chunks c
JOIN documents d ON d.doc_id = c.doc_id
WHERE d.bundle = %s AND d.path = ANY(%s) AND c.policy = 'struct'
ORDER BY c.chunk_id
"""

_DEFAULT_DSN = "postgresql://okf:okf@localhost:5434/okf"


def embedding_to_array(value) -> np.ndarray:
    """Turn a pgvector Vector (or list) into a 1-d float32 row.

    ``np.array(vector, dtype=np.float32)`` calls ``float(vector)`` and
    raises ``TypeError: float() argument must be a string or a real number, not 'Vector'``.
    """
    if isinstance(value, np.ndarray):
        arr = value.astype(np.float32, copy=False)
    elif hasattr(value, "to_list"):
        arr = np.asarray(value.to_list(), dtype=np.float32)
    elif hasattr(value, "tolist"):
        arr = np.asarray(value.tolist(), dtype=np.float32)
    else:
        arr = np.asarray(list(value), dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"embedding must be 1-d, got shape {arr.shape}")
    return arr


def order_structured_evidence(ranked_docs: list[tuple[str, float]], rows: list[tuple]) -> list[dict]:
    """Structured-lane evidence: the structured chunks of the ranked documents,
    lead sections first.

    Order is (section position within its document, document rank): every ranked
    document's first section in rank order, then every document's second section,
    and so on. The context budget therefore reaches all ranked documents before
    it spends tokens on later sections such as long sales tables.

    ``rows`` are ``(chunk_id, doc_id, path, heading_path, text, ord)``; each
    chunk carries its document's structured-ranking score.
    """
    by_doc: dict[str, list[tuple]] = defaultdict(list)
    for chunk_id, doc_id, path, heading_path, text, ord_ in rows:
        by_doc[path].append((ord_, chunk_id, doc_id, heading_path, text))
    keyed = []
    for rank, (path, score) in enumerate(ranked_docs):
        for position, (_ord, chunk_id, doc_id, heading_path, text) in enumerate(sorted(by_doc.get(path, []))):
            keyed.append(((position, rank), {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "doc_path": path,
                "heading_path": heading_path or [],
                "text": text,
                "score": float(score),
            }))
    return [chunk for _key, chunk in sorted(keyed, key=lambda kv: kv[0])]


def default_dsn() -> str:
    import os

    return os.environ.get("OKF_DSN", _DEFAULT_DSN)


def probe_postgres(conn_factory: Callable[[], Any]) -> bool:
    try:
        conn = conn_factory()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
            return True
        finally:
            conn.close()
    except Exception:
        return False


class WebBenchStore:
    def __init__(
        self,
        conn_factory: Callable[[], Any],
        encoder: BgeQueryEncoder,
        *,
        retrieve_override: Callable[[str, str, str, int], list[dict]] | None = None,
    ) -> None:
        self._conn_factory = conn_factory
        self._encoder = encoder
        self._retrieve_override = retrieve_override
        self._retriever_cache: dict[tuple[str, str], DenseRetriever] = {}
        self._metadata_cache: dict[str, Any] = {}
        self._bm25_available = _bm25_constructible()

    def available_arms(self) -> list[str]:
        arms = [ARM_FLAT]
        if self._bm25_available:
            arms.append(ARM_STRUCTURED)
        return arms

    def retrieve(self, corpus: str, arm: str, question: str, k: int) -> list[dict]:
        if self._retrieve_override is not None:
            return self._retrieve_override(corpus, arm, question, k)

        if arm == ARM_STRUCTURED:
            return self._retrieve_structured(corpus, question, k)
        if arm != ARM_FLAT:
            raise ValueError(f"unsupported arm: {arm}")

        if not self._encoder.available:
            raise RuntimeError("bge unavailable")

        retriever = self._get_retriever(corpus, "flat")
        hits = retriever.search(question, k)
        if not hits:
            return []

        chunk_ids = [chunk_id for _, chunk_id, _ in hits]
        score_by_chunk = {chunk_id: score for _, chunk_id, score in hits}
        return self._load_chunk_hits(chunk_ids, score_by_chunk)

    def _retrieve_structured(self, corpus: str, question: str, k: int) -> list[dict]:
        index = self._get_metadata_index(corpus)
        ranked = structured_rank(index, question, k)
        if not ranked:
            return []
        conn = self._conn_factory()
        try:
            with conn.cursor() as cur:
                cur.execute(_STRUCT_CHUNKS_FOR_DOCS_SQL, (corpus, [path for path, _ in ranked]))
                rows = cur.fetchall()
        finally:
            conn.close()
        return order_structured_evidence(ranked, rows)

    def _load_chunk_hits(
        self, chunk_ids: list[int], score_by_chunk: dict[int, float]
    ) -> list[dict]:
        conn = self._conn_factory()
        try:
            with conn.cursor() as cur:
                cur.execute(_CHUNK_DETAILS_SQL, (chunk_ids,))
                rows = cur.fetchall()
        finally:
            conn.close()

        by_id = {
            row[0]: {
                "chunk_id": row[0],
                "doc_id": row[1],
                "doc_path": row[2],
                "heading_path": row[3] or [],
                "text": row[4],
                "score": float(score_by_chunk[row[0]]),
            }
            for row in rows
        }
        return [by_id[cid] for cid in chunk_ids if cid in by_id]

    def list_documents(self, corpus: str) -> list[dict]:
        conn = self._conn_factory()
        try:
            with conn.cursor() as cur:
                cur.execute(_DOCUMENTS_SQL, (corpus,))
                rows = cur.fetchall()
                docs = []
                for path, title, n_flat, n_struct in rows:
                    identical = False
                    if n_flat == 1 and n_struct == 1:
                        identical = self._single_chunk_identical(cur, corpus, path)
                    docs.append(
                        {
                            "path": path,
                            "title": title,
                            "n_flat": int(n_flat),
                            "n_struct": int(n_struct),
                            "identical": identical,
                        }
                    )
                return docs
        finally:
            conn.close()

    def load_document_map(self, corpus: str, path: str) -> dict:
        conn = self._conn_factory()
        try:
            with conn.cursor() as cur:
                cur.execute(_DOC_ID_SQL, (corpus, path))
                row = cur.fetchone()
                if row is None:
                    raise ValueError(f"document not found: {path}")
                doc_id, title = row

                flat_chunks = self._load_path_chunks(cur, doc_id, "flat")
                struct_chunks = self._load_path_chunks(cur, doc_id, "struct")

                cur.execute(_EDGES_SQL, (doc_id, doc_id))
                authored_edges = [
                    {"src_path": src, "dst_path": dst, "edge_kind": kind}
                    for src, dst, kind in cur.fetchall()
                ]
        finally:
            conn.close()

        doc = map_document(
            path=path,
            title=title,
            flat_chunks=flat_chunks,
            struct_chunks=struct_chunks,
            authored_edges=authored_edges,
        )
        doc["overlap_ids"] = {
            "flat": {
                str(ch["chunk_id"]): overlap_ids(ch, struct_chunks)
                for ch in flat_chunks
            },
            "struct": {
                str(ch["chunk_id"]): overlap_ids(ch, flat_chunks)
                for ch in struct_chunks
            },
        }
        return doc

    def _get_retriever(self, corpus: str, policy: str) -> DenseRetriever:
        key = (corpus, policy)
        if key not in self._retriever_cache:
            conn = self._conn_factory()
            try:
                with conn.cursor() as cur:
                    cur.execute(_CHUNKS_SQL, (corpus, policy))
                    rows = cur.fetchall()
            finally:
                conn.close()

            if not rows:
                vectors = np.zeros((0, 768), dtype=np.float32)
                doc_ids: list[int] = []
                chunk_ids: list[int] = []
            else:
                vectors = np.stack([embedding_to_array(row[2]) for row in rows])
                doc_ids = [row[1] for row in rows]
                chunk_ids = [row[0] for row in rows]

            self._retriever_cache[key] = DenseRetriever(
                self._encoder.encode,
                vectors,
                doc_ids,
                chunk_ids,
            )
        return self._retriever_cache[key]

    def _get_metadata_index(self, corpus: str) -> Any:
        bm25_index = _import_bm25_index()
        if bm25_index is None:
            raise RuntimeError("bm25 unavailable")
        if corpus not in self._metadata_cache:
            conn = self._conn_factory()
            try:
                with conn.cursor() as cur:
                    cur.execute(_METADATA_DOCS_SQL, (corpus,))
                    rows = cur.fetchall()
            finally:
                conn.close()
            docs = [metadata_doc_dict(*row) for row in rows]
            self._metadata_cache[corpus] = bm25_index.build(metadata_pseudo_docs_v3(docs))
        return self._metadata_cache[corpus]

    @staticmethod
    def _load_path_chunks(cur, doc_id: int, policy: str) -> list[dict]:
        cur.execute(_PATH_CHUNKS_SQL, (doc_id, policy))
        return [
            {
                "chunk_id": row[0],
                "ord": row[1],
                "heading_path": row[2] or [],
                "text": row[3],
                "n_tokens": row[4],
            }
            for row in cur.fetchall()
        ]

    @staticmethod
    def _single_chunk_identical(cur, corpus: str, path: str) -> bool:
        cur.execute(_IDENTICAL_TEXTS_SQL, (corpus, path, "flat"))
        flat = [row[0] for row in cur.fetchall()]
        cur.execute(_IDENTICAL_TEXTS_SQL, (corpus, path, "struct"))
        struct = [row[0] for row in cur.fetchall()]
        flat_chunks = [{"text": t} for t in flat]
        struct_chunks = [{"text": t} for t in struct]
        return identical_chunk_pair(flat_chunks, struct_chunks)


def _import_bm25_index() -> Any | None:
    try:
        from okf_rag.retrieve.control_arms_v4 import BM25Index

        return BM25Index
    except Exception:
        return None


def _bm25_constructible() -> bool:
    bm25_index = _import_bm25_index()
    if bm25_index is None:
        return False
    try:
        bm25_index.build([])
        return True
    except Exception:
        return False
