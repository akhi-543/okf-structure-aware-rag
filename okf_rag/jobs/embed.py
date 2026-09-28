"""Pure file-in/file-out embedding job. Runs locally or on Kaggle. No DB."""
import argparse
from typing import Callable
import numpy as np

# `pyarrow` is imported lazily, inside the functions that need it (embed_table
# and main), rather than at module scope: on Windows, `import pyarrow` before
# `torch` (loaded lazily by sentence_transformers in _bge_encode) leaves the
# process's DLL search state such that torch's `c10.dll` fails to load with
# `OSError: [WinError 1114]`. See `scripts/ingest_corpus.py` for the full
# explanation and precedent.

def embed_table(table, encode: Callable, batch_size: int = 64):
    import pyarrow as pa

    texts = table.column("text").to_pylist()
    if not texts:
        # One-liner: np.concatenate([]) raises on zero batches -- an empty
        # input (e.g. a doc set that produced no chunks) must come back as
        # an empty table with a correctly-typed `vector` column, not crash.
        return table.append_column("vector", pa.array([], pa.list_(pa.float32())))
    vecs = [encode(texts[i:i + batch_size]) for i in range(0, len(texts), batch_size)]
    all_vecs = np.concatenate(vecs).astype(np.float32)
    return table.append_column("vector", pa.array(all_vecs.tolist(), pa.list_(pa.float32())))

def _bge_encode():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    return lambda texts: model.encode(texts, normalize_embeddings=True)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    # Load torch first (via _bge_encode) before importing pyarrow to avoid
    # Windows DLL initialization failure (see module docstring).
    encode = _bge_encode()
    # Now import pyarrow after torch is already loaded.
    import pyarrow.parquet as pq
    embedded = embed_table(pq.read_table(args.inp), encode)
    # Projection at write time (not a change to embed_table's contract):
    # the written parquet doesn't carry a second copy of
    # text/heading_path/etc., which Postgres already has from the ingest
    # side. `doc_id` does stay (finding 4 of the final review): slimming to
    # chunk_id + vector removed the only column that could tell a *reused*
    # chunk_id from the right one, which is what happens when the database
    # is dropped and recreated and BIGSERIAL restarts at 1. Two int64
    # columns next to a 768-float vector cost nothing, and
    # `okf_rag/ingest/load_embeddings.py` now requires doc_id and refuses a
    # parquet whose (chunk_id, doc_id) pairs disagree with the table.
    pq.write_table(embedded.select(["chunk_id", "doc_id", "vector"]), args.out)

if __name__ == "__main__":
    main()
