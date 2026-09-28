"""CPU embeddings for Brightmart pilot v3 (R1c + held-out queries).

Reads struct chunks of bundle synthetic_retail_pilot from Postgres READ-ONLY
and embeds contextual_text(title, heading_path, text). Never writes to the DB:
the vectors live only in the output parquet. Torch loads before pyarrow.

Usage:
    python scripts/embed_brightmart_ctx_v3.py
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

# Run this repository's packages even if another checkout providing `okf_rag` is
# pip-installed: `python scripts/x.py` puts scripts/, not the repo root, on sys.path.
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import hashlib
import json
from pathlib import Path

from okf_rag.jobs.model_lock import model_pin
from okf_rag.retrieve.brightmart_pilot_arms_v3 import contextual_text

BUNDLE = "synthetic_retail_pilot"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default=None)
    ap.add_argument("--questions", default="doc_synthetic/brightmart_heldout_questions.jsonl")
    ap.add_argument("--out-dir", default="results/brightmart-pilot-v3/embeddings")
    args = ap.parse_args()

    pin = model_pin("embedding")
    from sentence_transformers import SentenceTransformer  # torch first
    model = SentenceTransformer(pin.model_id, revision=pin.revision, device="cpu")

    from okf_rag.ingest.load_pg import get_conn
    conn = get_conn(args.dsn) if args.dsn else get_conn()
    with conn.cursor() as cur:
        cur.execute("""SELECT c.chunk_id, c.doc_id, d.title, c.heading_path, c.text
                       FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
                       WHERE d.bundle = %s AND c.policy = 'struct' ORDER BY c.chunk_id""", (BUNDLE,))
        rows = cur.fetchall()
    conn.close()
    if not rows:
        raise SystemExit(f"no struct chunks under bundle {BUNDLE!r}")

    texts = [contextual_text(t, list(hp or []), x) for _, _, t, hp, x in rows]
    vecs = model.encode(texts, normalize_embeddings=True, batch_size=32)

    qpath = Path(args.questions)
    qs = [json.loads(l) for l in qpath.read_text(encoding="utf-8").splitlines() if l.strip()]
    qvecs = model.encode([QUERY_PREFIX + q["text"] for q in qs], normalize_embeddings=True, batch_size=32)

    import pyarrow as pa
    import pyarrow.parquet as pq
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({
        "chunk_id": pa.array([r[0] for r in rows], pa.int64()),
        "doc_id": pa.array([r[1] for r in rows], pa.int64()),
        "vector": pa.array([v.tolist() for v in vecs], pa.list_(pa.float32())),
    }), out / "emb_struct_ctx.parquet")
    pq.write_table(pa.table({
        "query_id": [q["query_id"] for q in qs],
        "vector": pa.array([v.tolist() for v in qvecs], pa.list_(pa.float32())),
    }), out / "queries_heldout.parquet")
    manifest = {
        "model_id": pin.model_id, "revision": pin.revision, "device": "cpu",
        "normalize_embeddings": True, "query_instruction_prefix": QUERY_PREFIX,
        "passage_format": "contextual_text(title, heading_path, text)", "bundle": BUNDLE,
        "inputs": {qpath.as_posix(): hashlib.sha256(qpath.read_bytes()).hexdigest()},
        "rows": {"emb_struct_ctx.parquet": len(rows), "queries_heldout.parquet": len(qs)},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["rows"]))


if __name__ == "__main__":
    main()
