# scripts/embed_brightmart_pilot_v1.py
"""CPU embeddings for the Brightmart pilot (bundle synthetic_retail_pilot).

Writes chunk vectors in the exact format `okf_rag.ingest.load_embeddings`
expects (chunk_id, doc_id, vector) plus query vectors keyed by query_id.
Never connects to Postgres. Torch loads before pyarrow (Windows DLL rule).

Usage:
    python scripts/embed_brightmart_pilot_v1.py
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

BUNDLE = "synthetic_retail_pilot"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks-dir", default="results")
    ap.add_argument("--questions", default="doc_synthetic/brightmart_questions.jsonl")
    ap.add_argument("--out-dir", default="results/brightmart-pilot-v1/embeddings")
    args = ap.parse_args()

    pin = model_pin("embedding")
    from sentence_transformers import SentenceTransformer  # torch first
    model = SentenceTransformer(pin.model_id, revision=pin.revision, device="cpu")

    def encode(texts):
        return model.encode(texts, normalize_embeddings=True, batch_size=32)

    import pyarrow as pa
    import pyarrow.parquet as pq
    from okf_rag.jobs.embed import embed_table

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"model_id": pin.model_id, "revision": pin.revision, "device": "cpu",
                "normalize_embeddings": True, "passage_prefix": None,
                "query_instruction_prefix": QUERY_PREFIX, "bundle": BUNDLE, "inputs": {}, "rows": {}}

    for policy in ("flat", "struct"):
        src = Path(args.chunks_dir) / f"chunks_{BUNDLE}_{policy}.parquet"
        table = pq.read_table(src)
        emb = embed_table(table, encode).select(["chunk_id", "doc_id", "vector"])
        dst = out / f"emb_{policy}.parquet"
        pq.write_table(emb, dst)
        manifest["inputs"][src.as_posix()] = _sha(src)
        manifest["rows"][dst.name] = emb.num_rows

    qpath = Path(args.questions)
    qs = [json.loads(line) for line in qpath.read_text(encoding="utf-8").splitlines() if line.strip()]
    vecs = encode([QUERY_PREFIX + q["text"] for q in qs])
    qtable = pa.table({"query_id": [q["query_id"] for q in qs],
                       "vector": pa.array([v.tolist() for v in vecs], pa.list_(pa.float32()))})
    pq.write_table(qtable, out / "queries.parquet")
    manifest["inputs"][qpath.as_posix()] = _sha(qpath)
    manifest["rows"]["queries.parquet"] = qtable.num_rows

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["rows"]))


if __name__ == "__main__":
    main()
