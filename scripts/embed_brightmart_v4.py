"""CPU embeddings for one Brightmart v4 bundle (next iteration).

Writes, under --out-dir (default results/brightmart-v4/embeddings/<bundle>):
- emb_flat.parquet, emb_struct.parquet: chunk_id, doc_id, vector for
  `ingest_corpus.py --load-embeddings` (from the chunk parquets the ingest exported);
- emb_struct_hp.parquet: struct chunks embedded as heading path (without the page
  title) + text, the R1h input (T5); read from Postgres READ-ONLY and never loaded
  back into it;
- queries_dev.parquet, queries_heldout.parquet: query vectors (bge query prefix);
- manifest.json: model pin, rows, input hashes and wall-clock timings (T9).

Torch loads before pyarrow (Windows DLL rule). Order: ingest -> this script ->
load emb_flat/emb_struct -> run.

Usage:
    python scripts/embed_brightmart_v4.py --bundle synthetic_retail_v4
    python scripts/embed_brightmart_v4.py --bundle synthetic_retail_v4_noparent
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
import time
from pathlib import Path

from okf_rag.jobs.model_lock import model_pin
from okf_rag.retrieve.brightmart_v4_arms import heading_path_text

BUNDLES = ("synthetic_retail_v4", "synthetic_retail_v4_noparent")
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
QUESTION_FILES = {"dev": "doc_synthetic/brightmart_v4_dev_questions.jsonl",
                  "heldout": "doc_synthetic/brightmart_v4_heldout_questions.jsonl"}


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description="CPU embeddings for a Brightmart v4 bundle.")
    ap.add_argument("--bundle", choices=BUNDLES, required=True)
    ap.add_argument("--dsn", default=None)
    ap.add_argument("--chunks-dir", default="results")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    out = Path(args.out_dir or f"results/brightmart-v4/embeddings/{args.bundle}")

    pin = model_pin("embedding")
    from sentence_transformers import SentenceTransformer  # torch first
    t0 = time.perf_counter()
    model = SentenceTransformer(pin.model_id, revision=pin.revision, device="cpu")
    load_s = time.perf_counter() - t0

    def encode(texts):
        return model.encode(texts, normalize_embeddings=True, batch_size=32)

    from okf_rag.ingest.load_pg import get_conn
    conn = get_conn(args.dsn) if args.dsn else get_conn()
    with conn.cursor() as cur:
        cur.execute("""SELECT c.chunk_id, c.doc_id, d.title, c.heading_path, c.text
                       FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
                       WHERE d.bundle = %s AND c.policy = 'struct' ORDER BY c.chunk_id""", (args.bundle,))
        hp_rows = cur.fetchall()
    conn.close()
    if not hp_rows:
        raise SystemExit(f"no struct chunks under bundle {args.bundle!r}: run the ingest first")

    import pyarrow as pa
    import pyarrow.parquet as pq
    from okf_rag.jobs.embed import embed_table

    out.mkdir(parents=True, exist_ok=True)
    manifest = {"model_id": pin.model_id, "revision": pin.revision, "device": "cpu",
                "normalize_embeddings": True, "query_instruction_prefix": QUERY_PREFIX,
                "bundle": args.bundle, "inputs": {}, "rows": {}, "seconds": {"model_load": round(load_s, 2)}}

    for policy in ("flat", "struct"):
        src = Path(args.chunks_dir) / f"chunks_{args.bundle}_{policy}.parquet"
        t0 = time.perf_counter()
        emb = embed_table(pq.read_table(src), encode).select(["chunk_id", "doc_id", "vector"])
        manifest["seconds"][f"emb_{policy}"] = round(time.perf_counter() - t0, 2)
        pq.write_table(emb, out / f"emb_{policy}.parquet")
        manifest["inputs"][src.as_posix()] = _sha(src)
        manifest["rows"][f"emb_{policy}.parquet"] = emb.num_rows

    t0 = time.perf_counter()
    hp_vecs = encode([heading_path_text(t, list(hp or []), x) for _, _, t, hp, x in hp_rows])
    manifest["seconds"]["emb_struct_hp"] = round(time.perf_counter() - t0, 2)
    pq.write_table(pa.table({
        "chunk_id": pa.array([r[0] for r in hp_rows], pa.int64()),
        "doc_id": pa.array([r[1] for r in hp_rows], pa.int64()),
        "vector": pa.array([v.tolist() for v in hp_vecs], pa.list_(pa.float32())),
    }), out / "emb_struct_hp.parquet")
    manifest["rows"]["emb_struct_hp.parquet"] = len(hp_rows)
    manifest["passage_format_hp"] = "heading_path_text(title, heading_path, text)"

    for which, qfile in QUESTION_FILES.items():
        qpath = Path(qfile)
        qs = [json.loads(line) for line in qpath.read_text(encoding="utf-8").splitlines() if line.strip()]
        t0 = time.perf_counter()
        vecs = encode([QUERY_PREFIX + q["text"] for q in qs])
        manifest["seconds"][f"queries_{which}"] = round(time.perf_counter() - t0, 2)
        pq.write_table(pa.table({"query_id": [q["query_id"] for q in qs],
                                 "vector": pa.array([v.tolist() for v in vecs], pa.list_(pa.float32()))}),
                       out / f"queries_{which}.parquet")
        manifest["inputs"][qpath.as_posix()] = _sha(qpath)
        manifest["rows"][f"queries_{which}.parquet"] = len(qs)

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"rows": manifest["rows"], "seconds": manifest["seconds"]}))


if __name__ == "__main__":
    main()
