"""Brightmart pilot v3 run: v2 pipeline plus fixed structure arms (R1c/R2s/R3d/R4c).

Two modes:
    --mode dev      v2 questions, v2 query vectors. Overwritable. No verdict
                     (development set - not confirmatory).
    --mode heldout  Frozen held-out questions, held-out query vectors. Refuses
                     if results/brightmart-pilot-v3/verdict.json already exists.
                     Verdict uses only R1c/R2s/R3d/R4c (STRUCTURE_ARMS_V3).

Both modes read Postgres bundle synthetic_retail_pilot read-only. No writes,
no re-ingest. Order (A5): ingest -> embed -> embed_brightmart_ctx_v3 -> this
script. Refuses if emb_struct_ctx.parquet is stale relative to the bundle's
struct chunks (a re-ingest reassigns chunk_id via BIGSERIAL).

Usage:
    python -m scripts.run_brightmart_pilot_v3 --mode dev
    python -m scripts.run_brightmart_pilot_v3 --mode heldout
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

# Run this repository's packages even if another checkout providing `okf_rag` is
# pip-installed: `python scripts/x.py` puts scripts/, not the repo root, on sys.path.
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import csv
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

from okf_rag.eval.stats import bootstrap_mean_ci
from okf_rag.retrieve.brightmart_pilot_arms_v1 import (
    STRUCTURE_ARMS, collapse_to_docs, doc_metrics, metadata_pseudo_docs, verdict,
)
from okf_rag.retrieve.brightmart_pilot_arms_v3 import (
    STRUCTURE_ARMS_V3, doc_graph_arm, metadata_doc_dict, metadata_pseudo_docs_v3, structured_rank,
    undirected_neighbors,
)
from okf_rag.retrieve.control_arms_v4 import BM25Index, EdgeSet, hierarchy_prefix_retrieval, reciprocal_rank_fusion
from scripts.run_brightmart_pilot_v1 import (
    BUNDLE, K_CHUNKS, K_DOCS, _dense, _graph_arm, _graph_noop, _jsonl, _oracle_child_dir,
    _precompute_dense_matrices, _read_bundle, _sha, _to_float32, _verdict_md, build_manifest, score_runs,
)

V2_ARMS = ("R1", "R2", "R3a", "R4", "C_flat_hybrid")


def v3_arms(q_text: str, qvec: np.ndarray, ctx: dict) -> dict[str, list[str]]:
    """Pure: doc rankings for all ten arms given a prepared context. No I/O.

    ctx keys: flat, struct, struct_ctx (chunk-dict lists), matrices
    (_precompute_dense_matrices output), meta_v2 (BM25Index over
    metadata_pseudo_docs), meta_v3 (BM25Index over metadata_pseudo_docs_v3),
    flat_bm25 (BM25Index over flat chunks), edges (EdgeSet), neighbors
    (undirected_neighbors output).
    """
    matrices = ctx["matrices"]

    # v2 arms, computed exactly as run_brightmart_pilot_v1.main does.
    r0 = _dense(ctx["flat"], matrices["flat"], qvec)
    r1 = _dense(ctx["struct"], matrices["struct"], qvec)
    r0_docs = collapse_to_docs([c["doc_path"] for c in r0], K_DOCS)
    r1_docs = collapse_to_docs([c["doc_path"] for c in r1], K_DOCS)
    r2_docs = collapse_to_docs([p.split("meta:", 1)[1] for p, _ in ctx["meta_v2"].search(q_text, K_CHUNKS)], K_DOCS)
    r3a_docs = _graph_arm([c["chunk_id"] for c in r1], ctx["struct"], ctx["edges"])
    flat_bm25 = ctx["flat_bm25"]
    flat_bm25_lookup = {cid: flat_bm25.doc_paths[i] for i, cid in enumerate(flat_bm25.chunk_ids)}
    bm_flat = collapse_to_docs([flat_bm25_lookup[cid] for cid, _ in flat_bm25.search(q_text, K_CHUNKS)], K_DOCS)
    r4_docs = reciprocal_rank_fusion([r1_docs, r2_docs, r3a_docs], k=K_DOCS)
    c_flat_hybrid = reciprocal_rank_fusion([r0_docs, bm_flat], k=K_DOCS)

    # v3 fixed arms.
    r1c_chunks = _dense(ctx["struct_ctx"], matrices["struct_ctx"], qvec)
    r1c_chunk_paths = [c["doc_path"] for c in r1c_chunks]
    r1c_all_docs = collapse_to_docs(r1c_chunk_paths, len(r1c_chunk_paths))  # no K_DOCS cut, for R3d
    r1c_docs = collapse_to_docs(r1c_chunk_paths, K_DOCS)
    r3d_docs = doc_graph_arm(r1c_all_docs, ctx["neighbors"], k_docs=K_DOCS, n_seeds=5)
    r2s_docs = [p for p, _ in structured_rank(ctx["meta_v3"], q_text, K_DOCS, K_CHUNKS)]
    r4c_docs = reciprocal_rank_fusion([r1c_docs, r2s_docs, r3d_docs], k=K_DOCS)

    return {
        "R0": r0_docs, "R1": r1_docs, "R2": r2_docs, "R3a": r3a_docs, "R4": r4_docs,
        "C_flat_hybrid": c_flat_hybrid,
        "R1c": r1c_docs, "R2s": r2s_docs, "R3d": r3d_docs, "R4c": r4c_docs,
    }


def _annotate_v3_manifest(manifest: dict) -> dict:
    """Overwrite build_manifest()'s v1 structure_arms (R1/R2/R3a/R4 -- this is a
    v3 run) with the v3 verdict arms, and record the v2 arms separately as
    reference-only so the manifest never misreads as covering v1's arms."""
    manifest["structure_arms"] = list(STRUCTURE_ARMS_V3)
    manifest["reference_arms_v2"] = list(STRUCTURE_ARMS)
    return manifest


def _ctx_vectors_fresh(db_pairs, parquet_pairs) -> bool:
    """True iff emb_struct_ctx.parquet's (chunk_id, doc_id) pairs exactly match
    the bundle's struct chunks. A re-ingest reassigns chunk_id (BIGSERIAL), so
    any mismatch means the parquet was built against a different ingest."""
    return set(db_pairs) == set(parquet_pairs)


def _load_ctx_vectors(v3_emb_dir: Path):
    import pyarrow.parquet as pq
    t = pq.read_table(v3_emb_dir / "emb_struct_ctx.parquet").to_pydict()
    vecs_by_chunk = {int(cid): _to_float32(v) for cid, v in zip(t["chunk_id"], t["vector"])}
    doc_by_chunk = {int(cid): int(did) for cid, did in zip(t["chunk_id"], t["doc_id"])}
    return vecs_by_chunk, doc_by_chunk


def _build_ctx(conn, ctx_vecs_by_chunk, ctx_doc_by_chunk):
    docs, chunks, edge_rows = _read_bundle(conn)
    with conn.cursor() as cur:
        cur.execute("SELECT path, okf_type, status FROM documents WHERE bundle = %s", (BUNDLE,))
        okf_meta = {p: (ot, st) for p, ot, st in cur.fetchall()}
        cur.execute("""SELECT c.chunk_id, c.doc_id FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
                       WHERE d.bundle = %s AND c.policy = 'struct'""", (BUNDLE,))
        struct_chunk_doc = cur.fetchall()

    if not _ctx_vectors_fresh(struct_chunk_doc, list(ctx_doc_by_chunk.items())):
        raise SystemExit("R1c vectors stale: rerun scripts/embed_brightmart_ctx_v3.py")

    by_policy = defaultdict(list)
    for chunk_id, policy, path, text, emb in chunks:
        by_policy[policy].append({"chunk_id": str(chunk_id), "doc_path": path, "text": text, "vector": _to_float32(emb)})

    doc_path_by_id = {doc_id: path for doc_id, path, *_ in docs}
    struct_ctx_chunks = [
        {"chunk_id": str(chunk_id), "doc_path": doc_path_by_id[doc_id], "vector": ctx_vecs_by_chunk[chunk_id]}
        for chunk_id, doc_id in sorted(struct_chunk_doc)
    ]

    doc_dicts = [metadata_doc_dict(p, t, d, fm, okf_meta[p][0], okf_meta[p][1]) for _, p, t, d, fm in docs]

    matrices = _precompute_dense_matrices(
        {"flat": by_policy["flat"], "struct": by_policy["struct"], "struct_ctx": struct_ctx_chunks})

    ctx = {
        "flat": by_policy["flat"], "struct": by_policy["struct"], "struct_ctx": struct_ctx_chunks,
        "matrices": matrices,
        "meta_v2": BM25Index.build(metadata_pseudo_docs(doc_dicts)),
        "meta_v3": BM25Index.build(metadata_pseudo_docs_v3(doc_dicts)),
        "flat_bm25": BM25Index.build(by_policy["flat"]),
        "edges": EdgeSet.from_triples("authored", edge_rows),
        "neighbors": undirected_neighbors(edge_rows),
    }
    return ctx, docs, edge_rows, len(by_policy["flat"]), len(by_policy["struct"])


def _write_runs_and_per_query(out: Path, runs, rows):
    out.mkdir(parents=True, exist_ok=True)
    with (out / "runs.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for arm in sorted(runs):
            for qid in sorted(runs[arm]):
                f.write(json.dumps({"arm": arm, "query_id": qid, "ranked": runs[arm][qid]}) + "\n")
    keys = ["arm", "query_id", "stratum", "set", "P@1", "P@5", "P@10", "R@5", "R@10", "MRR@10", "nDCG@10"]
    with (out / "metrics_per_query.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _write_summary_csv(path: Path, rows, header_comment: str | None = None):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["arm"], r["stratum"], r["set"])].append(r["nDCG@10"])
    with path.open("w", encoding="utf-8", newline="") as f:
        if header_comment:
            f.write(header_comment + "\n")
        w = csv.writer(f)
        w.writerow(["arm", "stratum", "set", "n", "nDCG@10_mean", "ci_low", "ci_high"])
        for (arm, st, s), vals in sorted(groups.items()):
            ci = bootstrap_mean_ci(vals)
            w.writerow([arm, st, s, len(vals), f"{np.mean(vals):.4f}", f"{ci['ci_low']:.4f}", f"{ci['ci_high']:.4f}"])


def _write_dev_outputs(out: Path, runs, rows):
    _write_runs_and_per_query(out, runs, rows)
    _write_summary_csv(out / "dev_metrics_summary.csv", rows, header_comment="# development set - not confirmatory")


def _write_heldout_outputs(out: Path, runs, rows, v, oracle, qrels, docs, n_chunks_flat, n_chunks_struct,
                            n_edges, emb_manifest_for_build, input_hashes):
    _write_runs_and_per_query(out, runs, rows)
    _write_summary_csv(out / "metrics_summary.csv", rows)
    _write_summary_csv(out / "reference_v2_arms.csv", [r for r in rows if r["arm"] in V2_ARMS])
    with (out / "oracle_ceiling.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "capability", "query_id", "nDCG@10"])
        for qid, ranked in sorted(oracle.items()):
            w.writerow(["C_hier_oracle", "oracle_diagnostic", qid, f"{doc_metrics(ranked, qrels[qid])['nDCG@10']:.4f}"])
    (out / "verdict.json").write_text(json.dumps(v, indent=2), encoding="utf-8")
    (out / "verdict.md").write_text(_verdict_md(v, out.name), encoding="utf-8")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    manifest = build_manifest(git_head=head, n_docs=len(docs), n_chunks_flat=n_chunks_flat,
                               n_chunks_struct=n_chunks_struct, n_edges=n_edges, emb_manifest=emb_manifest_for_build,
                               input_hashes=input_hashes, arms=list(runs.keys()))
    manifest = _annotate_v3_manifest(manifest)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(_verdict_md(v, out.name))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dev", "heldout"], required=True)
    ap.add_argument("--dsn", default=None)
    ap.add_argument("--v2-emb-dir", default="results/brightmart-pilot-v1/embeddings")
    ap.add_argument("--v3-emb-dir", default="results/brightmart-pilot-v3/embeddings")
    args = ap.parse_args()

    v3_emb_dir = Path(args.v3_emb_dir)
    v3_emb_manifest_path = v3_emb_dir / "manifest.json"
    if not v3_emb_manifest_path.exists():
        raise SystemExit(f"embedding manifest not found at {v3_emb_manifest_path}; "
                          "run scripts/embed_brightmart_ctx_v3.py first")
    v3_emb_manifest = json.loads(v3_emb_manifest_path.read_text(encoding="utf-8"))

    if args.mode == "dev":
        out = Path("results/brightmart-pilot-v3-dev")
        qvec_path = Path(args.v2_emb_dir) / "queries.parquet"
        q_path = Path("doc_synthetic/brightmart_questions.jsonl")
        qrels_path = Path("doc_synthetic/brightmart_qrels.jsonl")
        gold_spans_path = Path("doc_synthetic/brightmart_gold_spans.jsonl")
    else:
        out = Path("results/brightmart-pilot-v3")
        if (out / "verdict.json").exists():
            raise SystemExit(f"{out}/verdict.json exists: v3 held-out run is one-shot")
        qvec_path = v3_emb_dir / "queries_heldout.parquet"
        q_path = Path("doc_synthetic/brightmart_heldout_questions.jsonl")
        qrels_path = Path("doc_synthetic/brightmart_heldout_qrels.jsonl")
        gold_spans_path = Path("doc_synthetic/brightmart_heldout_gold_spans.jsonl")

    from okf_rag.ingest.load_pg import get_conn
    conn = get_conn(args.dsn) if args.dsn else get_conn()

    ctx_vecs_by_chunk, ctx_doc_by_chunk = _load_ctx_vectors(v3_emb_dir)
    ctx, docs, edge_rows, n_chunks_flat, n_chunks_struct = _build_ctx(conn, ctx_vecs_by_chunk, ctx_doc_by_chunk)
    all_paths = [p for _, p, *_ in docs]

    import pyarrow.parquet as pq
    qt = pq.read_table(qvec_path).to_pydict()
    qvecs = {qid: _to_float32(v) for qid, v in zip(qt["query_id"], qt["vector"])}

    questions = _jsonl(q_path)
    qrels = defaultdict(set)
    for r in _jsonl(qrels_path):
        qrels[r["query_id"]].add(r["doc_path"])
    s4_paths = defaultdict(list)
    if args.mode == "heldout":
        for r in _jsonl(gold_spans_path):
            if r.get("path"):
                s4_paths[r["query_id"]].append(r["path"])

    runs = defaultdict(dict)
    oracle = {}
    for q in questions:
        qid, text, qvec = q["query_id"], q["text"], qvecs[q["query_id"]]
        for arm, ranked in v3_arms(text, qvec, ctx).items():
            runs[arm][qid] = ranked
        if args.mode == "heldout" and q["stratum"] == "S4" and s4_paths.get(qid):
            ranked = []
            for chain in s4_paths[qid]:
                child_dir = _oracle_child_dir(chain[-1])
                res = hierarchy_prefix_retrieval(ctx["struct"], all_paths, child_dir, K_CHUNKS,
                                                 selector_provenance="gold_s4_path")
                ranked += res.ranked_doc_paths
            oracle[qid] = collapse_to_docs(ranked, K_DOCS)

    if _graph_noop(runs):
        raise SystemExit("graph arm identical to R1 on all queries: wiring bug")
    if all(runs["R3d"][qid] == runs["R1c"][qid] for qid in runs["R1c"]):
        raise SystemExit("R3d identical to R1c on all queries: wiring bug")
    n_docs = len(all_paths)
    for qid, ranked in runs["R3d"].items():
        if len(ranked) != min(10, n_docs):
            raise SystemExit(f"R3d for {qid} has {len(ranked)} entries, expected {min(10, n_docs)}")

    rows, scores = score_runs(runs, questions, qrels)

    if args.mode == "dev":
        _write_dev_outputs(out, runs, rows)
        return

    v = verdict(scores, arms=STRUCTURE_ARMS_V3)
    emb_manifest_for_build = {
        "model_id": v3_emb_manifest["model_id"],
        "revision": v3_emb_manifest["revision"],
        "query_instruction_prefix": v3_emb_manifest["query_instruction_prefix"],
        "rows": {"queries.parquet": v3_emb_manifest["rows"]["queries_heldout.parquet"]},
    }
    input_hashes = {
        q_path.as_posix(): _sha(q_path),
        qrels_path.as_posix(): _sha(qrels_path),
        gold_spans_path.as_posix(): _sha(gold_spans_path),
        (v3_emb_dir / "manifest.json").as_posix(): _sha(v3_emb_dir / "manifest.json"),
        (Path(args.v2_emb_dir) / "manifest.json").as_posix(): _sha(Path(args.v2_emb_dir) / "manifest.json"),
    }
    _write_heldout_outputs(out, runs, rows, v, oracle, qrels, docs, n_chunks_flat, n_chunks_struct,
                           len(edge_rows), emb_manifest_for_build, input_hashes)


if __name__ == "__main__":
    main()
