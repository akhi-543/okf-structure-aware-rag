"""Brightmart pilot run: Postgres read-back -> arms -> doc-level metrics -> verdict.

Order (A5): ingest -> embed -> load embeddings -> this script. Refuses to run
if any chunk of the bundle lacks a vector (a re-ingest invalidated them).
Only bundle synthetic_retail_pilot is read. No gold field reaches a
non-oracle arm: gold is loaded only in `score_runs` and the oracle ceiling.

Usage:
    python scripts/run_brightmart_pilot_v1.py
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

# Run this repository's packages even if another checkout providing `okf_rag` is
# pip-installed: `python scripts/x.py` puts scripts/, not the repo root, on sys.path.
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import csv
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

from okf_rag.eval.stats import bootstrap_mean_ci
from okf_rag.retrieve.brightmart_pilot_arms_v1 import (
    STRUCTURE_ARMS, collapse_to_docs, doc_metrics, metadata_pseudo_docs, verdict,
)
from okf_rag.retrieve.control_arms_v4 import (
    BM25Index, EdgeSet, graph_expanded_retrieval, hierarchy_prefix_retrieval, reciprocal_rank_fusion,
)

BUNDLE = "synthetic_retail_pilot"
K_CHUNKS = 50
K_DOCS = 10
# Settings match the real-corpus runs (scripts/run_control_arms_v4.py ~line 149):
# k=10, seed_depth=5, hops=1, expansion_budget=5. At K_CHUNKS=50 with 25 seeds
# retained, the 25 seed chunks already cover >=10 distinct docs, so every
# graph-reached doc lands past position 10 and the arm is a silent no-op.
GRAPH = {"k": 10, "seed_depth": 5, "hops": 1, "expansion_budget": 5}


def _sha(path: Path) -> str:
    """Compute SHA256 hash of file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _to_float32(v) -> np.ndarray:
    """Convert vector to 1-D float32 ndarray.

    Handles pgvector.Vector (from psycopg), ndarray, or list.
    """
    if hasattr(v, 'to_numpy'):  # pgvector.Vector
        return np.asarray(v.to_numpy(), dtype=np.float32)
    return np.asarray(v, dtype=np.float32)


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _oracle_child_dir(parent_path: str) -> str:
    """Derive child directory from parent document path.

    Example: "regions/region-northeast.md" -> "regions/northeast/"
    Example: "departments/dept-home-garden.md" -> "departments/home-garden/"
    """
    prefix = parent_path.rsplit("/", 1)[0] + "/" if "/" in parent_path else parent_path
    parent_stem = parent_path.rsplit("/", 1)[-1].removesuffix(".md")
    child_dir = prefix + parent_stem.split("-", 1)[-1] + "/"
    return child_dir


def build_manifest(*, git_head: str, n_docs: int, n_chunks_flat: int, n_chunks_struct: int, n_edges: int,
                   emb_manifest: dict, input_hashes: dict, arms: list[str]) -> dict:
    """Build the run manifest with all required fields per spec §5.

    Args:
        git_head: git HEAD revision
        n_docs: total documents
        n_chunks_flat: flat policy chunks
        n_chunks_struct: struct policy chunks
        n_edges: authored edges
        emb_manifest: embedding manifest dict with model_id, revision, query_instruction_prefix, rows
        input_hashes: dict of input file paths to SHA256 hashes
        arms: list of arm names executed

    Returns:
        dict ready to serialize to manifest.json
    """
    n_chunks = n_chunks_flat + n_chunks_struct
    n_queries = emb_manifest["rows"]["queries.parquet"]

    return {
        "bundle": BUNDLE,
        "git_head": git_head,
        "inputs": input_hashes,
        "embedding": {
            "model_id": emb_manifest["model_id"],
            "revision": emb_manifest["revision"],
            "query_instruction_prefix": emb_manifest["query_instruction_prefix"],
        },
        "rows": {
            "n_docs": n_docs,
            "n_chunks": n_chunks,
            "n_chunks_flat": n_chunks_flat,
            "n_chunks_struct": n_chunks_struct,
            "n_queries": n_queries,
            "n_edges": n_edges,
        },
        "k_chunks": K_CHUNKS,
        "k_docs": K_DOCS,
        "graph": GRAPH,
        "arms": sorted(arms),
        "structure_arms": list(STRUCTURE_ARMS),
        "oracle_arms": ["C_hier_oracle"],
    }


def _graph_arm(r1_chunk_ids: list[str], struct_chunks: list[dict], edges: EdgeSet) -> list[str]:
    """R3a doc ranking: graph-expand R1's chunk ranking, then collapse to docs.

    Pure wrapper around `graph_expanded_retrieval` using the fixed GRAPH
    settings (see GRAPH docstring comment above) so main() and the unit
    tests call the exact same code path.
    """
    g = graph_expanded_retrieval(r1_chunk_ids, struct_chunks, edges, arm_name="R3a", **GRAPH)
    return collapse_to_docs(g.ranked_doc_paths, K_DOCS)


def _graph_noop(runs: dict[str, dict[str, list[str]]]) -> bool:
    """True iff R3a produced a doc ranking identical to R1 on every query."""
    return all(runs["R3a"][qid] == runs["R1"][qid] for qid in runs["R1"])


def score_runs(runs, questions, qrels):
    rows = []
    scores = {"S1_t": defaultdict(dict), "S34_t": defaultdict(dict), "S34_p": defaultdict(dict)}
    for arm, by_q in runs.items():
        for q in questions:
            qid, stratum = q["query_id"], q["stratum"]
            if stratum == "S5":
                continue
            m = doc_metrics(by_q[qid], qrels[qid])
            rows.append({"arm": arm, "query_id": qid, "stratum": stratum,
                         "set": "p" if qid.endswith("-p") else "t", **m})
            if stratum == "S1" and qid.endswith("-t"):
                scores["S1_t"][arm][qid] = m["nDCG@10"]
            elif stratum in ("S3", "S4"):
                scores["S34_p" if qid.endswith("-p") else "S34_t"][arm][qid] = m["nDCG@10"]
    return rows, {k: dict(v) for k, v in scores.items()}


def _read_bundle(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT doc_id, path, title, description, frontmatter FROM documents WHERE bundle = %s", (BUNDLE,))
        docs = cur.fetchall()
        cur.execute("""SELECT c.chunk_id, c.policy, d.path, c.text, c.embedding FROM chunks c
                       JOIN documents d ON d.doc_id = c.doc_id WHERE d.bundle = %s
                       ORDER BY c.chunk_id""", (BUNDLE,))
        chunks = cur.fetchall()
        cur.execute("""SELECT s.path, t.path FROM edges e JOIN documents s ON s.doc_id = e.src_doc
                       JOIN documents t ON t.doc_id = e.dst_doc
                       WHERE s.bundle = %s AND e.provenance = 'authored'""", (BUNDLE,))
        edges = cur.fetchall()
    if not docs:
        raise SystemExit(f"no documents under bundle {BUNDLE!r}: run the ingest first")
    missing = sum(1 for c in chunks if c[4] is None)
    if missing:
        raise SystemExit(f"{missing} chunks lack embeddings: run embed + load_embeddings (A5 order)")
    return docs, chunks, edges


def _precompute_dense_matrices(by_policy):
    """Precompute embedding matrices for dense retrieval."""
    matrices = {}
    for policy, chunks in by_policy.items():
        if chunks:
            matrices[policy] = np.stack([np.asarray(c["vector"], dtype=np.float32) for c in chunks])
        else:
            matrices[policy] = np.zeros((0, 0), dtype=np.float32)
    return matrices


def _dense(chunk_rows, mat, qvec):
    """Retrieve top-k chunks via cosine similarity on precomputed matrix."""
    if mat.shape[0] == 0:
        return []
    order = np.argsort(-(mat @ qvec), kind="stable")[:K_CHUNKS]
    return [chunk_rows[i] for i in order]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default=None)
    ap.add_argument("--emb-dir", default="results/brightmart-pilot-v1/embeddings")
    ap.add_argument("--out-dir", default="results/brightmart-pilot-v1")
    args = ap.parse_args()
    out = Path(args.out_dir)
    if (out / "verdict.json").exists():
        raise SystemExit(f"{out}/verdict.json exists: use a new versioned --out-dir")

    emb_dir = Path(args.emb_dir)
    emb_manifest_path = emb_dir / "manifest.json"
    if not emb_manifest_path.exists():
        raise SystemExit(f"embedding manifest not found at {emb_manifest_path}; run scripts/embed_brightmart_pilot_v1.py first")
    emb_manifest = json.loads(emb_manifest_path.read_text(encoding="utf-8"))

    from okf_rag.ingest.load_pg import get_conn
    conn = get_conn(args.dsn) if args.dsn else get_conn()
    docs, chunks, edge_rows = _read_bundle(conn)

    import pyarrow.parquet as pq
    qt = pq.read_table(emb_dir / "queries.parquet").to_pydict()
    qvecs = {qid: _to_float32(v) for qid, v in zip(qt["query_id"], qt["vector"])}

    questions = _jsonl(Path("doc_synthetic/brightmart_questions.jsonl"))
    qrels = defaultdict(set)
    for r in _jsonl(Path("doc_synthetic/brightmart_qrels.jsonl")):
        qrels[r["query_id"]].add(r["doc_path"])
    s4_paths = defaultdict(list)
    for r in _jsonl(Path("doc_synthetic/brightmart_gold_spans.jsonl")):
        if r.get("path"):
            s4_paths[r["query_id"]].append(r["path"])

    by_policy = defaultdict(list)
    for chunk_id, policy, path, text, emb in chunks:
        by_policy[policy].append({"chunk_id": str(chunk_id), "doc_path": path, "text": text, "vector": _to_float32(emb)})
    doc_dicts = [{"path": p, "title": t, "description": d, "tags": (fm or {}).get("tags", []),
                  "frontmatter": {k: v for k, v in (fm or {}).items() if k != "tags"}}
                 for _, p, t, d, fm in docs]
    meta_index = BM25Index.build(metadata_pseudo_docs(doc_dicts))
    flat_bm25 = BM25Index.build(by_policy["flat"])
    edges = EdgeSet.from_triples("authored", edge_rows)
    all_paths = [p for _, p, *_ in docs]

    # Precompute dense matrices and BM25 chunk_id -> doc_path lookup
    matrices = _precompute_dense_matrices(by_policy)
    flat_bm25_lookup = {cid: flat_bm25.doc_paths[i] for i, cid in enumerate(flat_bm25.chunk_ids)}

    # Compute input file hashes
    input_hashes = {
        "doc_synthetic/brightmart_questions.jsonl": _sha(Path("doc_synthetic/brightmart_questions.jsonl")),
        "doc_synthetic/brightmart_qrels.jsonl": _sha(Path("doc_synthetic/brightmart_qrels.jsonl")),
        "doc_synthetic/brightmart_gold_spans.jsonl": _sha(Path("doc_synthetic/brightmart_gold_spans.jsonl")),
        (emb_dir / "queries.parquet").as_posix(): _sha(emb_dir / "queries.parquet"),
    }

    runs = defaultdict(dict)
    oracle = {}
    for q in questions:
        qid, text, v = q["query_id"], q["text"], qvecs[q["query_id"]]
        r0 = _dense(by_policy["flat"], matrices["flat"], v)
        r1 = _dense(by_policy["struct"], matrices["struct"], v)
        r0_docs = collapse_to_docs([c["doc_path"] for c in r0], K_DOCS)
        r1_docs = collapse_to_docs([c["doc_path"] for c in r1], K_DOCS)
        r2_docs = collapse_to_docs([p.split("meta:", 1)[1] for p, _ in meta_index.search(text, K_CHUNKS)], K_DOCS)
        r3_docs = _graph_arm([c["chunk_id"] for c in r1], by_policy["struct"], edges)
        bm_flat = collapse_to_docs(
            [flat_bm25_lookup[cid] for cid, _ in flat_bm25.search(text, K_CHUNKS)], K_DOCS)
        runs["R0"][qid] = r0_docs
        runs["R1"][qid] = r1_docs
        runs["R2"][qid] = r2_docs
        runs["R3a"][qid] = r3_docs
        runs["R4"][qid] = reciprocal_rank_fusion([r1_docs, r2_docs, r3_docs], k=K_DOCS)
        runs["C_flat_hybrid"][qid] = reciprocal_rank_fusion([r0_docs, bm_flat], k=K_DOCS)
        if q["stratum"] == "S4" and s4_paths.get(qid):
            ranked = []
            for chain in s4_paths[qid]:
                child_dir = _oracle_child_dir(chain[-1])
                res = hierarchy_prefix_retrieval(by_policy["struct"], all_paths, child_dir, K_CHUNKS,
                                                 selector_provenance="gold_s4_path")
                ranked += res.ranked_doc_paths
            oracle[qid] = collapse_to_docs(ranked, K_DOCS)

    if _graph_noop(runs):
        raise SystemExit("graph arm identical to R1 on all queries: wiring bug")

    rows, scores = score_runs(runs, questions, qrels)
    v = verdict(scores)
    out.mkdir(parents=True, exist_ok=True)
    _write_outputs(out, runs, rows, v, oracle, qrels, conn, emb_manifest, input_hashes, len(by_policy["flat"]), len(by_policy["struct"]), len(edge_rows))


def _write_outputs(out, runs, rows, v, oracle, qrels, conn, emb_manifest, input_hashes, n_chunks_flat, n_chunks_struct, n_edges):
    with (out / "runs.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for arm in sorted(runs):
            for qid in sorted(runs[arm]):
                f.write(json.dumps({"arm": arm, "query_id": qid, "ranked": runs[arm][qid]}) + "\n")
    keys = ["arm", "query_id", "stratum", "set", "P@1", "P@5", "P@10", "R@5", "R@10", "MRR@10", "nDCG@10"]
    with (out / "metrics_per_query.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["arm"], r["stratum"], r["set"])].append(r["nDCG@10"])
    with (out / "metrics_summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "stratum", "set", "n", "nDCG@10_mean", "ci_low", "ci_high"])
        for (arm, st, s), vals in sorted(groups.items()):
            ci = bootstrap_mean_ci(vals)
            w.writerow([arm, st, s, len(vals), f"{np.mean(vals):.4f}", f"{ci['ci_low']:.4f}", f"{ci['ci_high']:.4f}"])
    with (out / "oracle_ceiling.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "capability", "query_id", "nDCG@10"])
        for qid, ranked in sorted(oracle.items()):
            w.writerow(["C_hier_oracle", "oracle_diagnostic", qid, f"{doc_metrics(ranked, qrels[qid])['nDCG@10']:.4f}"])
    (out / "verdict.json").write_text(json.dumps(v, indent=2), encoding="utf-8")
    (out / "verdict.md").write_text(_verdict_md(v, out.name), encoding="utf-8")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM documents WHERE bundle=%s", (BUNDLE,))
        n_docs = cur.fetchone()[0]
    manifest = build_manifest(git_head=head, n_docs=n_docs, n_chunks_flat=n_chunks_flat,
                               n_chunks_struct=n_chunks_struct, n_edges=n_edges, emb_manifest=emb_manifest,
                               input_hashes=input_hashes, arms=list(runs.keys()))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(_verdict_md(v, out.name))


def _verdict_md(v, out_name: str) -> str:
    def pf(x):
        return "PASS" if x else "FAIL"
    lines = [f"# Brightmart pilot - verdict ({out_name})", "",
             "Synthetic companion corpus, n = 20 per stratum (40 pooled for S3+S4). Companion evidence only;",
             "a FAIL on C2 may reflect low power rather than absence of effect.", "",
             f"Thresholds: `{json.dumps(v['thresholds'])}`", "",
             f"- **C1 flat competitive on S1:** {pf(v['C1']['pass'])}"]
    for a, r in v["C1"]["arms"].items():
        lines.append(f"  - R0 - {a}: delta {r['delta']:+.3f}, CI [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]")
    lines.append(f"- **C2 structure helps on S3+S4:** {pf(v['C2']['pass'])} (winners: {v['C2']['winning_arms'] or 'none'})")
    for a, r in v["C2"]["arms"].items():
        lines.append(f"  - {a} - R0: delta {r['delta']:+.3f}, CI [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}], "
                     f"p {r['p']:.4f}, Holm {'sig' if r['holm_significant'] else 'ns'}")
    lines.append(f"- **C3 survives paraphrase:** {pf(v['C3']['pass'])} {v['C3']['reason']}")
    for a, r in v["C3"]["arms"].items():
        lines.append(f"  - {a} - R0 (paraphrase): delta {r['delta']:+.3f} vs template {r['template_delta']:+.3f}, "
                     f"CI [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] -> {pf(r['pass'])}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
