"""Brightmart v4 evaluation run (next iteration, tests T1-T6 and T9).

Reads bundles synthetic_retail_v4 and synthetic_retail_v4_noparent from Postgres
READ-ONLY, runs every v4 configuration, scores documents and applies the
preregistered criteria (docs/brightmart_v4_prereg.md).

Modes:
    --mode dev      development questions -> results/brightmart-v4-dev (overwritable).
                    Chooses the R7f rrf_k on the development templates and writes it to
                    rrf_tuning.json; --freeze-tuning also writes configs/brightmart_v4_tuning.json
                    (once) for the held-out run.
    --mode heldout  frozen held-out questions -> results/brightmart-v4. One-shot: refuses if
                    verdict.json exists, if the corpus or held-out files differ from
                    configs/brightmart_v4_freeze.json, or if the tuning file is missing.
    --mode smoke    the held-out code path on the development questions ->
                    results/brightmart-v4-smoke (overwritable; for checking the pipeline
                    without looking at held-out results).

Optional GPU artifacts (scripts/llm_brightmart_v4.py), used when present:
    results/brightmart-v4/llm/graph_extracted.jsonl   -> R3x (T6)
    results/brightmart-v4/llm/filters_<set>.jsonl     -> R6l (T3, exploratory)

Order (A5): ingest -> embed_brightmart_v4 -> load embeddings -> [llm extract/filters] -> this.

Usage:
    python -m scripts.run_brightmart_v4 --mode dev
    python -m scripts.run_brightmart_v4 --mode heldout
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
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from okf_rag.eval.stats import bootstrap_mean_ci
from okf_rag.retrieve import brightmart_v4_arms as A
from okf_rag.retrieve.brightmart_pilot_arms_v3 import metadata_doc_dict, metadata_pseudo_docs_v3, structured_rank
from okf_rag.retrieve.control_arms_v4 import BM25Index, hierarchy_prefix_retrieval, reciprocal_rank_fusion
from okf_rag.transcode.brightmart_synthetic import corpus_sha256
from scripts.run_brightmart_pilot_v1 import _jsonl, _oracle_child_dir, _sha, _to_float32

STANDARD = "synthetic_retail_v4"
NOPARENT = "synthetic_retail_v4_noparent"
CORPUS_DIRS = {STANDARD: "results/corpus_v4", NOPARENT: "results/corpus_v4_noparent"}
FREEZE = Path("configs/brightmart_v4_freeze.json")
TUNING = Path("configs/brightmart_v4_tuning.json")
EMB_ROOT = Path("results/brightmart-v4/embeddings")
LLM_DIR = Path("results/brightmart-v4/llm")
QFILES = {s: {k: Path(f"doc_synthetic/brightmart_v4_{s}_{k}.jsonl") for k in ("questions", "qrels", "gold_spans")}
          for s in ("dev", "heldout")}
OUT = {"dev": Path("results/brightmart-v4-dev"), "heldout": Path("results/brightmart-v4"),
       "smoke": Path("results/brightmart-v4-smoke")}
NOPARENT_ARMS = ("R0", "R2s", "R3s", "R5p", "R6f")
T7_ARMS = ("R0", "R2s")
T7_TOP = 10
SUMMARY_METRICS = ("nDCG@10", "R@50", "F1@10")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_bundle(conn, bundle: str, emb_dir: Path) -> dict:
    with conn.cursor() as cur:
        cur.execute("""SELECT path, title, description, okf_type, status, frontmatter FROM documents
                       WHERE bundle = %s ORDER BY path""", (bundle,))
        docs = cur.fetchall()
        cur.execute("""SELECT c.chunk_id, c.doc_id, c.policy, d.path, c.heading_path, c.text, c.ord, c.embedding,
                              c.n_tokens
                       FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
                       WHERE d.bundle = %s ORDER BY c.chunk_id""", (bundle,))
        chunks = cur.fetchall()
        cur.execute("""SELECT s.path, t.path, e.edge_kind FROM edges e
                       JOIN documents s ON s.doc_id = e.src_doc JOIN documents t ON t.doc_id = e.dst_doc
                       WHERE s.bundle = %s AND e.provenance = 'authored' ORDER BY s.path, t.path, e.edge_kind""",
                    (bundle,))
        edges = [tuple(r) for r in cur.fetchall()]
    if not docs:
        raise SystemExit(f"no documents under bundle {bundle!r}: run the ingest first")
    if any(c[7] is None for c in chunks):
        raise SystemExit(f"{bundle}: chunks lack embeddings: run embed + load_embeddings (A5 order)")

    import pyarrow.parquet as pq
    hp = pq.read_table(emb_dir / "emb_struct_hp.parquet").to_pydict()
    hp_pairs = {(int(c), int(d)) for c, d in zip(hp["chunk_id"], hp["doc_id"])}
    struct_pairs = {(c[0], c[1]) for c in chunks if c[2] == "struct"}
    if hp_pairs != struct_pairs:
        raise SystemExit(f"{bundle}: emb_struct_hp.parquet is stale: rerun scripts/embed_brightmart_v4.py")
    hp_vec = {int(c): _to_float32(v) for c, v in zip(hp["chunk_id"], hp["vector"])}

    b = {"bundle": bundle, "edges": edges, "t_build": {}}
    b["meta"] = {p: {"title": t, "type": ot, "status": st, "fm": fm or {}} for p, t, _, ot, st, fm in docs}
    t0 = time.perf_counter()
    b["meta_index"] = BM25Index.build(metadata_pseudo_docs_v3(
        [metadata_doc_dict(p, t, d, fm, ot, st) for p, t, d, ot, st, fm in docs]))
    b["t_build"]["R2s"] = time.perf_counter() - t0
    by_policy = defaultdict(list)
    for cid, did, policy, path, hpath, text, ord_, emb, ntok in chunks:
        by_policy[policy].append({"chunk_id": cid, "doc_id": did, "doc_path": path, "heading_path": hpath or [],
                                  "text": text, "ord": ord_, "vector": _to_float32(emb), "n_tokens": ntok})
    b["flat"], b["struct"] = by_policy["flat"], by_policy["struct"]
    t0 = time.perf_counter()
    b["mat_flat"] = np.stack([c["vector"] for c in b["flat"]])
    b["t_build"]["R0"] = time.perf_counter() - t0
    b["mat_struct"] = np.stack([c["vector"] for c in b["struct"]])
    t0 = time.perf_counter()
    b["mat_hp"] = np.stack([hp_vec[c["chunk_id"]] for c in b["struct"]])
    b["t_build"]["R1h"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    b["children"] = A.children_by_parent(edges)
    b["neighbors"] = A.undirected_adjacency(edges)
    b["t_build"]["graph"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    b["ftypes"] = A.field_types(b["meta"])
    b["t_build"]["R6f"] = time.perf_counter() - t0
    b["struct_doc_idx"] = defaultdict(list)
    for i, c in enumerate(b["struct"]):
        b["struct_doc_idx"][c["doc_path"]].append(i)
    b["sizes"] = {
        "flat_chunks": len(b["flat"]), "struct_chunks": len(b["struct"]), "edges": len(edges),
        "dense_bytes_flat": int(b["mat_flat"].nbytes), "dense_bytes_struct": int(b["mat_struct"].nbytes),
        "bm25_postings_meta": sum(len(tf) for tf in b["meta_index"]._tf),
        "bm25_vocab_meta": len(b["meta_index"]._df),
    }
    return b


def load_query_vectors(emb_dir: Path, which: str) -> dict[str, np.ndarray]:
    import pyarrow.parquet as pq
    t = pq.read_table(emb_dir / f"queries_{which}.parquet").to_pydict()
    return {qid: _to_float32(v) for qid, v in zip(t["query_id"], t["vector"])}


def load_extracted_edges(path: Path, known: set[str]) -> list[tuple[str, str, str]] | None:
    if not path.exists():
        return None
    edges = []
    for r in _jsonl(path):
        for e in r.get("edges", []):
            if e["src"] in known and e["dst"] in known:
                edges.append((e["src"], e["dst"], e["kind"]))
    return sorted(set(edges))


def load_llm_filters(path: Path) -> dict[str, dict] | None:
    if not path.exists():
        return None
    return {r["query_id"]: r.get("parsed") or {"type": None, "filters": []} for r in _jsonl(path)}


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------

def _dense_docs(chunks: list[dict], mat: np.ndarray, qvec: np.ndarray, k: int = A.K_DOCS) -> list[str]:
    order = np.argsort(-(mat @ qvec), kind="stable")
    return list(dict.fromkeys(chunks[i]["doc_path"] for i in order))[:k]


def run_arms(b: dict, q: dict, qvec: np.ndarray, rrf_k: int, arms: tuple[str, ...] | None,
             extracted: list | None, llm_filter: dict | None, lat: dict) -> dict[str, list[str]]:
    """Doc rankings (up to 50) for every configuration on one question; latencies (s) into `lat`."""
    text = q["text"]
    want = set(arms) if arms else None
    out: dict[str, list[str]] = {}

    def timed(name, fn):
        if want is not None and name not in want:
            return None
        t0 = time.perf_counter()
        res = fn()
        lat[name].append(time.perf_counter() - t0)
        out[name] = res
        return res

    n_docs = len(b["meta"])
    t0 = time.perf_counter()
    r2s_pairs = structured_rank(b["meta_index"], text, n_docs, n_docs)
    r2s_time = time.perf_counter() - t0
    r2s_scores = dict(r2s_pairs)
    r2s_ranked = [p for p, _ in r2s_pairs]
    out["R2s"] = r2s_ranked[:A.K_DOCS]
    lat["R2s"].append(r2s_time)

    r0 = timed("R0", lambda: _dense_docs(b["flat"], b["mat_flat"], qvec))
    timed("R1h", lambda: _dense_docs(b["struct"], b["mat_hp"], qvec))
    timed("R3s", lambda: A.weighted_expansion(r2s_pairs[:A.K_DOCS], b["edges"]))
    if extracted is not None:
        timed("R3x", lambda: A.weighted_expansion(r2s_pairs[:A.K_DOCS], extracted))
    timed("R5p", lambda: A.parent_aware(text, r2s_scores, r2s_ranked, b["meta"], b["children"]))
    timed("R6f", lambda: A.filter_rank(A.parse_filters(text, b["meta"], b["ftypes"]), b["meta"],
                                       r2s_scores, r2s_ranked))
    if llm_filter is not None:
        timed("R6l", lambda: A.filter_rank(llm_filter, b["meta"], r2s_scores, r2s_ranked))
    if want is None or {"R7f", "R7h"} & want:
        r0 = r0 if r0 is not None else _dense_docs(b["flat"], b["mat_flat"], qvec)
        r7f = timed("R7f", lambda: reciprocal_rank_fusion([r2s_ranked[:A.K_DOCS], r0], k=A.K_DOCS, rrf_k=rrf_k))

        def _r7h():
            sims = b["mat_struct"] @ qvec
            dense_doc = {p: float(sims[idx].max()) for p, idx in b["struct_doc_idx"].items()}
            return A.two_step(text, b["meta"], b["neighbors"], dense_doc, r7f)
        timed("R7h", _r7h)
    return out


def oracle_rank(spans: list[dict], all_paths: list[str]) -> list[str] | None:
    chains = [s["path"] for s in spans if s.get("path")]
    if not chains:
        return None
    ranked = []
    for chain in chains:
        res = hierarchy_prefix_retrieval([], all_paths, _oracle_child_dir(chain[-1]), A.K_DOCS,
                                         selector_provenance="gold_s4_path")
        ranked += res.ranked_doc_paths
    return list(dict.fromkeys(ranked))[:A.K_DOCS]


def t7_passages(b: dict, arm: str, ranked_docs: list[str], qvec: np.ndarray) -> list[dict]:
    """Evidence order for answer generation: R0 = top flat chunks by cosine; R2s = the
    structured chunks of the top documents, lead sections first (web bench order)."""
    if arm == "R0":
        order = np.argsort(-(b["mat_flat"] @ qvec), kind="stable")[:T7_TOP]
        return [{"doc_path": b["flat"][i]["doc_path"], "chunk_id": b["flat"][i]["chunk_id"],
                 "heading_path": [], "text": b["flat"][i]["text"]} for i in order]
    keyed = []
    for rank, path in enumerate(ranked_docs[:T7_TOP]):
        idx = sorted(b["struct_doc_idx"].get(path, []), key=lambda i: b["struct"][i]["ord"])
        for pos, i in enumerate(idx):
            c = b["struct"][i]
            keyed.append(((pos, rank), {"doc_path": path, "chunk_id": c["chunk_id"],
                                        "heading_path": list(c["heading_path"]), "text": c["text"]}))
    return [c for _, c in sorted(keyed, key=lambda kv: kv[0])]


# ---------------------------------------------------------------------------
# Scoring and outputs
# ---------------------------------------------------------------------------

def score(runs: dict, questions: list[dict], qrels: dict) -> list[dict]:
    rows = []
    for (bundle, arm), by_q in sorted(runs.items()):
        for q in questions:
            qid = q["query_id"]
            if q["stratum"] == "S5" or qid not in by_q:
                continue
            m = A.doc_metrics_v4(by_q[qid], qrels[qid])
            rows.append({"bundle": bundle, "arm": arm, "query_id": qid, "stratum": q["stratum"],
                         "set": "p" if qid.endswith("-p") else "t", "condition": q.get("condition", ""), **m})
    return rows


def cells_from_rows(rows: list[dict]) -> dict:
    cells = {k: defaultdict(dict) for k in ("S34_t", "S34_p", "S4_np_t", "S3num_t", "S2_t", "S1to4_t", "S1_t")}
    for r in rows:
        v, arm, qid, st, s = r["nDCG@10"], r["arm"], r["query_id"], r["stratum"], r["set"]
        if r["bundle"] == NOPARENT:
            if st == "S4" and s == "t":
                cells["S4_np_t"][arm][qid] = v
            continue
        if s == "t":
            cells["S1to4_t"][arm][qid] = v
            if st in ("S3", "S4"):
                cells["S34_t"][arm][qid] = v
            if st == "S3" and r["condition"] == "numeric":
                cells["S3num_t"][arm][qid] = v
            if st == "S2":
                cells["S2_t"][arm][qid] = v
            if st == "S1":
                cells["S1_t"][arm][qid] = v
        elif st in ("S3", "S4"):
            cells["S34_p"][arm][qid] = v
    return {k: dict(v) for k, v in cells.items()}


def choose_rrf_k(b, questions, qvecs, qrels) -> tuple[int, dict]:
    """R7f rrf_k from RRF_GRID by mean dev-template nDCG@10 over S1-S4 (ties: larger k)."""
    means = {}
    lat = defaultdict(list)
    for k in A.RRF_GRID:
        vals = []
        for q in questions:
            if q["stratum"] == "S5" or not q["query_id"].endswith("-t"):
                continue
            r = run_arms(b, q, qvecs[q["query_id"]], k, ("R0", "R7f"), None, None, lat)["R7f"]
            vals.append(A.doc_metrics_v4(r, qrels[q["query_id"]])["nDCG@10"])
        means[k] = float(np.mean(vals))
    best = max(A.RRF_GRID, key=lambda k: (round(means[k], 12), k))
    return best, means


def write_csv(path: Path, rows: list[dict], keys: list[str], comment: str | None = None) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        if comment:
            f.write(comment + "\n")
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def summary_rows(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["bundle"], r["arm"], r["stratum"], r["set"])].append(r)
        if r["stratum"] == "S3" and r["condition"] == "numeric":
            groups[(r["bundle"], r["arm"], "S3num", r["set"])].append(r)
    out = []
    for (bundle, arm, st, s), rs in sorted(groups.items()):
        row = {"bundle": bundle, "arm": arm, "stratum": st, "set": s, "n": len(rs)}
        for m in SUMMARY_METRICS:
            vals = [r[m] for r in rs]
            ci = bootstrap_mean_ci(vals)
            row[f"{m}_mean"] = f"{np.mean(vals):.4f}"
            row[f"{m}_ci_low"] = f"{ci['ci_low']:.4f}"
            row[f"{m}_ci_high"] = f"{ci['ci_high']:.4f}"
        out.append(row)
    return out


def efficiency_rows(bundles: dict, lat: dict, emb_manifests: dict, llm_manifest: dict | None) -> list[dict]:
    rows = []
    for (bundle, arm), vals in sorted(lat.items()):
        b = bundles[bundle]
        ms = np.array(vals) * 1000.0
        build = b["t_build"].get(arm, 0.0)
        if arm in ("R3s", "R5p", "R7h", "R3x"):
            build += b["t_build"]["graph"] + b["t_build"]["R2s"]
        if arm in ("R6f", "R6l", "R7f"):
            build += b["t_build"]["R2s"]
        emb = emb_manifests[bundle]["seconds"]
        emb_s = {"R0": emb.get("emb_flat", 0.0), "R1h": emb.get("emb_struct_hp", 0.0),
                 "R7f": emb.get("emb_flat", 0.0), "R7h": emb.get("emb_flat", 0.0) + emb.get("emb_struct", 0.0)}.get(arm, 0.0)
        size = {"R0": b["sizes"]["dense_bytes_flat"], "R1h": b["sizes"]["dense_bytes_struct"],
                "R7f": b["sizes"]["dense_bytes_flat"], "R7h": b["sizes"]["dense_bytes_flat"] + b["sizes"]["dense_bytes_struct"]
                }.get(arm, 0)
        llm_tokens = 0
        if llm_manifest and arm == "R3x":
            llm_tokens = llm_manifest.get("extract", {}).get("prompt_tokens", 0) + llm_manifest.get("extract", {}).get("completion_tokens", 0)
        rows.append({"bundle": bundle, "arm": arm, "queries": len(vals),
                     "index_build_s": f"{build:.4f}", "embedding_s": f"{emb_s:.2f}",
                     "dense_index_bytes": size, "bm25_postings": b["sizes"]["bm25_postings_meta"] if arm != "R0" and arm != "R1h" else 0,
                     "latency_p50_ms": f"{np.percentile(ms, 50):.3f}", "latency_p95_ms": f"{np.percentile(ms, 95):.3f}",
                     "index_llm_tokens": llm_tokens})
    return rows


def edge_agreement(authored: list, extracted: list) -> dict:
    pair = lambda e: tuple(sorted((e[0], e[1])))
    a = {pair(e) for e in authored if e[0] != e[1]}
    x = {pair(e) for e in extracted if e[0] != e[1]}
    tp = len(a & x)
    return {"authored_pairs": len(a), "extracted_pairs": len(x), "shared_pairs": tp,
            "precision": tp / len(x) if x else 0.0, "recall": tp / len(a) if a else 0.0}


def verdict_md(v: dict, out_name: str) -> str:
    pf = lambda x: "PASS" if x else "FAIL"
    fmt = lambda r: (f"delta {r['delta']:+.3f}, CI [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}], p {r['p']:.4f}"
                     + (f", Holm {'sig' if r['holm_significant'] else 'ns'}" if "holm_significant" in r else "")
                     + f", n {r['n']}")
    L = [f"# Brightmart v4 - verdict ({out_name})", "",
         "Preregistration: docs/brightmart_v4_prereg.md. nDCG@10, paired bootstrap 10,000, seed 7, Holm alpha 0.05.", "",
         f"Thresholds: `{json.dumps(v['thresholds'])}`", "",
         "## Main family (standard corpus, S3+S4 templates, vs R0)", ""]
    L += [f"- {k}: {fmt(r)}" for k, r in v["main_family_S34_t"].items()]
    L += ["", f"## T1 scale: {pf(v['T1']['pass'])}",
          f"- R2s - R0 (templates): {fmt(v['T1']['R2s-R0'])}",
          f"- R2s - R0 (paraphrases): {fmt(v['T1']['paraphrase'])}",
          "", f"## T2 hierarchy without parent names: {pf(v['T2']['pass'])}"]
    L += [f"- {k} (noparent S4 templates): {fmt(r)}" for k, r in v["T2"]["family"].items()]
    L += ["", f"## T3 numeric/date conditions: {pf(v['T3']['pass'])}"]
    L += [f"- {k} (S3 numeric templates): {fmt(r)}" for k, r in v["T3"]["family"].items()]
    if v["T3"]["exploratory_R6l"]:
        L += [f"- exploratory {k}: {fmt(r)}" for k, r in v["T3"]["exploratory_R6l"].items()]
    L += ["", f"## T4 multi-hop fusion: {pf(v['T4']['pass'])}"]
    for name, r in v["T4"]["variants"].items():
        L += [f"- {name}: S2 {fmt(r['S2'])}; S3+S4 {fmt(r['S34'])} -> {pf(r['pass'])}"]
    L += ["", "## T5 contextual chunks and link expansion"]
    L += [f"- {k}: {fmt(r)} -> {'kept' if r['kept'] else 'not kept'}" for k, r in v["T5"].items()]
    L += ["", "## T6 authored minus extracted (R3s - R3x)"]
    if v["T6_authored_minus_extracted"]:
        L += [f"- {k}: {fmt(r)}" for k, r in v["T6_authored_minus_extracted"].items()]
        for k, val in v["exploratory"].get("T6_edges", {}).items():
            L.append(f"- {k}: {val}")
    else:
        L.append("- not run (no extracted edge set)")
    if v.get("S1_t_vs_R0"):
        L += ["", "## S1 direct facts (templates, vs R0)"] + [f"- {k}: {fmt(r)}" for k, r in v["S1_t_vs_R0"].items()]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def check_freeze() -> dict:
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    now = {"standard": corpus_sha256(Path(CORPUS_DIRS[STANDARD])),
           "noparent": corpus_sha256(Path(CORPUS_DIRS[NOPARENT]))}
    bad = [k for k, v in frozen["corpus_sha256"].items() if now.get(k) != v]
    bad += [p for p, h in frozen["heldout_files"].items() if _sha(Path(p)) != h]
    for bundle, variant in ((STANDARD, "standard"), (NOPARENT, "noparent")):
        stats = Path(f"results/transcode_stats_{bundle}.json")
        if not stats.exists() or json.loads(stats.read_text(encoding="utf-8")).get("corpus_sha256") != frozen["corpus_sha256"][variant]:
            bad.append(f"ingested {bundle} (transcode stats) != frozen corpus")
    if bad:
        raise SystemExit(f"freeze check failed: {bad}")
    return frozen


def main() -> None:
    ap = argparse.ArgumentParser(description="Brightmart v4 evaluation run.")
    ap.add_argument("--mode", choices=["dev", "heldout", "smoke"], required=True)
    ap.add_argument("--dsn", default=None)
    ap.add_argument("--emb-root", default=str(EMB_ROOT))
    ap.add_argument("--llm-dir", default=str(LLM_DIR))
    ap.add_argument("--freeze-tuning", action="store_true",
                    help="dev mode: also write configs/brightmart_v4_tuning.json (refuses if it exists)")
    args = ap.parse_args()

    out = OUT[args.mode]
    qset = "heldout" if args.mode == "heldout" else "dev"
    frozen = None
    if args.mode == "heldout":
        if (out / "verdict.json").exists():
            raise SystemExit(f"{out}/verdict.json exists: the v4 held-out run is one-shot")
        frozen = check_freeze()
    if args.mode in ("heldout", "smoke") and not TUNING.exists():
        raise SystemExit(f"{TUNING} missing: run --mode dev --freeze-tuning first")

    emb_root, llm_dir = Path(args.emb_root), Path(args.llm_dir)
    from okf_rag.ingest.load_pg import get_conn
    conn = get_conn(args.dsn) if args.dsn else get_conn()
    bundles = {bn: load_bundle(conn, bn, emb_root / bn) for bn in (STANDARD, NOPARENT)}
    conn.close()
    emb_manifests = {bn: json.loads((emb_root / bn / "manifest.json").read_text(encoding="utf-8")) for bn in bundles}
    qvecs = {bn: load_query_vectors(emb_root / bn, qset) for bn in bundles}

    questions = _jsonl(QFILES[qset]["questions"])
    qrels = defaultdict(set)
    for r in _jsonl(QFILES[qset]["qrels"]):
        qrels[r["query_id"]].add(r["doc_path"])
    spans = defaultdict(list)
    for r in _jsonl(QFILES[qset]["gold_spans"]):
        spans[r["query_id"]].append(r)

    if args.mode == "dev":
        rrf_k, rrf_means = choose_rrf_k(bundles[STANDARD], questions, qvecs[STANDARD], qrels)
    else:
        rrf_k = json.loads(TUNING.read_text(encoding="utf-8"))["rrf_k"]
        rrf_means = None

    std = bundles[STANDARD]
    extracted = load_extracted_edges(llm_dir / "graph_extracted.jsonl", set(std["meta"]))
    llm_filters = load_llm_filters(llm_dir / f"filters_{qset}.jsonl")
    llm_manifest_path = llm_dir / "manifest.json"
    llm_manifest = json.loads(llm_manifest_path.read_text(encoding="utf-8")) if llm_manifest_path.exists() else None

    runs = defaultdict(dict)
    lat = defaultdict(list)
    oracle = {}
    contexts = []
    for q in questions:
        qid = q["query_id"]
        for bn, arms in ((STANDARD, None), (NOPARENT, NOPARENT_ARMS)):
            b = bundles[bn]
            blat = defaultdict(list)
            lf = llm_filters.get(qid) if (llm_filters is not None and bn == STANDARD) else None
            res = run_arms(b, q, qvecs[bn][qid], rrf_k, arms, extracted if bn == STANDARD else None, lf, blat)
            for arm, ranked in res.items():
                if arms is None or arm in arms:
                    runs[(bn, arm)][qid] = ranked
            for arm, v in blat.items():
                lat[(bn, arm)] += v
            if q["stratum"] == "S4":
                o = oracle_rank(spans[qid], sorted(b["meta"]))
                if o is not None:
                    oracle[(bn, qid)] = o
        if qid.endswith("-t"):
            for arm in T7_ARMS:
                contexts.append({"query_id": qid, "arm": arm,
                                 "passages": t7_passages(std, arm, runs[(STANDARD, arm)][qid], qvecs[STANDARD][qid])})

    rows = score(runs, questions, qrels)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "runs.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for (bn, arm), by_q in sorted(runs.items()):
            for qid in sorted(by_q):
                f.write(json.dumps({"bundle": bn, "arm": arm, "query_id": qid, "ranked": by_q[qid]}) + "\n")
    with (out / "t7_contexts.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for c in contexts:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    keys = ["bundle", "arm", "query_id", "stratum", "set", "condition", *A.METRIC_KEYS]
    write_csv(out / "metrics_per_query.csv", rows, keys)
    summ = summary_rows(rows)
    skeys = ["bundle", "arm", "stratum", "set", "n"] + [f"{m}_{x}" for m in SUMMARY_METRICS for x in ("mean", "ci_low", "ci_high")]
    comment = "# development set - not confirmatory" if args.mode != "heldout" else None
    write_csv(out / "metrics_summary.csv", summ, skeys, comment)
    orows = [{"bundle": bn, "arm": "C_hier_oracle", "capability": "oracle_diagnostic", "query_id": qid,
              "nDCG@10": f"{A.doc_metrics_v4(r, qrels[qid])['nDCG@10']:.4f}"} for (bn, qid), r in sorted(oracle.items())]
    write_csv(out / "oracle_ceiling.csv", orows, ["bundle", "arm", "capability", "query_id", "nDCG@10"])
    write_csv(out / "efficiency.csv", efficiency_rows(bundles, lat, emb_manifests, llm_manifest),
              ["bundle", "arm", "queries", "index_build_s", "embedding_s", "dense_index_bytes", "bm25_postings",
               "latency_p50_ms", "latency_p95_ms", "index_llm_tokens"])

    manifest = {
        "mode": args.mode, "question_set": qset, "rrf_k": rrf_k, "rrf_dev_means": rrf_means,
        "inputs": {p.as_posix(): _sha(p) for p in QFILES[qset].values()},
        "embeddings": {bn: {"model_id": m["model_id"], "revision": m["revision"], "rows": m["rows"],
                            "manifest_sha256": _sha(emb_root / bn / "manifest.json")} for bn, m in emb_manifests.items()},
        "corpus_sha256": {bn: corpus_sha256(Path(d)) for bn, d in CORPUS_DIRS.items()},
        "sizes": {bn: b["sizes"] for bn, b in bundles.items()},
        "n_docs": {bn: len(b["meta"]) for bn, b in bundles.items()},
        "arms": sorted({f"{bn}:{arm}" for bn, arm in runs}),
        "extracted_edges": len(extracted) if extracted is not None else None,
        "llm_filters": len(llm_filters) if llm_filters is not None else None,
        "llm_manifest_sha256": _sha(llm_manifest_path) if llm_manifest else None,
        "freeze": frozen,
    }
    if args.mode == "dev":
        (out / "rrf_tuning.json").write_text(json.dumps({"rrf_k": rrf_k, "dev_means": rrf_means}, indent=2), encoding="utf-8")
        if args.freeze_tuning:
            if TUNING.exists():
                raise SystemExit(f"{TUNING} exists: tuning is already frozen")
            TUNING.write_text(json.dumps({"rrf_k": rrf_k, "dev_means": {str(k): v for k, v in rrf_means.items()},
                                          "chosen_on": "development templates S1-S4"}, indent=2) + "\n",
                              encoding="utf-8", newline="\n")
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"dev run written to {out}; rrf_k={rrf_k} (dev means {rrf_means})")
        return

    exploratory = {}
    if extracted is not None:
        exploratory["T6_edges"] = edge_agreement(std["edges"], extracted)
        if llm_manifest:
            exploratory["T6_edges"]["extraction"] = llm_manifest.get("extract")
    v = A.verdict_v4(cells_from_rows(rows), exploratory)
    v["rrf_k"] = rrf_k
    (out / "verdict.json").write_text(json.dumps(v, indent=2), encoding="utf-8")
    (out / "verdict.md").write_text(verdict_md(v, out.name), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(verdict_md(v, out.name))


if __name__ == "__main__":
    main()
