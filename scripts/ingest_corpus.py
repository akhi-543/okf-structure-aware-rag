"""Real-corpus ingest pipeline (Task 3, spec S1): clone-on-disk -> OKF
bundle -> Postgres rows -> chunk parquets.

Pipeline order (each stage prints progress to stdout):
  1. Verify `--repo-dir` HEAD matches the pinned commit in
     `configs/corpora.lock` (refuses to proceed on a mismatch unless
     `--allow-unpinned` is given).
  2. `transcode(repo_dir)` -> docs, edges, link-resolution stats.
  3. `write_bundle` -> `--bundle-out`; `validate_bundle` must return `[]`.
  4. `get_conn` + `init_schema`; `load_documents`; `load_edges(..., "authored", ids)`.
  5. Build a real `count_tokens` from `AutoTokenizer.from_pretrained(
     "BAAI/bge-base-en-v1.5")` -- constructed HERE, never inside
     `okf_rag/ingest/`, which stays free of any model/tokenizer import
     (that is the project's scientific premise: the chunking/flattening
     logic under test must not silently depend on which tokenizer happens
     to be installed). `--fast-tokens` swaps in a word-count fallback for
     quick smoke runs only -- the numbers recorded as the real Phase A
     deliverable must come from a run WITHOUT it.
  6. Per doc: chunk both policies (C-flat, C-struct) with the injected
     `count_tokens`, adapt to row tuples (`okf_rag.ingest.adapters`), and
     `load_chunks` each policy. Commits every `_COMMIT_EVERY` docs so a
     long run doesn't hold one multi-thousand-document transaction open.
  7. Export `results/chunks_{bundle}_flat.parquet` and
     `results/chunks_{bundle}_struct.parquet` (columns `chunk_id, doc_id,
     text`) by reading back the rows just loaded -- both policies get
     embedded downstream.
  8. Persist `results/transcode_stats_{bundle}.json` (docs, edges,
     resolved, unresolved split into in-scope/out-of-scope, rate,
     unresolved_samples, and, for github_docs, the Liquid
     unresolved/truncated prefix tallies).

`{bundle}` is `--bundle-name`, defaulting to `--corpus`. It is the single
identifier every artifact of a run is keyed on -- the Postgres `bundle`
column, the two parquets, and the stats file -- and `--allow-unpinned` /
`--fast-tokens` are refused against a canonical corpus name, so a fixture or
smoke run cannot overwrite the real corpus's rows or artifacts (see
`_resolve_bundle_name`).

A5 rule: a re-ingest reassigns `chunk_id`s (chunks are deleted and
re-inserted per (doc, policy) -- see `load_pg.load_chunks` -- and
`chunk_id` is a Postgres `BIGSERIAL`, so a re-run's rows get fresh ids,
never the old ones). Every downstream artifact keyed on `chunk_id`
(embeddings parquet, contexts, retrieval runs) is invalidated by a
re-ingest and must be regenerated. Order is always:
ingest -> embed -> load embeddings -> retrieve.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

# Run this repository's packages even if another checkout providing `okf_rag` is
# pip-installed: `python scripts/x.py` puts scripts/, not the repo root, on sys.path.
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Callable


# `pyarrow` is imported lazily, inside `_export_chunk_parquet` (step 7)
# rather than at module scope: on this Windows environment, `import
# pyarrow` before `transformers`/`torch` (loaded lazily in
# `_build_count_tokens`, step 5) leaves the process's DLL search state such
# that torch's `c10.dll` fails to load with `OSError: [WinError 1114]` --
# 100% reproducible, confirmed by isolating the two import orders in a
# throwaway `python -c`. Loading transformers first (step 5) and pyarrow
# second (step 7) avoids the conflict entirely; `--fast-tokens` runs never
# import transformers at all, so pyarrow loads with nothing to conflict with
# either way.

from okf_rag.ingest.adapters import flat_rows, struct_rows
from okf_rag.ingest.chunk import chunk_flat, chunk_struct
from okf_rag.ingest.flatten import flatten_doc
from okf_rag.ingest.load_embeddings import create_hnsw_index, load_embeddings
from okf_rag.ingest.load_pg import get_conn, init_schema, load_chunks, load_documents, load_edges
from okf_rag.transcode import brightmart_synthetic
from okf_rag.transcode.okf_writer import write_bundle
from okf_rag.transcode.validator import validate_bundle

# Docs processed between commits in the per-doc chunk-loading stage (step
# 6). Small enough that a killed run loses at most this many docs' worth of
# chunk rows, large enough that a multi-thousand-document corpus isn't one
# commit per doc (needless round-trip overhead on a long run).
_COMMIT_EVERY = 200

# Docs between progress lines, so a stalled run is visible without flooding
# stdout for a 3,700-document corpus.
_PROGRESS_EVERY = 200

_CORPORA: dict[str, dict] = {
    # Synthetic companion corpus (doc_synthetic/). Lives in this repo, so there
    # is no git pin: `lock_key` None makes `_verify_pin` record the content
    # hash instead. `_resolve_bundle_name` fixes its bundle to the one scratch
    # identifier below and refuses --fast-tokens.
    "brightmart": {
        "transcode": brightmart_synthetic.transcode, "lock_key": None,
        "in_scope": lambda href: True,
        "bundle": "synthetic_retail_pilot",
    },
    # Brightmart v4 (next iteration): rendered into results/ by
    # doc_synthetic/brightmart_v4_render.py, same OKF-shaped markdown as the pilot.
    "brightmart_v4": {
        "transcode": brightmart_synthetic.transcode, "lock_key": None,
        "in_scope": lambda href: True,
        "bundle": "synthetic_retail_v4",
    },
    "brightmart_v4_noparent": {
        "transcode": brightmart_synthetic.transcode, "lock_key": None,
        "in_scope": lambda href: True,
        "bundle": "synthetic_retail_v4_noparent",
    },
}

_A5_NOTICE = (
    "A5: this ingest reassigned chunk_id (delete-then-insert per (doc, policy), "
    "chunk_id is a Postgres BIGSERIAL) -- every downstream artifact keyed on "
    "chunk_id (embeddings parquet, contexts, retrieval runs) is now invalidated "
    "and must be regenerated. Order is always: ingest -> embed -> load embeddings -> retrieve."
)


def _git_head(repo_dir: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _resolve_bundle_name(
    corpus: str, bundle_name: str | None, allow_unpinned: bool, fast_tokens: bool
) -> str:
    """Decide the identifier this run writes under, and refuse the unsafe
    combinations (finding 3).

    The Postgres `bundle` column, the exported parquet names and the stats
    filename all key on one identifier, which used to be `--corpus` with no
    way to override it. So a fixture smoke run --

        --corpus github_docs --repo-dir tests/fixtures/ghd --allow-unpinned --fast-tokens

    -- upserted fixture rows into the *real* corpus's bundle, delete-then-
    inserted its edges and chunks, and reassigned every chunk_id. That is
    not hypothetical: it happened during this phase and silently replaced a
    real landing page with an 11-line stub. The mitigation adopted at the
    time was a documentation rule (CLAUDE.md); this is the code guard.

    Two flags mark a run as not-the-deliverable, and neither may write to a
    canonical bundle name:

    - `--allow-unpinned`: the corpus checkout is not at the pinned commit,
      so whatever is written is not reproducible from `configs/corpora.lock`.
    - `--fast-tokens`: token counts come from `len(text.split())`, not the
      real tokenizer, so `chunks.text` boundaries and `n_tokens` are
      different numbers from the ones the study reports -- and they
      overwrite the real rows in place.

    `--bundle-name` is the escape hatch: name the scratch run something
    obviously scratch (`github_docs_smoke`) and it proceeds.
    """
    fixed = _CORPORA.get(corpus, {}).get("bundle")
    if fixed is not None:
        if bundle_name != fixed:
            raise ValueError(
                f"corpus {corpus!r} may only be written under --bundle-name {fixed} "
                f"(got {bundle_name!r})"
            )
        if fast_tokens:
            raise ValueError(f"corpus {corpus!r} refuses --fast-tokens: real tokenizer only")
        return fixed
    name = bundle_name or corpus
    if name in _CORPORA:
        unsafe = [
            flag for flag, on in (("--allow-unpinned", allow_unpinned), ("--fast-tokens", fast_tokens))
            if on
        ]
        if unsafe:
            raise ValueError(
                f"refusing to write to the canonical bundle {name!r} with "
                f"{' and '.join(unsafe)}: this run's rows would replace the real corpus's "
                f"documents, edges and chunks in place (and reassign every chunk_id). "
                f"Pass --bundle-name with an obviously-scratch identifier, e.g. "
                f"--bundle-name {name}_smoke."
            )
    return name


def _verify_pin(repo_dir: Path, corpus: str, allow_unpinned: bool) -> None:
    """The Brightmart corpus lives in this repo, so there is no git pin: the
    reproducibility record is the corpus content hash, printed here and
    written into the stats file."""
    sha = brightmart_synthetic.corpus_sha256(repo_dir)
    print(f"[1/8] no git pin for {corpus!r}; corpus_sha256={sha}")


def _build_count_tokens(fast_tokens: bool) -> Callable[[str], int]:
    if fast_tokens:
        print("[5/8] --fast-tokens: using word-count fallback (smoke run only)")
        return lambda text: len(text.split())
    print("[5/8] loading AutoTokenizer.from_pretrained('BAAI/bge-base-en-v1.5') ...")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
    print("[5/8] tokenizer ready")

    def count_tokens(text: str) -> int:
        # verbose=False suppresses the per-call "sequence length longer
        # than the specified maximum" warning: chunk_flat's binary search
        # deliberately probes windows wider than the final 512-token
        # target while narrowing down, so tokenizing an over-length probe
        # is expected, not a bug to report on every call.
        return len(tokenizer.encode(text, add_special_tokens=False, verbose=False))

    return count_tokens


def _stats_to_dict(corpus: str, docs, edges, stats, bundle: str | None = None) -> dict:
    edge_kinds = Counter(e.kind for e in edges)
    # Finding 6: one "unresolved" bucket measured two different things
    # across the two corpora and could not tell them apart. home-assistant's
    # 56.45% is dominated by links to `/docs/`, `/blog/`, `/voice_control/`
    # -- targets deliberately outside the pinned `source/_integrations/`
    # subset, i.e. working links to pages this study does not ingest.
    # github/docs' 79.63% was a resolver bug (finding 1). Reporting them as
    # one number is why nobody caught that bug when the rate was recorded.
    in_scope = _CORPORA[corpus]["in_scope"]
    out_of_scope = sum(n for href, n in stats.unresolved_hrefs.items() if not in_scope(href))
    out = {
        "corpus": corpus,
        # The identifier this run actually wrote under -- equal to `corpus`
        # for a normal run, different for a scratch run (finding 3).
        "bundle": bundle or corpus,
        "docs": len(docs),
        "edges": len(edges),
        "edges_by_kind": dict(edge_kinds),
        "resolved": stats.resolved,
        "unresolved": stats.unresolved,
        # Target is a real page outside the corpus subset this study
        # ingests: expected, not a defect.
        "unresolved_out_of_scope": out_of_scope,
        # Target looks like a page we DID ingest and the resolver still
        # missed it: this is the number to watch.
        "unresolved_in_scope": stats.unresolved - out_of_scope,
        "rate": stats.rate,
        # json.dumps serializes a tuple as a JSON array already; no explicit
        # list() conversion needed for the (src, href) pairs.
        "unresolved_samples": stats.unresolved_samples,
    }
    if stats.liquid is not None:
        out["liquid"] = {
            "unresolved_by_prefix": stats.liquid.unresolved_by_prefix,
            "truncated_by_prefix": stats.liquid.truncated_by_prefix,
        }
    return out


def _export_chunk_parquet(conn, doc_ids: list[int], policy: str, out_path: Path) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    with conn.cursor() as cur:
        cur.execute(
            "SELECT chunk_id, doc_id, text FROM chunks WHERE doc_id = ANY(%s) AND policy = %s "
            "ORDER BY doc_id, ord",
            (doc_ids, policy),
        )
        rows = cur.fetchall()
    table = pa.table({
        "chunk_id": pa.array([r[0] for r in rows], type=pa.int64()),
        "doc_id": pa.array([r[1] for r in rows], type=pa.int64()),
        "text": pa.array([r[2] for r in rows], type=pa.string()),
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path)
    return len(rows)


def run(args: argparse.Namespace) -> None:
    corpus = args.corpus
    # First statement in the function on purpose (finding 3): an unsafe
    # invocation must not reach the pin check, the transcoder, or the
    # database.
    bundle_name = _resolve_bundle_name(
        corpus, args.bundle_name, args.allow_unpinned, args.fast_tokens
    )
    repo_dir = Path(args.repo_dir)
    bundle_out = Path(args.bundle_out)
    out_dir = Path(args.out_dir)

    _verify_pin(repo_dir, corpus, args.allow_unpinned)

    print(f"[2/8] transcoding {repo_dir} ...")
    transcode = _CORPORA[corpus]["transcode"]
    docs, edges, stats = transcode(repo_dir)
    print(
        f"[2/8] transcode done: docs={len(docs)} edges={len(edges)} "
        f"resolved={stats.resolved} unresolved={stats.unresolved} rate={stats.rate:.2%}"
    )

    print(f"[3/8] writing bundle to {bundle_out} ...")
    write_bundle(docs, bundle_out)
    errs = validate_bundle(bundle_out)
    if errs:
        print(f"[3/8] *** bundle validation FAILED with {len(errs)} error(s) ***", file=sys.stderr)
        for e in errs[:20]:
            print(f"    {e}", file=sys.stderr)
        sys.exit(1)
    print("[3/8] bundle conformant (OKF v0.2 s11)")

    print(f"[4/8] connecting to {args.dsn or '(default DSN)'} ...")
    conn = get_conn(args.dsn) if args.dsn else get_conn()
    init_schema(conn)
    ids = load_documents(conn, bundle_name, docs)
    load_edges(conn, edges, "authored", ids)
    conn.commit()
    print(f"[4/8] loaded {len(ids)} documents and {len(edges)} authored edges "
          f"under bundle {bundle_name!r}")

    count_tokens = _build_count_tokens(args.fast_tokens)

    print(f"[6/8] chunking + loading {len(docs)} docs (both policies) ...")
    flat_total = 0
    struct_total = 0
    for i, d in enumerate(docs, start=1):
        doc_id = ids[d.path]
        flat_chunks = chunk_flat(flatten_doc(d), count_tokens)
        load_chunks(conn, doc_id, "flat", flat_rows(flat_chunks, count_tokens))
        flat_total += len(flat_chunks)

        struct_chunks = chunk_struct(d, count_tokens)
        load_chunks(conn, doc_id, "struct", struct_rows(struct_chunks, count_tokens))
        struct_total += len(struct_chunks)

        if i % _COMMIT_EVERY == 0:
            conn.commit()
        if i % _PROGRESS_EVERY == 0 or i == len(docs):
            print(f"[6/8] {i}/{len(docs)} docs chunked (flat={flat_total} struct={struct_total})")
    conn.commit()
    print(f"[6/8] done: flat_chunks={flat_total} struct_chunks={struct_total}")

    print("[7/8] exporting chunk parquets ...")
    doc_ids = list(ids.values())
    out_dir.mkdir(parents=True, exist_ok=True)
    # Named by `bundle_name`, not `corpus` (finding 3): every artifact a
    # run writes is keyed on the one identifier, so a scratch run cannot
    # overwrite the real corpus's parquets or stats file either. Identical
    # filenames to before for a normal run, where bundle_name == corpus.
    n_flat = _export_chunk_parquet(conn, doc_ids, "flat", out_dir / f"chunks_{bundle_name}_flat.parquet")
    n_struct = _export_chunk_parquet(conn, doc_ids, "struct", out_dir / f"chunks_{bundle_name}_struct.parquet")
    print(f"[7/8] wrote chunks_{bundle_name}_flat.parquet ({n_flat} rows), "
          f"chunks_{bundle_name}_struct.parquet ({n_struct} rows)")

    print("[8/8] persisting transcode stats ...")
    stats_path = out_dir / f"transcode_stats_{bundle_name}.json"
    stats_dict = _stats_to_dict(corpus, docs, edges, stats, bundle=bundle_name)
    if _CORPORA[corpus]["lock_key"] is None:
        stats_dict["corpus_sha256"] = brightmart_synthetic.corpus_sha256(repo_dir)
    stats_path.write_text(json.dumps(stats_dict, indent=2), encoding="utf-8")
    print(f"[8/8] wrote {stats_path}")

    print()
    print(_A5_NOTICE)


def run_load_embeddings(args: argparse.Namespace) -> None:
    """`--load-embeddings PARQUET` mode: load a `jobs/embed.py` output
    parquet (`chunk_id, doc_id, vector`) into `chunks.embedding` and build
    the HNSW index, without re-running transcode/chunking. This is the "one command"
    half of the round trip described in this module's docstring (`ingest ->
    embed -> load embeddings -> retrieve`) -- the embed step runs
    out-of-process (GPU, Kaggle) and its parquet gets pulled back locally
    from HF Hub; this mode is what loads it into the same Postgres the
    ingest step already populated.

    Deliberately does not call transcode/write_bundle/chunk -- re-running
    those would reassign chunk_ids (A5) and immediately invalidate the very
    parquet this mode is loading.
    """
    parquet_path = Path(args.load_embeddings)
    print(f"[load-embeddings] connecting to {args.dsn or '(default DSN)'} ...")
    conn = get_conn(args.dsn) if args.dsn else get_conn()
    init_schema(conn)
    print(f"[load-embeddings] loading {parquet_path} ...")
    n = load_embeddings(conn, parquet_path)
    conn.commit()
    print(f"[load-embeddings] updated {n} chunk(s)")
    print("[load-embeddings] building HNSW index (idx_emb) ...")
    create_hnsw_index(conn)
    print("[load-embeddings] done")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    # Not required at the argparse level (validated below instead):
    # --load-embeddings runs a separate mode that skips transcode/chunking
    # entirely and needs none of these.
    ap.add_argument("--corpus", choices=sorted(_CORPORA))
    ap.add_argument("--repo-dir")
    # Required (when not in --load-embeddings mode), not defaulted (fix
    # round 1, Important): the brief's CLI contract lists `--bundle-out
    # PATH` without brackets -- the same convention that marks `--dsn` and
    # `--out-dir` optional below by wrapping them in `[...]` -- so
    # `--bundle-out` is required. A prior version defaulted to a hardcoded
    # `D:/pro/corpora/bundles/{corpus}` path, which is non-portable outside
    # this one development machine.
    ap.add_argument("--bundle-out")
    # Finding 3: the Postgres `bundle`, the exported parquet names and the
    # stats filename all key on this one identifier. It defaults to
    # --corpus (so a normal run is unchanged), and --allow-unpinned /
    # --fast-tokens are refused against a canonical name -- see
    # _resolve_bundle_name.
    ap.add_argument(
        "--bundle-name", default=None,
        help="Identifier this run writes under (Postgres bundle, parquet and stats "
             "filenames). Defaults to --corpus. Required to be an obviously-scratch "
             "value when --allow-unpinned or --fast-tokens is given.",
    )
    ap.add_argument("--dsn", default=None)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--allow-unpinned", action="store_true")
    ap.add_argument("--fast-tokens", action="store_true")
    ap.add_argument(
        "--load-embeddings", default=None, metavar="PARQUET",
        help="Load a chunk_id/doc_id/vector parquet (okf_rag/jobs/embed.py output) into "
             "chunks.embedding and build the HNSW index, instead of running the "
             "full ingest pipeline. Runs on its own -- no ingest arguments.",
    )
    args = ap.parse_args()
    if args.load_embeddings:
        # Finding 7: this mode skips transcode and chunking entirely, so
        # every ingest argument it was happy to accept was then silently
        # ignored -- on the one command that writes production vectors.
        # Same failure mode as finding 3, so it gets the same treatment:
        # say no rather than do something other than what was asked.
        ignored = [
            flag for flag, value in (
                ("--corpus", args.corpus), ("--repo-dir", args.repo_dir),
                ("--bundle-out", args.bundle_out), ("--bundle-name", args.bundle_name),
                ("--allow-unpinned", args.allow_unpinned), ("--fast-tokens", args.fast_tokens),
            ) if value
        ]
        if ignored:
            ap.error(
                f"--load-embeddings runs on its own and ignores {', '.join(ignored)}. "
                "Drop those arguments, or run the ingest pipeline separately (without "
                "--load-embeddings)."
            )
        run_load_embeddings(args)
        return
    if not (args.corpus and args.repo_dir and args.bundle_out):
        ap.error("--corpus, --repo-dir, and --bundle-out are required unless --load-embeddings is given")
    # Run the bundle-name guard here too (finding 3), so an operator mistake
    # is reported the way argparse reports any other bad invocation -- exit
    # 2 with a usage line, not a traceback. `run()` re-checks it as its
    # first statement, which is what protects a programmatic caller;
    # `_resolve_bundle_name` is pure, so calling it twice costs nothing.
    # Deliberately not a `try/except ValueError` around `run()`: that would
    # also swallow an unrelated ValueError from deep in the pipeline and
    # report a real bug as a usage error.
    try:
        _resolve_bundle_name(args.corpus, args.bundle_name, args.allow_unpinned, args.fast_tokens)
    except ValueError as e:
        ap.error(str(e))
    run(args)


if __name__ == "__main__":
    main()
