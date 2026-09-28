"""Retrieval control arms for isolating what authored structure contributes.

Why this module exists
----------------------
The v4 pilot compared R0 (dense over flat chunks) against R1 (dense over
structured chunks). That comparison varies *chunk representation only*.
It cannot answer whether authored structure — frontmatter metadata, path
hierarchy, authored cross-document links — helps retrieval, because no arm
in it uses any of those signals. This module supplies the missing arms.

Every arm here is deterministic and CPU-only: given the same inputs it
returns the same ranking, with ties broken by chunk id, so a local run is
reproducible without a GPU.

Honesty constraints these arms are built to respect
---------------------------------------------------
**Oracle arms are labelled as oracles.** `metadata_filter_retrieval` and
`hierarchy_prefix_retrieval` consume a structured filter or a path prefix.
When that filter comes from the query's gold annotation, the arm is not
doing natural-language retrieval — it is being *handed* the answer's
selector. Such an arm measures a capability ceiling ("if a system could
perfectly infer this filter, how well would authored metadata serve it?"),
never ordinary retrieval quality. Each result therefore carries
`capability='oracle_diagnostic'` and the provenance of its selector, and
`ORACLE_ARMS` names them so a report cannot quietly average them together
with natural-language arms.

**No gold answer or evidence id ever reaches ranking.** The scoring
functions take a query string plus corpus data. `expected_doc_paths`,
`evidence`, `required_facts`, and `sufficient_evidence_sets` are not
parameters of any ranking function in this module. The oracle arms take a
`filter`/`prefix` selector only, and record where it came from.

**Graph arms hold everything else fixed.** `graph_expanded_retrieval`
takes an already-computed seed ranking and expands it along authored edges.
Seed retriever, seed depth, traversal depth, ranking rule, and the final k
are all explicit parameters, so an authored-graph arm and any future
extracted-graph arm can be run with identical settings and remain
comparable. `EdgeSet` carries a `provenance` string and refuses to mix
sources: an authored edge set and an extracted edge set are separate
objects and are never silently unioned.

**Hybrid before graph.** `reciprocal_rank_fusion` provides the dense+BM25
control. Without it, any gain from a graph arm over dense-only could simply
be the lexical signal that BM25 supplies more cheaply, and attributing it
to authored structure would be wrong.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

#: Arms that consume a gold-derived selector. Never report these as
#: ordinary natural-language retrieval.
ORACLE_ARMS: frozenset[str] = frozenset(
    {"C_meta_oracle", "C_hier_oracle"}
)

#: Token pattern for the lexical arm. Keeps digits and intra-token dots,
#: hyphens and underscores together, so `llms.txt`, `bge-base-en-v1.5` and
#: `device_class` survive as single tokens instead of shattering into
#: meaningless fragments. Identifier integrity is the whole point in these
#: two corpora.
_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercased identifier-preserving tokens for lexical scoring.

    Lowercasing is applied HERE and only here, inside the lexical arm,
    where case-insensitive term matching is the established behaviour of
    BM25. It is not applied to evidence alignment, where case is meaning.
    """
    return [m.group(0).lower() for m in _TOKEN.finditer(text)]


@dataclass
class BM25Index:
    """Okapi BM25 over chunk texts. Deterministic, no external dependency.

    Built over one (corpus, policy) slice at a time so a lexical arm is
    never accidentally scored across corpora. `k1` and `b` use the standard
    defaults; they are constructor arguments so an ablation can vary them
    explicitly rather than by editing code.
    """

    chunk_ids: list[str]
    doc_paths: list[str]
    k1: float = 1.2
    b: float = 0.75
    _df: Counter = field(default_factory=Counter, repr=False)
    _tf: list[Counter] = field(default_factory=list, repr=False)
    _lengths: list[int] = field(default_factory=list, repr=False)
    _avg_len: float = 0.0

    @classmethod
    def build(cls, chunks: Sequence[dict], k1: float = 1.2, b: float = 0.75) -> "BM25Index":
        index = cls(
            chunk_ids=[str(c["chunk_id"]) for c in chunks],
            doc_paths=[c["doc_path"] for c in chunks],
            k1=k1,
            b=b,
        )
        for chunk in chunks:
            tokens = tokenize(chunk["text"])
            counts = Counter(tokens)
            index._tf.append(counts)
            index._lengths.append(len(tokens))
            index._df.update(counts.keys())
        index._avg_len = (
            sum(index._lengths) / len(index._lengths) if index._lengths else 0.0
        )
        return index

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Return [(chunk_id, score)] best-first, ties broken by chunk id."""
        if k <= 0:
            raise ValueError("k must be positive")
        terms = tokenize(query)
        n = len(self.chunk_ids)
        if not terms or not n:
            return []
        scores = [0.0] * n
        # Sorted, not set order: set iteration follows the per-process string hash seed,
        # which changes the float summation order and flips near-ties between runs.
        for term in sorted(set(terms)):
            df = self._df.get(term, 0)
            if not df:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for i, counts in enumerate(self._tf):
                freq = counts.get(term)
                if not freq:
                    continue
                denom = freq + self.k1 * (
                    1 - self.b + self.b * self._lengths[i] / (self._avg_len or 1.0)
                )
                scores[i] += idf * freq * (self.k1 + 1) / denom
        ranked = sorted(
            (i for i in range(n) if scores[i] > 0),
            key=lambda i: (-scores[i], self.chunk_ids[i]),
        )
        return [(self.chunk_ids[i], scores[i]) for i in ranked[:k]]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], k: int, rrf_k: int = 60
) -> list[str]:
    """Fuse ranked chunk-id lists by reciprocal rank.

    Score of an id is sum over input rankings of 1/(rrf_k + rank), rank
    1-based. RRF is used rather than score interpolation on purpose: dense
    cosine similarities and BM25 scores are on incomparable scales, and any
    weighted sum of them would need a tuned weight that this development
    stage has no held-out data to fit. RRF needs no such weight.

    Ties are broken by chunk id so the fusion is fully deterministic.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        seen: set[str] = set()
        for rank, chunk_id in enumerate(ranking, start=1):
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            scores[chunk_id] += 1.0 / (rrf_k + rank)
    return [
        cid for cid, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    ][:k]


@dataclass(frozen=True)
class EdgeSet:
    """Directed document-to-document edges with a single declared source.

    `provenance` is mandatory and describes where every edge came from
    (e.g. `"authored:v3-source-catalog.json"`). Two EdgeSets are never
    merged by this module: an authored-versus-extracted comparison is only
    meaningful if both were actually built and run separately, so mixing
    them into one traversal would destroy the very contrast it claims to
    measure.
    """

    provenance: str
    adjacency: dict[str, tuple[str, ...]]

    @classmethod
    def from_triples(
        cls, provenance: str, triples: Iterable[tuple[str, str]]
    ) -> "EdgeSet":
        if not provenance or not provenance.strip():
            raise ValueError("EdgeSet requires a nonempty provenance string")
        built: dict[str, list[str]] = defaultdict(list)
        for source, target in triples:
            if target not in built[source]:
                built[source].append(target)
        return cls(provenance, {k: tuple(sorted(v)) for k, v in built.items()})

    def neighbors(self, doc_path: str) -> tuple[str, ...]:
        return self.adjacency.get(doc_path, ())


@dataclass
class ArmResult:
    """One arm's ranking plus the labels a report needs to not mislead."""

    arm: str
    ranked_chunk_ids: list[str]
    ranked_doc_paths: list[str]
    capability: str
    selector_provenance: str = ""
    notes: str = ""

    @property
    def is_oracle(self) -> bool:
        return self.arm in ORACLE_ARMS


def _doc_order(chunks: Sequence[dict], chunk_ids: Sequence[str]) -> list[str]:
    by_id = {str(c["chunk_id"]): c["doc_path"] for c in chunks}
    seen: list[str] = []
    for cid in chunk_ids:
        path = by_id[cid]
        if path not in seen:
            seen.append(path)
    return seen


def lexical_retrieval(
    index: BM25Index, chunks: Sequence[dict], question: str, k: int
) -> ArmResult:
    """C_bm25: lexical-only control. Natural language in, no gold signal."""
    ranked = [cid for cid, _ in index.search(question, k)]
    return ArmResult(
        arm="C_bm25",
        ranked_chunk_ids=ranked,
        ranked_doc_paths=_doc_order(chunks, ranked),
        capability="natural_language_retrieval",
        notes="Okapi BM25 over the same chunk slice as the dense arm.",
    )


def hybrid_fusion_retrieval(
    dense_ranked_chunk_ids: Sequence[str],
    index: BM25Index,
    chunks: Sequence[dict],
    question: str,
    k: int,
    rrf_k: int = 60,
) -> ArmResult:
    """C_hybrid: dense + BM25 fused by RRF.

    This arm must be run before attributing any gain to graph structure.
    A graph arm that beats dense-only but not dense+BM25 has demonstrated
    a lexical gap, not a structural one.
    """
    lexical_ranked = [cid for cid, _ in index.search(question, k)]
    fused = reciprocal_rank_fusion([list(dense_ranked_chunk_ids), lexical_ranked], k, rrf_k)
    return ArmResult(
        arm="C_hybrid",
        ranked_chunk_ids=fused,
        ranked_doc_paths=_doc_order(chunks, fused),
        capability="natural_language_retrieval",
        notes=f"Reciprocal rank fusion of dense and BM25 rankings (rrf_k={rrf_k}).",
    )


def metadata_filter_retrieval(
    chunks: Sequence[dict],
    doc_metadata: dict[str, dict],
    metadata_filter: dict,
    k: int,
    selector_provenance: str,
) -> ArmResult:
    """C_meta_oracle (S3): select documents by an authored metadata filter.

    ORACLE. `metadata_filter` in the v4 pilot comes straight from the
    query's gold `filter` block, so this arm is handed the exact selector a
    real system would have to infer from the question. It answers "can
    authored frontmatter metadata express this set at all, and cleanly?" —
    a capability ceiling — and must never be reported as natural-language
    retrieval performance.

    Supported operators: `exact_set` (unordered set equality on a list-
    valued field) and `contains` (membership in a list-valued field).
    Unordered set equality is used deliberately: the v4 correction made
    category comparisons order-insensitive, and an order-sensitive
    comparison here would silently reintroduce that bug.
    """
    field_name = metadata_filter["field"]
    operator = metadata_filter["operator"]
    value = metadata_filter["value"]
    if operator not in ("exact_set", "contains"):
        raise ValueError(f"Unsupported metadata operator {operator!r}")

    matched: list[str] = []
    for doc_path, meta in sorted(doc_metadata.items()):
        raw = meta.get(field_name)
        if raw is None:
            continue
        values = raw if isinstance(raw, list) else [raw]
        if operator == "exact_set" and set(values) == set(value):
            matched.append(doc_path)
        elif operator == "contains" and value in values:
            matched.append(doc_path)

    matched = matched[:k]
    chunk_ids = [
        str(c["chunk_id"]) for c in chunks if c["doc_path"] in set(matched)
    ]
    truncated = len(chunk_ids) > k
    return ArmResult(
        arm="C_meta_oracle",
        ranked_chunk_ids=chunk_ids[:k],
        ranked_doc_paths=matched,
        capability="oracle_diagnostic",
        selector_provenance=selector_provenance,
        notes=(
            "Oracle capability diagnostic: the structured filter was supplied "
            "from the gold annotation, not inferred from the question. "
            f"Selected {len(matched)} document(s) carrying {len(chunk_ids)} chunk(s); "
            f"chunk list capped at k={k} (truncated={truncated}) so `k` means the "
            "same thing here as in every other arm."
        ),
    )


def hierarchy_prefix_retrieval(
    chunks: Sequence[dict],
    all_doc_paths: Sequence[str],
    prefix: str,
    k: int,
    selector_provenance: str,
    include_descendants: bool = False,
) -> ArmResult:
    """C_hier_oracle (S4): select documents under an authored path prefix.

    ORACLE whenever `prefix` is derived from the gold expected paths. The
    v4 S4 questions state the prefix in their own text ("All documents
    under account-and-profile/get-started/"), so a prefix parsed from the
    question alone would be a legitimate natural-language arm — but that is
    a property of these particular development questions, not of the
    stratum, and conflating the two would overstate the arm. The caller
    declares which it did via `selector_provenance`; the arm stays labelled
    oracle so a report never averages it into natural-language results by
    accident.

    `include_descendants=False` matches the v4 gold sets, which list only
    immediate members of the named directory.
    """
    if not prefix:
        raise ValueError("prefix must be nonempty")
    matched = []
    for path in sorted(all_doc_paths):
        if not path.startswith(prefix):
            continue
        remainder = path[len(prefix) :]
        if not include_descendants and "/" in remainder:
            continue
        matched.append(path)
    matched = matched[:k]
    chunk_ids = [
        str(c["chunk_id"]) for c in chunks if c["doc_path"] in set(matched)
    ]
    truncated = len(chunk_ids) > k
    return ArmResult(
        arm="C_hier_oracle",
        ranked_chunk_ids=chunk_ids[:k],
        ranked_doc_paths=matched,
        capability="oracle_diagnostic",
        selector_provenance=selector_provenance,
        notes=(
            "Oracle capability diagnostic: path prefix supplied rather than "
            f"inferred (include_descendants={include_descendants}). "
            f"Selected {len(matched)} document(s) carrying {len(chunk_ids)} chunk(s); "
            f"chunk list capped at k={k} (truncated={truncated}) so `k` means the "
            "same thing here as in every other arm."
        ),
    )


def graph_expanded_retrieval(
    seed_ranked_chunk_ids: Sequence[str],
    chunks: Sequence[dict],
    edges: EdgeSet,
    k: int,
    seed_depth: int,
    hops: int = 1,
    arm_name: str = "C_graph_authored",
    expansion_budget: int | None = None,
) -> ArmResult:
    """Expand a fixed seed ranking along one declared edge set.

    Everything that is not the graph is pinned by the caller and recorded:
    the seed ranking is passed in already computed, `seed_depth` fixes how
    many seed passages are allowed to expand, `hops` fixes traversal depth,
    and `k` fixes the final budget. Two graph arms differing only in
    `edges.provenance` are therefore directly comparable, which is the only
    way an authored-versus-extracted claim could ever be made honestly.

    `expansion_budget` reserves that many of the `k` slots for graph-reached
    chunks, filling the remaining `k - expansion_budget` from the seed
    ranking. This parameter is NOT a tuning knob to be quietly optimised --
    it exists because without it the arm is silently a no-op. The dense
    baseline already returns exactly `k` seeds, so appending expansions
    after them and truncating to `k` yields a ranking byte-identical to the
    seed ranking, and the arm would report "no effect from authored edges"
    when in fact it never got to place a single edge-reached passage. Any
    reported comparison must state the budget it used.

    Ranking rule (fixed, not tuned): retained seeds keep their original
    order and precede expansions; expansions are ordered by (hop distance,
    the rank of the seed that reached them, doc_path), then contribute
    their document's chunks in chunk-id order. No score is invented for an
    expanded chunk, because there is no comparable score to invent -- the
    graph contributes ordering, not similarity.
    """
    if k <= 0 or seed_depth <= 0 or hops <= 0:
        raise ValueError("k, seed_depth and hops must be positive")
    if expansion_budget is not None and not 0 <= expansion_budget < k:
        raise ValueError("expansion_budget must be in [0, k)")
    by_id = {str(c["chunk_id"]): c for c in chunks}
    by_doc: dict[str, list[str]] = defaultdict(list)
    for chunk in chunks:
        by_doc[chunk["doc_path"]].append(str(chunk["chunk_id"]))
    for ids in by_doc.values():
        ids.sort()

    all_seeds = list(seed_ranked_chunk_ids)[:k]
    seed_docs = []
    for cid in all_seeds[:seed_depth]:
        path = by_id[cid]["doc_path"]
        if path not in seed_docs:
            seed_docs.append(path)

    reached: dict[str, tuple[int, int, str]] = {}
    frontier = [(path, rank) for rank, path in enumerate(seed_docs)]
    visited = set(seed_docs)
    for hop in range(1, hops + 1):
        next_frontier: list[tuple[str, int]] = []
        for path, seed_rank in frontier:
            for neighbor in edges.neighbors(path):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                reached[neighbor] = (hop, seed_rank, neighbor)
                next_frontier.append((neighbor, seed_rank))
        frontier = next_frontier
        if not frontier:
            break

    kept_seeds = all_seeds if expansion_budget is None else all_seeds[: k - expansion_budget]
    ranked = list(kept_seeds)
    seen = set(ranked)
    expansions: list[str] = []
    for path in sorted(reached, key=lambda p: reached[p]):
        for cid in by_doc.get(path, ()):
            if cid not in seen:
                seen.add(cid)
                expansions.append(cid)
    if expansion_budget is not None:
        expansions = expansions[:expansion_budget]
    ranked = (ranked + expansions)[:k]
    return ArmResult(
        arm=arm_name,
        ranked_chunk_ids=ranked,
        ranked_doc_paths=_doc_order(chunks, ranked),
        capability="natural_language_retrieval",
        selector_provenance=edges.provenance,
        notes=(
            f"Seed ranking held fixed; {len(seed_docs)} seed documents expanded "
            f"{hops} hop(s) over edges from {edges.provenance!r}; k={k}, "
            f"seed_depth={seed_depth}, expansion_budget={expansion_budget}, "
            f"seeds_kept={len(kept_seeds)}, expansions_placed={len(expansions)}."
        ),
    )
