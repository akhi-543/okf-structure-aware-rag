"""Pure helpers for the Brightmart v4 configurations (next iteration, T1-T7).

Preregistration: docs/brightmart_v4_prereg.md. The constants below must equal it.
Nothing here touches Postgres, models or files, and no gold reaches any ranking
function: every function takes the question text, corpus metadata, authored (or
extracted) edges and precomputed scores only.

Document metadata (`docs_meta`) maps a bundle-relative path to
``{"title", "type", "status", "fm"}`` where ``fm`` is the stored frontmatter
(`documents.frontmatter`, i.e. every source key except type/title/description/
status/timestamp).
"""
from __future__ import annotations

import re
from collections import defaultdict

from okf_rag.eval.ir_metrics import mrr_at_k, ndcg_at_k, precision_at_k, recall_at_k
from okf_rag.eval.stats import holm, paired_bootstrap
from okf_rag.retrieve.brightmart_pilot_arms_v3 import normalize
from okf_rag.transcode.edge_weights import EDGE_WEIGHTS

K_DOCS = 50
MAIN_FAMILY = ("R2s", "R1h", "R3s", "R5p", "R6f", "R7f", "R7h")
T2_FAMILY = ("R5p", "R2s")
T4_ARMS = ("R7f", "R7h")
T5_ARMS = ("R1h", "R3s")
RRF_GRID = (1, 3, 10, 30, 60)
EXPANSION = {"lambda": 0.25, "n_seeds": 5}  # dev-tuned (prereg amendment 1)
TWO_STEP_MAX_ENTITIES = 3
THRESHOLDS_V4 = {
    "metric": "nDCG@10",
    "n_boot": 10000,
    "seed": 7,
    "alpha": 0.05,
    "c3_ratio": 0.5,
    "t4_s2_margin": -0.05,
    "t7_hhem_faithful": 0.5,
}

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

METRIC_KEYS = ("P@1", "P@10", "R@10", "R@50", "MRR@10", "nDCG@10", "F1@10")


def doc_metrics_v4(ranked: list[str], gold: set[str]) -> dict[str, float]:
    p10, r10 = precision_at_k(ranked, gold, 10), recall_at_k(ranked, gold, 10)
    return {
        "P@1": precision_at_k(ranked, gold, 1),
        "P@10": p10,
        "R@10": r10,
        "R@50": recall_at_k(ranked, gold, 50),
        "MRR@10": mrr_at_k(ranked, gold, 10),
        "nDCG@10": ndcg_at_k(ranked, gold, 10),
        "F1@10": 2 * p10 * r10 / (p10 + r10) if p10 + r10 else 0.0,
    }


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def heading_path_text(title: str, heading_path: list[str], text: str) -> str:
    """T5 contextual chunk text: the section heading path without the page title, then the text."""
    path = list(heading_path or [])
    if path and path[0] == title:
        path = path[1:]
    return f"{' > '.join(path)}\n{text}" if path else text


def _contains_seq(tokens: list[str], seq: list[str]) -> bool:
    n = len(seq)
    return n > 0 and any(tokens[i:i + n] == seq for i in range(len(tokens) - n + 1))


def title_matches(question: str, docs_meta: dict, exclude_types=("home", "schema")) -> list[str]:
    """Documents whose full normalized title occurs as a token sequence in the question,
    longest title first (ties by path)."""
    q = normalize(question)
    hits = [(len(normalize(m["title"])), p) for p, m in docs_meta.items()
            if m["type"] not in exclude_types and _contains_seq(q, normalize(m["title"]))]
    return [p for _, p in sorted(hits, key=lambda x: (-x[0], x[1]))]


# ---------------------------------------------------------------------------
# Target type (shared by R5p and R6f)
# ---------------------------------------------------------------------------

TYPE_WORDS = {
    "store": ("store", "stores", "location", "locations", "supercenter", "supercenters"),
    "promotion": ("promotion", "promotions", "promo", "promos"),
    "supplier": ("supplier", "suppliers", "vendor", "vendors"),
    "department": ("department", "departments"),
    "category": ("category", "categories"),
    "region": ("region", "regions"),
    "policy": ("policy", "policies"),
}
_WORD_TYPE = {w: t for t, ws in TYPE_WORDS.items() for w in ws}
_CUES = {"which", "what", "list", "name"}
_DETERMINERS = {"the", "all", "every", "of", "our", "ones", "each"}
# Words that end the noun phrase after a cue (verbs, auxiliaries, prepositions, adverbs).
_PHRASE_STOP = {
    "are", "is", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "can", "could",
    "will", "would", "should", "may", "might", "in", "at", "on", "with", "that", "which", "who", "where",
    "from", "to", "for", "by", "under", "over", "belong", "belongs", "run", "runs", "opened", "offer",
    "offers", "give", "gives", "take", "takes", "start", "starts", "begin", "begins", "operate", "operates",
    "sell", "sells", "lack", "lacks", "include", "includes", "fall", "falls", "come", "comes", "make",
    "makes", "measure", "exceed", "exceeds", "located", "based", "headquartered", "first", "currently",
    "sized", "carried", "launching", "running", "being", "bigger", "larger", "smaller",
}
_FALLBACK_TYPES = ("store", "promotion", "supplier", "category", "department")


def target_type(question: str) -> str | None:
    """The document type a list question asks for, or None.

    After a cue word (which/what/list/name) the head of the following noun phrase is
    its last type word before the phrase ends ("Which Mountain region locations first
    opened ..." -> store). Otherwise the first store/promotion/supplier/category/
    department word; a question starting with "where" asks for stores."""
    words = re.findall(r"[a-z&'-]+", question.lower())
    for i, w in enumerate(words):
        if w not in _CUES:
            continue
        found = None
        for x in words[i + 1:i + 7]:
            if x in _DETERMINERS and found is None:
                continue
            if x in _PHRASE_STOP:
                break
            found = _WORD_TYPE.get(x, found)
        if found:
            return found
    for w in words:
        if _WORD_TYPE.get(w) in _FALLBACK_TYPES:
            return _WORD_TYPE[w]
    if words and words[0] == "where":
        return "store"
    return None


# ---------------------------------------------------------------------------
# R3s / R3x: typed-weight link expansion seeded from a ranking (T5, T6)
# ---------------------------------------------------------------------------

def weighted_expansion(seed_ranked: list[tuple[str, float]], edges: list[tuple[str, str, str]],
                       k: int = K_DOCS, lam: float = EXPANSION["lambda"],
                       n_seeds: int = EXPANSION["n_seeds"], weights: dict | None = None) -> list[str]:
    """score(d) = s(d) + lam * max over seeds e adjacent to d of w(kind) * s(e).

    `seed_ranked` is (path, score) best first; s = score / max score (0 for documents
    outside the list). Edges are undirected; parallel edges between one pair count once
    with their largest weight. The boost is the best single seed's, not a sum over seeds, so
    a hub page adjacent to every seed does not outrank the seeds' own hits. Ties: seed-list
    order, then path.
    """
    weights = weights or EDGE_WEIGHTS
    if not seed_ranked:
        return []
    top = max(s for _, s in seed_ranked) or 1.0
    base = {p: s / top for p, s in seed_ranked}
    order = {p: i for i, (p, _) in enumerate(seed_ranked)}
    adj: dict[str, dict[str, float]] = defaultdict(dict)
    for src, dst, kind in edges:
        if src == dst:
            continue
        w = weights[kind]
        adj[src][dst] = max(adj[src].get(dst, 0.0), w)
        adj[dst][src] = max(adj[dst].get(src, 0.0), w)
    boost: dict[str, float] = defaultdict(float)
    for seed, _ in seed_ranked[:n_seeds]:
        for nb, w in adj.get(seed, {}).items():
            boost[nb] = max(boost[nb], lam * w * base[seed])
    score = dict(base)
    for p, v in boost.items():
        score[p] = score.get(p, 0.0) + v
    ranked = sorted(score, key=lambda p: (-score[p], order.get(p, len(order)), p))
    return ranked[:k]


# ---------------------------------------------------------------------------
# R5p: parent-aware traversal (T2)
# ---------------------------------------------------------------------------

def children_by_parent(edges: list[tuple[str, str, str]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for src, dst, kind in edges:
        if kind == "child":
            out[src].append(dst)
    return {p: sorted(set(c)) for p, c in out.items()}


def parent_aware(question: str, r2s_scores: dict[str, float], r2s_ranked: list[str], docs_meta: dict,
                 children: dict[str, list[str]], k: int = K_DOCS) -> list[str]:
    """Resolve parents named in the question, return their children (ordered by R2s
    score), then the R2s ranking. Falls back to R2s when no parent resolves or the
    question does not ask for the parents' child type."""
    want = target_type(question)
    if want is None:
        return r2s_ranked[:k]
    parents = [p for p in title_matches(question, docs_meta)
               if p in children and docs_meta[p]["type"] != "home"]
    kids = sorted({c for p in parents for c in children[p] if docs_meta[c]["type"] == want},
                  key=lambda c: (-r2s_scores.get(c, 0.0), c))
    if not kids:
        return r2s_ranked[:k]
    return list(dict.fromkeys(kids + list(r2s_ranked)))[:k]


# ---------------------------------------------------------------------------
# R6f: rule-based query-to-filter (T3)
# ---------------------------------------------------------------------------

MONTHS = ("january", "february", "march", "april", "may", "june", "july", "august", "september",
          "october", "november", "december")
CATEGORICAL_FIELDS = ("region", "store_format", "store_status", "country", "dept_group", "department",
                      "supplier", "formats")
_NEG = {"no", "not", "non", "without", "lack", "lacking", "neither", "nor", "outside", "except",
        "excluding", "don", "doesn", "other", "rather"}
_BOOL_CUES = {"has_pharmacy": ("pharmacy", "prescription"),
              "has_fuel": ("fuel", "gas", "pump", "tank")}
_BEFORE_OPS = (  # phrase ending right before the number -> op (longest phrases first)
    ("larger than", ">"), ("bigger than", ">"), ("more than", ">"), ("greater than", ">"),
    ("later than", ">"), ("exceeding", ">"), ("exceed", ">"), ("over", ">"), ("above", ">"), ("after", ">"),
    ("at least", ">="), ("since", ">="),
    ("smaller than", "<"), ("less than", "<"), ("fewer than", "<"), ("earlier than", "<"),
    ("prior to", "<"), ("under", "<"), ("below", "<"), ("before", "<"),
    ("at most", "<="),
)
_AFTER_OPS = (  # phrase shortly after the number -> op
    ("or more", ">="), ("or higher", ">="), ("or later", ">="), ("or above", ">="), ("and up", ">="),
    ("onward", ">="), ("or less", "<="), ("or lower", "<="), ("or earlier", "<="), ("or below", "<="),
)


def field_types(docs_meta: dict) -> dict[str, dict[str, str]]:
    """type -> field -> kind (int/bool/date/list/str) for fields present on at least half
    of the documents of that type."""
    by_type: dict[str, list[dict]] = defaultdict(list)
    for m in docs_meta.values():
        by_type[m["type"]].append(m["fm"])
    out: dict[str, dict[str, str]] = {}
    for t, fms in by_type.items():
        counts: dict[str, int] = defaultdict(int)
        kinds: dict[str, str] = {}
        for fm in fms:
            for key, v in fm.items():
                counts[key] += 1
                if isinstance(v, bool):
                    kinds[key] = "bool"
                elif isinstance(v, int):
                    kinds[key] = "int"
                elif isinstance(v, list):
                    kinds[key] = "list"
                elif isinstance(v, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):
                    kinds[key] = "date"
                else:
                    kinds[key] = "str"
        out[t] = {key: kinds[key] for key, n in counts.items() if 2 * n >= len(fms)}
    return out


def categorical_values(docs_meta: dict, doc_type: str, field: str) -> list[str]:
    vals = set()
    for m in docs_meta.values():
        if m["type"] != doc_type or field not in m["fm"]:
            continue
        v = m["fm"][field]
        vals.update(v if isinstance(v, list) else [v])
    return sorted(str(x) for x in vals)


def _negated(tokens: list[str], start: int, window: int = 4) -> bool:
    return any(t in _NEG for t in tokens[max(0, start - window):start])


def _number(s: str) -> int:
    return int(s.replace(",", ""))


def _op_before(text_before: str, text_after: str) -> str | None:
    """Comparison implied by the words around a number: a trailing phrase such as
    'or more' wins, then the phrase that ends right before the number."""
    after = text_after[:16]
    for phrase, op in _AFTER_OPS:
        if phrase in after:
            return op
    before = text_before.rstrip()
    for phrase, op in _BEFORE_OPS:
        if before.endswith(phrase):
            return op
    return None


def _numeric_filters(q: str, field: str, pattern: str) -> list[dict]:
    """Numeric conditions on `field` for numbers matched by `pattern` (one capture group)."""
    out = []
    between = re.search(rf"(?:between|from)\s+{pattern}\s+(?:and|to|through)\s+{pattern}", q)
    if between:
        return [{"field": field, "op": "between", "value": [_number(between.group(1)), _number(between.group(2))]}]
    between = re.search(rf"{pattern}\s+(?:through|to)\s+{pattern}", q)
    if between:
        return [{"field": field, "op": "between", "value": [_number(between.group(1)), _number(between.group(2))]}]
    for m in re.finditer(pattern, q):
        op = _op_before(q[:m.start()], q[m.end():])
        if op is None and re.search(r"\bin\s*$", q[:m.start()]):
            op = "=="
        if op:
            out.append({"field": field, "op": op, "value": _number(m.group(1))})
    return out


def parse_filters(question: str, docs_meta: dict, ftypes: dict | None = None) -> dict:
    """Question -> {"type": str|None, "filters": [{"field", "op", "value"}]}.

    Ops: ==, !=, >, >=, <, <=, between (inclusive), in, not_in, only (list field holds
    exactly the values), month_overlap, start_month, end_month. Only fields that exist
    on the target type are used.
    """
    ftypes = ftypes if ftypes is not None else field_types(docs_meta)
    t = target_type(question)
    if t is None or t not in ftypes:
        return {"type": t, "filters": []}
    fields = ftypes[t]
    q = question.lower()
    tokens = normalize(question)
    filters: list[dict] = []

    for field in CATEGORICAL_FIELDS:
        if field not in fields:
            continue
        pos, neg = [], []
        for value in categorical_values(docs_meta, t, field):
            seq = normalize(value)
            for i in range(len(tokens) - len(seq) + 1):
                if tokens[i:i + len(seq)] == seq:
                    if seq == ["open"] and tokens[i + 1:i + 2] == ["since"]:
                        continue  # "been open since 2004" dates the store, not its status
                    (neg if _negated(tokens, i, 3) else pos).append(value)
                    break
        exclusive = fields[field] == "list" and re.search(r"\b(only|limited to|exclusively|alone)\b", q)
        if len(set(pos)) == len(categorical_values(docs_meta, t, field)) > 1:
            pos = []  # every value named: a choice question ("food, health or general"), not a filter
        if pos and exclusive:
            filters.append({"field": field, "op": "only", "value": sorted(set(pos))})
        elif pos:
            filters.append({"field": field, "op": "in", "value": sorted(set(pos))})
        if neg:
            filters.append({"field": field, "op": "not_in", "value": sorted(set(neg))})

    for field, cues in _BOOL_CUES.items():
        if fields.get(field) != "bool":
            continue
        for i, tok in enumerate(tokens):
            if tok in cues:
                filters.append({"field": field, "op": "==", "value": not _negated(tokens, i)})
                break

    if fields.get("opened_year") == "int":
        filters += _numeric_filters(q, "opened_year", r"\b((?:19|20)\d{2})\b(?!,\d)")
    if fields.get("sq_ft") == "int" and re.search(r"square f(?:ee|oo)t|sq\.? ?ft", q):
        filters += _numeric_filters(q, "sq_ft", r"\b(\d{1,3}(?:,\d{3})+|\d{5,})\b")
    if fields.get("discount_pct") == "int":
        filters += _numeric_filters(q, "discount_pct", r"\b(\d{1,2})\s*(?:%|percent)")
    if fields.get("start") == "date" and fields.get("end") == "date":
        for mi, month in enumerate(MONTHS, start=1):
            if re.search(rf"\b{month}\b", q):
                if re.search(r"\b(start|starts|begin|begins|kick off|kicks off|launch|launching|launches)\b", q):
                    filters.append({"field": "start", "op": "start_month", "value": mi})
                elif re.search(r"\b(end|ends|finish|finishes)\b", q):
                    filters.append({"field": "end", "op": "end_month", "value": mi})
                else:
                    filters.append({"field": "start", "op": "month_overlap", "value": mi})
    return {"type": t, "filters": filters}


def _month(date: str) -> int:
    return int(date[5:7])


def _matches(fm: dict, flt: dict) -> bool:
    field, op, value = flt["field"], flt["op"], flt["value"]
    if op == "month_overlap":
        return "start" in fm and "end" in fm and _month(fm["start"]) <= value <= _month(fm["end"])
    if op == "start_month":
        return "start" in fm and _month(fm["start"]) == value
    if op == "end_month":
        return "end" in fm and _month(fm["end"]) == value
    if field not in fm:
        return False
    v = fm[field]
    vals = [str(x) for x in v] if isinstance(v, list) else [str(v)]
    try:
        if op == "in":
            return any(x in vals for x in value)
        if op == "only":
            return sorted(set(vals)) == sorted(set(value))
        if op == "not_in":
            return not any(x in vals for x in value)
        if op == "==":
            return v == value
        if op == "!=":
            return v != value
        if op == ">":
            return v > value
        if op == ">=":
            return v >= value
        if op == "<":
            return v < value
        if op == "<=":
            return v <= value
        if op == "between":
            return value[0] <= v <= value[1]
    except TypeError:
        return False
    raise ValueError(f"unknown op {op!r}")


def apply_filters(parsed: dict, docs_meta: dict) -> list[str]:
    """Paths of documents of the parsed type that satisfy every filter (sorted)."""
    t, flts = parsed.get("type"), parsed.get("filters") or []
    return sorted(p for p, m in docs_meta.items()
                  if m["type"] == t and all(_matches(m["fm"], f) for f in flts))


def filter_rank(parsed: dict, docs_meta: dict, r2s_scores: dict[str, float], r2s_ranked: list[str],
                k: int = K_DOCS) -> list[str]:
    """Documents passing the parsed filters (R2s score order), then the R2s ranking.
    R2s alone when no condition was parsed or nothing matches."""
    if not parsed.get("filters"):
        return r2s_ranked[:k]
    hits = apply_filters(parsed, docs_meta)
    if not hits:
        return r2s_ranked[:k]
    hits.sort(key=lambda p: (-r2s_scores.get(p, 0.0), p))
    return list(dict.fromkeys(hits + list(r2s_ranked)))[:k]


# ---------------------------------------------------------------------------
# R7h: two-step entity -> authored links (T4)
# ---------------------------------------------------------------------------

def undirected_adjacency(edges: list[tuple[str, str, str]]) -> dict[str, set[str]]:
    nb: dict[str, set[str]] = defaultdict(set)
    for src, dst, _ in edges:
        if src != dst:
            nb[src].add(dst)
            nb[dst].add(src)
    return nb


def two_step(question: str, docs_meta: dict, neighbors: dict[str, set[str]],
             dense_doc_scores: dict[str, float], fallback: list[str], k: int = K_DOCS) -> list[str]:
    """Entities named in the question, then their authored neighbours by dense score,
    then the fallback ranking."""
    entities = title_matches(question, docs_meta)[:TWO_STEP_MAX_ENTITIES]
    if not entities:
        return fallback[:k]
    nbs = {n for e in entities for n in neighbors.get(e, ()) if n not in entities}
    ordered = sorted(nbs, key=lambda p: (-dense_doc_scores.get(p, -1.0), p))
    return list(dict.fromkeys(entities + ordered + list(fallback)))[:k]


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------

def _pair(scores: dict[str, dict[str, float]], base: str, arm: str) -> dict:
    """Paired bootstrap of (arm - base) over the query ids both have."""
    qids = sorted(set(scores[base]) & set(scores[arm]))
    return paired_bootstrap([scores[base][q] for q in qids], [scores[arm][q] for q in qids],
                            n_boot=THRESHOLDS_V4["n_boot"], seed=THRESHOLDS_V4["seed"]) | {"n": len(qids)}


def _family(scores, comparisons: list[tuple[str, str]]) -> dict:
    res = {f"{a}-{b}": _pair(scores, b, a) for a, b in comparisons}
    sig = holm({name: r["p"] for name, r in res.items()}, alpha=THRESHOLDS_V4["alpha"])
    for name, r in res.items():
        r["holm_significant"] = bool(sig[name])
    return res


def verdict_v4(cells: dict[str, dict[str, dict[str, float]]], exploratory: dict | None = None) -> dict:
    """Apply T1-T5 (and the T6 report) to per-cell nDCG@10 scores.

    `cells` keys: "S34_t", "S34_p", "S4_np_t" (noparent corpus), "S3num_t", "S2_t", "S1to4_t";
    each maps arm -> query_id -> nDCG@10. Arms missing from a cell are skipped in reports.
    """
    th = THRESHOLDS_V4
    main = _family(cells["S34_t"], [(a, "R0") for a in MAIN_FAMILY])
    r2s = main["R2s-R0"]
    t1_para = _pair(cells["S34_p"], "R0", "R2s")
    t1_pass = (r2s["delta"] > 0 and r2s["ci_low"] > 0 and r2s["holm_significant"]
               and t1_para["ci_low"] > 0 and t1_para["delta"] >= th["c3_ratio"] * r2s["delta"])

    t2 = _family(cells["S4_np_t"], [(a, "R0") for a in T2_FAMILY])
    t2_pass = t2["R5p-R0"]["ci_low"] > 0 and t2["R5p-R0"]["holm_significant"]

    t3 = _family(cells["S3num_t"], [("R6f", "R0"), ("R6f", "R2s")])
    t3_pass = all(r["ci_low"] > 0 and r["holm_significant"] for r in t3.values())
    t3_expl = ({f"R6l-{b}": _pair(cells["S3num_t"], b, "R6l") for b in ("R0", "R2s")}
               if "R6l" in cells["S3num_t"] else None)

    t4 = {}
    for v in T4_ARMS:
        s2 = _pair(cells["S2_t"], "R0", v)
        s34 = main[f"{v}-R0"]
        ok = s2["ci_low"] > th["t4_s2_margin"] and s34["ci_low"] > 0 and s34["holm_significant"]
        t4[v] = {"S2": s2, "S34": s34, "pass": ok}
    t5 = {v: {**main[f"{v}-R0"], "kept": main[f"{v}-R0"]["ci_low"] > 0} for v in T5_ARMS}

    t6 = None
    if "R3x" in cells["S34_t"]:
        t6 = {"S34_t": _pair(cells["S34_t"], "R3x", "R3s"), "S1to4_t": _pair(cells["S1to4_t"], "R3x", "R3s")}

    s1 = {a: _pair(cells["S1_t"], "R0", a) for a in MAIN_FAMILY} if "S1_t" in cells else None
    return {
        "thresholds": dict(th),
        "main_family_S34_t": main,
        "T1": {"pass": bool(t1_pass), "R2s-R0": r2s, "paraphrase": t1_para},
        "T2": {"pass": bool(t2_pass), "family": t2},
        "T3": {"pass": bool(t3_pass), "family": t3, "exploratory_R6l": t3_expl},
        "T4": {"pass": any(x["pass"] for x in t4.values()), "variants": t4},
        "T5": t5,
        "T6_authored_minus_extracted": t6,
        "S1_t_vs_R0": s1,
        "exploratory": exploratory or {},
    }
