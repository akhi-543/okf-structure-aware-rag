"""GPU steps of the Brightmart v4 study (T6 graph extraction, T3 LLM filters, T7 answers).

Subcommands (models from configs/models.lock; greedy decoding):
    extract-graph        T6: extract parent/mention links for every standard-corpus page with the
                         `graph_extraction` pin -> results/brightmart-v4/llm/graph_extracted.jsonl
    filters --set S      T3 (R6l): question -> JSON metadata filter with the `query_filter` pin
                         -> results/brightmart-v4/llm/filters_<S>.jsonl
    answer --set S       T7: pack each question's R0 / R2s context (from the run's t7_contexts.jsonl)
                         to 2,048 tokens and answer with the `answer_generator` pin
    judge --set S        T7: HHEM-2.1 (`faithfulness_judge`) and MiniCheck-7B
                         (`faithfulness_judge_minicheck`) scores for every answer
    score --set S        T7: correctness, faithfulness, abstention and the preregistered verdict (CPU)

Order: extract-graph and filters before `run_brightmart_v4 --mode heldout`; answer, judge and score
after it. `--model-override` (smoke tests only) requires an --out-dir whose name contains "smoke".
Never touches Postgres.

Usage:
    python scripts/llm_brightmart_v4.py extract-graph
    python scripts/llm_brightmart_v4.py filters --set heldout
    python scripts/llm_brightmart_v4.py answer --set heldout
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

from okf_rag.eval import answer_scoring_v4 as S
from okf_rag.jobs import brightmart_v4_llm as L
from okf_rag.jobs.model_lock import model_pin
from okf_rag.retrieve.brightmart_v4_arms import THRESHOLDS_V4

CORPUS = Path("results/corpus_v4")
LLM_OUT = Path("results/brightmart-v4/llm")
T7_OUT = Path("results/brightmart-v4/t7")
CONTEXTS = {"heldout": Path("results/brightmart-v4/t7_contexts.jsonl"),
            "dev": Path("results/brightmart-v4-dev/t7_contexts.jsonl")}
PACK_BUDGET = 2048
MINICHECK_HEADROOM_GIB = 5  # per GPU, for the 7B full-sequence forward pass (OOM on 2x T4 without it)
MINICHECK_CONFIG = {"attn_implementation": "sdpa"}  # the checkpoint ships "eager"; see _minicheck_scores


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _update_manifest(out_dir: Path, key: str, value: dict) -> None:
    path = out_dir / "manifest.json"
    m = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    m[key] = value
    out_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(m, indent=2), encoding="utf-8")


def _questions(which: str) -> dict[str, dict]:
    return {q["query_id"]: q for q in _jsonl(Path(f"doc_synthetic/brightmart_v4_{which}_questions.jsonl"))}


def _corpus_meta():
    """Standard-corpus documents via the transcoder (no database): path -> meta, plus ConceptDocs."""
    from okf_rag.transcode.brightmart_synthetic import transcode
    docs, _, _ = transcode(CORPUS)
    meta = {d.path: {"title": d.title, "type": d.okf_type, "status": d.status, "fm": d.x_source} for d in docs}
    return docs, meta


# --------------------------------------------------------------------------- models

def _dtype_kw() -> str:
    import transformers
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    return "dtype" if (major, minor) >= (4, 56) else "torch_dtype"


def max_memory_map(totals_bytes: list[int], headroom_gib: float) -> dict[int, str]:
    """Per-GPU weight budget that leaves `headroom_gib` free on every GPU for activations
    (a full-sequence forward pass of a 7B model needs several GiB beyond its weights)."""
    return {i: f"{max(1, int(t / 2**30 - headroom_gib))}GiB" for i, t in enumerate(totals_bytes)}


def load_causal_lm(role: str, override: str | None, headroom_gib: float | None = None,
                   config_overrides: dict | None = None):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    pin = model_pin(role)
    model_id, revision = (override, None) if override else (pin.model_id, pin.revision)
    trust = "MiniCheck" in model_id
    cuda = torch.cuda.is_available()
    dtype = torch.float16 if (cuda and pin.dtype == "float16") else torch.float32
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=trust)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = {"revision": revision, "trust_remote_code": trust, _dtype_kw(): dtype}
    if cuda:
        kw["device_map"] = "auto"
        if headroom_gib:
            kw["max_memory"] = max_memory_map(
                [torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())],
                headroom_gib)
    if config_overrides:
        config = AutoConfig.from_pretrained(model_id, revision=revision, trust_remote_code=trust)
        for key, value in config_overrides.items():
            setattr(config, key, value)
        kw["config"] = config
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw).eval()
    info = {"role": role, "model_id": model_id, "revision": revision, "dtype": str(dtype), "cuda": cuda,
            "override": bool(override), "config_overrides": config_overrides}
    return tok, model, info


def chat(tok, messages: list[dict]) -> str:
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def generate(tok, model, prompts: list[str], max_new_tokens: int, batch_size: int) -> list[dict]:
    """Greedy batched generation (left padding, longest prompts first)."""
    import torch
    torch.manual_seed(0)
    device = next(model.parameters()).device
    order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
    out: list[dict] = [None] * len(prompts)  # type: ignore[list-item]
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                                 pad_token_id=tok.pad_token_id)
        width = enc["input_ids"].shape[1]
        for j, i in enumerate(idx):
            cont = gen[j, width:]
            out[i] = {"text": tok.decode(cont, skip_special_tokens=True).strip(),
                      "prompt_tokens": int(enc["attention_mask"][j].sum()),
                      "completion_tokens": int((cont != tok.pad_token_id).sum())}
        print(f"  generated {min(start + batch_size, len(order))}/{len(order)}", flush=True)
    return out


def _out_dir(args, default: Path) -> Path:
    if args.model_override and (not args.out_dir or "smoke" not in args.out_dir):
        raise SystemExit("--model-override requires an --out-dir whose name contains 'smoke'")
    return Path(args.out_dir) if args.out_dir else default


# --------------------------------------------------------------------------- steps

def cmd_extract_graph(args) -> None:
    from okf_rag.ingest.flatten import flatten_doc
    out_dir = _out_dir(args, LLM_OUT)
    docs, meta = _corpus_meta()
    if args.limit:
        docs = docs[:args.limit]
    title_to_path = {m["title"]: p for p, m in meta.items()}
    catalog = sorted(title_to_path)
    tok, model, info = load_causal_lm("graph_extraction", args.model_override)
    prompts = [chat(tok, L.graph_prompt(d.title, flatten_doc(d), catalog)) for d in docs]
    t0 = time.perf_counter()
    gens = generate(tok, model, prompts, model_pin("graph_extraction").max_new_tokens, args.batch_size)
    seconds = time.perf_counter() - t0
    rows = []
    for d, g in zip(docs, gens):
        parsed = L.parse_graph_output(g["text"], d.path, title_to_path)
        rows.append({"path": d.path, "raw": g["text"], **parsed,
                     "prompt_tokens": g["prompt_tokens"], "completion_tokens": g["completion_tokens"]})
    _write_jsonl(out_dir / "graph_extracted.jsonl", rows)
    _update_manifest(out_dir, "extract", {
        **info, "docs": len(rows), "edges": sum(len(r["edges"]) for r in rows),
        "parse_ok": sum(r["parse_ok"] for r in rows), "unknown_titles": sum(len(r["unknown_titles"]) for r in rows),
        "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
        "completion_tokens": sum(r["completion_tokens"] for r in rows),
        "seconds": round(seconds, 1), "batch_size": args.batch_size, "corpus": CORPUS.as_posix()})
    print(f"extracted {sum(len(r['edges']) for r in rows)} edges from {len(rows)} pages in {seconds:.0f}s")


def cmd_filters(args) -> None:
    out_dir = _out_dir(args, LLM_OUT)
    _, meta = _corpus_meta()
    schema = L.filter_schema(meta)
    kinds = L.schema_kinds(meta)
    qs = list(_questions(args.set).values())
    if args.limit:
        qs = qs[:args.limit]
    tok, model, info = load_causal_lm("query_filter", args.model_override)
    prompts = [chat(tok, L.filter_prompt(q["text"], schema)) for q in qs]
    t0 = time.perf_counter()
    gens = generate(tok, model, prompts, model_pin("query_filter").max_new_tokens, args.batch_size)
    rows = [{"query_id": q["query_id"], "raw": g["text"], "parsed": L.parse_filter_output(g["text"], kinds),
             "prompt_tokens": g["prompt_tokens"], "completion_tokens": g["completion_tokens"]}
            for q, g in zip(qs, gens)]
    _write_jsonl(out_dir / f"filters_{args.set}.jsonl", rows)
    _update_manifest(out_dir, f"filters_{args.set}", {
        **info, "questions": len(rows), "with_filters": sum(bool(r["parsed"]["filters"]) for r in rows),
        "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
        "completion_tokens": sum(r["completion_tokens"] for r in rows),
        "seconds": round(time.perf_counter() - t0, 1), "batch_size": args.batch_size})
    print(f"filters for {len(rows)} questions")


def cmd_answer(args) -> None:
    from okf_web_bench.okf_web_bench_compare import numbered_documentation, pack_context
    out_dir = _out_dir(args, T7_OUT)
    contexts = _jsonl(Path(args.contexts) if args.contexts else CONTEXTS[args.set])
    qs = _questions(args.set)
    if args.limit:
        keep = sorted({c["query_id"] for c in contexts})[:args.limit]
        contexts = [c for c in contexts if c["query_id"] in keep]
    tok, model, info = load_causal_lm("answer_generator", args.model_override)
    items = []
    for c in contexts:
        packed, parts = pack_context(c["passages"], tok, PACK_BUDGET)
        doc = numbered_documentation(parts)
        items.append({"query_id": c["query_id"], "arm": c["arm"], "stratum": qs[c["query_id"]]["stratum"],
                      "context": doc, "tokens_in": len(tok.encode(packed, add_special_tokens=False)),
                      "n_passages": len(parts),
                      "sources": [p["doc_path"] for p in parts]})
    prompts = [chat(tok, L.answer_prompt(qs[it["query_id"]]["text"], it["context"])) for it in items]
    t0 = time.perf_counter()
    gens = generate(tok, model, prompts, model_pin("answer_generator").max_new_tokens, args.batch_size)
    for it, g in zip(items, gens):
        it.update({"answer": g["text"], "prompt_tokens": g["prompt_tokens"], "completion_tokens": g["completion_tokens"]})
    _write_jsonl(out_dir / f"answers_{args.set}.jsonl", items)
    _update_manifest(out_dir, f"answers_{args.set}", {
        **info, "answers": len(items), "pack_budget": PACK_BUDGET, "system_prompt": L.ANSWER_SYSTEM,
        "seconds": round(time.perf_counter() - t0, 1), "batch_size": args.batch_size,
        "contexts_sha256": _sha(Path(args.contexts) if args.contexts else CONTEXTS[args.set])})
    print(f"{len(items)} answers")


HHEM_FOUNDATION = "google/flan-t5-base"
HHEM_PROMPT = "<pad> Determine if the hypothesis is true given the premise?\n\nPremise: {text1}\n\nHypothesis: {text2}"


def load_hhem():
    """HHEM-2.1-Open without its remote code (which breaks on transformers 5): the pinned
    weights are a T5ForTokenClassification over flan-t5-base under the `t5.` prefix, scored
    exactly as the model's own `predict` does (token 0, softmax, P(consistent))."""
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoTokenizer, T5ForTokenClassification
    pin = model_pin("faithfulness_judge")
    weights = load_file(hf_hub_download(pin.model_id, "model.safetensors", revision=pin.revision))
    state = {k[len("t5."):]: v for k, v in weights.items() if k.startswith("t5.")}
    model = T5ForTokenClassification(AutoConfig.from_pretrained(HHEM_FOUNDATION))
    res = model.load_state_dict(state, strict=False)
    if res.unexpected_keys or set(res.missing_keys) - {"transformer.encoder.embed_tokens.weight"}:
        raise RuntimeError(f"HHEM weights do not fit T5ForTokenClassification: {res}")
    with torch.no_grad():
        model.transformer.encoder.embed_tokens.weight.copy_(state["transformer.shared.weight"])
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    return AutoTokenizer.from_pretrained(HHEM_FOUNDATION), model


def hhem_predict(tok, model, pairs: list[tuple[str, str]], batch_size: int = 8) -> list[float]:
    import torch
    out = []
    for i in range(0, len(pairs), batch_size):
        texts = [HHEM_PROMPT.format(text1=p, text2=h) for p, h in pairs[i:i + batch_size]]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            logits = model(**enc).logits[:, 0, :]
        out += [float(x) for x in torch.softmax(logits, dim=-1)[:, 1].cpu()]
    return out


def _hhem_scores(pairs: list[tuple[str, str]], batch_size: int) -> list[float]:
    tok, model = load_hhem()
    return hhem_predict(tok, model, pairs, batch_size)


def last_token_logits(model, ids):
    """Next-token logits for the last position only: runs the decoder body, then the output
    head on one position, instead of materializing (and upcasting) logits for every position."""
    import torch
    body = getattr(model, "model", None)
    head = getattr(model, "output", None) or getattr(model, "lm_head", None)
    with torch.no_grad():
        if body is None or head is None:
            return model(**ids, use_cache=False).logits[0, -1].float()
        hidden = body(**ids, use_cache=False)[0]
        return head(hidden[:, -1:, :])[0, -1].float()


def _minicheck_scores(docs_claims: list[tuple[str, str]]) -> list[float]:
    """Support probability P('Yes') from the next-token distribution (top 5), as MiniCheck does."""
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # The checkpoint's config selects "eager" attention, which materializes a 32 x L x L fp32 score
    # matrix per layer (out of memory on T4s); its own modeling code also ships an SDPA path.
    tok, model, _ = load_causal_lm("faithfulness_judge_minicheck", None, headroom_gib=MINICHECK_HEADROOM_GIB,
                                   config_overrides=MINICHECK_CONFIG)
    device = next(model.parameters()).device
    out = []
    for n, (doc, claim) in enumerate(docs_claims, start=1):
        ids = tok(chat(tok, L.minicheck_prompt(doc, claim)), return_tensors="pt", add_special_tokens=False).to(device)
        logits = last_token_logits(model, ids)
        if n % 100 == 0:
            print(f"  minicheck {n}/{len(docs_claims)}", flush=True)
        probs = torch.softmax(logits, dim=-1)
        top = torch.topk(probs, 5)
        out.append(float(sum(p for p, t in zip(top.values.tolist(), top.indices.tolist())
                             if tok.decode([t]).strip().lower() == "yes")))
    del model
    return out


def cmd_judge(args) -> None:
    out_dir = Path(args.out_dir) if args.out_dir else T7_OUT
    t0 = time.perf_counter()
    if args.minicheck_only:
        # Resume: keep the recorded HHEM scores (the primary judge) and add MiniCheck only.
        src = Path(args.from_dir) / f"judged_{args.set}.jsonl"
        if out_dir.resolve() == Path(args.from_dir).resolve():
            raise SystemExit("--minicheck-only writes to a new --out-dir, never over the scored run")
        rows = _jsonl(src)
        todo = [i for i, r in enumerate(rows) if not S.abstained(r["answer"])]
        if any("hhem" not in rows[i] for i in todo):
            raise SystemExit(f"{src} lacks HHEM scores: run judge without --minicheck-only first")
    else:
        rows = _jsonl(out_dir / f"answers_{args.set}.jsonl")
        todo = [i for i, r in enumerate(rows) if not S.abstained(r["answer"])]
        hh = _hhem_scores([(rows[i]["context"], S.strip_citations(rows[i]["answer"])) for i in todo], args.batch_size)
        for i, s in zip(todo, hh):
            rows[i]["hhem"] = s
    status = "ok"
    try:
        if args.skip_minicheck:
            raise RuntimeError("skipped (--skip-minicheck)")
        claims = [(i, c) for i in todo for c in S.claim_sentences(rows[i]["answer"])]
        mc = _minicheck_scores([(rows[i]["context"], c) for i, c in claims])
        per = {}
        for (i, _), s in zip(claims, mc):
            per[i] = min(per.get(i, 1.0), s)
        for i in todo:
            rows[i]["minicheck"] = per.get(i)
    except Exception as exc:  # judge unavailable on this runtime: recorded, HHEM stays primary
        status = f"unavailable: {type(exc).__name__}: {exc}"[:500]
    _write_jsonl(out_dir / f"judged_{args.set}.jsonl", rows)
    _update_manifest(out_dir, f"judge_{args.set}", {
        "hhem": model_pin("faithfulness_judge").model_id, "minicheck": model_pin("faithfulness_judge_minicheck").model_id,
        "minicheck_status": status, "judged": len(todo), "seconds": round(time.perf_counter() - t0, 1),
        "minicheck_only_from": args.from_dir if args.minicheck_only else None})
    print(f"judged {len(todo)} answers; minicheck {status}")


def cmd_score(args) -> None:
    out_dir = Path(args.out_dir) if args.out_dir else T7_OUT
    verdict_path = out_dir / f"t7_verdict_{args.set}.json"
    if args.set == "heldout" and verdict_path.exists():
        raise SystemExit(f"{verdict_path} exists: the T7 held-out verdict is one-shot")
    _, meta = _corpus_meta()
    qs = _questions(args.set)
    cands: dict[str, dict[str, str]] = {}
    for p, m in meta.items():
        c = cands.setdefault(m["type"], {})
        c[m["title"]] = m["title"]
        if m["type"] == "store":
            c[m["title"].replace("Brightmart ", "")] = m["title"]
    th = THRESHOLDS_V4["t7_hhem_faithful"]
    rows = []
    for r in _jsonl(out_dir / f"judged_{args.set}.jsonl"):
        q = qs[r["query_id"]]
        if q["stratum"] in ("S3", "S4"):
            correct = S.set_f1(r["answer"], q["answer_names"], cands[q["target_type"]])
        elif q["stratum"] in ("S1", "S2"):
            correct = S.fact_correct(r["answer"], q["reference_answer"])
        else:
            correct = float(S.abstained(r["answer"]))
        mc = r.get("minicheck")
        rows.append({"query_id": r["query_id"], "arm": r["arm"], "stratum": q["stratum"], "correct": correct,
                     "faithful": S.faithful(r.get("hhem"), r["answer"], th),
                     "faithful_minicheck": 1.0 if S.abstained(r["answer"]) else (float(mc >= 0.5) if mc is not None else None),
                     "abstained": float(S.abstained(r["answer"])), "tokens_in": r["tokens_in"]})
    v = S.t7_verdict(rows, th)
    mc_rows = [r for r in rows if r["faithful_minicheck"] is not None]
    if len(mc_rows) == len(rows):
        v["secondary_minicheck_S34"] = S.t7_verdict(
            [{**r, "faithful": r["faithful_minicheck"]} for r in rows], th)["S34_faithful"]
    v["context_tokens_mean"] = {arm: sum(r["tokens_in"] for r in rows if r["arm"] == arm) /
                                max(1, sum(1 for r in rows if r["arm"] == arm)) for arm in ("R0", "R2s")}
    _write_jsonl(out_dir / f"scored_{args.set}.jsonl", rows)
    verdict_path.write_text(json.dumps(v, indent=2), encoding="utf-8")
    fmt = lambda r: ("n/a (no items)" if r is None else
                     f"R0 {r['R0_mean']:.3f} vs R2s {r['R2s_mean']:.3f}, delta {r['delta']:+.3f}, "
                     f"CI [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}], n {r['n']}")
    md = [f"# Brightmart v4 - T7 answer quality ({args.set})", "",
          f"**T7: {'PASS' if v['pass'] else 'FAIL'}** (preregistered: S3+S4 F1 and faithful-rate CI lower bounds > 0, "
          "S5 abstention not lower)", "",
          f"- S3+S4 answer F1: {fmt(v['S34_correct_F1'])}",
          f"- S3+S4 faithful (HHEM >= {th}): {fmt(v['S34_faithful'])}",
          f"- S1+S2 correct: {fmt(v['S12_correct'])}",
          f"- S5 abstention: {fmt(v['S5_abstention'])}"]
    if "secondary_minicheck_S34" in v:
        md.append(f"- S3+S4 faithful (MiniCheck, secondary): {fmt(v['secondary_minicheck_S34'])}")
    md.append(f"- mean context tokens: {v['context_tokens_mean']}")
    (out_dir / f"t7_verdict_{args.set}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


def main() -> None:
    ap = argparse.ArgumentParser(description="GPU steps of the Brightmart v4 study.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("extract-graph", "filters", "answer", "judge", "score"):
        p = sub.add_parser(name)
        p.add_argument("--set", choices=["dev", "heldout"], default="heldout")
        p.add_argument("--batch-size", type=int, default=4)
        p.add_argument("--limit", type=int, default=0, help="first N items only (smoke tests)")
        p.add_argument("--model-override", default=None, help="smoke tests only; requires a *smoke* --out-dir")
        p.add_argument("--out-dir", default=None)
        p.add_argument("--contexts", default=None, help="answer: t7_contexts.jsonl to use")
        p.add_argument("--skip-minicheck", action="store_true", help="judge: HHEM only (smoke tests)")
        p.add_argument("--minicheck-only", action="store_true",
                       help="judge: keep the HHEM scores in --from-dir and add MiniCheck (writes to a new --out-dir)")
        p.add_argument("--from-dir", default=str(T7_OUT), help="judge --minicheck-only: directory of the scored run")
    args = ap.parse_args()
    {"extract-graph": cmd_extract_graph, "filters": cmd_filters, "answer": cmd_answer,
     "judge": cmd_judge, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
