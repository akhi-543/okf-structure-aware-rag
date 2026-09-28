"""Validator for the Brightmart v4 corpora and question sets.

Checks:
- rendering is deterministic and the files under results/corpus_v4* equal a fresh render;
- both variants: frontmatter, unique titles, every parent chain reaches home, every body
  link resolves; the noparent variant carries no child->parent naming (T2);
- filler prose under ## Notes has no digits and no entity names;
- entity names used in answer scoring are not substrings of one another;
- both question sets: counts, paraphrase pairing, gold documents exist, S1/S2 spans are
  body lines of their gold documents, S3/S4 gold equals a re-execution of gold_sql,
  S4 hierarchy paths chain to home, S5 terms are absent from both variants, and no
  question text is shared between the sets;
- the freeze file (configs/brightmart_v4_freeze.json) matches the corpus hashes and the
  held-out files. `--freeze` writes it; do that once, before any configuration is run
  on the held-out set.

Usage:
    python doc_synthetic/brightmart_v4_validate.py            # validate (and check freeze)
    python doc_synthetic/brightmart_v4_validate.py --freeze   # validate, then write the freeze file
"""

import argparse
import datetime
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import brightmart_v4_seed as seed
import brightmart_v4_render as R
import brightmart_v4_questions_build as B

ROOT = Path(__file__).resolve().parents[1]
FREEZE_PATH = ROOT / 'configs' / 'brightmart_v4_freeze.json'
HELDOUT_FILES = [B.OUT['heldout'][k] for k in ('questions', 'qrels', 'gold_spans')]
_LINK = re.compile(r'\]\((/[^)]+)\)')


def corpus_sha256_of(docs: dict[str, str]) -> str:
    """Same digest as okf_rag.transcode.brightmart_synthetic.corpus_sha256 over the written files."""
    h = hashlib.sha256()
    for path in sorted(p.lstrip('/') for p in docs):
        h.update(path.encode('utf-8'))
        h.update(b'\0')
        h.update(docs['/' + path].encode('utf-8'))
        h.update(b'\0')
    return h.hexdigest()


def file_sha256(path) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def _split(text):
    _, fm, body = text.split('---\n', 2)
    return yaml.safe_load(fm), body


def check_corpus(variant: str, docs: dict[str, str], f: dict) -> list[str]:
    errs = []
    parsed = {p: _split(t) for p, t in docs.items()}
    titles = defaultdict(list)
    for p, (fm, _) in parsed.items():
        titles[fm['title']].append(p)
        for key in ('type', 'title', 'description', 'tags', 'status', 'timestamp'):
            if key not in fm:
                errs.append(f'{variant}: {p} lacks {key}')
    errs += [f'{variant}: duplicate title {t!r}: {ps}' for t, ps in titles.items() if len(ps) > 1]
    for p, (fm, body) in parsed.items():
        seen, cur = set(), p
        while cur != R.HOME_PATH:
            parent = parsed[cur][0].get('parent')
            if parent not in parsed or cur in seen:
                errs.append(f'{variant}: broken parent chain at {cur} ({parent})')
                break
            seen.add(cur)
            cur = parent
        for href in _LINK.findall(body):
            if href not in parsed:
                errs.append(f'{variant}: {p} links to missing {href}')
        notes = body.split('\n## Notes\n', 1)
        if len(notes) != 2:
            errs.append(f'{variant}: {p} has no Notes section')
    if variant == 'noparent':
        for p, (fm, body) in parsed.items():
            kind = fm['type']
            if kind == 'store' or (kind == 'archive' and 'store_id' in fm):
                if 'region' in fm or any(t in f['regions'] for t in fm['tags']):
                    errs.append(f'noparent: {p} still names its region')
                if '[Back to ' in body:
                    errs.append(f'noparent: {p} still links back to its region')
            if kind == 'category':
                if 'department' in fm or any(t in f['depts'] for t in fm['tags']):
                    errs.append(f'noparent: {p} still names its department')
                if 'This category is part of' in body:
                    errs.append(f'noparent: {p} still states its department')
    return errs


def entity_names(f: dict) -> list[str]:
    names = [s[2] for s in f['stores'].values()] + [R.short_name(s[2]) for s in f['stores'].values()]
    names += [s[3] for s in f['stores'].values()]
    names += [r[1] for r in f['regions'].values()] + [r[2] for r in f['regions'].values()]
    names += [d[1] for d in f['depts'].values()] + [c[2] for c in f['categories'].values()]
    names += [s[1] for s in f['suppliers'].values()] + [p[1] for p in f['promos'].values()]
    names += [p[1] for p in f['policies'].values()]
    names += sorted({m for sd in f['store_depts'].values() for m in sd.values()})
    return names


def check_filler(docs: dict[str, str], f: dict) -> list[str]:
    errs = []
    names = entity_names(f)
    for p, text in docs.items():
        notes = text.split('\n## Notes\n', 1)[1]
        if re.search(r'\d', notes):
            errs.append(f'filler in {p} contains a digit')
        for n in names:
            if re.search(rf'\b{re.escape(n)}\b', notes):
                errs.append(f'filler in {p} names entity {n!r}')
    return errs


def check_name_substrings(f: dict) -> list[str]:
    """Answer scoring matches names by substring; no name may contain another of its kind."""
    errs = []
    groups = {
        'store short name': [R.short_name(s[2]).lower() for s in f['stores'].values()],
        'category': [c[2].lower() for c in f['categories'].values()],
        'supplier': [s[1].lower() for s in f['suppliers'].values()],
        'promotion': [p[1].lower() for p in f['promos'].values()],
        'department': [d[1].lower() for d in f['depts'].values()],
    }
    for label, names in groups.items():
        for a in names:
            for b in names:
                if a != b and re.search(rf'\b{re.escape(a)}\b', b):
                    errs.append(f'{label} {a!r} occurs inside {b!r}')
    return errs


def _jsonl(path):
    return [json.loads(x) for x in (ROOT / path).read_text(encoding='utf-8').splitlines() if x.strip()]


def check_questions(which: str, conn, docs: dict[str, str], docs_np: dict[str, str]) -> list[str]:
    errs = []
    qs = _jsonl(B.OUT[which]['questions'])
    qrels = defaultdict(list)
    for r in _jsonl(B.OUT[which]['qrels']):
        qrels[r['query_id']].append(r['doc_path'])
    spans = defaultdict(list)
    for r in _jsonl(B.OUT[which]['gold_spans']):
        spans[r['query_id']].append(r)
    ids = {q['query_id'] for q in qs}
    counts = defaultdict(int)
    for q in qs:
        counts[q['stratum']] += 1
    if dict(counts) != {s: 80 for s in ('S1', 'S2', 'S3', 'S4', 'S5')}:
        errs.append(f'{which}: stratum counts {dict(counts)}')
    prefix = B.ID_PREFIX[which]
    ctx = B.Ctx(conn)
    for q in qs:
        qid = q['query_id']
        if not qid.startswith(prefix + 's'):
            errs.append(f'{which}: bad id {qid}')
        twin = qid[:-1] + ('p' if qid.endswith('t') else 't')
        if twin not in ids:
            errs.append(f'{which}: {qid} has no paraphrase twin')
        if sorted(qrels[qid]) != sorted(q['source_paths']):
            errs.append(f'{which}: {qid} qrels differ from source_paths')
        for p in q['source_paths']:
            if '/' + p not in docs:
                errs.append(f'{which}: {qid} gold {p} missing from corpus')
        st = q['stratum']
        if st in ('S1', 'S2'):
            doc_spans = [s for s in spans[qid] if s.get('doc_path')]
            if len(doc_spans) != len(q['source_paths']):
                errs.append(f'{which}: {qid} has {len(doc_spans)} spans for {len(q["source_paths"])} gold docs')
            for s in doc_spans:
                if s['span'] not in B.body_lines(docs['/' + s['doc_path']]):
                    errs.append(f'{which}: {qid} span not a body line of {s["doc_path"]}')
            if st == 'S2':
                types = {_split(docs['/' + p])[0]['type'] for p in q['source_paths']}
                if len(q['source_paths']) != 2 or len(types) != 2:
                    errs.append(f'{which}: {qid} S2 gold is not two documents of different types')
        if st in ('S3', 'S4'):
            sql = [s for s in spans[qid] if s.get('gold_sql')]
            if len(sql) != 1:
                errs.append(f'{which}: {qid} lacks gold_sql')
                continue
            rows = [r[0] for r in conn.execute(sql[0]['gold_sql'], sql[0]['gold_sql_params']).fetchall()]
            ents = [ctx.entity(e) for e in rows]
            if [p for p, _ in ents] != q['source_paths'] or [n for _, n in ents] != q['answer_names']:
                errs.append(f'{which}: {qid} gold differs from re-executed SQL')
            if st == 'S3' and q.get('condition') not in ('numeric', 'categorical'):
                errs.append(f'{which}: {qid} S3 without condition')
            if st == 'S4':
                chains = [s['path'] for s in spans[qid] if s.get('path')]
                if not chains:
                    errs.append(f'{which}: {qid} S4 without hierarchy path')
                for chain in chains:
                    if chain[0] != B.HOME or any('/' + p not in docs for p in chain):
                        errs.append(f'{which}: {qid} bad hierarchy path {chain}')
        if st == 'S5':
            terms = [s['absent_term'] for s in spans[qid] if s.get('absent_term')]
            if len(terms) != 1 or q['source_paths']:
                errs.append(f'{which}: {qid} S5 record malformed')
            for d in (docs, docs_np):
                if terms and any(terms[0].lower() in t.lower() for t in d.values()):
                    errs.append(f'{which}: {qid} absent term {terms[0]!r} occurs in the corpus')
    texts = [q['text'] for q in qs]
    if len(set(texts)) != len(texts):
        errs.append(f'{which}: duplicate question texts')
    return errs


def freeze_record(docs: dict[str, str], docs_np: dict[str, str]) -> dict:
    return {
        'frozen_on': datetime.date.today().isoformat(),
        'corpus_sha256': {'standard': corpus_sha256_of(docs), 'noparent': corpus_sha256_of(docs_np)},
        'heldout_files': {p: file_sha256(p) for p in HELDOUT_FILES},
    }


def check_freeze(docs, docs_np) -> list[str]:
    if not FREEZE_PATH.exists():
        return ['freeze file missing: run with --freeze once before any held-out run']
    frozen = json.loads(FREEZE_PATH.read_text(encoding='utf-8'))
    now = freeze_record(docs, docs_np)
    errs = []
    for k in ('corpus_sha256', 'heldout_files'):
        for name, digest in frozen[k].items():
            if now[k].get(name) != digest:
                errs.append(f'freeze mismatch: {k}/{name} changed since {frozen["frozen_on"]}')
    return errs


def validate() -> tuple[list[str], dict, dict]:
    conn = seed.connect()
    f = R.load_facts(conn)
    docs = R.render(conn, 'standard')
    docs_np = R.render(conn, 'noparent')
    errs = []
    if docs != R.render(conn, 'standard') or docs_np != R.render(conn, 'noparent'):
        errs.append('rendering is not deterministic')
    for variant, d in (('standard', docs), ('noparent', docs_np)):
        out = ROOT / R.OUT_DIRS[variant]
        on_disk = {'/' + p.relative_to(out).as_posix(): p.read_bytes().decode('utf-8')
                   for p in sorted(out.rglob('*.md'))} if out.exists() else {}
        if on_disk != d:
            errs.append(f'{variant}: files under {R.OUT_DIRS[variant]} differ from a fresh render '
                        f'(run python doc_synthetic/brightmart_v4_render.py)')
        errs += check_corpus(variant, d, f)
        errs += check_filler(d, f)
    if set(docs) != set(docs_np):
        errs.append('variants do not share their paths')
    errs += check_name_substrings(f)
    for which in B.SETS:
        errs += check_questions(which, conn, docs, docs_np)
    dev_texts = {q['text'] for q in _jsonl(B.OUT['dev']['questions'])}
    shared = dev_texts & {q['text'] for q in _jsonl(B.OUT['heldout']['questions'])}
    errs += [f'text shared by dev and heldout: {t!r}' for t in sorted(shared)]
    return errs, docs, docs_np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--freeze', action='store_true', help='write configs/brightmart_v4_freeze.json after validating')
    args = ap.parse_args()
    errs, docs, docs_np = validate()
    if args.freeze and not errs:
        if FREEZE_PATH.exists():
            sys.exit(f'{FREEZE_PATH} exists: the held-out set is already frozen')
        FREEZE_PATH.write_text(json.dumps(freeze_record(docs, docs_np), indent=2) + '\n', encoding='utf-8', newline='\n')
        print(f'froze {FREEZE_PATH.relative_to(ROOT).as_posix()}')
    errs += check_freeze(docs, docs_np)
    if errs:
        for e in errs[:50]:
            print(e)
        sys.exit(f'FAILED: {len(errs)} error(s)')
    print(f'OK: {len(docs)} docs x 2 variants, 400 dev + 400 heldout questions')


if __name__ == '__main__':
    main()
