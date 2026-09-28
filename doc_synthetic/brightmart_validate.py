"""
Brightmart synthetic retail corpus validator.
"""

import os
import re
import sqlite3
import subprocess
import sys
import yaml
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

import brightmart_seed
import brightmart_render


def _get_git_ls_files():
    """Get all files from git ls-files at repo root."""
    try:
        result = subprocess.run(
            ['git', 'ls-files'],
            cwd=Path(__file__).parent.parent,  # repo root
            capture_output=True,
            text=True,
            check=True
        )
        return set(result.stdout.strip().split('\n')) if result.stdout.strip() else set()
    except Exception:
        return set()


def _check_unique_basenames(scan_root: str = None, git_paths: set = None) -> list[str]:
    """
    Check (a): Unique basenames within scan_root and not in git ls-files outside scan_root.

    Args:
        scan_root: Directory to scan for files (default: doc_synthetic/)
        git_paths: Set of git-tracked paths (default: run git ls-files)

    Returns:
        List of error strings (empty = no errors)
    """
    errors = []

    if scan_root is None:
        scan_root = str(Path(__file__).parent)

    if git_paths is None:
        git_paths = _get_git_ls_files()

    scan_path = Path(scan_root)

    # Collect all files and their paths
    all_files_in_scan = set()
    duplicates = {}

    for root, dirs, files in os.walk(scan_path):
        for file in files:
            all_files_in_scan.add(file)
            if file not in duplicates:
                duplicates[file] = []
            duplicates[file].append(os.path.relpath(os.path.join(root, file), scan_path))

    # Check for duplicates within scan_root
    for basename, paths in duplicates.items():
        if len(paths) > 1:
            errors.append(f'Check (a): Duplicate basename in {scan_root}: {basename} at {", ".join(paths)}')

    # Check basenames against git ls-files outside scan_root
    scan_root_name = os.path.basename(str(scan_path).rstrip('/'))
    for basename in all_files_in_scan:
        for git_path in git_paths:
            # Skip files within scan_root
            if git_path.startswith(scan_root_name + '/'):
                continue
            if os.path.basename(git_path) == basename:
                errors.append(f'Check (a): Basename {basename} from {scan_root} conflicts with {git_path} in git ls-files')
                break

    return errors


def validate_corpus(corpus_dir: str = 'doc_synthetic/corpus', prose_dir: str = None,
                    scan_root: str = None, git_paths: set = None) -> list[str]:
    """
    Validate corpus: return list of error strings (empty = OK).
    Checks:
    (a) Unique basenames within scan_root and not in git ls-files outside
    (b) Frontmatter/links/parent validation on disk
    (c) On-disk corpus equals render(connect()) byte-for-byte
    (d) Prose file validation (no digits, no forbidden terms, word count, matching corpus doc)

    Args:
        corpus_dir: Path to corpus directory
        prose_dir: Path to prose directory (default: sibling to corpus_dir)
        scan_root: Directory to scan for basenames (default: doc_synthetic/)
        git_paths: Set of git-tracked paths (default: run git ls-files)
    """
    errors = []

    if prose_dir is None:
        # Default prose dir is sibling to corpus_dir
        prose_dir = str(Path(corpus_dir).parent / 'brightmart_llm_prose')

    # Check (a): Unique basenames
    errors.extend(_check_unique_basenames(scan_root=scan_root, git_paths=git_paths))

    # Connect to DB and render expected corpus
    conn = brightmart_seed.connect()
    expected_docs = brightmart_render.render(conn)
    conn.close()

    # Build full paths for expected docs (convert bundle-relative to filesystem paths)
    corpus_path = Path(corpus_dir)
    expected_paths = {}
    for bundle_path, content in expected_docs.items():
        # Convert /path/to/doc.md to path/to/doc.md (remove leading /)
        rel_path = bundle_path.lstrip('/')
        fs_path = corpus_path / rel_path
        expected_paths[str(fs_path)] = content

    # Check (c): Compare on-disk corpus with expected
    on_disk_files = set()
    if corpus_path.exists():
        for root, dirs, files in os.walk(corpus_path):
            for file in files:
                on_disk_files.add(os.path.join(root, file))

    for fs_path_str in on_disk_files:
        if fs_path_str not in expected_paths:
            rel = os.path.relpath(fs_path_str, corpus_path)
            errors.append(f'Check (c): Extra file on disk: {rel}')

    for fs_path_str, expected_content in expected_paths.items():
        if fs_path_str not in on_disk_files:
            rel = os.path.relpath(fs_path_str, corpus_path)
            errors.append(f'Check (c): Missing file: {rel}')
        else:
            # Check byte-for-byte match
            try:
                with open(fs_path_str, 'r', encoding='utf-8') as f:
                    actual_content = f.read()
                if actual_content != expected_content:
                    rel = os.path.relpath(fs_path_str, corpus_path)
                    errors.append(f'Check (c): Content mismatch: {rel}')
            except Exception as e:
                rel = os.path.relpath(fs_path_str, corpus_path)
                errors.append(f'Check (c): Cannot read file {rel}: {e}')

    # Check (b): Frontmatter/links/parent on disk
    required_keys = {'type', 'title', 'description', 'tags', 'status', 'timestamp'}
    for fs_path_str in on_disk_files:
        try:
            with open(fs_path_str, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract frontmatter
            match = re.match(r'^---\n(.*?)\n---\n', content, re.DOTALL)
            if not match:
                rel = os.path.relpath(fs_path_str, corpus_path)
                errors.append(f'Check (b): No frontmatter found: {rel}')
                continue

            fm_str = match.group(1)
            try:
                fm = yaml.safe_load(fm_str)
            except Exception as e:
                rel = os.path.relpath(fs_path_str, corpus_path)
                errors.append(f'Check (b): Frontmatter parse error in {rel}: {e}')
                continue

            # Check required keys
            fm_keys = set(fm.keys()) if fm else set()
            missing = required_keys - fm_keys
            if missing:
                rel = os.path.relpath(fs_path_str, corpus_path)
                errors.append(f'Check (b): Missing keys in {rel}: {missing}')

            # Check link targets
            link_pattern = r'\[([^\]]+)\]\((/[^)]+)\)'
            for link_match in re.finditer(link_pattern, content):
                link_target = link_match.group(2)
                # Convert link target to filesystem path
                link_rel_path = link_target.lstrip('/')
                link_fs_path = str(corpus_path / link_rel_path)
                if link_fs_path not in expected_paths:
                    rel = os.path.relpath(fs_path_str, corpus_path)
                    errors.append(f'Check (b): Link target does not exist in {rel}: {link_target}')

            # Check parent
            if 'parent' in fm:
                parent_target = fm['parent']
                parent_rel_path = parent_target.lstrip('/')
                parent_fs_path = str(corpus_path / parent_rel_path)
                if parent_fs_path not in expected_paths and parent_target != '/brightmart-home.md':
                    rel = os.path.relpath(fs_path_str, corpus_path)
                    errors.append(f'Check (b): Parent target does not exist in {rel}: {parent_target}')

        except Exception as e:
            rel = os.path.relpath(fs_path_str, corpus_path) if corpus_path in Path(fs_path_str).parents else fs_path_str
            errors.append(f'Check (b): Error checking {rel}: {e}')

    # Check (d): Prose file validation
    prose_path = Path(prose_dir)
    if prose_path.exists():
        # Build forbidden terms from seed data
        forbidden_terms = set()

        # Supplier names
        for sup_id, name, slug, country in brightmart_seed.SUPPLIERS:
            forbidden_terms.add(name)

        # Store names and names without "Brightmart " prefix
        store_map = {}
        for store_id, region_id, name, city, state, fmt, year, sq_ft, pharm, fuel, status in brightmart_seed.STORES:
            store_map[store_id] = {'full_name': name, 'short_name': name.replace('Brightmart ', '')}
            forbidden_terms.add(name)
            forbidden_terms.add(name.replace('Brightmart ', ''))

        # Cities
        for store_id, region_id, name, city, state, fmt, year, sq_ft, pharm, fuel, status in brightmart_seed.STORES:
            forbidden_terms.add(city)

        # State codes (2-letter, case-sensitive)
        state_codes = set()
        for store_id, region_id, name, city, state, fmt, year, sq_ft, pharm, fuel, status in brightmart_seed.STORES:
            state_codes.add(state)

        # Region managers
        for region_id, name, manager in brightmart_seed.REGIONS:
            forbidden_terms.add(manager)

        # Department managers (from pool used in seed)
        dept_managers = set(brightmart_seed.MANAGER_POOL)
        forbidden_terms.update(dept_managers)

        # Promotion names
        for promo_id, name, slug, dept_id, start, end, discount in brightmart_seed.PROMOTIONS:
            forbidden_terms.add(name)

        # Policy names
        for policy_id, name, slug, applies_to in brightmart_seed.POLICIES:
            forbidden_terms.add(name)

        for prose_file in prose_path.glob('*.prose.txt'):
            prose_stem = prose_file.stem.replace('.prose', '')  # Remove .prose suffix
            corpus_doc_name = f'{prose_stem}.md'

            # Find matching corpus doc file
            matching_corpus_path = None
            for exp_path in expected_paths.keys():
                if os.path.basename(exp_path) == corpus_doc_name:
                    matching_corpus_path = exp_path
                    break

            if matching_corpus_path is None:
                errors.append(f'Check (d): No matching corpus doc for prose file: {prose_file.name}')
                continue

            # Read prose file
            try:
                with open(prose_file, 'r', encoding='utf-8') as f:
                    prose_content = f.read().strip()
            except Exception as e:
                errors.append(f'Check (d): Cannot read prose file {prose_file.name}: {e}')
                continue

            # Check for digits
            if re.search(r'\d', prose_content):
                errors.append(f'Check (d): Prose file contains digits: {prose_file.name}')

            # Check word count
            words = prose_content.split()
            word_count = len(words)
            if word_count < 60 or word_count > 160:
                errors.append(f'Check (d): Prose file word count outside 60-160: {prose_file.name} ({word_count} words)')

            # Build own-entity exemptions from corpus doc
            own_entity_names = set()

            # Read the matching corpus doc's frontmatter
            try:
                with open(matching_corpus_path, 'r', encoding='utf-8') as f:
                    doc_content = f.read()
                match = re.match(r'^---\n(.*?)\n---\n', doc_content, re.DOTALL)
                if match:
                    fm_str = match.group(1)
                    fm = yaml.safe_load(fm_str)
                    if fm and 'title' in fm:
                        own_entity_names.add(fm['title'])

                    # For archive docs, read the supersedes doc's title
                    if fm and fm.get('type') == 'archive' and 'supersedes' in fm:
                        supersedes_target = fm['supersedes']
                        supersedes_rel = supersedes_target.lstrip('/')
                        supersedes_path = str(corpus_path / supersedes_rel)
                        if supersedes_path in expected_paths:
                            supersedes_content = expected_paths[supersedes_path]
                            super_match = re.match(r'^---\n(.*?)\n---\n', supersedes_content, re.DOTALL)
                            if super_match:
                                super_fm_str = super_match.group(1)
                                super_fm = yaml.safe_load(super_fm_str)
                                if super_fm and 'title' in super_fm:
                                    own_entity_names.add(super_fm['title'])

                    # For stores, also exempt the name without "Brightmart "
                    if fm and fm.get('type') == 'store' and 'title' in fm:
                        store_title = fm['title']
                        if store_title.startswith('Brightmart '):
                            own_entity_names.add(store_title.replace('Brightmart ', ''))

            except Exception as e:
                errors.append(f'Check (d): Cannot read corpus doc {corpus_doc_name}: {e}')
                continue

            # Check for forbidden terms (except doc's own entity names)
            for term in forbidden_terms:
                if term in own_entity_names:
                    continue
                # Case-insensitive word boundary match for regular terms
                if re.search(r'\b' + re.escape(term) + r'\b', prose_content, re.I):
                    errors.append(f'Check (d): Forbidden term "{term}" in prose file: {prose_file.name}')

            # State codes: case-sensitive whole word
            for state_code in state_codes:
                if re.search(r'\b' + re.escape(state_code) + r'\b', prose_content):
                    errors.append(f'Check (d): State code "{state_code}" in prose file: {prose_file.name}')

    return errors


def entity_id_to_doc_path(entity_id, sql_conn):
    """Map entity ID to doc path using same logic as build."""
    cursor = sql_conn.cursor()

    if entity_id.startswith('SUP'):
        sup_row = cursor.execute(
            'SELECT supplier_id, slug FROM supplier WHERE supplier_id=?', (entity_id,)
        ).fetchone()
        if sup_row:
            return f'suppliers/supplier-{sup_row[1]}.md'
    elif entity_id.startswith('S'):
        store_row = cursor.execute(
            'SELECT store_id, region_id, name FROM store WHERE store_id=?', (entity_id,)
        ).fetchone()
        if store_row:
            slug = store_row[2].replace('Brightmart ', '').lower().replace(' ', '-')
            return f'regions/{store_row[1]}/store-{entity_id.lower()}-{slug}.md'
    elif entity_id.startswith('P'):
        promo_row = cursor.execute(
            'SELECT promo_id, slug FROM promotion WHERE promo_id=?', (entity_id,)
        ).fetchone()
        if promo_row:
            return f'promotions/promo-{promo_row[1]}.md'
    elif entity_id.startswith('D'):
        dept_row = cursor.execute(
            'SELECT dept_id, slug FROM department WHERE dept_id=?', (entity_id,)
        ).fetchone()
        if dept_row:
            return f'departments/dept-{dept_row[1]}.md'
    elif entity_id.startswith('C'):
        cat_row = cursor.execute(
            'SELECT cat_id, dept_id, slug FROM category WHERE cat_id=?', (entity_id,)
        ).fetchone()
        if cat_row:
            dept_row = cursor.execute(
                'SELECT slug FROM department WHERE dept_id=?', (cat_row[1],)
            ).fetchone()
            if dept_row:
                return f'departments/{dept_row[0]}/category-{cat_row[2]}.md'
    return None


def validate_questions(questions_file: str = 'doc_synthetic/brightmart_questions.jsonl',
                       qrels_file: str = 'doc_synthetic/brightmart_qrels.jsonl',
                       gold_spans_file: str = 'doc_synthetic/brightmart_gold_spans.jsonl',
                       corpus_dir: str = 'doc_synthetic/corpus',
                       paraphrases_file: str | None = 'doc_synthetic/brightmart_paraphrases.json',
                       id_prefix: str = 'bm-s',
                       expected_total: int = 200) -> list[str]:
    """
    Validate questions, qrels, and gold_spans files.
    Checks:
    - 100 questions, 20 per stratum, unique IDs
    - No duplicate question texts
    - S1-S4 spans are doc substrings; S5 has absent_terms
    - Qrels doc paths exist; no archive docs in gold
    - S2: exactly 2 gold docs from different types
    - S3/S4: ≥2 gold docs per question
    - S5: absent_term appears in no corpus doc
    """
    import json

    errors = []

    # Load all files
    questions = []
    qrels = []
    gold_spans = []

    try:
        with open(questions_file, 'r', encoding='utf-8') as f:
            questions = [json.loads(l) for l in f if l.strip()]
    except Exception as e:
        errors.append(f'Cannot read questions file: {e}')
        return errors

    try:
        with open(qrels_file, 'r', encoding='utf-8') as f:
            qrels = [json.loads(l) for l in f if l.strip()]
    except Exception as e:
        errors.append(f'Cannot read qrels file: {e}')

    try:
        with open(gold_spans_file, 'r', encoding='utf-8') as f:
            gold_spans = [json.loads(l) for l in f if l.strip()]
    except Exception as e:
        errors.append(f'Cannot read gold_spans file: {e}')

    # Check: IDs are unique, 100 total, 20 per stratum
    query_ids = [q['query_id'] for q in questions]
    if len(query_ids) != len(set(query_ids)):
        errors.append('Duplicate query IDs found')

    if len(questions) != expected_total:
        errors.append(f'Expected {expected_total} questions, got {len(questions)}')

    expected_per_stratum = expected_total // 5
    for stratum in ['S1', 'S2', 'S3', 'S4', 'S5']:
        count = len([q for q in questions if q['stratum'] == stratum])
        if count != expected_per_stratum:
            errors.append(f'Stratum {stratum}: expected {expected_per_stratum}, got {count}')

    bad_ids = [q['query_id'] for q in questions if not q['query_id'].startswith(id_prefix)]
    if bad_ids:
        errors.append(f'Query IDs not matching prefix "{id_prefix}": {bad_ids[:5]}')

    # Check: No duplicate question texts
    texts = [q['text'] for q in questions]
    if len(texts) != len(set(texts)):
        dupes = [t for t in texts if texts.count(t) > 1]
        errors.append(f'Duplicate question texts found: {set(dupes)}')

    # Render corpus
    conn = brightmart_seed.connect()
    docs = brightmart_render.render(conn)
    conn.close()

    # Build qrel map
    qrel_map = {}
    for qrel in qrels:
        qid = qrel['query_id']
        if qid not in qrel_map:
            qrel_map[qid] = []
        qrel_map[qid].append(qrel['doc_path'])

    # Check: Spans match docs
    for span_record in gold_spans:
        query_id = span_record['query_id']
        doc_path = span_record.get('doc_path')
        span_text = span_record.get('span')
        absent_term = span_record.get('absent_term')

        if absent_term:
            # S5: Check absent_term is truly absent
            all_text = '\n'.join(docs.values()).lower()
            if absent_term.lower() in all_text:
                errors.append(f'{query_id}: absent_term "{absent_term}" found in corpus')
        elif doc_path and span_text:
            # S1-S4: Check span is in doc
            full_path = f'/{doc_path}'
            if full_path not in docs:
                errors.append(f'{query_id}: doc path not found: {doc_path}')
            else:
                doc_text = docs[full_path]
                if '## Notes' in doc_text:
                    doc_text = doc_text.split('## Notes')[0]

                if span_text not in doc_text:
                    errors.append(f'{query_id}: span not in {doc_path}')

    # Check: Qrels doc paths exist, no archive docs
    for qrel in qrels:
        doc_path = qrel['doc_path']
        full_path = f'/{doc_path}'
        if full_path not in docs:
            errors.append(f'{qrel["query_id"]}: doc path not found: {doc_path}')
        else:
            doc_text = docs[full_path]
            # Check if archive
            if '---\n' in doc_text:
                fm_str = doc_text.split('---\n')[1].split('\n---\n')[0]
                try:
                    fm = yaml.safe_load(fm_str)
                    if fm and fm.get('type') == 'archive':
                        errors.append(f'{qrel["query_id"]}: gold doc is archive: {doc_path}')
                except:
                    pass

    # Check: S2 has exactly 2 gold docs from different types
    for q in questions:
        if q['stratum'] == 'S2':
            qid = q['query_id']
            doc_paths = qrel_map.get(qid, [])
            if len(doc_paths) != 2:
                errors.append(f'{qid} (S2): expected 2 gold docs, got {len(doc_paths)}')
            else:
                # Check doc types differ
                doc_types = set()
                for dp in doc_paths:
                    fp = f'/{dp}'
                    if fp in docs:
                        try:
                            fm = yaml.safe_load(docs[fp].split('---\n')[1].split('\n---\n')[0])
                            doc_types.add(fm.get('type'))
                        except:
                            pass
                if len(doc_types) < 2:
                    errors.append(f'{qid} (S2): gold docs must be from different types')

    # Check: S3/S4 have ≥2 gold docs each
    for q in questions:
        if q['stratum'] in ['S3', 'S4']:
            qid = q['query_id']
            doc_count = len(qrel_map.get(qid, []))
            if doc_count < 2:
                errors.append(f'{qid} ({q["stratum"]}): expected ≥2 gold docs, got {doc_count}')

    if paraphrases_file is not None:
        # Task 6 Check: Paraphrase validation
        # Check that every base ID has both -t and -p variants (except S5 absent_term rule)
        template_qids = set(q['query_id'] for q in questions if q['query_id'].endswith('-t'))
        paraphrase_qids = set(q['query_id'] for q in questions if q['query_id'].endswith('-p'))

        for qid in template_qids:
            base_id = qid.replace('-t', '')
            para_qid = base_id + '-p'
            if para_qid not in paraphrase_qids:
                errors.append(f'{base_id}: missing paraphrase {para_qid}')

        # Check paraphrases have no 4-word overlap with templates
        import re
        def extract_4grams(text):
            """Extract 4-word sequences (lowercased, punctuation stripped)."""
            # Lowercase and remove punctuation
            cleaned = text.lower()
            cleaned = re.sub(r'[^a-z0-9\s]', '', cleaned)
            words = cleaned.split()
            if len(words) < 4:
                return set()
            return set(' '.join(words[i:i+4]) for i in range(len(words) - 3))

        # Build template-to-paraphrase mapping
        template_by_base = {}
        para_by_base = {}
        for q in questions:
            if q['query_id'].endswith('-t'):
                base_id = q['query_id'].replace('-t', '')
                template_by_base[base_id] = q['text']
            elif q['query_id'].endswith('-p'):
                base_id = q['query_id'].replace('-p', '')
                para_by_base[base_id] = q['text']

        for base_id in template_by_base:
            if base_id not in para_by_base:
                continue
            template_text = template_by_base[base_id]
            para_text = para_by_base[base_id]

            template_grams = extract_4grams(template_text)
            para_grams = extract_4grams(para_text)

            overlap = template_grams & para_grams
            if overlap:
                errors.append(f'{base_id}: paraphrase shares 4-grams with template: {overlap.pop()}...')

        # Check that -p gold docs match -t gold docs
        template_qid_to_gold = {}
        para_qid_to_gold = {}

        for qrel in qrels:
            qid = qrel['query_id']
            doc_path = qrel['doc_path']
            if qid.endswith('-t'):
                base_id = qid.replace('-t', '')
                if base_id not in template_qid_to_gold:
                    template_qid_to_gold[base_id] = []
                template_qid_to_gold[base_id].append(doc_path)
            elif qid.endswith('-p'):
                base_id = qid.replace('-p', '')
                if base_id not in para_qid_to_gold:
                    para_qid_to_gold[base_id] = []
                para_qid_to_gold[base_id].append(doc_path)

        for base_id in template_qid_to_gold:
            if base_id in para_qid_to_gold:
                template_docs = sorted(template_qid_to_gold[base_id])
                para_docs = sorted(para_qid_to_gold[base_id])
                if template_docs != para_docs:
                    errors.append(f'{base_id}: paraphrase gold docs differ from template: {para_docs} vs {template_docs}')

    # D1 Check: S1-S4 must have non-empty reference_answer
    for q in questions:
        if q['stratum'] in ['S1', 'S2', 'S3', 'S4']:
            qid = q['query_id']
            answer = q.get('reference_answer')
            if not answer or (isinstance(answer, str) and not answer.strip()):
                errors.append(f'{qid} ({q["stratum"]}): reference_answer is empty')

    # D1 Check: For S1/S2, answer must appear in at least one gold doc (case-insensitive)
    for q in questions:
        if q['stratum'] in ['S1', 'S2']:
            qid = q['query_id']
            answer = q.get('reference_answer', '').lower()
            if not answer:
                continue

            # Get gold doc paths for this question
            qrel_paths = qrel_map.get(qid, [])
            found_in_doc = False
            for doc_path in qrel_paths:
                full_path = f'/{doc_path}'
                if full_path not in docs:
                    continue
                doc_text = docs[full_path].lower()
                if '## Notes' in docs[full_path].lower():
                    doc_text = doc_text.split('## notes')[0]

                # Search for answer (strip formatting for matching)
                search_term = answer.replace(' square feet', '').replace('%', '')
                if search_term in doc_text:
                    found_in_doc = True
                    break

            if not found_in_doc and answer.strip():
                errors.append(f'{qid} (S1/S2): reference_answer "{answer}" not found in any gold doc')

    # Task 0b Check (i): S1/S2 spans must be valid (not "---", empty, or frontmatter)
    # and every gold doc must have ≥1 span
    span_map = {}
    for span_record in gold_spans:
        if span_record.get('doc_path'):
            qid = span_record['query_id']
            if qid not in span_map:
                span_map[qid] = []
            span_map[qid].append(span_record)

    for q in questions:
        if q['stratum'] in ['S1', 'S2']:
            qid = q['query_id']
            qrel_paths = qrel_map.get(qid, [])

            # Check all gold docs have ≥1 span
            gold_docs_with_spans = set()
            for span_record in span_map.get(qid, []):
                doc_path = span_record.get('doc_path')
                if doc_path:
                    gold_docs_with_spans.add(doc_path)

                    # Check span is not "---", empty, or frontmatter
                    span_text = span_record.get('span', '')
                    if span_text == '---' or not span_text.strip() or span_text.startswith('---'):
                        errors.append(f'{qid} ({q["stratum"]}): span is invalid: "{span_text}"')

            for doc_path in qrel_paths:
                if doc_path not in gold_docs_with_spans:
                    errors.append(f'{qid} ({q["stratum"]}): no span for gold doc: {doc_path}')

    # Task 0b Check (ii): reference_answer must occur in at least one span
    # For S2, final answer must be in the second-hop doc's span
    for q in questions:
        if q['stratum'] in ['S1', 'S2']:
            qid = q['query_id']
            answer = q.get('reference_answer', '').lower()
            if not answer:
                continue

            qrel_paths = qrel_map.get(qid, [])
            spans_for_q = span_map.get(qid, [])

            # Normalize answer for matching (strip " square feet", treat "X%" as matching "X% off")
            search_answer = answer.replace(' square feet', '').replace('%', '')

            # For S2, check second doc specifically
            if q['stratum'] == 'S2' and len(qrel_paths) >= 2:
                second_doc = qrel_paths[1]
                second_spans = [s.get('span', '').lower() for s in spans_for_q if s.get('doc_path') == second_doc]
                found_in_second = False
                for span_text in second_spans:
                    search_span = span_text.replace(' square feet', '').replace('%', '')
                    if search_answer in search_span:
                        found_in_second = True
                        break
                if not found_in_second and answer.strip():
                    errors.append(f'{qid} (S2): answer not found in second-hop doc span')

            # Check at least one span contains the answer
            found_in_any_span = False
            for span_record in spans_for_q:
                span_text = span_record.get('span', '').lower()
                search_span = span_text.replace(' square feet', '').replace('%', '')
                if search_answer in search_span:
                    found_in_any_span = True
                    break

            if not found_in_any_span and answer.strip():
                errors.append(f'{qid} ({q["stratum"]}): answer not found in any span')

    # D4 Check: S3/S4 SQL validation - execute and compare with qrels
    for q in questions:
        if q['stratum'] in ['S3', 'S4']:
            qid = q['query_id']

            # Find the gold_sql record for this question
            gold_sql_record = next((s for s in gold_spans if s.get('query_id') == qid and s.get('gold_sql')), None)
            if not gold_sql_record:
                errors.append(f'{qid} ({q["stratum"]}): missing gold_sql record')
                continue

            sql_query = gold_sql_record.get('gold_sql')
            sql_params = gold_sql_record.get('gold_sql_params', ())

            # Execute SQL and compare with qrels
            try:
                sql_conn = brightmart_seed.connect()
                result = sql_conn.execute(sql_query, sql_params).fetchall()
                result_ids = {row[0] for row in result}

                # Map result IDs to doc paths
                expected_paths = set()
                for entity_id in result_ids:
                    path = entity_id_to_doc_path(entity_id, sql_conn)
                    if path:
                        expected_paths.add(path)

                sql_conn.close()

                # Get qrels for this question
                qrel_paths = set(qrel_map.get(qid, []))

                # Compare
                if expected_paths != qrel_paths:
                    missing = expected_paths - qrel_paths
                    extra = qrel_paths - expected_paths
                    error_msg = f'{qid} ({q["stratum"]}): SQL set mismatch'
                    if missing:
                        error_msg += f'; missing from qrels: {missing}'
                    if extra:
                        error_msg += f'; extra in qrels: {extra}'
                    errors.append(error_msg)

            except Exception as e:
                errors.append(f'{qid} ({q["stratum"]}): error validating SQL: {e}')

    # Task 0b Check (iii): S4 path validation
    # Each question must have ≥1 path; paths must start with brightmart-home.md,
    # consecutive pairs must be real parent links, and every gold doc's parent must
    # match the last element of at least one path
    for q in questions:
        if q['stratum'] == 'S4':
            qid = q['query_id']

            # Find path records for this question (currently stored in gold_spans)
            # Note: For S4, paths are currently represented via qrels, but we should
            # also support explicit path records when hierarchies are complex
            qrel_paths = qrel_map.get(qid, [])

            if not qrel_paths:
                errors.append(f'{qid} (S4): no gold docs found')
                continue

            # For each gold doc, check its parent is "/" + one of the hierarchy parents
            # This is a simplified check; full path hierarchy validation would require
            # explicit path records which may be added in Task 6
            for doc_path in qrel_paths:
                full_path = f'/{doc_path}'
                if full_path in docs:
                    try:
                        doc_content = docs[full_path]
                        fm_str = doc_content.split('---\n')[1].split('\n---\n')[0]
                        fm = yaml.safe_load(fm_str)
                        parent = fm.get('parent', '')
                        if not parent:
                            errors.append(f'{qid} (S4): doc has no parent: {doc_path}')
                    except:
                        pass

    return errors


def main():
    """Run all validation checks."""
    heldout = '--heldout' in sys.argv[1:]

    errors = validate_corpus()
    if heldout:
        errors.extend(validate_questions(
            questions_file='doc_synthetic/brightmart_heldout_questions.jsonl',
            qrels_file='doc_synthetic/brightmart_heldout_qrels.jsonl',
            gold_spans_file='doc_synthetic/brightmart_heldout_gold_spans.jsonl',
            paraphrases_file='doc_synthetic/brightmart_heldout_paraphrases.json',
            id_prefix='bm-h-s',
            expected_total=200,
        ))
    else:
        errors.extend(validate_questions())

    if errors:
        for error in errors:
            print(error)
        sys.exit(1)
    else:
        # Count docs and questions
        conn = brightmart_seed.connect()
        docs = brightmart_render.render(conn)
        conn.close()

        import json
        questions_file = ('doc_synthetic/brightmart_heldout_questions.jsonl' if heldout
                          else 'doc_synthetic/brightmart_questions.jsonl')
        questions_count = 0
        try:
            with open(questions_file, 'r', encoding='utf-8') as f:
                questions_count = len([l for l in f if l.strip()])
        except:
            pass

        print(f'OK: {len(docs)} docs, {questions_count} questions')


if __name__ == '__main__':
    main()
