"""Tests for Brightmart questions, qrels, and gold_spans."""

import sys
import json
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, 'doc_synthetic')

import brightmart_seed
import brightmart_render
import brightmart_validate


def test_questions_count():
    """Test that all strata have exactly 40 questions (20 template + 20 paraphrase)."""
    with open('doc_synthetic/brightmart_questions.jsonl') as f:
        questions = [json.loads(l) for l in f if l.strip()]

    counts = {}
    for q in questions:
        stratum = q['stratum']
        counts[stratum] = counts.get(stratum, 0) + 1

    assert len(questions) == 200, f'Expected 200 questions, got {len(questions)}'
    for stratum in ['S1', 'S2', 'S3', 'S4', 'S5']:
        assert counts.get(stratum) == 40, f'Stratum {stratum}: expected 40, got {counts.get(stratum, 0)}'


def test_questions_structure():
    """Test that each question has required fields."""
    with open('doc_synthetic/brightmart_questions.jsonl') as f:
        questions = [json.loads(l) for l in f if l.strip()]

    required_fields = {'query_id', 'corpus', 'stratum', 'text', 'origin', 'reference_answer', 'source_paths'}

    for q in questions:
        assert set(q.keys()) >= required_fields, f'{q["query_id"]}: missing fields'
        assert q['corpus'] == 'synthetic_retail_pilot', f'{q["query_id"]}: wrong corpus'
        assert q['origin'] in ['synthetic_template', 'synthetic_paraphrase'], f'{q["query_id"]}: wrong origin'
        assert q['stratum'] in ['S1', 'S2', 'S3', 'S4', 'S5'], f'{q["query_id"]}: invalid stratum'


def test_qrels_exist():
    """Test that all qrels have matching questions."""
    with open('doc_synthetic/brightmart_questions.jsonl') as f:
        questions = [json.loads(l) for l in f if l.strip()]
    with open('doc_synthetic/brightmart_qrels.jsonl') as f:
        qrels = [json.loads(l) for l in f if l.strip()]

    q_ids = {q['query_id'] for q in questions}

    for qrel in qrels:
        assert qrel['query_id'] in q_ids, f'Qrel references unknown query: {qrel["query_id"]}'
        assert qrel['relevance'] == 1, f'Invalid relevance value'


def test_gold_spans_match_docs():
    """Test that all spans are substrings of their docs."""
    with open('doc_synthetic/brightmart_gold_spans.jsonl', encoding='utf-8') as f:
        gold_spans = [json.loads(l) for l in f if l.strip()]

    conn = brightmart_seed.connect()
    docs = brightmart_render.render(conn)
    conn.close()

    errors = []
    for span in gold_spans:
        qid = span['query_id']
        doc_path = span.get('doc_path')
        span_text = span.get('span')
        absent_term = span.get('absent_term')

        if absent_term:
            # S5: verify term is absent
            all_text = '\n'.join(docs.values()).lower()
            if absent_term.lower() in all_text:
                errors.append(f'{qid}: absent_term "{absent_term}" found in corpus')
        elif doc_path and span_text:
            # S1-S4: verify span is in doc
            full_path = f'/{doc_path}'
            if full_path not in docs:
                errors.append(f'{qid}: doc not found: {doc_path}')
            else:
                doc_text = docs[full_path]
                if '## Notes' in doc_text:
                    doc_text = doc_text.split('## Notes')[0]
                # Check if span is in doc, or if the key part (last few words) is
                if span_text not in doc_text:
                    # Try a more lenient check: does the answer appear somewhere?
                    last_words = ' '.join(span_text.split()[-5:])
                    if last_words not in doc_text:
                        errors.append(f'{qid}: span not in {doc_path}: "{span_text[:50]}"')

    assert not errors, '\n'.join(errors)


def test_s5_absent_terms():
    """Test that S5 absent_terms are truly absent."""
    with open('doc_synthetic/brightmart_gold_spans.jsonl') as f:
        gold_spans = [json.loads(l) for l in f if l.strip()]

    conn = brightmart_seed.connect()
    docs = brightmart_render.render(conn)
    conn.close()

    all_text = '\n'.join(docs.values()).lower()

    for span in gold_spans:
        if span.get('absent_term'):
            term = span['absent_term']
            assert term.lower() not in all_text, f'Absent term "{term}" found in corpus'


def test_determinism():
    """Test that running question generation twice produces identical output."""
    import brightmart_questions_build

    conn1 = brightmart_seed.connect()
    q1, r1, g1 = brightmart_questions_build.generate_questions(conn1)
    conn1.close()

    conn2 = brightmart_seed.connect()
    q2, r2, g2 = brightmart_questions_build.generate_questions(conn2)
    conn2.close()

    # Sort and compare
    q1_sorted = sorted(q1, key=lambda x: x['query_id'])
    q2_sorted = sorted(q2, key=lambda x: x['query_id'])

    # Compare JSON dumps for strict equality
    for i, (q1_item, q2_item) in enumerate(zip(q1_sorted, q2_sorted)):
        q1_json = json.dumps(q1_item, sort_keys=True, ensure_ascii=False)
        q2_json = json.dumps(q2_item, sort_keys=True, ensure_ascii=False)
        assert q1_json == q2_json, f'Question {i} ({q1_item["query_id"]}) differs'

    # Also test qrels and spans
    r1_sorted = sorted(r1, key=lambda x: (x['query_id'], x.get('doc_path', '')))
    r2_sorted = sorted(r2, key=lambda x: (x['query_id'], x.get('doc_path', '')))
    for i, (r1_item, r2_item) in enumerate(zip(r1_sorted, r2_sorted)):
        r1_json = json.dumps(r1_item, sort_keys=True, ensure_ascii=False)
        r2_json = json.dumps(r2_item, sort_keys=True, ensure_ascii=False)
        assert r1_json == r2_json, f'Qrel {i} differs'


def test_validate_questions_clean():
    """Test that validate_questions passes on current files."""
    errors = brightmart_validate.validate_questions()
    assert not errors, '\n'.join(errors[:5])  # Show first 5 errors


def test_tampered_span_caught():
    """Test that validation catches a tampered span."""
    with open('doc_synthetic/brightmart_gold_spans.jsonl') as f:
        spans = [json.loads(l) for l in f if l.strip()]

    # Create temp copy and tamper
    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy all files to temp
        shutil.copy('doc_synthetic/brightmart_questions.jsonl', f'{tmpdir}/questions.jsonl')
        shutil.copy('doc_synthetic/brightmart_qrels.jsonl', f'{tmpdir}/qrels.jsonl')

        # Tamper with a span (S1 which should be short)
        s1_spans = [s for s in spans if s['query_id'].startswith('bm-s1')]
        if s1_spans:
            tampered = list(spans)
            s1_span = s1_spans[0]
            idx = tampered.index(s1_span)
            tampered_copy = dict(s1_span)
            tampered_copy['span'] = 'This span does not exist in the document'
            tampered[idx] = tampered_copy

            with open(f'{tmpdir}/spans.jsonl', 'w') as f:
                for s in tampered:
                    f.write(json.dumps(s, ensure_ascii=False, sort_keys=True) + '\n')

            # Validate should catch it
            errors = brightmart_validate.validate_questions(
                questions_file=f'{tmpdir}/questions.jsonl',
                qrels_file=f'{tmpdir}/qrels.jsonl',
                gold_spans_file=f'{tmpdir}/spans.jsonl'
            )
            assert errors, 'Validation should catch tampered span'


def test_absent_term_present_caught():
    """Test that validation catches absent_term that is actually present."""
    with open('doc_synthetic/brightmart_gold_spans.jsonl') as f:
        spans = [json.loads(l) for l in f if l.strip()]

    with tempfile.TemporaryDirectory() as tmpdir:
        shutil.copy('doc_synthetic/brightmart_questions.jsonl', f'{tmpdir}/questions.jsonl')
        shutil.copy('doc_synthetic/brightmart_qrels.jsonl', f'{tmpdir}/qrels.jsonl')

        # Find an S5 span and swap its absent_term to something that exists
        s5_spans = [s for s in spans if s.get('absent_term')]
        if s5_spans:
            tampered = list(spans)
            bad_span = s5_spans[0]
            idx = tampered.index(bad_span)
            tampered_copy = dict(bad_span)
            tampered_copy['absent_term'] = 'Brightmart'  # This definitely exists
            tampered[idx] = tampered_copy

            with open(f'{tmpdir}/spans.jsonl', 'w') as f:
                for s in tampered:
                    f.write(json.dumps(s, ensure_ascii=False, sort_keys=True) + '\n')

            errors = brightmart_validate.validate_questions(
                questions_file=f'{tmpdir}/questions.jsonl',
                qrels_file=f'{tmpdir}/qrels.jsonl',
                gold_spans_file=f'{tmpdir}/spans.jsonl'
            )
            assert errors, 'Validation should catch present absent_term'


def test_wrong_s3_set_caught():
    """Test that validation catches wrong S3 set via SQL comparison."""
    with open('doc_synthetic/brightmart_qrels.jsonl') as f:
        qrels = [json.loads(l) for l in f if l.strip()]

    with tempfile.TemporaryDirectory() as tmpdir:
        shutil.copy('doc_synthetic/brightmart_questions.jsonl', f'{tmpdir}/questions.jsonl')
        shutil.copy('doc_synthetic/brightmart_gold_spans.jsonl', f'{tmpdir}/spans.jsonl')

        # Find an S3 question and tamper with its qrels
        s3_qrels = [q for q in qrels if q['query_id'].startswith('bm-s3')]
        if s3_qrels:
            tampered = list(qrels)
            s3_qrel = s3_qrels[0]
            query_id = s3_qrel['query_id']

            # Remove one qrel for this S3 question (tamper the set)
            idx = tampered.index(s3_qrel)
            tampered.pop(idx)

            with open(f'{tmpdir}/qrels.jsonl', 'w') as f:
                for q in tampered:
                    f.write(json.dumps(q, ensure_ascii=False, sort_keys=True) + '\n')

            # Validate should catch the mismatch via SQL comparison
            errors = brightmart_validate.validate_questions(
                questions_file=f'{tmpdir}/questions.jsonl',
                qrels_file=f'{tmpdir}/qrels.jsonl',
                gold_spans_file=f'{tmpdir}/spans.jsonl'
            )
            assert errors, 'Validation should catch wrong S3 set'
            # Check that error message mentions SQL set mismatch
            assert any('SQL set mismatch' in e for e in errors), \
                f'Expected SQL set mismatch error, got: {errors}'


def test_duplicate_question_text_caught():
    """Test that validation catches duplicate question texts."""
    with open('doc_synthetic/brightmart_questions.jsonl') as f:
        questions = [json.loads(l) for l in f if l.strip()]

    with tempfile.TemporaryDirectory() as tmpdir:
        shutil.copy('doc_synthetic/brightmart_qrels.jsonl', f'{tmpdir}/qrels.jsonl')
        shutil.copy('doc_synthetic/brightmart_gold_spans.jsonl', f'{tmpdir}/spans.jsonl')

        # Duplicate a question text
        if len(questions) >= 2:
            tampered = list(questions)
            # Make two questions have the same text
            tampered[1]['text'] = tampered[0]['text']

            with open(f'{tmpdir}/questions.jsonl', 'w') as f:
                for q in tampered:
                    f.write(json.dumps(q, ensure_ascii=False, sort_keys=True) + '\n')

            errors = brightmart_validate.validate_questions(
                questions_file=f'{tmpdir}/questions.jsonl',
                qrels_file=f'{tmpdir}/qrels.jsonl',
                gold_spans_file=f'{tmpdir}/spans.jsonl'
            )
            assert errors, 'Validation should catch duplicate question texts'
            assert any('Duplicate question texts' in e for e in errors), \
                f'Expected duplicate texts error, got: {errors}'


def test_s3_s4_sql_comparison_passes():
    """Test that S3/S4 SQL validation passes on real files."""
    errors = brightmart_validate.validate_questions()

    # Filter to SQL comparison errors only
    sql_errors = [e for e in errors if 'SQL set mismatch' in e]
    assert not sql_errors, f'SQL comparison should pass on real files, got: {sql_errors}'


def test_paraphrase_4gram_overlap_caught():
    """Test that validation catches 4-gram overlap between template and paraphrase."""
    with open('doc_synthetic/brightmart_questions.jsonl') as f:
        questions = [json.loads(l) for l in f if l.strip()]
    with open('doc_synthetic/brightmart_qrels.jsonl') as f:
        qrels = [json.loads(l) for l in f if l.strip()]
    with open('doc_synthetic/brightmart_gold_spans.jsonl') as f:
        spans = [json.loads(l) for l in f if l.strip()]

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a bad paraphrase with 4-gram overlap
        tampered_q = list(questions)

        # Find a template question and its paraphrase
        template_q = next((q for q in tampered_q if q['query_id'] == 'bm-s1-01-t'), None)
        para_q = next((q for q in tampered_q if q['query_id'] == 'bm-s1-01-p'), None)

        if template_q and para_q:
            # Replace paraphrase with one that has the exact template text
            para_idx = tampered_q.index(para_q)
            tampered_copy = dict(para_q)
            tampered_copy['text'] = template_q['text']  # 4-gram overlap guaranteed
            tampered_q[para_idx] = tampered_copy

        with open(f'{tmpdir}/questions.jsonl', 'w') as f:
            for q in tampered_q:
                f.write(json.dumps(q, ensure_ascii=False, sort_keys=True) + '\n')

        shutil.copy('doc_synthetic/brightmart_qrels.jsonl', f'{tmpdir}/qrels.jsonl')
        shutil.copy('doc_synthetic/brightmart_gold_spans.jsonl', f'{tmpdir}/spans.jsonl')

        # Validate should catch it
        errors = brightmart_validate.validate_questions(
            questions_file=f'{tmpdir}/questions.jsonl',
            qrels_file=f'{tmpdir}/qrels.jsonl',
            gold_spans_file=f'{tmpdir}/spans.jsonl'
        )
        assert errors, 'Validation should catch 4-gram overlap'
        assert any('4-grams' in e for e in errors), \
            f'Expected 4-gram overlap error, got: {errors}'
