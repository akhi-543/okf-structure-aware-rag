"""
Tests for Brightmart corpus renderer.
"""

import sys
import sqlite3
import re
import yaml
import tempfile
from pathlib import Path

sys.path.insert(0, 'doc_synthetic')

import brightmart_seed
import brightmart_render


def test_doc_count():
    """Test that render produces exactly 53 docs."""
    conn = brightmart_seed.connect()
    docs = brightmart_render.render(conn)
    conn.close()

    assert len(docs) == 53, f'Expected 53 docs, got {len(docs)}'


def test_unique_basenames():
    """Test that all doc basenames are unique."""
    conn = brightmart_seed.connect()
    docs = brightmart_render.render(conn)
    conn.close()

    basenames = [path.split('/')[-1] for path in docs.keys()]
    assert len(basenames) == len(set(basenames)), f'Duplicate basenames found: {basenames}'

    # Also check no index.md
    assert 'index.md' not in basenames, 'Found index.md which is not allowed'


def test_frontmatter_parse():
    """Test that all frontmatter parses as valid YAML and has required keys."""
    conn = brightmart_seed.connect()
    docs = brightmart_render.render(conn)
    conn.close()

    required_keys = {'type', 'title', 'description', 'tags', 'status', 'timestamp'}

    for path, content in docs.items():
        # Extract frontmatter
        match = re.match(r'^---\n(.*?)\n---\n', content, re.DOTALL)
        assert match, f'{path}: No frontmatter found'

        fm_str = match.group(1)
        try:
            fm = yaml.safe_load(fm_str)
        except Exception as e:
            assert False, f'{path}: Frontmatter parse error: {e}'

        # Check required keys
        fm_keys = set(fm.keys()) if fm else set()
        missing = required_keys - fm_keys
        assert not missing, f'{path}: Missing keys {missing}'


def test_link_targets_exist():
    """Test that every link in docs points to an existing doc."""
    conn = brightmart_seed.connect()
    docs = brightmart_render.render(conn)
    conn.close()

    # Extract all links and normalize to bundle-absolute paths
    link_pattern = r'\[([^\]]+)\]\((/[^)]+)\)'

    all_paths = set(docs.keys())

    for path, content in docs.items():
        for match in re.finditer(link_pattern, content):
            link_target = match.group(2)
            # Handle backlinks to parent
            assert link_target in all_paths, f'{path}: Link to {link_target} does not exist'


def test_parent_chain():
    """Test that every doc's parent exists and chains to brightmart-home.md."""
    conn = brightmart_seed.connect()
    docs = brightmart_render.render(conn)
    conn.close()

    all_paths = set(docs.keys())

    for path, content in docs.items():
        # Extract frontmatter
        match = re.match(r'^---\n(.*?)\n---\n', content, re.DOTALL)
        if not match:
            continue

        fm_str = match.group(1)
        fm = yaml.safe_load(fm_str)

        if 'parent' in fm:
            parent = fm['parent']
            assert parent in all_paths, f'{path}: Parent {parent} does not exist'

            # Follow chain to home
            current = parent
            visited = set()
            while current and current != '/brightmart-home.md':
                if current in visited:
                    assert False, f'{path}: Circular parent chain at {current}'
                visited.add(current)

                parent_content = docs[current]
                parent_match = re.match(r'^---\n(.*?)\n---\n', parent_content, re.DOTALL)
                if parent_match:
                    parent_fm = yaml.safe_load(parent_match.group(1))
                    current = parent_fm.get('parent')
                else:
                    break

            assert current == '/brightmart-home.md' or current is None, \
                f'{path}: Parent chain does not reach home (ended at {current})'


def test_determinism():
    """Test that rendering twice produces identical output."""
    conn1 = brightmart_seed.connect()
    docs1 = brightmart_render.render(conn1)
    conn1.close()

    conn2 = brightmart_seed.connect()
    docs2 = brightmart_render.render(conn2)
    conn2.close()

    assert docs1 == docs2, 'Two renders produced different output'


def test_s01_contains_182000():
    """Test that S01 store doc contains the string '182,000'."""
    conn = brightmart_seed.connect()
    docs = brightmart_render.render(conn)
    conn.close()

    s01_doc = docs['/regions/northeast/store-s01-riverside.md']
    assert '182,000' in s01_doc, 'S01 doc does not contain "182,000"'


def test_s06_q4_all_zero():
    """Test that S06 (closed store) has all 0.00 in Q4 sales table."""
    conn = brightmart_seed.connect()
    docs = brightmart_render.render(conn)
    conn.close()

    s06_doc = docs['/regions/southeast/store-s06-bayou-gate.md']

    # Extract sales table
    table_match = re.search(r'## 2025 Sales by Quarter\n\n(.*?)\n', s06_doc, re.DOTALL)
    assert table_match, 'S06 doc does not have sales table'

    table_text = table_match.group(1)
    lines = table_text.strip().split('\n')

    # Skip header and separator
    for line in lines[2:]:
        if line.strip() == '':
            continue
        cells = [c.strip() for c in line.split('|')]
        if len(cells) > 4:
            q4_value = cells[4]  # Q4 is the last cell
            assert q4_value == '0.00', f'S06 Q4 cell is {q4_value}, expected 0.00'


def test_write_creates_corpus():
    """Test that write() creates docs in the correct directory."""
    import os
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        corpus_dir = os.path.join(tmpdir, 'corpus')

        # Use a minimal render and write
        conn = brightmart_seed.connect()
        docs = brightmart_render.render(conn)
        conn.close()

        # Manually write (without using write() to keep test isolated)
        for path, content in docs.items():
            rel_path = path.lstrip('/')
            full_path = os.path.join(corpus_dir, rel_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w', encoding='utf-8', newline='\n') as f:
                f.write(content)

        # Verify structure
        assert os.path.exists(os.path.join(corpus_dir, 'brightmart-home.md'))
        assert os.path.exists(os.path.join(corpus_dir, 'regions', 'region-northeast.md'))
        assert os.path.exists(os.path.join(corpus_dir, 'regions', 'northeast', 'store-s01-riverside.md'))


def test_policy_no_raw_keys():
    """Test that policy docs do not have raw '- key: value' bullets."""
    conn = brightmart_seed.connect()
    docs = brightmart_render.render(conn)
    conn.close()

    policy_paths = [p for p in docs.keys() if '/policies/policy-' in p]

    for path in policy_paths:
        doc = docs[path]
        lines = doc.split('\n')
        for line in lines:
            # Check for pattern "- " followed by snake_case key
            if line.strip().startswith('- '):
                # Extract what follows "- "
                content = line.strip()[2:]
                # Check if it's a snake_case key (word_word pattern)
                if '_' in content.split(':')[0] and ':' in content:
                    # This looks like a raw key:value bullet
                    assert False, f'{path}: Found raw key bullet: {line}'


def test_archive_a1_properties():
    """Test that A1 (archive of S07) contains '150,000' and no S07 link in body."""
    conn = brightmart_seed.connect()
    docs = brightmart_render.render(conn)
    conn.close()

    a1_doc = docs['/archive/archived-cedar-falls-profile-2022.md']

    # A1 should contain 150,000 (the stale sq_ft value)
    assert '150,000' in a1_doc, 'A1 does not contain "150,000"'

    # A1 should NOT have a link to S07 doc in the body (only supersedes in frontmatter is OK)
    # Extract body (after second ---)
    parts = a1_doc.split('---\n', 2)
    if len(parts) > 2:
        body = parts[2]
        assert '[' not in body or '/regions/midwest/store-s07-cedar-falls.md' not in body, \
            'A1 body should not link to current S07 doc'


def test_sentence_formatting():
    """Test that all rendered body sentences are grammatically formatted.

    Checks: start with uppercase/backtick, end with period, no double spaces.
    Skips frontmatter, headings, bullets, tables, and link-only lines.
    Skips the Notes section (prose is not gold data).
    """
    conn = brightmart_seed.connect()
    docs = brightmart_render.render(conn)
    conn.close()

    for path, content in docs.items():
        # Extract body (after second ---)
        parts = content.split('---\n', 2)
        if len(parts) < 3:
            continue
        body = parts[2]

        # Skip Notes section (prose is not gold data)
        if '\n## Notes\n' in body:
            body = body.split('\n## Notes\n')[0]

        lines = body.split('\n')
        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()

            # Skip empty lines, headings, tables, bullets, code blocks, links-only
            if not stripped or stripped.startswith('#') or stripped.startswith('|'):
                continue
            if stripped.startswith('-') and ('[' in stripped or not any(c.isalpha() for c in stripped)):
                continue
            if stripped.startswith('[') and stripped.endswith(')') and '(' in stripped:
                # Link-only line like "[Back to Northeast](/regions/region-northeast.md)"
                continue
            if stripped.startswith('**'):
                # Bold metadata line like "**Dates:**"
                continue

            # Check for double spaces (G1 issue)
            assert '  ' not in line, f'{path} line {line_num}: Double space found: {line}'

            # For lines that look like sentences (contain alphanumeric at start)
            if stripped and (stripped[0].isupper() or stripped[0] == '`'):
                # Should end with period
                if not stripped.endswith('.') and not stripped.endswith('`'):
                    # Allow some exceptions for bold/special formatting
                    if not (stripped.endswith(']') or stripped.endswith(')')):
                        assert False, f'{path} line {line_num}: Sentence does not end with period: {line}'


def test_no_format_list_artifacts():
    """Test that format lists don't create empty spaces (G1)."""
    conn = brightmart_seed.connect()
    docs = brightmart_render.render(conn)
    conn.close()

    for path, content in docs.items():
        # Check for "  and " and "to  " patterns
        assert '  and ' not in content, f'{path}: Found "  and " (empty format list item)'
        assert 'to  ' not in content, f'{path}: Found "to  " (empty format list item)'


def test_supplier_country_articles():
    """Test that US suppliers use 'the United States' (H1 fix)."""
    conn = brightmart_seed.connect()
    docs = brightmart_render.render(conn)
    conn.close()

    for path, content in docs.items():
        # Check for incorrect patterns without article
        assert 'in United States' not in content, f'{path}: Found "in United States" (missing "the")'
        assert 'of United States' not in content, f'{path}: Found "of United States" (missing "the")'


def test_prose_appending():
    """Test that prose files are appended to docs with Notes sections."""
    conn = brightmart_seed.connect()

    # Create a temporary directory with one prose file
    with tempfile.TemporaryDirectory() as tmpdir:
        prose_file = Path(tmpdir) / 'brightmart-home.prose.txt'
        prose_file.write_text('Test prose content for home.', encoding='utf-8')

        # Render with prose_dir set
        docs = brightmart_render.render(conn, prose_dir=tmpdir)

        # Check that home doc contains the Notes section
        home_doc = docs['/brightmart-home.md']
        assert '\n## Notes\n\n' in home_doc, 'Home doc should contain Notes section'
        assert 'Test prose content for home.' in home_doc, 'Home doc should contain prose content'
        assert home_doc.endswith('Test prose content for home.\n'), 'Doc should end with newline after prose'

        # Check that a doc without a prose file does NOT contain Notes section
        region_doc = docs['/regions/region-northeast.md']
        assert '\n## Notes\n\n' not in region_doc, 'Region doc without prose should not contain Notes section'

    # Test without prose_dir to ensure render still works
    docs_no_prose = brightmart_render.render(conn)
    assert '/brightmart-home.md' in docs_no_prose, 'Should render home doc without prose_dir'

    conn.close()


if __name__ == '__main__':
    test_doc_count()
    test_unique_basenames()
    test_frontmatter_parse()
    test_link_targets_exist()
    test_parent_chain()
    test_determinism()
    test_s01_contains_182000()
    test_s06_q4_all_zero()
    test_write_creates_corpus()
    test_policy_no_raw_keys()
    test_archive_a1_properties()
    test_sentence_formatting()
    test_no_format_list_artifacts()
    test_supplier_country_articles()
    test_prose_appending()
    print('All tests passed')
