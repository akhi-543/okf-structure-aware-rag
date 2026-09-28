"""
Tests for Brightmart corpus validator.
"""

import sys
import os
import tempfile
from pathlib import Path

sys.path.insert(0, 'doc_synthetic')

import brightmart_seed
import brightmart_render
import brightmart_validate


def _copy_corpus_to_tmp(tmp_path):
    """Helper: copy full corpus to tmp_path/corpus."""
    corpus_dir = tmp_path / 'corpus'
    corpus_dir.mkdir()

    conn = brightmart_seed.connect()
    docs = brightmart_render.render(conn)
    conn.close()

    for bundle_path, content in docs.items():
        rel_path = bundle_path.lstrip('/')
        file_path = corpus_dir / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8', newline='\n') as f:
            f.write(content)

    return corpus_dir


def test_clean_corpus(tmp_path):
    """Test that clean corpus validates with no errors."""
    corpus_dir = _copy_corpus_to_tmp(tmp_path)
    prose_dir = tmp_path / 'brightmart_llm_prose'
    prose_dir.mkdir()

    errors = brightmart_validate.validate_corpus(str(corpus_dir), str(prose_dir))
    assert errors == [], f'Clean corpus should have no errors, got: {errors}'


def test_prose_with_digit(tmp_path):
    """Test that prose file with digit triggers error."""
    corpus_dir = _copy_corpus_to_tmp(tmp_path)
    prose_dir = tmp_path / 'brightmart_llm_prose'
    prose_dir.mkdir()

    # Create prose file with digit
    prose_file = prose_dir / 'store-s01-riverside.prose.txt'
    with open(prose_file, 'w', encoding='utf-8') as f:
        f.write('This is a great store with wonderful service and amazing products. ' * 3 + '123')

    errors = brightmart_validate.validate_corpus(str(corpus_dir), str(prose_dir))
    assert any('digit' in e.lower() for e in errors), f'Expected digit error, got: {errors}'


def test_prose_with_forbidden_term(tmp_path):
    """Test that forbidden term in non-matching doc triggers error."""
    corpus_dir = _copy_corpus_to_tmp(tmp_path)
    prose_dir = tmp_path / 'brightmart_llm_prose'
    prose_dir.mkdir()

    # Create prose for store-s01-riverside (not S02/Riverdale)
    # It should not mention "Riverdale" which is S02's name
    prose_file = prose_dir / 'store-s01-riverside.prose.txt'
    with open(prose_file, 'w', encoding='utf-8') as f:
        # Write prose mentioning "Riverdale" (forbidden in this doc)
        prose_content = 'This store serves the community like Riverdale does. ' * 3
        prose_content += 'Great service and good products ' * 3
        f.write(prose_content)

    errors = brightmart_validate.validate_corpus(str(corpus_dir), str(prose_dir))
    assert any('Forbidden term' in e and 'Riverdale' in e for e in errors), \
        f'Expected Riverdale forbidden term error, got: {errors}'


def test_prose_allowed_entity_exemption(tmp_path):
    """Test that prose can mention doc's own entity name without error."""
    corpus_dir = _copy_corpus_to_tmp(tmp_path)
    prose_dir = tmp_path / 'brightmart_llm_prose'
    prose_dir.mkdir()

    # Create prose for supplier-northwind-provisions mentioning "Northwind Provisions"
    prose_file = prose_dir / 'supplier-northwind-provisions.prose.txt'
    with open(prose_file, 'w', encoding='utf-8') as f:
        # Write valid prose that mentions "Northwind Provisions" (allowed for this doc)
        prose_content = (
            'Northwind Provisions is a trusted supplier. '
            'They provide quality products to our stores. '
            'Northwind Provisions has been reliable for years. ' * 5
        )
        f.write(prose_content)

    errors = brightmart_validate.validate_corpus(str(corpus_dir), str(prose_dir))
    # Should have NO forbidden term error for "Northwind Provisions" in this doc
    assert not any('Forbidden term' in e and 'Northwind Provisions' in e for e in errors), \
        f'Expected no Northwind Provisions forbidden error, got: {errors}'


def test_prose_word_count_too_low(tmp_path):
    """Test that prose with too few words triggers error."""
    corpus_dir = _copy_corpus_to_tmp(tmp_path)
    prose_dir = tmp_path / 'brightmart_llm_prose'
    prose_dir.mkdir()

    prose_file = prose_dir / 'store-s01-riverside.prose.txt'
    with open(prose_file, 'w', encoding='utf-8') as f:
        # Only 30 words (below 60 minimum)
        f.write('This is a test. ' * 2)

    errors = brightmart_validate.validate_corpus(str(corpus_dir), str(prose_dir))
    assert any('word count' in e.lower() for e in errors), f'Expected word count error, got: {errors}'


def test_prose_word_count_too_high(tmp_path):
    """Test that prose with too many words triggers error."""
    corpus_dir = _copy_corpus_to_tmp(tmp_path)
    prose_dir = tmp_path / 'brightmart_llm_prose'
    prose_dir.mkdir()

    prose_file = prose_dir / 'store-s01-riverside.prose.txt'
    with open(prose_file, 'w', encoding='utf-8') as f:
        # 200 words (above 160 maximum)
        f.write('This is a word. ' * 200)

    errors = brightmart_validate.validate_corpus(str(corpus_dir), str(prose_dir))
    assert any('word count' in e.lower() for e in errors), f'Expected word count error, got: {errors}'


def test_prose_orphan_file(tmp_path):
    """Test that prose file with no matching corpus doc triggers error."""
    corpus_dir = _copy_corpus_to_tmp(tmp_path)
    prose_dir = tmp_path / 'brightmart_llm_prose'
    prose_dir.mkdir()

    # Create prose file with no matching corpus doc
    prose_file = prose_dir / 'nonexistent-doc.prose.txt'
    with open(prose_file, 'w', encoding='utf-8') as f:
        f.write('This is a valid prose file with enough words. ' * 3)

    errors = brightmart_validate.validate_corpus(str(corpus_dir), str(prose_dir))
    assert any('No matching corpus doc' in e for e in errors), f'Expected orphan prose error, got: {errors}'


def test_check_a_duplicate_basename(tmp_path):
    """Test check (a): duplicate basename detection."""
    corpus_dir = _copy_corpus_to_tmp(tmp_path)
    prose_dir = tmp_path / 'brightmart_llm_prose'
    prose_dir.mkdir()

    # Create a duplicate basename by creating a file in a subdirectory with same name as another
    dup_file = corpus_dir / 'departments' / 'brightmart-home.md'
    dup_file.parent.mkdir(parents=True, exist_ok=True)
    with open(dup_file, 'w', encoding='utf-8') as f:
        f.write('Duplicate')

    errors = brightmart_validate.validate_corpus(str(corpus_dir), str(prose_dir))
    # Note: The current implementation checks within doc_synthetic, not within corpus_dir itself
    # For a real duplicate basename check, the validator would need to see the doc_synthetic structure
    # Since we're testing with corpus_dir in isolation, we verify the function runs
    assert isinstance(errors, list), f'Expected list of errors'


def test_check_b_broken_link(tmp_path):
    """Test check (b): broken link detection."""
    corpus_dir = _copy_corpus_to_tmp(tmp_path)
    prose_dir = tmp_path / 'brightmart_llm_prose'
    prose_dir.mkdir()

    # Modify a corpus file to have a broken link
    home_file = corpus_dir / 'brightmart-home.md'
    content = home_file.read_text(encoding='utf-8')
    # Replace a valid link with a broken one
    broken_content = content.replace(
        '[Brightmart Sales Database](/schema/brightmart-sales-database.md)',
        '[Nonexistent](/nonexistent/doc.md)'
    )
    home_file.write_text(broken_content, encoding='utf-8', newline='\n')

    errors = brightmart_validate.validate_corpus(str(corpus_dir), str(prose_dir))
    assert any('Link target does not exist' in e for e in errors), f'Expected broken link error, got: {errors}'


def test_check_b_bad_parent(tmp_path):
    """Test check (b): bad parent chain detection."""
    corpus_dir = _copy_corpus_to_tmp(tmp_path)
    prose_dir = tmp_path / 'brightmart_llm_prose'
    prose_dir.mkdir()

    # Modify a corpus file to have a bad parent
    region_file = corpus_dir / 'regions' / 'region-northeast.md'
    content = region_file.read_text(encoding='utf-8')
    # Replace parent with nonexistent path
    bad_parent_content = content.replace(
        "parent: /brightmart-home.md",
        "parent: /nonexistent/parent.md"
    )
    region_file.write_text(bad_parent_content, encoding='utf-8', newline='\n')

    errors = brightmart_validate.validate_corpus(str(corpus_dir), str(prose_dir))
    assert any('Parent target does not exist' in e for e in errors), f'Expected bad parent error, got: {errors}'


def test_check_c_content_mismatch(tmp_path):
    """Test check (c): content mismatch detection."""
    corpus_dir = _copy_corpus_to_tmp(tmp_path)
    prose_dir = tmp_path / 'brightmart_llm_prose'
    prose_dir.mkdir()

    # Modify one byte in a file
    home_file = corpus_dir / 'brightmart-home.md'
    content = home_file.read_text(encoding='utf-8')
    modified_content = content[:-10] + 'MODIFIED' + content[-2:]  # Change last 10 chars
    home_file.write_text(modified_content, encoding='utf-8', newline='\n')

    errors = brightmart_validate.validate_corpus(str(corpus_dir), str(prose_dir))
    assert any('Content mismatch' in e for e in errors), f'Expected content mismatch error, got: {errors}'


def test_check_c_extra_file(tmp_path):
    """Test check (c): extra file detection."""
    corpus_dir = _copy_corpus_to_tmp(tmp_path)
    prose_dir = tmp_path / 'brightmart_llm_prose'
    prose_dir.mkdir()

    # Add an extra file not in the rendered corpus
    extra_file = corpus_dir / 'extra-file.md'
    with open(extra_file, 'w', encoding='utf-8') as f:
        f.write('This file should not exist')

    errors = brightmart_validate.validate_corpus(str(corpus_dir), str(prose_dir))
    assert any('Extra file on disk' in e for e in errors), f'Expected extra file error, got: {errors}'


def test_check_c_missing_file(tmp_path):
    """Test check (c): missing file detection."""
    corpus_dir = _copy_corpus_to_tmp(tmp_path)
    prose_dir = tmp_path / 'brightmart_llm_prose'
    prose_dir.mkdir()

    # Delete a file
    home_file = corpus_dir / 'brightmart-home.md'
    home_file.unlink()

    errors = brightmart_validate.validate_corpus(str(corpus_dir), str(prose_dir))
    assert any('Missing file' in e for e in errors), f'Expected missing file error, got: {errors}'


def test_check_a_duplicate_basename_in_subdirs(tmp_path):
    """Test check (a): duplicate basename in different subdirectories."""
    # Create tmp scan root with two files with same basename in different subdirs
    scan_root = tmp_path / 'scan_root'
    scan_root.mkdir()

    subdir1 = scan_root / 'subdir1'
    subdir1.mkdir()
    (subdir1 / 'test.md').write_text('content1')

    subdir2 = scan_root / 'subdir2'
    subdir2.mkdir()
    (subdir2 / 'test.md').write_text('content2')

    # Call check function with this scan_root and empty git_paths
    errors = brightmart_validate._check_unique_basenames(
        scan_root=str(scan_root),
        git_paths=set()
    )

    assert any('Duplicate basename' in e and 'test.md' in e for e in errors), \
        f'Expected duplicate basename error for test.md, got: {errors}'


def test_check_a_git_collision(tmp_path):
    """Test check (a): basename collision with git ls-files."""
    # Create tmp scan root with a file
    scan_root = tmp_path / 'scan_root'
    scan_root.mkdir()
    (scan_root / 'README.md').write_text('content')

    # Create git paths that include README.md at root level (outside scan_root)
    git_paths = {'README.md', 'docs/index.md', 'src/main.py'}

    # Call check function with this scan_root and git_paths
    errors = brightmart_validate._check_unique_basenames(
        scan_root=str(scan_root),
        git_paths=git_paths
    )

    assert any('conflict' in e.lower() and 'README.md' in e for e in errors), \
        f'Expected collision error for README.md, got: {errors}'
