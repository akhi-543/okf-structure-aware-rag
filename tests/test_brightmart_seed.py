import sys
import sqlite3
sys.path.insert(0, 'doc_synthetic')

import brightmart_seed


def test_row_counts():
    """Test that all tables have expected row counts."""
    conn = brightmart_seed.connect()
    cursor = conn.cursor()

    expected = {
        'region': 4,
        'store': 12,
        'department': 8,
        'supplier': 6,
        'category': 10,
        'promotion': 5,
        'policy': 3,
        'policy_fact': 7,
        'archive_doc': 3,
        'store_department': 60,
        'weekly_sales': 2944,
    }

    for table, count in expected.items():
        actual = cursor.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
        assert actual == count, f'{table}: expected {count}, got {actual}'

    conn.close()


def test_determinism():
    """Test that two connects() produce identical dumps."""
    conn1 = brightmart_seed.connect()
    dump1 = '\n'.join(conn1.iterdump())
    conn1.close()

    conn2 = brightmart_seed.connect()
    dump2 = '\n'.join(conn2.iterdump())
    conn2.close()

    assert dump1 == dump2, 'Dumps differ; randomness not deterministic'


def test_foreign_keys():
    """Test that foreign key constraints are satisfied."""
    conn = brightmart_seed.connect()
    cursor = conn.cursor()

    violations = cursor.execute('PRAGMA foreign_key_check').fetchall()
    assert not violations, f'Foreign key violations: {violations}'

    conn.close()


def test_d02_pharmacy_constraint():
    """Test that D02 is present iff has_pharmacy=1."""
    conn = brightmart_seed.connect()
    cursor = conn.cursor()

    # Get all stores and their pharmacy status
    stores = cursor.execute('SELECT store_id, has_pharmacy FROM store').fetchall()

    for store_id, has_pharmacy in stores:
        # Check if D02 exists for this store
        d02_exists = cursor.execute(
            'SELECT COUNT(*) FROM store_department WHERE store_id = ? AND dept_id = ?',
            (store_id, 'D02')
        ).fetchone()[0] > 0

        if has_pharmacy == 1:
            assert d02_exists, f'{store_id}: has_pharmacy=1 but D02 not found'
        else:
            assert not d02_exists, f'{store_id}: has_pharmacy=0 but D02 found'

    conn.close()


if __name__ == '__main__':
    test_row_counts()
    test_determinism()
    test_foreign_keys()
    test_d02_pharmacy_constraint()
    print('All tests passed')
