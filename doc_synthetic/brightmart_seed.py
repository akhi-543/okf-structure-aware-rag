import sqlite3
import random

# Determinism seed
RNG = random.Random(20260924)

# Constants
REGIONS = [
    ('northeast', 'Northeast', 'Dana Whitlock'),
    ('southeast', 'Southeast', 'Marcus Ebo'),
    ('midwest', 'Midwest', 'Priya Halvorsen'),
    ('west', 'West', 'Tomas Arriaga'),
]

STORES = [
    ('S01', 'northeast', 'Brightmart Riverside', 'Hartford', 'CT', 'supercenter', 2004, 182000, 1, 1, 'open'),
    ('S02', 'northeast', 'Brightmart Riverdale', 'Albany', 'NY', 'neighborhood', 2011, 41000, 1, 0, 'open'),
    ('S03', 'northeast', 'Brightmart Harbor Point', 'Portland', 'ME', 'express', 2019, 14000, 0, 0, 'open'),
    ('S04', 'southeast', 'Brightmart Magnolia Row', 'Savannah', 'GA', 'supercenter', 2008, 176000, 1, 1, 'open'),
    ('S05', 'southeast', 'Brightmart Palmetto Cross', 'Columbia', 'SC', 'neighborhood', 2015, 39000, 0, 1, 'remodeling'),
    ('S06', 'southeast', 'Brightmart Bayou Gate', 'Baton Rouge', 'LA', 'supercenter', 2001, 190000, 1, 1, 'closed'),
    ('S07', 'midwest', 'Brightmart Cedar Falls', 'Cedar Falls', 'IA', 'supercenter', 2006, 171000, 1, 1, 'open'),
    ('S08', 'midwest', 'Brightmart Lakeshore', 'Madison', 'WI', 'neighborhood', 2013, 43000, 1, 0, 'open'),
    ('S09', 'midwest', 'Brightmart Prairie Hub', 'Wichita', 'KS', 'express', 2020, 12500, 0, 1, 'open'),
    ('S10', 'west', 'Brightmart Cedar Park', 'Boise', 'ID', 'supercenter', 2009, 185000, 1, 1, 'open'),
    ('S11', 'west', 'Brightmart Desert Bloom', 'Tucson', 'AZ', 'neighborhood', 2016, 40500, 0, 0, 'open'),
    ('S12', 'west', 'Brightmart Summit Ridge', 'Reno', 'NV', 'express', 2021, 13000, 0, 1, 'remodeling'),
]

DEPARTMENTS = [
    ('D01', 'Grocery', 'grocery', 'food'),
    ('D02', 'Pharmacy', 'pharmacy', 'health'),
    ('D03', 'Electronics', 'electronics', 'general'),
    ('D04', 'Home & Garden', 'home-garden', 'general'),
    ('D05', 'Apparel', 'apparel', 'general'),
    ('D06', 'Toys', 'toys', 'general'),
    ('D07', 'Automotive', 'automotive', 'general'),
    ('D08', 'Bakery', 'bakery', 'food'),
]

SUPPLIERS = [
    ('SUP1', 'Northwind Provisions', 'northwind-provisions', 'United States'),
    ('SUP2', 'Kestrel Electronics', 'kestrel-electronics', 'Taiwan'),
    ('SUP3', 'Greenfield Farms Co-op', 'greenfield-farms-coop', 'United States'),
    ('SUP4', 'Greenfield Home Supply', 'greenfield-home-supply', 'Canada'),
    ('SUP5', 'Alder & Finch Textiles', 'alder-finch-textiles', 'Portugal'),
    ('SUP6', 'Copperline Auto Parts', 'copperline-auto-parts', 'Mexico'),
]

CATEGORIES = [
    ('C01', 'D01', 'Fresh Produce', 'fresh-produce', 'SUP3'),
    ('C02', 'D01', 'Dairy', 'dairy', 'SUP1'),
    ('C03', 'D02', 'OTC Medicines', 'otc-medicines', 'SUP1'),
    ('C04', 'D03', 'Televisions', 'televisions', 'SUP2'),
    ('C05', 'D03', 'Phones', 'phones', 'SUP2'),
    ('C06', 'D04', 'Patio Furniture', 'patio-furniture', 'SUP4'),
    ('C07', 'D05', 'Kids Apparel', 'kids-apparel', 'SUP5'),
    ('C08', 'D06', 'Board Games', 'board-games', 'SUP1'),
    ('C09', 'D07', 'Motor Oil', 'motor-oil', 'SUP6'),
    ('C10', 'D08', 'Artisan Bread', 'artisan-bread', 'SUP3'),
]

PROMOTIONS = [
    ('P01', 'Spring Garden Days', 'spring-garden-days', 'D04', '2025-03-15', '2025-04-15', 20),
    ('P02', 'Back to Class', 'back-to-class', 'D05', '2025-08-01', '2025-08-31', 15),
    ('P03', 'Big Screen Weekend', 'big-screen-weekend', 'D03', '2025-11-28', '2025-11-30', 25),
    ('P04', 'Fresh Friday', 'fresh-friday', 'D01', '2025-01-03', '2025-12-26', 10),
    ('P05', 'Winter Care', 'winter-care', 'D02', '2025-12-01', '2025-12-31', 12),
]

PROMOTION_FORMATS = [
    ('P01', 'supercenter'),
    ('P01', 'neighborhood'),
    ('P02', 'supercenter'),
    ('P03', 'supercenter'),
    ('P04', 'supercenter'),
    ('P04', 'neighborhood'),
    ('P04', 'express'),
    ('P05', 'supercenter'),
    ('P05', 'neighborhood'),
]

POLICIES = [
    ('POL1', 'Returns Policy', 'returns-policy', 'all stores'),
    ('POL2', 'Pharmacy Hours Policy', 'pharmacy-hours-policy', 'stores with pharmacy'),
    ('POL3', 'Fuel Rewards Policy', 'fuel-rewards-policy', 'stores with fuel'),
]

POLICY_FACTS = [
    ('POL1', 'standard_window_days', '30'),
    ('POL1', 'electronics_window_days', '15'),
    ('POL1', 'receipt_required', 'yes'),
    ('POL2', 'weekday_hours', '08:00-21:00'),
    ('POL2', 'sunday_hours', '10:00-18:00'),
    ('POL3', 'cents_off_per_gallon', '10'),
    ('POL3', 'spend_threshold_usd', '100'),
]

ARCHIVE_DOCS = [
    ('A1', 'archived-cedar-falls-profile-2022', 'S07', 'sq_ft', '150000', 'deprecated'),
    ('A2', 'draft-spring-garden-days-2026', 'P01', 'discount_pct', '30', 'draft'),
    ('A3', 'archived-returns-policy-v1', 'POL1', 'standard_window_days', '60', 'deprecated'),
]

MANAGER_POOL = [
    'Alice Chen', 'Bob Martinez', 'Carol Ramirez', 'David Kim', 'Emma Johnson',
    'Frank Miller', 'Grace Lee', 'Henry Brown', 'Iris Wilson', 'Jack Anderson',
    'Karen Thomas', 'Leo Garcia', 'Mona Rodriguez', 'Nathan Taylor', 'Olivia Moore',
    'Paul Jackson', 'Quinn White', 'Rachel Davis', 'Samuel Martin', 'Tina Harris',
    'Uma Patel', 'Victor Lewis', 'Wendy Clark', 'Xavier Young', 'Yuki Tanaka',
    'Zara Ahmed', 'Adrian Scott', 'Bella Green', 'Carlos Adams', 'Diane Nelson',
    'Ethan Baker', 'Fiona Hall', 'Gus Rivera', 'Hannah Carter', 'Ivan Roberts',
]

DEPT_WEIGHT = {
    'D01': 1.0,
    'D02': 0.45,
    'D03': 0.6,
    'D04': 0.35,
    'D05': 0.3,
    'D06': 0.2,
    'D07': 0.15,
    'D08': 0.12,
}


def _read_schema():
    """Read schema from file."""
    import os
    schema_path = os.path.join(os.path.dirname(__file__), 'brightmart_schema.sql')
    with open(schema_path, 'r') as f:
        return f.read()


def connect():
    """Create and populate in-memory Brightmart database."""
    conn = sqlite3.connect(':memory:')
    conn.execute('PRAGMA foreign_keys=ON')

    # Execute schema
    schema = _read_schema()
    conn.executescript(schema)

    # Insert regions
    conn.executemany('INSERT INTO region VALUES (?, ?, ?)', REGIONS)

    # Insert stores
    conn.executemany('INSERT INTO store VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', STORES)

    # Insert departments
    conn.executemany('INSERT INTO department VALUES (?, ?, ?, ?)', DEPARTMENTS)

    # Compute and insert store_department
    store_department_rows = []
    rng = random.Random(20260924)
    manager_pool_copy = list(MANAGER_POOL)

    # Create lookup maps
    store_by_id = {s[0]: s for s in STORES}

    for store_id, region_id, name, city, state, store_format, opened_year, sq_ft, has_pharmacy, has_fuel, status in STORES:
        for dept_id, dept_name, slug, group in DEPARTMENTS:
            # Determine if this department exists in this store
            if store_format == 'supercenter':
                include = True
            elif store_format == 'neighborhood':
                include = dept_id in ('D01', 'D04', 'D08') or (dept_id == 'D02' and has_pharmacy == 1)
            elif store_format == 'express':
                include = dept_id in ('D01', 'D08')
            else:
                include = False

            if include:
                dept_manager = rng.choice(manager_pool_copy)
                store_department_rows.append((store_id, dept_id, dept_manager))

    conn.executemany('INSERT INTO store_department VALUES (?, ?, ?)', store_department_rows)

    # Insert suppliers
    conn.executemany('INSERT INTO supplier VALUES (?, ?, ?, ?)', SUPPLIERS)

    # Insert categories
    conn.executemany('INSERT INTO category VALUES (?, ?, ?, ?, ?)', CATEGORIES)

    # Insert promotions
    conn.executemany('INSERT INTO promotion VALUES (?, ?, ?, ?, ?, ?, ?)', PROMOTIONS)

    # Insert promotion_format
    conn.executemany('INSERT INTO promotion_format VALUES (?, ?)', PROMOTION_FORMATS)

    # Insert policies
    conn.executemany('INSERT INTO policy VALUES (?, ?, ?, ?)', POLICIES)

    # Insert policy_fact
    conn.executemany('INSERT INTO policy_fact VALUES (?, ?, ?)', POLICY_FACTS)

    # Insert archive_doc
    conn.executemany('INSERT INTO archive_doc VALUES (?, ?, ?, ?, ?, ?)', ARCHIVE_DOCS)

    # Generate and insert weekly_sales
    weekly_sales_rows = []
    rng = random.Random(20260924)

    for store_id, region_id, name, city, state, store_format, opened_year, sq_ft, has_pharmacy, has_fuel, status in STORES:
        for dept_id, dept_name, slug, group in DEPARTMENTS:
            # Check if this store_department exists
            if store_format == 'supercenter':
                include = True
            elif store_format == 'neighborhood':
                include = dept_id in ('D01', 'D04', 'D08') or (dept_id == 'D02' and has_pharmacy == 1)
            elif store_format == 'express':
                include = dept_id in ('D01', 'D08')
            else:
                include = False

            if not include:
                continue

            # Determine base sales by format
            base_by_format = {'supercenter': 42000, 'neighborhood': 15000, 'express': 6000}
            base = base_by_format[store_format] * DEPT_WEIGHT[dept_id]

            # Determine max weeks
            max_weeks = 52
            if store_id == 'S06' and status == 'closed':
                max_weeks = 30

            for week in range(1, max_weeks + 1):
                sales = round(base * rng.uniform(0.8, 1.2), 2)
                weekly_sales_rows.append((store_id, dept_id, week, sales))

    conn.executemany('INSERT INTO weekly_sales VALUES (?, ?, ?, ?)', weekly_sales_rows)

    conn.commit()
    return conn


if __name__ == '__main__':
    import sys

    conn = connect()

    # Print row counts
    cursor = conn.cursor()
    tables = [
        'region', 'store', 'department', 'supplier', 'category',
        'promotion', 'policy', 'policy_fact', 'archive_doc', 'store_department', 'weekly_sales'
    ]

    for table in tables:
        count = cursor.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
        print(f'{table}: {count}')

    # Write to file if path provided
    if len(sys.argv) > 1:
        output_path = sys.argv[1]
        output_conn = sqlite3.connect(output_path)
        conn.backup(output_conn)
        output_conn.close()
        print(f'Database written to {output_path}')

    conn.close()
