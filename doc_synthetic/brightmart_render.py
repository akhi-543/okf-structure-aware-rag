"""
Brightmart synthetic retail corpus renderer.
Renders 53 markdown docs from in-memory SQLite database.
"""

import sqlite3
import random
import os
import shutil
import yaml


def render(conn: sqlite3.Connection, prose_dir: str = None) -> dict[str, str]:
    """
    Render corpus docs from database connection.
    Returns dict mapping bundle-absolute path -> full markdown text.
    If prose_dir is provided, appends prose files as Notes sections.
    """
    # Initialize RNG for template choice (consumed in sorted path order)
    rng = random.Random(20260924)

    cursor = conn.cursor()
    docs = {}

    # Fetch all data
    regions = {r[0]: r for r in cursor.execute(
        'SELECT region_id, name, manager FROM region ORDER BY region_id'
    ).fetchall()}

    stores = {s[0]: s for s in cursor.execute(
        'SELECT store_id, region_id, name, city, state, format, opened_year, sq_ft, has_pharmacy, has_fuel, status FROM store ORDER BY store_id'
    ).fetchall()}

    depts = {d[0]: d for d in cursor.execute(
        'SELECT dept_id, name, slug, "group" FROM department ORDER BY dept_id'
    ).fetchall()}

    suppliers = {s[0]: s for s in cursor.execute(
        'SELECT supplier_id, name, slug, country FROM supplier ORDER BY supplier_id'
    ).fetchall()}

    categories = {c[0]: c for c in cursor.execute(
        'SELECT cat_id, dept_id, name, slug, supplier_id FROM category ORDER BY cat_id'
    ).fetchall()}

    promos = {p[0]: p for p in cursor.execute(
        'SELECT promo_id, name, slug, dept_id, start, end, discount_pct FROM promotion ORDER BY promo_id'
    ).fetchall()}

    promo_formats = {}
    for pf in cursor.execute('SELECT promo_id, format FROM promotion_format').fetchall():
        if pf[0] not in promo_formats:
            promo_formats[pf[0]] = []
        promo_formats[pf[0]].append(pf[1])

    policies = {p[0]: p for p in cursor.execute(
        'SELECT policy_id, name, slug, applies_to FROM policy ORDER BY policy_id'
    ).fetchall()}

    policy_facts = {}
    for pf in cursor.execute('SELECT policy_id, key, value FROM policy_fact').fetchall():
        if pf[0] not in policy_facts:
            policy_facts[pf[0]] = {}
        policy_facts[pf[0]][pf[1]] = pf[2]

    archives = {a[0]: a for a in cursor.execute(
        'SELECT archive_id, slug, subject, stale_key, stale_value, status FROM archive_doc ORDER BY archive_id'
    ).fetchall()}

    store_depts = {}
    for sd in cursor.execute('SELECT store_id, dept_id, dept_manager FROM store_department').fetchall():
        if sd[0] not in store_depts:
            store_depts[sd[0]] = {}
        store_depts[sd[0]][sd[1]] = sd[2]

    # Helper: get weekly sales for a store-dept across weeks, sum by quarter
    def get_quarterly_sales(store_id, dept_id):
        """Returns [Q1_sum, Q2_sum, Q3_sum, Q4_sum] as formatted strings."""
        sales = cursor.execute(
            'SELECT week, sales FROM weekly_sales WHERE store_id = ? AND dept_id = ? ORDER BY week',
            (store_id, dept_id)
        ).fetchall()
        sales_by_week = {w: s for w, s in sales}

        # Q1: weeks 1-13, Q2: 14-26, Q3: 27-39, Q4: 40-52
        # For S06 (closed), only weeks 1-30 exist
        quarters = []
        for q, (start, end) in enumerate([(1, 13), (14, 26), (27, 39), (40, 52)], 1):
            q_total = sum(sales_by_week.get(w, 0.0) for w in range(start, end + 1))
            quarters.append(f'{q_total:,.2f}')

        return quarters

    # Generate all docs in sorted path order (for RNG consumption)
    docs_to_gen = []

    # Home
    docs_to_gen.append(('/brightmart-home.md', 'home', None))

    # Regions
    for region_id in sorted(regions.keys()):
        docs_to_gen.append((f'/regions/region-{region_id}.md', 'region', region_id))

    # Stores
    for store_id in sorted(stores.keys()):
        store = stores[store_id]
        region_id = store[1]
        name = store[2]
        name_slug = name.replace('Brightmart ', '').lower().replace(' ', '-')
        path = f'/regions/{region_id}/store-{store_id.lower()}-{name_slug}.md'
        docs_to_gen.append((path, 'store', store_id))

    # Departments
    for dept_id in sorted(depts.keys()):
        dept = depts[dept_id]
        path = f'/departments/dept-{dept[2]}.md'
        docs_to_gen.append((path, 'department', dept_id))

    # Categories
    for cat_id in sorted(categories.keys()):
        cat = categories[cat_id]
        path = f'/departments/{depts[cat[1]][2]}/category-{cat[3]}.md'
        docs_to_gen.append((path, 'category', cat_id))

    # Suppliers
    for supplier_id in sorted(suppliers.keys()):
        supp = suppliers[supplier_id]
        path = f'/suppliers/supplier-{supp[2]}.md'
        docs_to_gen.append((path, 'supplier', supplier_id))

    # Promotions
    for promo_id in sorted(promos.keys()):
        promo = promos[promo_id]
        path = f'/promotions/promo-{promo[2]}.md'
        docs_to_gen.append((path, 'promotion', promo_id))

    # Policies
    for policy_id in sorted(policies.keys()):
        policy = policies[policy_id]
        path = f'/policies/policy-{policy[2]}.md'
        docs_to_gen.append((path, 'policy', policy_id))

    # Archives
    for archive_id in sorted(archives.keys()):
        archive = archives[archive_id]
        path = f'/archive/{archive[1]}.md'
        docs_to_gen.append((path, 'archive', archive_id))

    # Schema
    docs_to_gen.append(('/schema/brightmart-sales-database.md', 'schema', None))

    # Render each doc
    for path, doc_type, key in docs_to_gen:
        if doc_type == 'home':
            docs[path] = _render_home(rng, regions, depts, suppliers, promos, policies)
        elif doc_type == 'region':
            docs[path] = _render_region(rng, key, regions, stores)
        elif doc_type == 'store':
            docs[path] = _render_store(rng, key, stores, regions, depts, store_depts, get_quarterly_sales)
        elif doc_type == 'department':
            docs[path] = _render_department(rng, key, depts, categories, stores, store_depts)
        elif doc_type == 'category':
            docs[path] = _render_category(rng, key, categories, depts, suppliers)
        elif doc_type == 'supplier':
            docs[path] = _render_supplier(rng, key, suppliers, categories, depts)
        elif doc_type == 'promotion':
            docs[path] = _render_promotion(rng, key, promos, depts, promo_formats)
        elif doc_type == 'policy':
            docs[path] = _render_policy(rng, key, policies, policy_facts)
        elif doc_type == 'archive':
            docs[path] = _render_archive(rng, key, archives, stores, regions, depts, promos, policies, promo_formats, policy_facts, store_depts, get_quarterly_sales)
        elif doc_type == 'schema':
            docs[path] = _render_schema(rng)

    # Append prose if prose_dir is provided
    if prose_dir is None:
        prose_dir = os.path.join(os.path.dirname(__file__), 'brightmart_llm_prose')

    if os.path.exists(prose_dir):
        for path, content in docs.items():
            # Extract stem from path (e.g., /regions/region-northeast.md -> region-northeast)
            rel_path = path.lstrip('/')
            stem = os.path.splitext(rel_path)[0].split('/')[-1]
            prose_file = os.path.join(prose_dir, f'{stem}.prose.txt')

            if os.path.exists(prose_file):
                with open(prose_file, 'r', encoding='utf-8') as f:
                    prose_content = f.read().strip()
                # Append prose with Notes section
                docs[path] = content + f'\n## Notes\n\n{prose_content}\n'

    return docs


def _render_home(rng, regions, depts, suppliers, promos, policies):
    """Render home document."""
    fm = {
        'type': 'home',
        'title': 'Brightmart Synthetic Retail Wiki',
        'description': 'Central hub for the Brightmart synthetic retail corpus.',
        'tags': ['home', 'synthetic'],
        'status': 'stable',
        'timestamp': '2026-01-15',
    }

    body = '# Brightmart Synthetic Retail Wiki\n\n'
    body += 'Welcome to the Brightmart synthetic retail dataset documentation.\n\n'

    body += '## Regions\n\n'
    for rid in sorted(regions.keys()):
        body += f'- [{regions[rid][1]}](/regions/region-{rid}.md)\n'

    body += '\n## Departments\n\n'
    for did in sorted(depts.keys()):
        body += f'- [{depts[did][1]}](/departments/dept-{depts[did][2]}.md)\n'

    body += '\n## Suppliers\n\n'
    for sid in sorted(suppliers.keys()):
        body += f'- [{suppliers[sid][1]}](/suppliers/supplier-{suppliers[sid][2]}.md)\n'

    body += '\n## Promotions\n\n'
    for pid in sorted(promos.keys()):
        body += f'- [{promos[pid][1]}](/promotions/promo-{promos[pid][2]}.md)\n'

    body += '\n## Policies\n\n'
    for pol_id in sorted(policies.keys()):
        body += f'- [{policies[pol_id][1]}](/policies/policy-{policies[pol_id][2]}.md)\n'

    body += '\n## Schema\n\n'
    body += '- [Brightmart Sales Database](/schema/brightmart-sales-database.md)\n'

    return _build_markdown(fm, body)


def _render_region(rng, region_id, regions, stores):
    """Render region document."""
    region = regions[region_id]

    fm = {
        'type': 'region',
        'title': region[1],
        'description': f'Regional hub for {region[1]}, managed by {region[2]}.',
        'tags': ['region', region_id],
        'status': 'stable',
        'timestamp': '2026-01-15',
        'parent': '/brightmart-home.md',
        'region_id': region_id,
        'manager': region[2],
    }

    body = f'# {region[1]}\n\n'

    # Manager sentence
    manager_templates = [
        'The {region} region is managed by {manager}.',
        '{manager} oversees the {region} region.',
        'Regional director {manager} leads the {region} region.',
    ]
    body += rng.choice(manager_templates).format(region=region[1], manager=region[2]) + '\n\n'

    # Stores
    body += '## Stores\n\n'
    region_stores = sorted([s for s in stores.values() if s[1] == region_id], key=lambda x: x[0])
    for store in region_stores:
        store_id = store[0]
        name = store[2]
        name_slug = name.replace('Brightmart ', '').lower().replace(' ', '-')
        path = f'/regions/{region_id}/store-{store_id.lower()}-{name_slug}.md'
        body += f'- [{name}]({path})\n'

    return _build_markdown(fm, body)


def _render_store(rng, store_id, stores, regions, depts, store_depts, get_quarterly_sales, sq_ft_override=None):
    """Render store document. If sq_ft_override is provided, use it for archive variants."""
    store = stores[store_id]
    region_id = store[1]
    name = store[2]
    city = store[3]
    state = store[4]
    fmt = store[5]
    opened_year = store[6]
    sq_ft = sq_ft_override if sq_ft_override is not None else store[7]
    has_pharmacy = store[8]
    has_fuel = store[9]
    status = store[10]

    name_slug = name.replace('Brightmart ', '').lower().replace(' ', '-')

    fm = {
        'type': 'store',
        'title': name,
        'description': f'Profile of {name}, a {fmt} in {city}, {state}.',
        'tags': ['store', store_id, region_id, fmt],
        'status': 'stable',
        'timestamp': '2026-01-15',
        'parent': f'/regions/region-{region_id}.md',
        'store_id': store_id,
        'region': regions[region_id][1],
        'store_format': fmt,
        'opened_year': opened_year,
        'sq_ft': sq_ft,
        'has_pharmacy': bool(has_pharmacy),
        'has_fuel': bool(has_fuel),
        'store_status': status,
        'departments': [depts[did][1] for did in sorted(store_depts.get(store_id, {}).keys())],
    }

    body = f'# {name}\n\n'

    # Location sentence
    location_templates = [
        f'{name} is located in {city}, {state}.',
        f'This store is situated in {city}, {state}.',
        f'You will find this store in {city}, {state}.',
    ]
    body += rng.choice(location_templates) + '\n\n'

    # Format sentence
    format_templates = [
        f'{name} is a {fmt} format store.',
        f'Operating as a {fmt}, {name} offers a curated selection of products.',
        f'This {fmt} location provides comprehensive retail services.',
    ]
    body += rng.choice(format_templates) + '\n\n'

    # Year sentence (G5: Fix missing comma)
    year_templates = [
        f'{name} opened its doors in {opened_year}.',
        f'Established in {opened_year}, this store has served the community.',
        f'Since {opened_year}, this facility has been operating.',
    ]
    body += rng.choice(year_templates) + '\n\n'

    # Size sentence
    size_templates = [
        f'{name} occupies {sq_ft:,} square feet of retail space.',
        f'The sales floor at {name} covers {sq_ft:,} square feet.',
        f'{name} spans {sq_ft:,} square feet.',
    ]
    body += rng.choice(size_templates) + '\n\n'

    # Pharmacy
    if has_pharmacy:
        pharm_templates = [
            f'{name} includes a full-service pharmacy.',
            'Pharmacy services are available on-site.',
            'A pharmacy operates within this location.',
        ]
        body += rng.choice(pharm_templates) + '\n\n'
    else:
        pharm_templates = [
            f'{name} does not operate a pharmacy.',
            'No pharmacy services are offered at this location.',
            'This store does not have a pharmacy.',
        ]
        body += rng.choice(pharm_templates) + '\n\n'

    # Fuel
    if has_fuel:
        fuel_templates = [
            'Fuel services are available at the pumps.',
            'Customers can fill up at the fuel station.',
            f'{name} operates a fuel station.',
        ]
        body += rng.choice(fuel_templates) + '\n\n'
    else:
        fuel_templates = [
            'No fuel station is available at this location.',
            'Fuel services are not offered here.',
            f'{name} does not have a fuel station.',
        ]
        body += rng.choice(fuel_templates) + '\n\n'

    # Status
    if status == 'open':
        status_templates = [
            f'{name} is currently open and operating normally.',
            'This location is open to customers.',
            'The store is operational and welcoming shoppers.',
        ]
    elif status == 'closed':
        status_templates = [
            f'{name} is permanently closed.',
            'This location has ceased operations.',
            'The store is no longer in operation.',
        ]
    else:  # remodeling
        status_templates = [
            f'{name} is currently undergoing remodeling.',
            'Renovations are in progress at this location.',
            'This store is closed for renovations.',
        ]
    body += rng.choice(status_templates) + '\n\n'

    # Departments
    body += '## Departments\n\n'
    dept_ids = sorted(store_depts.get(store_id, {}).keys())
    for did in dept_ids:
        dept = depts[did]
        manager = store_depts[store_id][did]
        body += f'- [{dept[1]}](/departments/dept-{dept[2]}.md) – managed by {manager}\n'

    # Quarterly sales table (omit for archive variants in some cases)
    body += '\n## 2025 Sales by Quarter\n\n'
    body += '| Department | Q1 | Q2 | Q3 | Q4 |\n'
    body += '|---|---|---|---|---|\n'
    for did in dept_ids:
        dept = depts[did]
        quarters = get_quarterly_sales(store_id, did)
        body += f'| {dept[1]} | {quarters[0]} | {quarters[1]} | {quarters[2]} | {quarters[3]} |\n'

    body += f'\n[Back to {regions[region_id][1]}](/regions/region-{region_id}.md)\n'

    return _build_markdown(fm, body)


def _render_department(rng, dept_id, depts, categories, stores, store_depts):
    """Render department document."""
    dept = depts[dept_id]

    fm = {
        'type': 'department',
        'title': dept[1],
        'description': f'The {dept[1]} department focuses on {dept[3]} products.',
        'tags': ['department', dept_id, dept[3]],
        'status': 'stable',
        'timestamp': '2026-01-15',
        'parent': '/brightmart-home.md',
        'dept_id': dept_id,
        'dept_group': dept[3],
    }

    body = f'# {dept[1]}\n\n'

    # Group sentence
    group_templates = [
        f'The {dept[1]} department belongs to the {dept[3]} category group.',
        f'Part of our {dept[3]} family, {dept[1]} focuses on specialized products.',
        f'This department specializes in {dept[3]} products and services.',
    ]
    body += rng.choice(group_templates) + '\n\n'

    # Categories
    dept_cats = sorted([c for c in categories.values() if c[1] == dept_id], key=lambda x: x[0])
    if dept_cats:
        body += '## Categories\n\n'
        for cat in dept_cats:
            body += f'- [{cat[2]}](/departments/{dept[2]}/category-{cat[3]}.md)\n'
        body += '\n'

    # Stores carrying this dept
    body += '## Available at\n\n'
    stores_with_dept = sorted(set(
        s[0] for s in stores.values()
        if s[0] in store_depts and dept_id in store_depts[s[0]]
    ))
    for store_id in stores_with_dept:
        store = stores[store_id]
        name_slug = store[2].replace('Brightmart ', '').lower().replace(' ', '-')
        region_id = store[1]
        path = f'/regions/{region_id}/store-{store_id.lower()}-{name_slug}.md'
        body += f'- [{store[2]}]({path})\n'

    return _build_markdown(fm, body)


def _render_category(rng, cat_id, categories, depts, suppliers):
    """Render category document."""
    cat = categories[cat_id]
    dept = depts[cat[1]]
    supplier = suppliers[cat[4]]

    fm = {
        'type': 'category',
        'title': cat[2],
        'description': f'{cat[2]} category offered through Brightmart stores.',
        'tags': ['category', cat[3], cat[1]],
        'status': 'stable',
        'timestamp': '2026-01-15',
        'parent': f'/departments/dept-{dept[2]}.md',
        'department': dept[1],
        'supplier': supplier[1],
    }

    body = f'# {cat[2]}\n\n'

    # Department link
    body += f'This category is part of the [{dept[1]}](/departments/dept-{dept[2]}.md) department.\n\n'

    # Supplier sentence
    supplier_templates = [
        f'{cat[2]} is sourced from [{supplier[1]}](/suppliers/supplier-{supplier[2]}.md).',
        f'[{supplier[1]}](/suppliers/supplier-{supplier[2]}.md) supplies these products.',
        f'Our partner [{supplier[1]}](/suppliers/supplier-{supplier[2]}.md) provides {cat[2]} items.',
    ]
    body += rng.choice(supplier_templates) + '\n'

    return _build_markdown(fm, body)


def _render_supplier(rng, supplier_id, suppliers, categories, depts):
    """Render supplier document."""
    supplier = suppliers[supplier_id]

    # H1: Use "the United States" for US suppliers in description and body
    country_for_text = supplier[3]
    if country_for_text == 'United States':
        country_for_text = 'the United States'

    fm = {
        'type': 'supplier',
        'title': supplier[1],
        'description': f'Supplier {supplier[1]} based in {country_for_text}.',
        'tags': ['supplier', supplier_id, supplier[3]],
        'status': 'stable',
        'timestamp': '2026-01-15',
        'parent': '/brightmart-home.md',
        'country': supplier[3],
    }

    body = f'# {supplier[1]}\n\n'

    # Country sentence (H1: "the United States" for US suppliers)
    country = country_for_text

    country_templates = [
        f'{supplier[1]} is based in {country}.',
        f'{supplier[1]} operates out of {country}.',
        f'{supplier[1]} is headquartered in {country}.',
    ]
    body += rng.choice(country_templates) + '\n\n'

    # Categories
    supp_cats = sorted([c for c in categories.values() if c[4] == supplier_id], key=lambda x: x[0])
    if supp_cats:
        body += '## Categories Supplied\n\n'
        for cat in supp_cats:
            dept = depts[cat[1]]
            body += f'- [{cat[2]}](/departments/{dept[2]}/category-{cat[3]}.md)\n'

    return _build_markdown(fm, body)


def _format_list(items):
    """Format list as 'a, b, and c' with proper commas and 'and'."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _render_promotion(rng, promo_id, promos, depts, promo_formats, start_override=None, end_override=None, discount_override=None):
    """Render promotion document. Overrides allow archive variants."""
    promo = promos[promo_id]
    dept = depts[promo[3]]
    formats = promo_formats.get(promo_id, [])

    start = start_override if start_override is not None else promo[4]
    end = end_override if end_override is not None else promo[5]
    discount = discount_override if discount_override is not None else promo[6]

    fm = {
        'type': 'promotion',
        'title': promo[1],
        'description': f'{promo[1]} promotion offering {discount}% discount.',
        'tags': ['promotion', promo_id, promo[3]],
        'status': 'stable',
        'timestamp': '2026-01-15',
        'parent': '/brightmart-home.md',
        'department': dept[1],
        'formats': formats,
        'start': start,
        'end': end,
        'discount_pct': discount,
    }

    body = f'# {promo[1]}\n\n'

    # Department link
    body += f'This promotion applies to the [{dept[1]}](/departments/dept-{dept[2]}.md) department.\n\n'

    # Dates
    body += f'**Dates:** {start} to {end}\n\n'

    # Discount
    body += f'**Discount:** {discount}% off\n\n'

    # Formats sentence
    if formats:
        fmt_str = _format_list(formats)
        format_templates = [
            f'{promo[1]} is available at {fmt_str} formats.',
            f'This promotion applies to {fmt_str} store locations.',
            f'{fmt_str.capitalize()} stores are valid for {promo[1]}.',
        ]
        body += rng.choice(format_templates) + '\n'

    return _build_markdown(fm, body)


def _render_policy(rng, policy_id, policies, policy_facts, facts_override=None):
    """Render policy document. facts_override allows archive variants."""
    policy = policies[policy_id]
    facts = facts_override if facts_override is not None else policy_facts.get(policy_id, {})

    fm = {
        'type': 'policy',
        'title': policy[1],
        'description': f'{policy[1]} for {policy[3]}.',
        'tags': ['policy', policy_id],
        'status': 'stable',
        'timestamp': '2026-01-15',
        'parent': '/brightmart-home.md',
        'applies_to': policy[3],
    }

    body = f'# {policy[1]}\n\n'

    # Policy facts as grammatical sentences with templates
    for key in sorted(facts.keys()):
        value = facts[key]

        # Build sentence templates for each key
        if key == 'standard_window_days':
            templates = [
                f'Under the {policy[1]}, most items can be returned within {value} days of purchase.',
                f'The {policy[1]} allows returns within {value} days.',
                f'{value} days is the standard return window for the {policy[1]}.',
            ]
        elif key == 'electronics_window_days':
            templates = [
                f'Electronics have a {value}-day return window under the {policy[1]}.',
                f'The {policy[1]} provides {value} days for electronics returns.',
                f'Electronics purchased through Brightmart can be returned within {value} days.',
            ]
        elif key == 'receipt_required':
            # G2: Handle yes/no meaning, not literal value
            if value.lower() == 'yes':
                templates = [
                    f'Under the {policy[1]}, a receipt is required for all returns.',
                    f'The {policy[1]} requires a receipt when returning items.',
                    f'A receipt must be presented to return items under this policy.',
                ]
            else:
                templates = [
                    f'Under the {policy[1]}, a receipt is not required for returns.',
                    f'The {policy[1]} allows returns without a receipt.',
                    f'Items can be returned without a receipt under this policy.',
                ]
        elif key == 'weekday_hours':
            # G3: Distinguish pharmacy hours from store hours
            templates = [
                f'Under the {policy[1]}, pharmacies are open {value} on weekdays.',
                f'Weekday pharmacy hours are {value} per the {policy[1]}.',
                f'Monday through Friday, pharmacy hours are {value}.',
            ]
        elif key == 'sunday_hours':
            # G3: Distinguish pharmacy hours from store hours
            templates = [
                f'Under the {policy[1]}, pharmacies are open {value} on Sundays.',
                f'On Sundays, pharmacy hours are {value} per the {policy[1]}.',
                f'The pharmacy operates {value} on Sundays under this policy.',
            ]
        elif key == 'cents_off_per_gallon':
            templates = [
                f'The {policy[1]} offers {value} cents off per gallon.',
                f'Customers receive {value} cents off each gallon under this program.',
                f'{value} cents per gallon is the standard discount.',
            ]
        elif key == 'spend_threshold_usd':
            templates = [
                f'The {policy[1]} requires a minimum purchase of ${value}.',
                f'Rewards apply after spending ${value}.',
                f'Customers must spend ${value} to qualify for benefits.',
            ]
        else:
            # Fallback for unknown keys
            templates = [f'{key}: {value}']

        body += rng.choice(templates) + '\n'

    return _build_markdown(fm, body)


def _render_archive(rng, archive_id, archives, stores, regions, depts, promos, policies, promo_formats, policy_facts, store_depts, get_quarterly_sales):
    """Render archive document as near-duplicate distractor."""
    archive = archives[archive_id]
    subject = archive[2]
    stale_key = archive[3]
    stale_value = archive[4]
    status = archive[5]

    # Determine what entity this archives and build full doc
    if subject.startswith('S'):
        # Archive of a store (A1: S07 with sq_ft 150000)
        store = stores[subject]
        current_path = f'/regions/{store[1]}/store-{subject.lower()}-{store[2].replace("Brightmart ", "").lower().replace(" ", "-")}.md'

        # Render as full store profile with stale value
        sq_ft_val = int(stale_value)
        doc_body = _render_store(
            rng, subject, stores, regions, depts, store_depts, get_quarterly_sales,
            sq_ft_override=sq_ft_val
        )

        # Extract frontmatter and body
        parts = doc_body.split('---\n', 2)
        fm_str = parts[1]
        body_content = parts[2]

        # Update frontmatter for archive
        fm = yaml.safe_load(fm_str)
        fm['type'] = 'archive'
        fm['status'] = status
        fm['supersedes'] = current_path
        del fm['parent']
        fm['parent'] = '/brightmart-home.md'

        # Add year marker to body (G5: Fix double blank line)
        year_note = 'This is a 2022 profile of this store.'
        body_content = body_content.replace(f'# {store[2]}\n\n', f'# {store[2]} (2022 profile)\n\n{year_note}\n\n')

        final_body = body_content

    elif subject.startswith('POL'):
        # Archive of a policy (A3: POL1 with standard_window_days 60)
        policy = policies[subject]
        current_path = f'/policies/policy-{policy[2]}.md'

        # Build facts override
        facts = policy_facts.get(subject, {}).copy()
        facts[stale_key] = stale_value

        doc_body = _render_policy(rng, subject, policies, policy_facts, facts_override=facts)

        # Extract frontmatter and body
        parts = doc_body.split('---\n', 2)
        fm_str = parts[1]
        body_content = parts[2]

        # Update frontmatter for archive
        fm = yaml.safe_load(fm_str)
        fm['type'] = 'archive'
        fm['status'] = status
        fm['supersedes'] = current_path
        del fm['parent']
        fm['parent'] = '/brightmart-home.md'

        # Add version marker to body (G5: Fix double blank line)
        year_note = 'This is version 1 of this policy.'
        body_content = body_content.replace(f'# {policy[1]}\n\n', f'# {policy[1]} (version 1)\n\n{year_note}\n\n')

        final_body = body_content

    else:
        # Archive of a promotion (A2: P01 with discount_pct 30 and 2026 dates)
        promo = promos[subject]
        current_path = f'/promotions/promo-{promo[2]}.md'

        # Parse dates from archive (they're stale, use 2026 dates)
        start_val = '2026-03-15'
        end_val = '2026-04-15'
        discount_val = int(stale_value)

        doc_body = _render_promotion(
            rng, subject, promos, depts, promo_formats,
            start_override=start_val, end_override=end_val, discount_override=discount_val
        )

        # Extract frontmatter and body
        parts = doc_body.split('---\n', 2)
        fm_str = parts[1]
        body_content = parts[2]

        # Update frontmatter for archive
        fm = yaml.safe_load(fm_str)
        fm['type'] = 'archive'
        fm['status'] = status
        fm['supersedes'] = current_path
        del fm['parent']
        fm['parent'] = '/brightmart-home.md'

        # Add draft marker to body (G5: Fix double blank line)
        year_note = 'This is a 2026 draft version of this promotion.'
        body_content = body_content.replace(f'# {promo[1]}\n\n', f'# {promo[1]} (2026 draft)\n\n{year_note}\n\n')

        final_body = body_content

    # Build final markdown
    fm_yaml = yaml.safe_dump(fm, sort_keys=False, allow_unicode=True)
    return f'---\n{fm_yaml}---\n{final_body}'


def _render_schema(rng):
    """Render schema document."""
    fm = {
        'type': 'schema',
        'title': 'Brightmart Sales Database Schema',
        'description': 'Database schema for the Brightmart synthetic retail dataset.',
        'tags': ['schema', 'database'],
        'status': 'stable',
        'timestamp': '2026-01-15',
        'parent': '/brightmart-home.md',
    }

    body = '# Brightmart Sales Database Schema\n\n'

    # Table schemas with FKs
    body += '### region\n\n'
    body += '- `region_id` (PK): TEXT\n'
    body += '- `name`: TEXT\n'
    body += '- `manager`: TEXT\n\n'

    body += '### store\n\n'
    body += '- `store_id` (PK): TEXT\n'
    body += '- `region_id`: TEXT\n'
    body += '- `name`: TEXT\n'
    body += '- `city`: TEXT\n'
    body += '- `state`: TEXT\n'
    body += '- `format`: TEXT\n'
    body += '- `opened_year`: INTEGER\n'
    body += '- `sq_ft`: INTEGER\n'
    body += '- `has_pharmacy`: INTEGER\n'
    body += '- `has_fuel`: INTEGER\n'
    body += '- `status`: TEXT\n'
    body += '`store.region_id` references `region.region_id`.\n\n'

    body += '### department\n\n'
    body += '- `dept_id` (PK): TEXT\n'
    body += '- `name`: TEXT\n'
    body += '- `slug`: TEXT\n'
    body += '- `group`: TEXT\n\n'

    body += '### store_department\n\n'
    body += '- `store_id`: TEXT\n'
    body += '- `dept_id`: TEXT\n'
    body += '- `dept_manager`: TEXT\n'
    body += '`store_department.store_id` references `store.store_id`.\n'
    body += '`store_department.dept_id` references `department.dept_id`.\n\n'

    body += '### supplier\n\n'
    body += '- `supplier_id` (PK): TEXT\n'
    body += '- `name`: TEXT\n'
    body += '- `slug`: TEXT\n'
    body += '- `country`: TEXT\n\n'

    body += '### category\n\n'
    body += '- `cat_id` (PK): TEXT\n'
    body += '- `dept_id`: TEXT\n'
    body += '- `name`: TEXT\n'
    body += '- `slug`: TEXT\n'
    body += '- `supplier_id`: TEXT\n'
    body += '`category.dept_id` references `department.dept_id`.\n'
    body += '`category.supplier_id` references `supplier.supplier_id`.\n\n'

    body += '### weekly_sales\n\n'
    body += '- `store_id`: TEXT\n'
    body += '- `dept_id`: TEXT\n'
    body += '- `week`: INTEGER\n'
    body += '- `sales`: REAL\n'
    body += '`weekly_sales.store_id` references `store.store_id`.\n'
    body += '`weekly_sales.dept_id` references `department.dept_id`.\n\n'

    body += '### promotion\n\n'
    body += '- `promo_id` (PK): TEXT\n'
    body += '- `name`: TEXT\n'
    body += '- `slug`: TEXT\n'
    body += '- `dept_id`: TEXT\n'
    body += '- `start`: TEXT\n'
    body += '- `end`: TEXT\n'
    body += '- `discount_pct`: INTEGER\n'
    body += '`promotion.dept_id` references `department.dept_id`.\n\n'

    body += '### promotion_format\n\n'
    body += '- `promo_id`: TEXT\n'
    body += '- `format`: TEXT\n'
    body += '`promotion_format.promo_id` references `promotion.promo_id`.\n\n'

    body += '### policy\n\n'
    body += '- `policy_id` (PK): TEXT\n'
    body += '- `name`: TEXT\n'
    body += '- `slug`: TEXT\n'
    body += '- `applies_to`: TEXT\n\n'

    body += '### policy_fact\n\n'
    body += '- `policy_id`: TEXT\n'
    body += '- `key`: TEXT\n'
    body += '- `value`: TEXT\n'
    body += '`policy_fact.policy_id` references `policy.policy_id`.\n\n'

    body += '### archive_doc\n\n'
    body += '- `archive_id` (PK): TEXT\n'
    body += '- `slug`: TEXT\n'
    body += '- `subject`: TEXT\n'
    body += '- `stale_key`: TEXT\n'
    body += '- `stale_value`: TEXT\n'
    body += '- `status`: TEXT\n\n'

    return _build_markdown(fm, body)


def _build_markdown(frontmatter: dict, body: str) -> str:
    """Build markdown with YAML frontmatter."""
    fm_str = yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True)
    return f'---\n{fm_str}---\n{body}'


def write(out_dir: str = 'doc_synthetic/corpus') -> None:
    """Delete corpus directory and write all rendered docs."""
    import sys
    sys.path.insert(0, 'doc_synthetic')
    import brightmart_seed

    # Remove corpus directory if it exists
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir, ignore_errors=True)

    # Connect and render
    conn = brightmart_seed.connect()
    docs = render(conn)
    conn.close()

    # Write all docs
    for path, content in docs.items():
        # Remove leading slash and build full path
        rel_path = path.lstrip('/')
        full_path = os.path.join(out_dir, rel_path)

        # Create parent directories
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        # Write file with UTF-8, Unix line endings
        with open(full_path, 'w', encoding='utf-8', newline='\n') as f:
            f.write(content)


if __name__ == '__main__':
    write()
    print('Brightmart corpus written')

    # Count docs
    import sys
    sys.path.insert(0, 'doc_synthetic')
    import brightmart_seed
    conn = brightmart_seed.connect()
    docs = render(conn)
    print(f'Total docs: {len(docs)}')
    conn.close()
