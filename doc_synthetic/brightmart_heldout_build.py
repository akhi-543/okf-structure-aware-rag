"""
Brightmart held-out question set builder (v3 pilot, Task 1).

Builds 100 held-out questions (20 per stratum S1-S5) from the binding list in
docs/superpowers/specs/2026-09-24-brightmart-v3-design.md ("Held-out question
set" section). Gold (doc paths, spans, SQL, hierarchy paths) is computed from
the DB/render, never hand-typed, mirroring brightmart_questions_build.py's
(v2) patterns.

IDs: bm-h-s{n}-{nn}-t (template) / -p (paraphrase, added by add_paraphrases).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import brightmart_seed
import brightmart_render
import brightmart_validate as bv


# ---------------------------------------------------------------------------
# Binding list (spec "Held-out question set" section, IDs 01-20 per stratum)
# ---------------------------------------------------------------------------

S1_ITEMS = [
    {"id": "01", "text": "How many square feet does Brightmart Magnolia Row occupy?",
     "kind": "store_attr", "store": "S04", "attr": "sq_ft"},
    {"id": "02", "text": "How many square feet does Brightmart Cedar Park occupy?",
     "kind": "store_attr", "store": "S10", "attr": "sq_ft"},
    {"id": "03", "text": "In which city is Brightmart Riverside located?",
     "kind": "store_attr", "store": "S01", "attr": "city"},
    {"id": "04", "text": "In which city is Brightmart Palmetto Cross located?",
     "kind": "store_attr", "store": "S05", "attr": "city"},
    {"id": "05", "text": "In what year did Brightmart Riverdale open?",
     "kind": "store_attr", "store": "S02", "attr": "opened_year"},
    {"id": "06", "text": "In what year did Brightmart Cedar Falls open?",
     "kind": "store_attr", "store": "S07", "attr": "opened_year"},
    {"id": "07", "text": "Who manages the Bakery department at Brightmart Riverside?",
     "kind": "dept_manager", "store": "S01", "dept": "D08"},
    {"id": "08", "text": "Who manages the Electronics department at Brightmart Cedar Park?",
     "kind": "dept_manager", "store": "S10", "dept": "D03"},
    {"id": "09", "text": "What were Brightmart Magnolia Row's Pharmacy sales in Q1 2025?",
     "kind": "sales", "store": "S04", "dept": "D02", "quarter": 1},
    {"id": "10", "text": "What were Brightmart Prairie Hub's Bakery sales in Q3 2025?",
     "kind": "sales", "store": "S09", "dept": "D08", "quarter": 3},
    {"id": "11", "text": "What discount does the Fresh Friday promotion offer?",
     "kind": "promo_discount", "promo": "P04"},
    {"id": "12", "text": "When does the Winter Care promotion run?",
     "kind": "promo_dates", "promo": "P05"},
    {"id": "13", "text": "What discount does the Back to Class promotion offer?",
     "kind": "promo_discount", "promo": "P02"},
    {"id": "14", "text": "Does the Returns Policy require a receipt?",
     "kind": "policy_fact", "policy": "POL1", "key": "receipt_required"},
    {"id": "15", "text": "What are the weekday pharmacy hours under the Pharmacy Hours Policy?",
     "kind": "policy_fact", "policy": "POL2", "key": "weekday_hours"},
    {"id": "16", "text": "How much must a customer spend to qualify under the Fuel Rewards Policy?",
     "kind": "policy_fact", "policy": "POL3", "key": "spend_threshold_usd"},
    {"id": "17", "text": "In which country is Alder & Finch Textiles headquartered?",
     "kind": "supplier_country", "supplier": "SUP5"},
    {"id": "18", "text": "In which country is Greenfield Home Supply based?",
     "kind": "supplier_country", "supplier": "SUP4"},
    {"id": "19", "text": "What is the current operating status of Brightmart Bayou Gate?",
     "kind": "store_attr", "store": "S06", "attr": "status"},
    {"id": "20", "text": "Who is the regional manager of the Southeast region?",
     "kind": "region_manager", "region": "southeast"},
]

S2_ITEMS = [
    {"id": "01", "text": "In which country is the supplier of Dairy based?",
     "kind": "category_supplier", "category": "C02"},
    {"id": "02", "text": "In which country is the supplier of OTC Medicines based?",
     "kind": "category_supplier", "category": "C03"},
    {"id": "03", "text": "In which country is the supplier of Phones based?",
     "kind": "category_supplier", "category": "C05"},
    {"id": "04", "text": "In which country is the supplier of Artisan Bread based?",
     "kind": "category_supplier", "category": "C10"},
    {"id": "05", "text": "Which regional manager oversees Brightmart Riverdale?",
     "kind": "store_region", "store": "S02"},
    {"id": "06", "text": "Which regional manager oversees Brightmart Harbor Point?",
     "kind": "store_region", "store": "S03"},
    {"id": "07", "text": "Which regional manager oversees Brightmart Magnolia Row?",
     "kind": "store_region", "store": "S04"},
    {"id": "08", "text": "Which regional manager oversees Brightmart Bayou Gate?",
     "kind": "store_region", "store": "S06"},
    {"id": "09", "text": "Which regional manager oversees Brightmart Lakeshore?",
     "kind": "store_region", "store": "S08"},
    {"id": "10", "text": "Which regional manager oversees Brightmart Desert Bloom?",
     "kind": "store_region", "store": "S11"},
    {"id": "11", "text": "What discount does the promotion for the department that sells Dairy offer?",
     "kind": "category_promo", "category": "C02"},
    {"id": "12", "text": "What discount does the promotion for the department that sells Phones offer?",
     "kind": "category_promo", "category": "C05"},
    {"id": "13", "text": "Which supplier provides the category sold in the Pharmacy department?",
     "kind": "dept_category", "dept": "D02"},
    {"id": "14", "text": "Which supplier provides the category sold in the Home & Garden department?",
     "kind": "dept_category", "dept": "D04"},
    {"id": "15", "text": "Which supplier provides the category sold in the Apparel department?",
     "kind": "dept_category", "dept": "D05"},
    {"id": "16", "text": "Which department group does the department that sells Fresh Produce belong to?",
     "kind": "category_deptgroup", "category": "C01"},
    {"id": "17", "text": "Which department group does the department that sells Televisions belong to?",
     "kind": "category_deptgroup", "category": "C04"},
    {"id": "18", "text": "Which department group does the department that sells Patio Furniture belong to?",
     "kind": "category_deptgroup", "category": "C06"},
    {"id": "19", "text": "Which department group does the department that sells Motor Oil belong to?",
     "kind": "category_deptgroup", "category": "C09"},
    {"id": "20", "text": "Which department group does the department that sells Board Games belong to?",
     "kind": "category_deptgroup", "category": "C08"},
]

S3_ITEMS = [
    {"id": "01", "text": "Which stores have a pharmacy but no fuel station?",
     "sql": 'SELECT store_id FROM store WHERE has_pharmacy=? AND has_fuel=? ORDER BY store_id', "params": (1, 0)},
    {"id": "02", "text": "Which stores are not currently open?",
     "sql": 'SELECT store_id FROM store WHERE status!=? ORDER BY store_id', "params": ('open',)},
    {"id": "03", "text": "Which neighborhood stores have no pharmacy?",
     "sql": 'SELECT store_id FROM store WHERE format=? AND has_pharmacy=? ORDER BY store_id', "params": ('neighborhood', 0)},
    {"id": "04", "text": "Which stores opened between 2005 and 2012?",
     "sql": 'SELECT store_id FROM store WHERE opened_year BETWEEN ? AND ? ORDER BY store_id', "params": (2005, 2012)},
    {"id": "05", "text": "Which stores opened after 2018?",
     "sql": 'SELECT store_id FROM store WHERE opened_year>? ORDER BY store_id', "params": (2018,)},
    {"id": "06", "text": "Which stores are between 30,000 and 50,000 square feet?",
     "sql": 'SELECT store_id FROM store WHERE sq_ft BETWEEN ? AND ? ORDER BY store_id', "params": (30000, 50000)},
    {"id": "07", "text": "Which stores are larger than 180,000 square feet?",
     "sql": 'SELECT store_id FROM store WHERE sq_ft>? ORDER BY store_id', "params": (180000,)},
    {"id": "08", "text": "Which Southeast stores have a fuel station?",
     "sql": 'SELECT store_id FROM store WHERE region_id=? AND has_fuel=? ORDER BY store_id', "params": ('southeast', 1)},
    {"id": "09", "text": "Which Northeast stores have no fuel station?",
     "sql": 'SELECT store_id FROM store WHERE region_id=? AND has_fuel=? ORDER BY store_id', "params": ('northeast', 0)},
    {"id": "10", "text": "Which West stores are open?",
     "sql": 'SELECT store_id FROM store WHERE region_id=? AND status=? ORDER BY store_id', "params": ('west', 'open')},
    {"id": "11", "text": "Which Midwest stores have a pharmacy?",
     "sql": 'SELECT store_id FROM store WHERE region_id=? AND has_pharmacy=? ORDER BY store_id', "params": ('midwest', 1)},
    {"id": "12", "text": "Which open stores have neither a pharmacy nor a fuel station?",
     "sql": 'SELECT store_id FROM store WHERE status=? AND has_pharmacy=? AND has_fuel=? ORDER BY store_id', "params": ('open', 0, 0)},
    {"id": "13", "text": "Which supercenters opened before 2005?",
     "sql": 'SELECT store_id FROM store WHERE format=? AND opened_year<? ORDER BY store_id', "params": ('supercenter', 2005)},
    {"id": "14", "text": "Which promotions run at both supercenter and neighborhood stores?",
     "sql": "SELECT promo_id FROM promotion_format WHERE format IN ('supercenter','neighborhood') "
            "GROUP BY promo_id HAVING COUNT(DISTINCT format)=2 ORDER BY promo_id", "params": ()},
    {"id": "15", "text": "Which promotions offer a discount of less than 15%?",
     "sql": 'SELECT promo_id FROM promotion WHERE discount_pct<? ORDER BY promo_id', "params": (15,)},
    {"id": "16", "text": "Which promotions run during December 2025?",
     "sql": 'SELECT promo_id FROM promotion WHERE start<=? AND end>=? ORDER BY promo_id',
     "params": ('2025-12-31', '2025-12-01')},
    {"id": "17", "text": "Which categories are supplied by Northwind Provisions?",
     "sql": 'SELECT cat_id FROM category WHERE supplier_id=? ORDER BY cat_id', "params": ('SUP1',)},
    {"id": "18", "text": "Which categories are supplied by Greenfield Farms Co-op?",
     "sql": 'SELECT cat_id FROM category WHERE supplier_id=? ORDER BY cat_id', "params": ('SUP3',)},
    {"id": "19", "text": "Which categories are supplied by Kestrel Electronics?",
     "sql": 'SELECT cat_id FROM category WHERE supplier_id=? ORDER BY cat_id', "params": ('SUP2',)},
    {"id": "20", "text": "Which stores in the Southeast or Midwest regions are neighborhood stores?",
     "sql": 'SELECT store_id FROM store WHERE region_id IN (?,?) AND format=? ORDER BY store_id',
     "params": ('southeast', 'midwest', 'neighborhood')},
]

S4_ITEMS = [
    {"id": "01", "text": "Which categories are in the Grocery or Bakery departments?",
     "sql": 'SELECT cat_id FROM category WHERE dept_id IN (?,?) ORDER BY cat_id', "params": ('D01', 'D08'), "entity_type": "category"},
    {"id": "02", "text": "Which categories are in the Electronics or Toys departments?",
     "sql": 'SELECT cat_id FROM category WHERE dept_id IN (?,?) ORDER BY cat_id', "params": ('D03', 'D06'), "entity_type": "category"},
    {"id": "03", "text": "Which categories are in the Apparel, Automotive or Pharmacy departments?",
     "sql": 'SELECT cat_id FROM category WHERE dept_id IN (?,?,?) ORDER BY cat_id', "params": ('D05', 'D07', 'D02'), "entity_type": "category"},
    {"id": "04", "text": "Which categories are in the Home & Garden or Electronics departments?",
     "sql": 'SELECT cat_id FROM category WHERE dept_id IN (?,?) ORDER BY cat_id', "params": ('D04', 'D03'), "entity_type": "category"},
    {"id": "05", "text": "Which stores in the Northeast region are open?",
     "sql": 'SELECT store_id FROM store WHERE region_id=? AND status=? ORDER BY store_id', "params": ('northeast', 'open'), "entity_type": "store"},
    {"id": "06", "text": "Which stores in the Southeast region have a pharmacy?",
     "sql": 'SELECT store_id FROM store WHERE region_id=? AND has_pharmacy=? ORDER BY store_id', "params": ('southeast', 1), "entity_type": "store"},
    {"id": "07", "text": "Which stores in the West region have a fuel station?",
     "sql": 'SELECT store_id FROM store WHERE region_id=? AND has_fuel=? ORDER BY store_id', "params": ('west', 1), "entity_type": "store"},
    {"id": "08", "text": "Which stores in the Midwest region opened before 2015?",
     "sql": 'SELECT store_id FROM store WHERE region_id=? AND opened_year<? ORDER BY store_id', "params": ('midwest', 2015), "entity_type": "store"},
    {"id": "09", "text": "Which stores in the Northeast region are not express stores?",
     "sql": 'SELECT store_id FROM store WHERE region_id=? AND format!=? ORDER BY store_id', "params": ('northeast', 'express'), "entity_type": "store"},
    {"id": "10", "text": "Which stores in the West region opened after 2010?",
     "sql": 'SELECT store_id FROM store WHERE region_id=? AND opened_year>? ORDER BY store_id', "params": ('west', 2010), "entity_type": "store"},
    {"id": "11", "text": "Which stores are in the Northeast or West regions?",
     "sql": 'SELECT store_id FROM store WHERE region_id IN (?,?) ORDER BY store_id', "params": ('northeast', 'west'), "entity_type": "store"},
    {"id": "12", "text": "Which stores are in the Midwest or Southeast regions?",
     "sql": 'SELECT store_id FROM store WHERE region_id IN (?,?) ORDER BY store_id', "params": ('midwest', 'southeast'), "entity_type": "store"},
    {"id": "13", "text": "Which supercenters are in the Midwest or West regions?",
     "sql": 'SELECT store_id FROM store WHERE region_id IN (?,?) AND format=? ORDER BY store_id', "params": ('midwest', 'west', 'supercenter'), "entity_type": "store"},
    {"id": "14", "text": "Which neighborhood stores are in the Northeast or Southeast regions?",
     "sql": 'SELECT store_id FROM store WHERE region_id IN (?,?) AND format=? ORDER BY store_id', "params": ('northeast', 'southeast', 'neighborhood'), "entity_type": "store"},
    {"id": "15", "text": "Which express stores are in the Northeast or Midwest regions?",
     "sql": 'SELECT store_id FROM store WHERE region_id IN (?,?) AND format=? ORDER BY store_id', "params": ('northeast', 'midwest', 'express'), "entity_type": "store"},
    {"id": "16", "text": "Which stores with a pharmacy are in the West or Midwest regions?",
     "sql": 'SELECT store_id FROM store WHERE region_id IN (?,?) AND has_pharmacy=? ORDER BY store_id', "params": ('west', 'midwest', 1), "entity_type": "store"},
    {"id": "17", "text": "Which categories belong to departments in the health or food groups?",
     "sql": 'SELECT cat_id FROM category WHERE dept_id IN (SELECT dept_id FROM department WHERE "group" IN (?,?)) ORDER BY cat_id',
     "params": ('health', 'food'), "entity_type": "category"},
    {"id": "18", "text": "Which categories under general-group departments are supplied by Kestrel Electronics?",
     "sql": 'SELECT cat_id FROM category WHERE supplier_id=? AND dept_id IN (SELECT dept_id FROM department WHERE "group"=?) ORDER BY cat_id',
     "params": ('SUP2', 'general'), "entity_type": "category"},
    {"id": "19", "text": "Which stores in the Southeast region opened after 2005?",
     "sql": 'SELECT store_id FROM store WHERE region_id=? AND opened_year>? ORDER BY store_id', "params": ('southeast', 2005), "entity_type": "store"},
    {"id": "20", "text": "Which stores in the Northeast or Southeast regions have a fuel station?",
     "sql": 'SELECT store_id FROM store WHERE region_id IN (?,?) AND has_fuel=? ORDER BY store_id', "params": ('northeast', 'southeast', 1), "entity_type": "store"},
]

S5_ITEMS = [
    {"id": "01", "text": "What is the floor area of the Brightmart store in Chicago?", "absent_term": "Chicago"},
    {"id": "02", "text": "Who supplies the Frozen Desserts category?", "absent_term": "Frozen Desserts"},
    {"id": "03", "text": "Which supplier is based in Germany?", "absent_term": "Germany"},
    {"id": "04", "text": "When did the Brightmart store in Florida open?", "absent_term": "Florida"},
    {"id": "05", "text": "What were Brightmart Lakeshore's Bakery sales in 2023?", "absent_term": "2023"},
    {"id": "06", "text": "What is the fax number of Brightmart Riverdale?", "absent_term": "fax"},
    {"id": "07", "text": "What discount does the Summer Grill Fest promotion offer?", "absent_term": "Summer Grill Fest"},
    {"id": "08", "text": "Who manages the Northwest region?", "absent_term": "Northwest"},
    {"id": "09", "text": "How many parking spaces does Brightmart Cedar Park have?", "absent_term": "parking"},
    {"id": "10", "text": "Which store offers curbside pickup?", "absent_term": "curbside"},
    {"id": "11", "text": "Who is the chief executive officer of Brightmart?", "absent_term": "chief executive"},
    {"id": "12", "text": "Who manages the Garden Center department at Brightmart Riverside?", "absent_term": "Garden Center"},
    {"id": "13", "text": "What is the annual revenue of Kestrel Electronics?", "absent_term": "annual revenue"},
    {"id": "14", "text": "What is the street address of Brightmart Summit Ridge?", "absent_term": "street address"},
    {"id": "15", "text": "What discount does the Back to School Tech promotion offer?", "absent_term": "Back to School Tech"},
    {"id": "16", "text": "In which state is Brightmart Willow Creek located?", "absent_term": "Willow Creek"},
    {"id": "17", "text": "What are the pharmacy hours on Saturdays?", "absent_term": "Saturday"},
    {"id": "18", "text": "How many cents per gallon does Brightmart Riverside charge for diesel?", "absent_term": "diesel"},
    {"id": "19", "text": "Who supplies the Pet Food category?", "absent_term": "Pet Food"},
    {"id": "20", "text": "When is Brightmart Bayou Gate scheduled to reopen?", "absent_term": "reopen"},
]


# ---------------------------------------------------------------------------
# Span / path helpers (mirror brightmart_questions_build.py's patterns)
# ---------------------------------------------------------------------------

def find_span(doc_text, search_term, prefer_context=None):
    """Find first line containing search_term in doc text (skip frontmatter/Notes)."""
    if '---\n' in doc_text:
        parts = doc_text.split('---\n')
        if len(parts) >= 3:
            doc_text = parts[2]
    if '## Notes' in doc_text:
        doc_text = doc_text.split('## Notes')[0]
    lines_with_term = [l.strip() for l in doc_text.split('\n') if search_term in l.strip()]
    if prefer_context:
        for line in lines_with_term:
            if prefer_context in line:
                return line
    return lines_with_term[0] if lines_with_term else None


def find_s2_span(docs, doc_path, search_terms):
    """Find line in the doc containing one of search_terms (case-insensitive)."""
    doc_text = docs.get(f'/{doc_path}', '')
    if doc_text.count('---\n') >= 2:
        doc_text = doc_text.split('---\n', 2)[2]
    if '## Notes' in doc_text:
        doc_text = doc_text.split('## Notes')[0]
    for line in doc_text.split('\n'):
        line_stripped = line.strip()
        if not line_stripped or line_stripped.startswith('#'):
            continue
        line_lower = line_stripped.lower()
        for term in search_terms:
            if term.lower() in line_lower:
                return line_stripped
    return None


def region_path(region_id):
    return f'regions/region-{region_id}.md'


def policy_path(cursor, policy_id):
    """entity_id_to_doc_path has no policy branch ('POL...' also starts with 'P')."""
    slug = cursor.execute('SELECT slug FROM policy WHERE policy_id=?', (policy_id,)).fetchone()[0]
    return f'policies/policy-{slug}.md'


def display_name(cursor, entity_id):
    """Display name for an entity id (mirrors entity_id_to_doc_path's prefix dispatch)."""
    if entity_id.startswith('SUP'):
        row = cursor.execute('SELECT name FROM supplier WHERE supplier_id=?', (entity_id,)).fetchone()
    elif entity_id.startswith('S'):
        row = cursor.execute('SELECT name FROM store WHERE store_id=?', (entity_id,)).fetchone()
    elif entity_id.startswith('P'):
        row = cursor.execute('SELECT name FROM promotion WHERE promo_id=?', (entity_id,)).fetchone()
    elif entity_id.startswith('D'):
        row = cursor.execute('SELECT name FROM department WHERE dept_id=?', (entity_id,)).fetchone()
    elif entity_id.startswith('C'):
        row = cursor.execute('SELECT name FROM category WHERE cat_id=?', (entity_id,)).fetchone()
    else:
        row = None
    return row[0] if row else None


def get_quarterly_sales(cursor, store_id, dept_id):
    sales = cursor.execute(
        'SELECT week, sales FROM weekly_sales WHERE store_id = ? AND dept_id = ? ORDER BY week',
        (store_id, dept_id)
    ).fetchall()
    sales_by_week = {w: s for w, s in sales}
    quarters = []
    for start, end in [(1, 13), (14, 26), (27, 39), (40, 52)]:
        q_total = sum(sales_by_week.get(w, 0.0) for w in range(start, end + 1))
        quarters.append(q_total)
    return quarters


# ---------------------------------------------------------------------------
# build()
# ---------------------------------------------------------------------------

def _s1(conn, cursor, docs, item):
    qid = f"bm-h-s1-{item['id']}-t"
    kind = item['kind']
    gold_doc = None
    answer = None
    span = None

    if kind == 'store_attr':
        store = cursor.execute(
            'SELECT store_id, region_id, name, city, opened_year, sq_ft, status '
            'FROM store WHERE store_id=?', (item['store'],)
        ).fetchone()
        store_id, region_id, name, city, opened_year, sq_ft, status = store
        gold_doc = bv.entity_id_to_doc_path(store_id, conn)
        doc_text = docs[f'/{gold_doc}']
        attr = item['attr']
        if attr == 'sq_ft':
            answer = f'{sq_ft:,} square feet'
            span = find_span(doc_text, f'{sq_ft:,}')
        elif attr == 'city':
            answer = city
            span = find_span(doc_text, city)
        elif attr == 'opened_year':
            answer = str(opened_year)
            span = find_span(doc_text, answer)
        elif attr == 'status':
            answer = status
            span = find_span(doc_text, status)
            if span is None:
                # The rng-selected body sentence for this status value doesn't
                # always literally contain the status word (e.g. 'closed' can
                # render as "no longer in operation."). Frontmatter always has
                # the literal `store_status: {status}` line; fall back to it.
                for line in doc_text.split('\n'):
                    if f'store_status: {status}' in line.strip():
                        span = line.strip()
                        break

    elif kind == 'dept_manager':
        gold_doc = bv.entity_id_to_doc_path(item['store'], conn)
        doc_text = docs[f'/{gold_doc}']
        manager = cursor.execute(
            'SELECT dept_manager FROM store_department WHERE store_id=? AND dept_id=?',
            (item['store'], item['dept'])
        ).fetchone()
        if manager:
            answer = manager[0]
            span = find_span(doc_text, answer)

    elif kind == 'sales':
        gold_doc = bv.entity_id_to_doc_path(item['store'], conn)
        doc_text = docs[f'/{gold_doc}']
        dept_name = cursor.execute('SELECT name FROM department WHERE dept_id=?', (item['dept'],)).fetchone()[0]
        quarters = get_quarterly_sales(cursor, item['store'], item['dept'])
        amount = quarters[item['quarter'] - 1]
        answer = f'{amount:,.2f}'
        for line in doc_text.split('\n'):
            if dept_name in line and '|' in line and answer in line:
                span = line.strip()
                break

    elif kind == 'promo_discount':
        gold_doc = bv.entity_id_to_doc_path(item['promo'], conn)
        doc_text = docs[f'/{gold_doc}']
        discount = cursor.execute('SELECT discount_pct FROM promotion WHERE promo_id=?', (item['promo'],)).fetchone()[0]
        answer = f'{discount}%'
        span = find_span(doc_text, str(discount))

    elif kind == 'promo_dates':
        gold_doc = bv.entity_id_to_doc_path(item['promo'], conn)
        doc_text = docs[f'/{gold_doc}']
        start, end = cursor.execute('SELECT start, end FROM promotion WHERE promo_id=?', (item['promo'],)).fetchone()
        answer = f'{start} to {end}'
        span = find_span(doc_text, start)

    elif kind == 'policy_fact':
        gold_doc = policy_path(cursor, item['policy'])
        doc_text = docs[f'/{gold_doc}']
        value = cursor.execute(
            'SELECT value FROM policy_fact WHERE policy_id=? AND key=?', (item['policy'], item['key'])
        ).fetchone()[0]
        if item['key'] == 'receipt_required':
            # The rng-selected sentence states the fact in prose (e.g. "A
            # receipt must be presented...") and never literally contains the
            # raw DB value ('yes'/'no'); use the sentence itself as the
            # answer so it's guaranteed to be a substring of its own span.
            span = find_span(doc_text, 'receipt')
            answer = span
        elif item['key'] == 'weekday_hours':
            answer = value
            span = find_span(doc_text, value)
        elif item['key'] == 'spend_threshold_usd':
            answer = f'${value}'
            span = find_span(doc_text, f'${value}')

    elif kind == 'supplier_country':
        gold_doc = bv.entity_id_to_doc_path(item['supplier'], conn)
        doc_text = docs[f'/{gold_doc}']
        country = cursor.execute('SELECT country FROM supplier WHERE supplier_id=?', (item['supplier'],)).fetchone()[0]
        answer = country
        span = find_span(doc_text, country)

    elif kind == 'region_manager':
        gold_doc = region_path(item['region'])
        doc_text = docs[f'/{gold_doc}']
        manager = cursor.execute('SELECT manager FROM region WHERE region_id=?', (item['region'],)).fetchone()[0]
        answer = manager
        span = find_span(doc_text, manager)

    if not (span and answer and gold_doc):
        raise ValueError(f'{qid}: could not compute gold (span={span!r}, answer={answer!r}, doc={gold_doc!r})')

    question = {
        'corpus': 'synthetic_retail_pilot',
        'origin': 'synthetic_template',
        'query_id': qid,
        'reference_answer': answer,
        'reference_claim': None,
        'source_paths': [gold_doc],
        'stratum': 'S1',
        'text': item['text'],
    }
    qrels = [{'doc_path': gold_doc, 'query_id': qid, 'relevance': 1}]
    spans = [{'doc_path': gold_doc, 'query_id': qid, 'span': span}]
    return question, qrels, spans


def _s2(conn, cursor, docs, item):
    qid = f"bm-h-s2-{item['id']}-t"
    kind = item['kind']
    gold_docs = []
    answer = None
    spans_by_doc = {}

    if kind == 'category_supplier':
        cat_id = item['category']
        supplier_id = cursor.execute('SELECT supplier_id FROM category WHERE cat_id=?', (cat_id,)).fetchone()[0]
        supplier_name, country = cursor.execute(
            'SELECT name, country FROM supplier WHERE supplier_id=?', (supplier_id,)
        ).fetchone()
        answer = country
        gold_docs = [bv.entity_id_to_doc_path(cat_id, conn), bv.entity_id_to_doc_path(supplier_id, conn)]
        span = find_s2_span(docs, gold_docs[0], [supplier_name])
        if span:
            spans_by_doc[gold_docs[0]] = span
        span = find_s2_span(docs, gold_docs[1], [answer])
        if span:
            spans_by_doc[gold_docs[1]] = span

    elif kind == 'store_region':
        store_id = item['store']
        region_id = cursor.execute('SELECT region_id FROM store WHERE store_id=?', (store_id,)).fetchone()[0]
        region_name, manager = cursor.execute(
            'SELECT name, manager FROM region WHERE region_id=?', (region_id,)
        ).fetchone()
        answer = manager
        gold_docs = [bv.entity_id_to_doc_path(store_id, conn), region_path(region_id)]
        span = find_s2_span(docs, gold_docs[0], [region_name, f'[Back to {region_name}]'])
        if span:
            spans_by_doc[gold_docs[0]] = span
        span = find_s2_span(docs, gold_docs[1], [answer])
        if span:
            spans_by_doc[gold_docs[1]] = span

    elif kind == 'category_promo':
        cat_id = item['category']
        dept_id, = cursor.execute('SELECT dept_id FROM category WHERE cat_id=?', (cat_id,)).fetchone()
        dept_name = cursor.execute('SELECT name FROM department WHERE dept_id=?', (dept_id,)).fetchone()[0]
        promo_row = cursor.execute('SELECT promo_id, discount_pct FROM promotion WHERE dept_id=?', (dept_id,)).fetchone()
        if promo_row:
            promo_id, discount = promo_row
            answer = f'{discount}%'
            gold_docs = [bv.entity_id_to_doc_path(cat_id, conn), bv.entity_id_to_doc_path(promo_id, conn)]
            span = find_s2_span(docs, gold_docs[0], [dept_name])
            if span:
                spans_by_doc[gold_docs[0]] = span
            span = find_s2_span(docs, gold_docs[1], [f'{discount}% off'])
            if span:
                spans_by_doc[gold_docs[1]] = span

    elif kind == 'dept_category':
        dept_id = item['dept']
        cat_row = cursor.execute('SELECT cat_id, name, supplier_id FROM category WHERE dept_id=? ORDER BY cat_id', (dept_id,)).fetchone()
        if cat_row:
            cat_id, cat_name, supplier_id = cat_row
            supplier_name = cursor.execute('SELECT name FROM supplier WHERE supplier_id=?', (supplier_id,)).fetchone()[0]
            answer = supplier_name
            gold_docs = [bv.entity_id_to_doc_path(dept_id, conn), bv.entity_id_to_doc_path(cat_id, conn)]
            span = find_s2_span(docs, gold_docs[0], [cat_name])
            if span:
                spans_by_doc[gold_docs[0]] = span
            span = find_s2_span(docs, gold_docs[1], [supplier_name])
            if span:
                spans_by_doc[gold_docs[1]] = span

    elif kind == 'category_deptgroup':
        cat_id = item['category']
        dept_id, = cursor.execute('SELECT dept_id FROM category WHERE cat_id=?', (cat_id,)).fetchone()
        dept_name, group = cursor.execute('SELECT name, "group" FROM department WHERE dept_id=?', (dept_id,)).fetchone()
        answer = group
        gold_docs = [bv.entity_id_to_doc_path(cat_id, conn), bv.entity_id_to_doc_path(dept_id, conn)]
        span = find_s2_span(docs, gold_docs[0], [dept_name])
        if span:
            spans_by_doc[gold_docs[0]] = span
        span = find_s2_span(docs, gold_docs[1], [group])
        if span:
            spans_by_doc[gold_docs[1]] = span

    if not (len(gold_docs) == 2 and answer):
        raise ValueError(f'{qid}: could not compute gold (gold_docs={gold_docs!r}, answer={answer!r})')

    question = {
        'corpus': 'synthetic_retail_pilot',
        'origin': 'synthetic_template',
        'query_id': qid,
        'reference_answer': answer,
        'reference_claim': None,
        'source_paths': gold_docs,
        'stratum': 'S2',
        'text': item['text'],
    }
    qrels = []
    spans = []
    for gold_doc in gold_docs:
        qrels.append({'doc_path': gold_doc, 'query_id': qid, 'relevance': 1})
        if gold_doc in spans_by_doc:
            spans.append({'doc_path': gold_doc, 'query_id': qid, 'span': spans_by_doc[gold_doc]})
    return question, qrels, spans


def _run_set(cursor, sql, params):
    """Execute SQL, return list of (entity_id) in query order."""
    return [row[0] for row in cursor.execute(sql, params).fetchall()]


def _set_question(stratum, item, conn, cursor, docs):
    """Shared S3/S4 logic: run the set SQL, resolve entities to doc paths/display
    names, build the question dict, qrels, and per-doc first-line spans. Does
    NOT append the gold_sql span record - callers append that themselves (S4
    appends it after its own hierarchy path records, to preserve S4's original
    span order: per-doc spans, then path records, then gold_sql)."""
    qid = f"bm-h-{stratum.lower()}-{item['id']}-t"
    entity_ids = _run_set(cursor, item['sql'], item['params'])
    gold_docs = []
    display_names = []
    for entity_id in entity_ids:
        path = bv.entity_id_to_doc_path(entity_id, conn)
        name = display_name(cursor, entity_id)
        if path and name:
            gold_docs.append(path)
            display_names.append(name)

    if len(gold_docs) < 2:
        raise ValueError(f'{qid}: expected >=2 gold docs, got {gold_docs!r}')

    answer = ', '.join(display_names)
    question = {
        'corpus': 'synthetic_retail_pilot',
        'origin': 'synthetic_template',
        'query_id': qid,
        'reference_answer': answer,
        'reference_claim': None,
        'source_paths': gold_docs,
        'stratum': stratum,
        'text': item['text'],
    }
    qrels = []
    spans = []
    for gold_doc in gold_docs:
        qrels.append({'doc_path': gold_doc, 'query_id': qid, 'relevance': 1})
        doc_text = docs.get(f'/{gold_doc}', '')
        if '## Notes' in doc_text:
            doc_text = doc_text.split('## Notes')[0]
        for line in doc_text.split('\n'):
            if line.strip() and not line.startswith('#'):
                spans.append({'doc_path': gold_doc, 'query_id': qid, 'span': line.strip()})
                break
    return question, qrels, spans


def _s3(conn, cursor, docs, item):
    qid = f"bm-h-s3-{item['id']}-t"
    question, qrels, spans = _set_question('S3', item, conn, cursor, docs)
    spans.append({'query_id': qid, 'doc_path': None, 'gold_sql': item['sql'], 'gold_sql_params': list(item['params'])})
    return question, qrels, spans


def _s4(conn, cursor, docs, item):
    qid = f"bm-h-s4-{item['id']}-t"
    question, qrels, spans = _set_question('S4', item, conn, cursor, docs)

    # Hierarchy path records: parent = region (stores) or department (categories),
    # computed from the actual result rows (never hand-typed).
    entity_ids = _run_set(cursor, item['sql'], item['params'])
    entity_type = item['entity_type']
    placeholders = ','.join('?' for _ in entity_ids)
    if entity_type == 'store':
        parent_ids = sorted({row[0] for row in cursor.execute(
            f'SELECT DISTINCT region_id FROM store WHERE store_id IN ({placeholders})', entity_ids
        ).fetchall()})
        parent_paths = [region_path(pid) for pid in parent_ids]
    else:
        parent_rows = cursor.execute(
            f'SELECT DISTINCT dept_id FROM category WHERE cat_id IN ({placeholders})', entity_ids
        ).fetchall()
        parent_ids = sorted({row[0] for row in parent_rows})
        parent_paths = [bv.entity_id_to_doc_path(pid, conn) for pid in parent_ids]

    for parent_path in parent_paths:
        spans.append({'query_id': qid, 'doc_path': None, 'path': ['brightmart-home.md', parent_path]})

    spans.append({'query_id': qid, 'doc_path': None, 'gold_sql': item['sql'], 'gold_sql_params': list(item['params'])})
    return question, qrels, spans


def _s5(conn, cursor, docs, item):
    qid = f"bm-h-s5-{item['id']}-t"
    absent_term = item['absent_term']
    all_text = '\n'.join(docs.values()).lower()
    if absent_term.lower() in all_text:
        raise ValueError(f'{qid}: absent_term {absent_term!r} found in corpus')

    question = {
        'corpus': 'synthetic_retail_pilot',
        'origin': 'synthetic_template',
        'query_id': qid,
        'reference_answer': None,
        'reference_claim': None,
        'source_paths': [],
        'stratum': 'S5',
        'text': item['text'],
    }
    spans = [{'absent_term': absent_term, 'query_id': qid}]
    return question, [], spans


def build(conn):
    """Build the 100 held-out template ('-t') questions/qrels/gold_spans."""
    cursor = conn.cursor()
    docs = brightmart_render.render(conn)

    questions = []
    qrels = []
    gold_spans = []

    for item in S1_ITEMS:
        q, r, s = _s1(conn, cursor, docs, item)
        questions.append(q)
        qrels.extend(r)
        gold_spans.extend(s)

    for item in S2_ITEMS:
        q, r, s = _s2(conn, cursor, docs, item)
        questions.append(q)
        qrels.extend(r)
        gold_spans.extend(s)

    for item in S3_ITEMS:
        q, r, s = _s3(conn, cursor, docs, item)
        questions.append(q)
        qrels.extend(r)
        gold_spans.extend(s)

    for item in S4_ITEMS:
        q, r, s = _s4(conn, cursor, docs, item)
        questions.append(q)
        qrels.extend(r)
        gold_spans.extend(s)

    for item in S5_ITEMS:
        q, r, s = _s5(conn, cursor, docs, item)
        questions.append(q)
        qrels.extend(r)
        gold_spans.extend(s)

    return questions, qrels, gold_spans


def add_paraphrases(questions, qrels, spans, paraphrases: dict):
    """Add -p paraphrase records for every -t question with a matching paraphrase.

    Copies gold (qrels, spans, gold_sql, path records) to the -p id unchanged.
    Returns combined (questions, qrels, spans) including both -t and -p records.
    """
    para_questions = []
    para_qrels = []
    para_spans = []

    for q in questions:
        qid = q['query_id']
        if not qid.endswith('-t'):
            continue
        base_id = qid[:-2]
        if base_id not in paraphrases:
            continue

        para_q = dict(q)
        para_qid = base_id + '-p'
        para_q['query_id'] = para_qid
        para_q['origin'] = 'synthetic_paraphrase'
        para_q['text'] = paraphrases[base_id]
        para_questions.append(para_q)

        for qrel in qrels:
            if qrel['query_id'] == qid:
                para_qrel = dict(qrel)
                para_qrel['query_id'] = para_qid
                para_qrels.append(para_qrel)

        for span in spans:
            if span['query_id'] == qid:
                para_span = dict(span)
                para_span['query_id'] = para_qid
                para_spans.append(para_span)

    return questions + para_questions, qrels + para_qrels, spans + para_spans


def v2_overlap(texts, v2_texts):
    """Return the subset of `texts` that match (whitespace/case-insensitive) any v2 text."""
    norm_v2 = {' '.join(t.lower().split()) for t in v2_texts}
    return [t for t in texts if ' '.join(t.lower().split()) in norm_v2]


if __name__ == '__main__':
    conn = brightmart_seed.connect()
    questions, qrels, gold_spans = build(conn)
    conn.close()

    paraphrases_path = 'doc_synthetic/brightmart_heldout_paraphrases.json'
    if os.path.exists(paraphrases_path):
        with open(paraphrases_path, 'r', encoding='utf-8') as f:
            paraphrases = json.load(f)
        questions, qrels, gold_spans = add_paraphrases(questions, qrels, gold_spans, paraphrases)

    with open('doc_synthetic/brightmart_questions.jsonl', 'r', encoding='utf-8') as f:
        v2_texts = [json.loads(l)['text'] for l in f if l.strip()]

    overlap = v2_overlap([q['text'] for q in questions], v2_texts)
    if overlap:
        print('ERROR: held-out text(s) overlap the v2 question set:')
        for t in overlap:
            print(f'  {t!r}')
        sys.exit(1)

    def _write(path, records):
        with open(path, 'w', encoding='utf-8', newline='\n') as f:
            for r in sorted(records, key=lambda x: x['query_id']):
                f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n')

    _write('doc_synthetic/brightmart_heldout_questions.jsonl', questions)
    _write('doc_synthetic/brightmart_heldout_qrels.jsonl', qrels)
    _write('doc_synthetic/brightmart_heldout_gold_spans.jsonl', gold_spans)

    print(f'Wrote {len(questions)} held-out questions')
    for n in range(1, 6):
        stratum = f'S{n}'
        print(f'  {stratum}: {len([q for q in questions if q["stratum"] == stratum])}')
