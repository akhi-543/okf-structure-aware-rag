"""
Brightmart synthetic questions generator.
Builds 5 strata (S1-S5) x 20 questions each = 100 total from binding list.
Answers computed from SQL + rendered corpus, never hand-typed.

D4 Note: S3/S4 reference_answer entries now carry gold_sql field (stored on
first span record with doc_path=null, query_id=qid) so validator can recompute.
"""

import json
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import brightmart_seed
import brightmart_render


# Binding list questions (from task-5-question-list.md)
S1_QUESTIONS = [
    ('01', 'How many square feet does Brightmart Riverside occupy?', 'S01', None, None),
    ('02', 'How many square feet does Brightmart Riverdale occupy?', 'S02', None, None),
    ('03', 'What is the current floor area of Brightmart Cedar Falls?', 'S07', None, None),
    ('04', 'In which city is Brightmart Cedar Park located?', 'S10', None, None),
    ('05', 'In which city is Brightmart Cedar Falls located?', 'S07', None, None),
    ('06', 'In what year did Brightmart Harbor Point open?', 'S03', None, None),
    ('07', 'In what year did Brightmart Bayou Gate open?', 'S06', None, None),
    ('08', 'Who manages the Grocery department at Brightmart Magnolia Row?', 'S04', 'D01', None),
    ('09', 'Who manages the Pharmacy department at Brightmart Lakeshore?', 'S08', 'D02', None),
    ('10', 'What were Brightmart Riverside\'s Electronics sales in Q2 2025?', 'S01', 'D03', 2),
    ('11', 'What were Brightmart Desert Bloom\'s Grocery sales in Q4 2025?', 'S11', 'D01', 4),
    ('12', 'What discount does the Spring Garden Days promotion offer?', 'P01', None, None),
    ('13', 'When does the Back to Class promotion run?', 'P02', None, None),
    ('14', 'What discount does the Big Screen Weekend promotion offer?', 'P03', None, None),
    ('15', 'How many days do customers have to return most items under the Returns Policy?', 'POL1', None, None),
    ('16', 'How long is the return window for electronics under the Returns Policy?', 'POL1', None, None),
    ('17', 'What are the Sunday pharmacy hours under the Pharmacy Hours Policy?', 'POL2', None, None),
    ('18', 'How many cents off per gallon does the Fuel Rewards Policy give?', 'POL3', None, None),
    ('19', 'In which country is Kestrel Electronics headquartered?', 'SUP2', None, None),
    ('20', 'What is the current operating status of Brightmart Summit Ridge?', 'S12', None, None),
]

S2_QUESTIONS = [
    ('01', 'In which country is the supplier of Fresh Produce based?', 'C01', 'category_supplier'),
    ('02', 'In which country is the supplier of Televisions based?', 'C04', 'category_supplier'),
    ('03', 'In which country is the supplier of Patio Furniture based?', 'C06', 'category_supplier'),
    ('04', 'In which country is the supplier of Kids Apparel based?', 'C07', 'category_supplier'),
    ('05', 'In which country is the supplier of Motor Oil based?', 'C09', 'category_supplier'),
    ('06', 'In which country is the supplier of Board Games based?', 'C08', 'category_supplier'),
    ('07', 'Who is the regional manager responsible for Brightmart Riverside?', 'S01', 'store_region'),
    ('08', 'Who is the regional manager responsible for Brightmart Palmetto Cross?', 'S05', 'store_region'),
    ('09', 'Who is the regional manager responsible for Brightmart Cedar Falls?', 'S07', 'store_region'),
    ('10', 'Who is the regional manager responsible for Brightmart Prairie Hub?', 'S09', 'store_region'),
    ('11', 'Who is the regional manager responsible for Brightmart Cedar Park?', 'S10', 'store_region'),
    ('12', 'Who is the regional manager responsible for Brightmart Summit Ridge?', 'S12', 'store_region'),
    ('13', 'What discount does the promotion for the department that sells Fresh Produce offer?', 'C01', 'category_promo'),
    ('14', 'What discount does the promotion for the department that sells Televisions offer?', 'C04', 'category_promo'),
    ('15', 'What discount does the promotion for the department that sells Patio Furniture offer?', 'C06', 'category_promo'),
    ('16', 'What discount does the promotion for the department that sells Kids Apparel offer?', 'C07', 'category_promo'),
    ('17', 'What discount does the promotion for the department that sells OTC Medicines offer?', 'C03', 'category_promo'),
    ('18', 'Which supplier provides the category sold in the Automotive department?', 'D07', 'dept_category'),
    ('19', 'Which supplier provides the category sold in the Toys department?', 'D06', 'dept_category'),
    ('20', 'Which supplier provides the category sold in the Bakery department?', 'D08', 'dept_category'),
]

S3_QUESTIONS = [
    ('01', 'Which stores are supercenters with a pharmacy?', 'SELECT store_id FROM store WHERE format=? AND has_pharmacy=? ORDER BY store_id', ('supercenter', 1)),
    ('02', 'Which neighborhood stores have a pharmacy?', 'SELECT store_id FROM store WHERE format=? AND has_pharmacy=? ORDER BY store_id', ('neighborhood', 1)),
    ('03', 'Which stores have a fuel station but no pharmacy?', 'SELECT store_id FROM store WHERE has_fuel=? AND has_pharmacy=? ORDER BY store_id', (1, 0)),
    ('04', 'Which stores have neither a pharmacy nor a fuel station?', 'SELECT store_id FROM store WHERE has_pharmacy=? AND has_fuel=? ORDER BY store_id', (0, 0)),
    ('05', 'Which express stores have a fuel station?', 'SELECT store_id FROM store WHERE format=? AND has_fuel=? ORDER BY store_id', ('express', 1)),
    ('06', 'Which stores are currently remodeling?', 'SELECT store_id FROM store WHERE status=? ORDER BY store_id', ('remodeling',)),
    ('07', 'Which Midwest stores have a fuel station?', 'SELECT store_id FROM store WHERE region_id=? AND has_fuel=? ORDER BY store_id', ('midwest', 1)),
    ('08', 'Which West stores have a fuel station?', 'SELECT store_id FROM store WHERE region_id=? AND has_fuel=? ORDER BY store_id', ('west', 1)),
    ('09', 'Which Northeast stores have a pharmacy?', 'SELECT store_id FROM store WHERE region_id=? AND has_pharmacy=? ORDER BY store_id', ('northeast', 1)),
    ('10', 'Which stores opened before 2010?', 'SELECT store_id FROM store WHERE opened_year<? ORDER BY store_id', (2010,)),
    ('11', 'Which stores opened in 2015 or later?', 'SELECT store_id FROM store WHERE opened_year>=? ORDER BY store_id', (2015,)),
    ('12', 'Which stores are larger than 100,000 square feet?', 'SELECT store_id FROM store WHERE sq_ft>? ORDER BY store_id', (100000,)),
    ('13', 'Which stores are smaller than 20,000 square feet?', 'SELECT store_id FROM store WHERE sq_ft<? ORDER BY store_id', (20000,)),
    ('14', 'Which promotions run at neighborhood stores?', 'SELECT DISTINCT promo_id FROM promotion_format WHERE format=? ORDER BY promo_id', ('neighborhood',)),
    ('15', 'Which promotions are limited to supercenter stores?', 'SELECT promo_id FROM promotion WHERE promo_id NOT IN (SELECT promo_id FROM promotion_format WHERE format!=?) ORDER BY promo_id', ('supercenter',)),
    ('16', 'Which promotions offer a discount of 15% or more?', 'SELECT promo_id FROM promotion WHERE discount_pct>=? ORDER BY promo_id', (15,)),
    ('17', 'Which suppliers are based in the United States?', 'SELECT supplier_id FROM supplier WHERE country=? ORDER BY supplier_id', ('United States',)),
    ('18', 'Which suppliers are based outside the United States?', 'SELECT supplier_id FROM supplier WHERE country!=? ORDER BY supplier_id', ('United States',)),
    ('19', 'Which departments belong to the food group?', 'SELECT dept_id FROM department WHERE "group"=? ORDER BY dept_id', ('food',)),
    ('20', 'Which open supercenters have a fuel station?', 'SELECT store_id FROM store WHERE status=? AND format=? AND has_fuel=? ORDER BY store_id', ('open', 'supercenter', 1)),
]

S4_QUESTIONS = [
    ('01', 'Which stores belong to the Northeast region?', 'northeast', 'region'),
    ('02', 'Which stores belong to the Southeast region?', 'southeast', 'region'),
    ('03', 'Which stores belong to the Midwest region?', 'midwest', 'region'),
    ('04', 'Which stores belong to the West region?', 'west', 'region'),
    ('05', 'Which categories are in the Grocery department?', 'D01', 'department'),
    ('06', 'Which categories are in the Electronics department?', 'D03', 'department'),
    ('07', 'Which categories fall under food-group departments?', 'food', 'group'),
    ('08', 'Which categories fall under general-group departments?', 'general', 'group'),
    ('09', 'Which stores in the Northeast region have a pharmacy?', 'northeast', 'region_pharmacy'),
    ('10', 'Which stores in the Southeast region are supercenters?', 'southeast', 'region_format'),
    ('11', 'Which stores in the Midwest region opened after 2010?', 'midwest', 'region_year'),
    ('12', 'Which stores in the West region are not supercenters?', 'west', 'region_not_supercenter'),
    ('13', 'Which categories are in the Home & Garden or Apparel departments?', ('D04', 'D05'), 'multi_department'),
    ('14', 'Which categories are in the Toys or Automotive departments?', ('D06', 'D07'), 'multi_department'),
    ('15', 'Which categories are in the Pharmacy or Bakery departments?', ('D02', 'D08'), 'multi_department'),
    ('16', 'Which stores are in the Northeast or Midwest regions?', ('northeast', 'midwest'), 'multi_region'),
    ('17', 'Which stores are in the Southeast or West regions?', ('southeast', 'west'), 'multi_region'),
    ('18', 'Which supercenters are in the Northeast or Southeast regions?', ('northeast', 'southeast'), 'multi_region_format'),
    ('19', 'Which express stores are in the Midwest or West regions?', ('midwest', 'west'), 'multi_region_format_express'),
    ('20', 'Which neighborhood stores are in the Midwest or West regions?', ('midwest', 'west'), 'multi_region_format_neighborhood'),
]

S5_QUESTIONS = [
    ('01', 'What is the floor area of the Brightmart store in Denver?', 'Denver'),
    ('02', 'Who supplies the Frozen Foods category?', 'Frozen Foods'),
    ('03', 'Which supplier is based in Japan?', 'Japan'),
    ('04', 'When did the Brightmart store in Texas open?', 'Texas'),
    ('05', 'What were Brightmart Riverside\'s Grocery sales in 2024?', '2024'),
    ('06', 'What is the phone number of Brightmart Lakeshore?', 'phone number'),
    ('07', 'What discount does the Holiday Toy Drive promotion offer?', 'Holiday Toy Drive'),
    ('08', 'Who manages the Southwest region?', 'Southwest'),
    ('09', 'How many loyalty points does a customer earn per dollar spent?', 'loyalty points'),
    ('10', 'Which store has a drive-through pharmacy?', 'drive-through'),
    ('11', 'What is the email address of the Kestrel Electronics account team?', 'email'),
    ('12', 'Who manages the Seafood department at Brightmart Magnolia Row?', 'Seafood'),
    ('13', 'How many employees work at Brightmart Cedar Park?', 'employees'),
    ('14', 'What is the ZIP code of Brightmart Harbor Point?', 'ZIP code'),
    ('15', 'What discount does the Pet Supplies promotion offer?', 'Pet Supplies'),
    ('16', 'In which state is Brightmart Oak Hollow located?', 'Oak Hollow'),
    ('17', 'What are the pharmacy hours on public holidays?', 'public holiday'),
    ('18', 'What is the price of a gallon of fuel at Brightmart Riverside?', 'fuel price'),
    ('19', 'Who supplies the Garden Tools category?', 'Garden Tools'),
    ('20', 'On what date did Brightmart Bayou Gate close?', 'closing date'),
]


def generate_questions(conn):
    """Generate all 100 questions (20 per stratum S1-S5)."""
    cursor = conn.cursor()
    docs = brightmart_render.render(conn)

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

    policies = {p[0]: p for p in cursor.execute(
        'SELECT policy_id, name, slug, applies_to FROM policy ORDER BY policy_id'
    ).fetchall()}

    policy_facts = {}
    for pf in cursor.execute('SELECT policy_id, key, value FROM policy_fact').fetchall():
        if pf[0] not in policy_facts:
            policy_facts[pf[0]] = {}
        policy_facts[pf[0]][pf[1]] = pf[2]

    store_depts = {}
    for sd in cursor.execute('SELECT store_id, dept_id, dept_manager FROM store_department').fetchall():
        if sd[0] not in store_depts:
            store_depts[sd[0]] = {}
        store_depts[sd[0]][sd[1]] = sd[2]

    # Helpers
    def get_quarterly_sales(store_id, dept_id):
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

    def find_span(doc_text, search_term, prefer_context=None):
        """Find first line containing search_term in doc text (ignoring frontmatter and Notes)."""
        # Skip frontmatter
        if '---\n' in doc_text:
            parts = doc_text.split('---\n')
            if len(parts) >= 3:
                doc_text = parts[2]
        # Skip Notes section
        if '## Notes' in doc_text:
            doc_text = doc_text.split('## Notes')[0]
        lines_with_term = []
        for line in doc_text.split('\n'):
            line_s = line.strip()
            if search_term in line_s:
                lines_with_term.append(line_s)

        if prefer_context and lines_with_term:
            for line in lines_with_term:
                if prefer_context in line:
                    return line

        return lines_with_term[0] if lines_with_term else None

    def store_path(store_id):
        s = stores[store_id]
        slug = s[2].replace('Brightmart ', '').lower().replace(' ', '-')
        return f'regions/{s[1]}/store-{store_id.lower()}-{slug}.md'

    def store_display_name(store_id):
        return stores[store_id][2]

    def promo_path(promo_id):
        return f'promotions/promo-{promos[promo_id][2]}.md'

    def policy_path(policy_id):
        return f'policies/policy-{policies[policy_id][2]}.md'

    def supplier_path(supplier_id):
        return f'suppliers/supplier-{suppliers[supplier_id][2]}.md'

    def supplier_display_name(supplier_id):
        return suppliers[supplier_id][1]

    def category_path(cat_id):
        cat = categories[cat_id]
        dept_slug = depts[cat[1]][2]
        return f'departments/{dept_slug}/category-{cat[3]}.md'

    def category_display_name(cat_id):
        return categories[cat_id][2]

    def dept_path(dept_id):
        return f'departments/dept-{depts[dept_id][2]}.md'

    def region_path(region_id):
        return f'regions/region-{region_id}.md'

    questions = []
    qrels = []
    gold_spans = []

    # S1: Direct fact questions
    for q_id, text, entity_id, dept_id, quarter in S1_QUESTIONS:
        query_id = f'bm-s1-{q_id}-t'
        gold_doc = None
        answer = None
        span = None

        if entity_id.startswith('SUP'):  # Supplier (before S to avoid collision)
            supplier = suppliers[entity_id]
            gold_doc = supplier_path(entity_id)
            doc_text = docs[f'/{gold_doc}']
            answer = supplier[3]
            span = find_span(doc_text, answer)

        elif entity_id.startswith('S'):  # Store
            store = stores[entity_id]
            gold_doc = store_path(entity_id)
            doc_text = docs[f'/{gold_doc}']

            if 'square feet' in text or 'floor area' in text:
                answer = f'{store[7]:,} square feet'
                # Search for both comma and non-comma versions
                span = find_span(doc_text, f'{store[7]:,}') or find_span(doc_text, str(store[7]))
            elif 'city' in text:
                answer = store[3]
                span = find_span(doc_text, store[3])
            elif 'year' in text or 'opened' in text:
                answer = str(store[6])
                span = find_span(doc_text, answer)
            elif 'status' in text:
                answer = store[10]
                span = find_span(doc_text, answer)
            elif dept_id and 'manages' in text:
                manager = store_depts.get(entity_id, {}).get(dept_id)
                if manager:
                    answer = manager
                    span = find_span(doc_text, manager)
            elif dept_id and quarter:
                quarters = get_quarterly_sales(entity_id, dept_id)
                dept_name = depts[dept_id][1]
                amount = quarters[quarter - 1]
                answer = f'{amount:,.2f}'
                for line in doc_text.split('\n'):
                    if dept_name in line and '|' in line and answer in line:
                        span = line.strip()
                        break

        elif entity_id.startswith('POL'):  # Policy (before P to avoid collision)
            policy = policies[entity_id]
            gold_doc = policy_path(entity_id)
            doc_text = docs[f'/{gold_doc}']
            facts = policy_facts.get(entity_id, {})

            text_lower = text.lower()
            context = None
            if 'return' in text_lower and 'electronics' in text_lower:
                answer = facts.get('electronics_window_days', '')
                context = 'electronics'
            elif 'return' in text_lower and 'days' in text_lower:
                answer = facts.get('standard_window_days', '')
                context = 'days'
            elif 'sunday' in text_lower:
                answer = facts.get('sunday_hours', '')
                context = 'sunday'
            elif 'cents' in text_lower:
                answer = facts.get('cents_off_per_gallon', '')
                context = 'gallon'

            if answer:
                span = find_span(doc_text, answer, context)

        elif entity_id.startswith('P'):  # Promotion
            promo = promos[entity_id]
            gold_doc = promo_path(entity_id)
            doc_text = docs[f'/{gold_doc}']

            if 'discount' in text:
                answer = f'{promo[6]}%'
                span = find_span(doc_text, str(promo[6]))
            elif 'run' in text or 'dates' in text:
                answer = f'{promo[4]} to {promo[5]}'
                span = find_span(doc_text, promo[4])

        if span and answer and gold_doc:
            questions.append({
                'corpus': 'synthetic_retail_pilot',
                'origin': 'synthetic_template',
                'query_id': query_id,
                'reference_answer': answer,
                'reference_claim': None,
                'source_paths': [gold_doc],
                'stratum': 'S1',
                'text': text,
            })
            qrels.append({'doc_path': gold_doc, 'query_id': query_id, 'relevance': 1})
            gold_spans.append({'doc_path': gold_doc, 'query_id': query_id, 'span': span})

    # E1: S2 span finding helper - find line containing the answer
    def find_s2_span(doc_path, search_terms):
        """Find span in S2 doc that contains one of the search terms (case-insensitive)."""
        doc_text = docs.get(f'/{doc_path}', '')
        if '---\n' in doc_text:
            # Skip frontmatter
            doc_text = doc_text.split('---\n', 2)[2] if doc_text.count('---\n') >= 2 else doc_text
        if '## Notes' in doc_text:
            doc_text = doc_text.split('## Notes')[0]

        # Search for lines containing any of the search terms
        for line in doc_text.split('\n'):
            line_stripped = line.strip()
            if not line_stripped or line_stripped.startswith('#'):
                continue
            line_lower = line_stripped.lower()
            for term in search_terms:
                if term.lower() in line_lower:
                    return line_stripped
        return None

    # S2: Entity linking (2 gold docs) — E1 fix: real body spans
    for q_id, text, entity_id, link_type in S2_QUESTIONS:
        query_id = f'bm-s2-{q_id}-t'
        gold_docs = []
        answer = None
        spans_by_doc = {}

        if link_type == 'category_supplier':
            # 01-06: category + supplier; answer = supplier country
            cat = categories[entity_id]
            supplier_id = cat[4]
            supplier = suppliers[supplier_id]
            answer = supplier[3]  # country
            cat_name = categories[entity_id][2]
            gold_docs = [category_path(entity_id), supplier_path(supplier_id)]

            # Category: look for supplier name; Supplier: look for country
            span = find_s2_span(gold_docs[0], [supplier[1]])
            if span:
                spans_by_doc[gold_docs[0]] = span
            span = find_s2_span(gold_docs[1], [answer])
            if span:
                spans_by_doc[gold_docs[1]] = span

        elif link_type == 'store_region':
            # 07-12: store + region; answer = region manager name
            store = stores[entity_id]
            region_id = store[1]
            region = regions[region_id]
            answer = region[2]  # manager
            region_name = regions[region_id][1]
            gold_docs = [store_path(entity_id), region_path(region_id)]

            # Store: look for region name or link; Region: look for manager
            span = find_s2_span(gold_docs[0], [region_name, f'[Back to {region_name}]'])
            if span:
                spans_by_doc[gold_docs[0]] = span
            span = find_s2_span(gold_docs[1], [answer])
            if span:
                spans_by_doc[gold_docs[1]] = span

        elif link_type == 'category_promo':
            # 13-17: category + promo; answer = promo discount
            cat = categories[entity_id]
            dept_id = cat[1]
            promo_id = next((p for p, promo in promos.items() if promo[3] == dept_id), None)
            if promo_id:
                promo = promos[promo_id]
                answer = f'{promo[6]}%'
                dept_name = depts[dept_id][1]
                gold_docs = [category_path(entity_id), promo_path(promo_id)]

                # Category: look for department name; Promo: look for "X% off"
                span = find_s2_span(gold_docs[0], [dept_name])
                if span:
                    spans_by_doc[gold_docs[0]] = span
                span = find_s2_span(gold_docs[1], [f'{promo[6]}% off'])
                if span:
                    spans_by_doc[gold_docs[1]] = span

        elif link_type == 'dept_category':
            # E1 fix: 18-20: dept + category; answer = supplier name
            cat_for_dept = next((c for c in categories.values() if c[1] == entity_id), None)
            if cat_for_dept:
                supplier_id = cat_for_dept[4]
                supplier = suppliers[supplier_id]
                answer = supplier[1]  # supplier name
                cat_name = cat_for_dept[2]
                gold_docs = [dept_path(entity_id), category_path(cat_for_dept[0])]

                # Dept: look for category name; Category: look for supplier name
                span = find_s2_span(gold_docs[0], [cat_name])
                if span:
                    spans_by_doc[gold_docs[0]] = span
                span = find_s2_span(gold_docs[1], [supplier[1]])
                if span:
                    spans_by_doc[gold_docs[1]] = span

        if len(gold_docs) == 2 and answer:
            questions.append({
                'corpus': 'synthetic_retail_pilot',
                'origin': 'synthetic_template',
                'query_id': query_id,
                'reference_answer': answer,
                'reference_claim': None,
                'source_paths': gold_docs,
                'stratum': 'S2',
                'text': text,
            })
            for gold_doc in gold_docs:
                qrels.append({'doc_path': gold_doc, 'query_id': query_id, 'relevance': 1})
                if gold_doc in spans_by_doc:
                    gold_spans.append({'doc_path': gold_doc, 'query_id': query_id, 'span': spans_by_doc[gold_doc]})

    # S3: Metadata set — D3 & D4 fixes
    for q_id, text, sql_query, params in S3_QUESTIONS:
        query_id = f'bm-s3-{q_id}-t'
        result = cursor.execute(sql_query, params).fetchall()
        gold_docs = []
        display_names = []

        if result:
            for row in result:
                entity_id = row[0]
                display_name = None

                if entity_id.startswith('SUP'):
                    gold_docs.append(supplier_path(entity_id))
                    display_name = supplier_display_name(entity_id)
                elif entity_id.startswith('S'):
                    gold_docs.append(store_path(entity_id))
                    display_name = store_display_name(entity_id)
                elif entity_id.startswith('P'):
                    gold_docs.append(promo_path(entity_id))
                    display_name = promos[entity_id][1]
                elif entity_id.startswith('D'):
                    gold_docs.append(dept_path(entity_id))
                    display_name = depts[entity_id][1]

                if display_name:
                    display_names.append(display_name)

        if len(gold_docs) >= 2:
            # D3 fix: use display names, not IDs
            answer = ', '.join(display_names)
            questions.append({
                'corpus': 'synthetic_retail_pilot',
                'origin': 'synthetic_template',
                'query_id': query_id,
                'reference_answer': answer,
                'reference_claim': None,
                'source_paths': gold_docs,
                'stratum': 'S3',
                'text': text,
            })
            for gold_doc in gold_docs:
                qrels.append({'doc_path': gold_doc, 'query_id': query_id, 'relevance': 1})
                doc_text = docs.get(f'/{gold_doc}', '')
                if '## Notes' in doc_text:
                    doc_text = doc_text.split('## Notes')[0]
                for line in doc_text.split('\n'):
                    if line.strip() and not line.startswith('#'):
                        gold_spans.append({'doc_path': gold_doc, 'query_id': query_id, 'span': line.strip()})
                        break

            # D4 fix: store gold_sql on a record with doc_path=null
            gold_spans.append({
                'query_id': query_id,
                'doc_path': None,
                'gold_sql': sql_query,
                'gold_sql_params': params,
            })

    # S4: Hierarchy — D3 & D4 fixes
    for q_id, text, parent_id, hierarchy_type in S4_QUESTIONS:
        query_id = f'bm-s4-{q_id}-t'
        gold_docs = []
        display_names = []
        sql_query = None
        sql_params = None

        if hierarchy_type == 'region':
            region_stores = [s for s in stores.values() if s[1] == parent_id]
            for store in sorted(region_stores, key=lambda x: x[0]):
                gold_docs.append(store_path(store[0]))
                display_names.append(store_display_name(store[0]))
            sql_query = 'SELECT store_id FROM store WHERE region_id=? ORDER BY store_id'
            sql_params = (parent_id,)

        elif hierarchy_type == 'department':
            dept_cats = [c for c in categories.values() if c[1] == parent_id]
            for cat in sorted(dept_cats, key=lambda x: x[0]):
                gold_docs.append(category_path(cat[0]))
                display_names.append(category_display_name(cat[0]))
            sql_query = 'SELECT cat_id FROM category WHERE dept_id=? ORDER BY cat_id'
            sql_params = (parent_id,)

        elif hierarchy_type == 'group':
            group_depts = [d for d in depts.values() if d[3] == parent_id]
            for dept in group_depts:
                dept_cats = [c for c in categories.values() if c[1] == dept[0]]
                for cat in sorted(dept_cats, key=lambda x: x[0]):
                    gold_docs.append(category_path(cat[0]))
                    display_names.append(category_display_name(cat[0]))
            sql_query = 'SELECT cat_id FROM category WHERE dept_id IN (SELECT dept_id FROM department WHERE "group"=?) ORDER BY cat_id'
            sql_params = (parent_id,)

        elif hierarchy_type == 'region_pharmacy':
            region_stores = [s for s in stores.values() if s[1] == parent_id and s[8]]
            for store in sorted(region_stores, key=lambda x: x[0]):
                gold_docs.append(store_path(store[0]))
                display_names.append(store_display_name(store[0]))
            sql_query = 'SELECT store_id FROM store WHERE region_id=? AND has_pharmacy=? ORDER BY store_id'
            sql_params = (parent_id, 1)

        elif hierarchy_type == 'region_format':
            region_stores = [s for s in stores.values() if s[1] == parent_id and s[5] == 'supercenter']
            for store in sorted(region_stores, key=lambda x: x[0]):
                gold_docs.append(store_path(store[0]))
                display_names.append(store_display_name(store[0]))
            sql_query = 'SELECT store_id FROM store WHERE region_id=? AND format=? ORDER BY store_id'
            sql_params = (parent_id, 'supercenter')

        elif hierarchy_type == 'region_year':
            region_stores = [s for s in stores.values() if s[1] == parent_id and s[6] > 2010]
            for store in sorted(region_stores, key=lambda x: x[0]):
                gold_docs.append(store_path(store[0]))
                display_names.append(store_display_name(store[0]))
            sql_query = 'SELECT store_id FROM store WHERE region_id=? AND opened_year>? ORDER BY store_id'
            sql_params = (parent_id, 2010)

        elif hierarchy_type == 'region_not_supercenter':
            region_stores = [s for s in stores.values() if s[1] == parent_id and s[5] != 'supercenter']
            for store in sorted(region_stores, key=lambda x: x[0]):
                gold_docs.append(store_path(store[0]))
                display_names.append(store_display_name(store[0]))
            sql_query = 'SELECT store_id FROM store WHERE region_id=? AND format!=? ORDER BY store_id'
            sql_params = (parent_id, 'supercenter')

        elif hierarchy_type == 'multi_department':
            for dept_id in parent_id:
                dept_cats = [c for c in categories.values() if c[1] == dept_id]
                for cat in sorted(dept_cats, key=lambda x: x[0]):
                    gold_docs.append(category_path(cat[0]))
                    display_names.append(category_display_name(cat[0]))
            sql_query = 'SELECT cat_id FROM category WHERE dept_id IN (' + ','.join(['?'] * len(parent_id)) + ') ORDER BY cat_id'
            sql_params = tuple(parent_id)

        elif hierarchy_type == 'multi_region':
            for region_id in parent_id:
                region_stores = [s for s in stores.values() if s[1] == region_id]
                for store in sorted(region_stores, key=lambda x: x[0]):
                    gold_docs.append(store_path(store[0]))
                    display_names.append(store_display_name(store[0]))
            sql_query = 'SELECT store_id FROM store WHERE region_id IN (' + ','.join(['?'] * len(parent_id)) + ') ORDER BY store_id'
            sql_params = tuple(parent_id)

        elif hierarchy_type == 'multi_region_format':
            for region_id in parent_id:
                region_stores = [s for s in stores.values() if s[1] == region_id and s[5] == 'supercenter']
                for store in sorted(region_stores, key=lambda x: x[0]):
                    gold_docs.append(store_path(store[0]))
                    display_names.append(store_display_name(store[0]))
            sql_query = 'SELECT store_id FROM store WHERE region_id IN (' + ','.join(['?'] * len(parent_id)) + ') AND format=? ORDER BY store_id'
            sql_params = tuple(list(parent_id) + ['supercenter'])

        elif hierarchy_type == 'multi_region_format_express':
            for region_id in parent_id:
                region_stores = [s for s in stores.values() if s[1] == region_id and s[5] == 'express']
                for store in sorted(region_stores, key=lambda x: x[0]):
                    gold_docs.append(store_path(store[0]))
                    display_names.append(store_display_name(store[0]))
            sql_query = 'SELECT store_id FROM store WHERE region_id IN (' + ','.join(['?'] * len(parent_id)) + ') AND format=? ORDER BY store_id'
            sql_params = tuple(list(parent_id) + ['express'])

        elif hierarchy_type == 'multi_region_format_neighborhood':
            for region_id in parent_id:
                region_stores = [s for s in stores.values() if s[1] == region_id and s[5] == 'neighborhood']
                for store in sorted(region_stores, key=lambda x: x[0]):
                    gold_docs.append(store_path(store[0]))
                    display_names.append(store_display_name(store[0]))
            sql_query = 'SELECT store_id FROM store WHERE region_id IN (' + ','.join(['?'] * len(parent_id)) + ') AND format=? ORDER BY store_id'
            sql_params = tuple(list(parent_id) + ['neighborhood'])

        if len(gold_docs) >= 2:
            # D3 fix: use display names, not IDs
            answer = ', '.join(display_names)
            questions.append({
                'corpus': 'synthetic_retail_pilot',
                'origin': 'synthetic_template',
                'query_id': query_id,
                'reference_answer': answer,
                'reference_claim': None,
                'source_paths': gold_docs,
                'stratum': 'S4',
                'text': text,
            })
            for gold_doc in gold_docs:
                qrels.append({'doc_path': gold_doc, 'query_id': query_id, 'relevance': 1})
                doc_text = docs.get(f'/{gold_doc}', '')
                if '## Notes' in doc_text:
                    doc_text = doc_text.split('## Notes')[0]
                for line in doc_text.split('\n'):
                    if line.strip() and not line.startswith('#'):
                        gold_spans.append({'doc_path': gold_doc, 'query_id': query_id, 'span': line.strip()})
                        break

            # E2 fix: store hierarchy path records
            if hierarchy_type in ['region', 'region_pharmacy', 'region_format', 'region_year', 'region_not_supercenter']:
                # Single parent region
                path = ['brightmart-home.md', f'regions/region-{parent_id}.md']
                gold_spans.append({'query_id': query_id, 'doc_path': None, 'path': path})
            elif hierarchy_type in ['department']:
                # Single parent department
                path = ['brightmart-home.md', f'departments/dept-{depts[parent_id][2]}.md']
                gold_spans.append({'query_id': query_id, 'doc_path': None, 'path': path})
            elif hierarchy_type == 'group':
                # Group: one path per parent department
                for dept in [d for d in depts.values() if d[3] == parent_id]:
                    path = ['brightmart-home.md', f'departments/dept-{dept[2]}.md']
                    gold_spans.append({'query_id': query_id, 'doc_path': None, 'path': path})
            elif hierarchy_type == 'multi_region':
                # Multi-region: one path per parent region
                for region_id in parent_id:
                    path = ['brightmart-home.md', f'regions/region-{region_id}.md']
                    gold_spans.append({'query_id': query_id, 'doc_path': None, 'path': path})
            elif hierarchy_type == 'multi_department':
                # Multi-department: one path per parent department
                for dept_id in parent_id:
                    path = ['brightmart-home.md', f'departments/dept-{depts[dept_id][2]}.md']
                    gold_spans.append({'query_id': query_id, 'doc_path': None, 'path': path})
            elif hierarchy_type in ['multi_region_format', 'multi_region_format_express', 'multi_region_format_neighborhood']:
                # Multi-region with format: one path per parent region
                for region_id in parent_id:
                    path = ['brightmart-home.md', f'regions/region-{region_id}.md']
                    gold_spans.append({'query_id': query_id, 'doc_path': None, 'path': path})

            # D4 fix: store gold_sql on a record with doc_path=null
            if sql_query and sql_params is not None:
                gold_spans.append({
                    'query_id': query_id,
                    'doc_path': None,
                    'gold_sql': sql_query,
                    'gold_sql_params': sql_params,
                })

    # S5: Unanswerable
    for q_id, text, absent_term in S5_QUESTIONS:
        query_id = f'bm-s5-{q_id}-t'

        found_in_corpus = False
        for doc_text in docs.values():
            if absent_term.lower() in doc_text.lower():
                found_in_corpus = True
                break

        if not found_in_corpus:
            questions.append({
                'corpus': 'synthetic_retail_pilot',
                'origin': 'synthetic_template',
                'query_id': query_id,
                'reference_answer': None,
                'reference_claim': None,
                'source_paths': [],
                'stratum': 'S5',
                'text': text,
            })
            gold_spans.append({'absent_term': absent_term, 'query_id': query_id})

    return questions, qrels, gold_spans


if __name__ == '__main__':
    conn = brightmart_seed.connect()
    questions, qrels, gold_spans = generate_questions(conn)
    conn.close()

    # Load paraphrases
    with open('doc_synthetic/brightmart_paraphrases.json', 'r', encoding='utf-8') as f:
        paraphrases = json.load(f)

    # Add paraphrase questions for S1-S4
    paraphrase_questions = []
    paraphrase_qrels = []
    paraphrase_spans = []

    for q in questions:
        if q['stratum'] in ['S1', 'S2', 'S3', 'S4', 'S5']:
            # Extract base ID (remove -t suffix)
            base_id = q['query_id'].replace('-t', '')

            if base_id in paraphrases:
                para_text = paraphrases[base_id]
                para_q = q.copy()
                para_q['query_id'] = base_id + '-p'
                para_q['origin'] = 'synthetic_paraphrase'
                para_q['text'] = para_text
                paraphrase_questions.append(para_q)

                # Copy qrels for paraphrase
                for qrel in qrels:
                    if qrel['query_id'] == q['query_id']:
                        para_qrel = qrel.copy()
                        para_qrel['query_id'] = base_id + '-p'
                        paraphrase_qrels.append(para_qrel)

                # Copy gold_spans for paraphrase
                for span in gold_spans:
                    if span['query_id'] == q['query_id']:
                        para_span = span.copy()
                        para_span['query_id'] = base_id + '-p'
                        paraphrase_spans.append(para_span)

    # Combine template and paraphrase questions
    all_questions = questions + paraphrase_questions
    all_qrels = qrels + paraphrase_qrels
    all_gold_spans = gold_spans + paraphrase_spans

    # Write JSONL files (sorted by query_id, preserving doc order within each question)
    questions_sorted = sorted(all_questions, key=lambda x: x['query_id'])
    # For qrels and spans, maintain original order within same question_id
    # Group by query_id while preserving order
    qrels_by_qid = {}
    qrels_order = []
    for qrel in all_qrels:
        qid = qrel['query_id']
        if qid not in qrels_by_qid:
            qrels_by_qid[qid] = []
            qrels_order.append(qid)
        qrels_by_qid[qid].append(qrel)
    qrels_sorted = []
    for qid in sorted(qrels_order):
        qrels_sorted.extend(qrels_by_qid[qid])

    gold_spans_by_qid = {}
    spans_order = []
    for span in all_gold_spans:
        qid = span['query_id']
        if qid not in gold_spans_by_qid:
            gold_spans_by_qid[qid] = []
            spans_order.append(qid)
        gold_spans_by_qid[qid].append(span)
    gold_spans_sorted = []
    for qid in sorted(spans_order):
        gold_spans_sorted.extend(gold_spans_by_qid[qid])

    with open('doc_synthetic/brightmart_questions.jsonl', 'w', encoding='utf-8', newline='\n') as f:
        for q in questions_sorted:
            f.write(json.dumps(q, ensure_ascii=False, sort_keys=True) + '\n')

    with open('doc_synthetic/brightmart_qrels.jsonl', 'w', encoding='utf-8', newline='\n') as f:
        for qrel in qrels_sorted:
            f.write(json.dumps(qrel, ensure_ascii=False, sort_keys=True) + '\n')

    with open('doc_synthetic/brightmart_gold_spans.jsonl', 'w', encoding='utf-8', newline='\n') as f:
        for span in gold_spans_sorted:
            f.write(json.dumps(span, ensure_ascii=False, sort_keys=True) + '\n')

    print(f'Generated {len(questions_sorted)} questions')
    print(f'  S1: {len([q for q in questions_sorted if q["stratum"] == "S1"])}')
    print(f'  S2: {len([q for q in questions_sorted if q["stratum"] == "S2"])}')
    print(f'  S3: {len([q for q in questions_sorted if q["stratum"] == "S3"])}')
    print(f'  S4: {len([q for q in questions_sorted if q["stratum"] == "S4"])}')
    print(f'  S5: {len([q for q in questions_sorted if q["stratum"] == "S5"])}')
