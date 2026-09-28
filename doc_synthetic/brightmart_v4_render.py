"""Brightmart v4 corpus renderer (next iteration, test plan T1/T2).

Renders about 348 markdown documents from `brightmart_v4_seed.connect()` in
two variants that share every path and every sentence:

- ``standard``: children name their parent in their own metadata (a store
  carries ``region``, a category carries ``department``), like the pilot.
- ``noparent`` (T2): the store ``region`` field and region tag, the store's
  "Back to <region>" link, the category ``department`` field and department tag,
  and the category's "part of the <department> department" sentence are
  removed. The relation then exists only as the authored ``parent`` link and
  the parent page's child list.

The corpora are generated outputs, written under ``results/`` (not
committed): ``python doc_synthetic/brightmart_v4_render.py`` writes
``results/corpus_v4`` and ``results/corpus_v4_noparent``. Rendering is
deterministic; `brightmart_v4_validate.py` checks the files on disk equal a
fresh render and that the corpus hashes match the frozen values.

Filler prose under ``## Notes`` is composed from fixed per-type sentence
banks; it carries no digits and no entity names.
"""

import os
import random
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import brightmart_v4_seed as seed
from brightmart_render import _build_markdown, _format_list

VARIANTS = ('standard', 'noparent')
OUT_DIRS = {'standard': 'results/corpus_v4', 'noparent': 'results/corpus_v4_noparent'}
TIMESTAMP = '2026-01-15'
MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
          'September', 'October', 'November', 'December']
QUARTER_WEEKS = [(1, 13), (14, 26), (27, 39), (40, 52)]

FILLER = {
    'home': [
        'This wiki gathers the reference pages that store teams consult most often.',
        'Pages are maintained by the operations group and reviewed on a regular cycle.',
        'Each section links to the pages that describe it in more detail.',
        'Readers are encouraged to report outdated information to the page owner.',
        'The structure of the wiki mirrors the way the business is organized.',
        'New pages are added when a new area of the business needs documentation.',
        'Older versions of pages are kept for reference in a separate area.',
        'Navigation starts here and branches into the main areas of the company.',
    ],
    'region': [
        'The regional office coordinates staffing, merchandising and store visits.',
        'Regional teams meet with store leaders to review performance and plans.',
        'Seasonal planning for the region starts well before each holiday period.',
        'The region shares best practices through a monthly leadership call.',
        'Store openings and remodels in the region are planned by a dedicated team.',
        'Regional leadership tracks customer feedback and follows up on complaints.',
        'The region works closely with distribution centers to keep shelves stocked.',
        'Training for new store managers is organized at the regional level.',
        'Community partnerships are coordinated by the regional marketing lead.',
        'The regional office publishes a short update for store teams every week.',
    ],
    'store': [
        'The store team focuses on friendly service and well-stocked shelves.',
        'Shoppers often mention the easy parking and clear signage.',
        'The layout was designed so that everyday essentials are easy to find.',
        'Store associates receive regular training on customer service.',
        'Seasonal displays near the entrance change throughout the year.',
        'The store supports local community events and school programs.',
        'Checkout lanes are staffed to keep waiting times short during busy hours.',
        'Customers can ask any associate for help locating a product.',
        'The management team walks the floor each morning before opening.',
        'Online orders can be collected at the customer service desk.',
        'The store keeps a quiet hour on weekday mornings for shoppers who prefer calm.',
        'Local suppliers are featured in a small showcase near the front of the store.',
    ],
    'sales_report': [
        'Figures are compiled from the weekly sales ledger and summed by quarter.',
        'Amounts are reported in dollars and rounded to the nearest cent.',
        'Departments that the store does not carry are not listed in the table.',
        'The finance team reviews these figures before they are published.',
        'Quarterly totals help store leaders compare seasonal performance.',
        'Questions about the figures should be directed to the finance team.',
        'The report is refreshed once the books for the year are closed.',
        'Comparisons across stores should account for differences in store format.',
    ],
    'department': [
        'Department leads work with buyers to plan the assortment for each season.',
        'Product placement follows a planogram shared across stores.',
        'The department reviews customer feedback to refine its selection.',
        'Associates in this department receive product knowledge training.',
        'Shelf space is adjusted when new products are introduced.',
        'The department coordinates with marketing on featured items.',
        'Inventory levels are monitored daily to avoid empty shelves.',
        'Private label items are offered alongside national brands.',
        'The department follows the company guidelines for product safety.',
        'Merchandising teams refresh the department displays several times a year.',
    ],
    'category': [
        'The category assortment is reviewed with the buying team each quarter.',
        'Popular items in this category are replenished on a frequent schedule.',
        'Customers can request items that are not currently on the shelf.',
        'Pricing for the category is benchmarked against local competitors.',
        'New products are tested in a few stores before a wider rollout.',
        'Product labels in this category follow the company style guide.',
        'The category manager monitors quality reports from stores.',
        'Seasonal items in this category rotate throughout the year.',
        'Shelf tags highlight items that are part of a current promotion.',
        'Slow-moving items are reviewed and may be replaced with new products.',
    ],
    'supplier': [
        'The supplier relationship is managed by the central buying team.',
        'Deliveries are scheduled to match store replenishment cycles.',
        'The supplier participates in the annual vendor quality review.',
        'Contracts are renewed after a review of service and product quality.',
        'The buying team meets with the supplier to plan seasonal volumes.',
        'Product recalls are coordinated directly with the supplier.',
        'The supplier follows the company code of conduct for vendors.',
        'Shipping documents are checked when goods arrive at distribution centers.',
        'New products from the supplier go through a tasting or testing panel.',
        'Invoices are processed by the accounts payable team.',
    ],
    'promotion': [
        'Signage for the promotion is sent to stores ahead of the start date.',
        'The promotion is advertised in the weekly flyer and online.',
        'Discounts are applied automatically at checkout.',
        'Store teams set up promotional displays before the event begins.',
        'Customers should check shelf tags for eligible items.',
        'The promotion cannot be combined with certain other offers.',
        'Results of the promotion are reviewed by the marketing team afterwards.',
        'Stock levels are increased for featured items during the event.',
        'Associates are briefed on the promotion details before launch.',
        'Questions about eligibility can be answered at the service desk.',
    ],
    'policy': [
        'The policy applies consistently across participating stores.',
        'Store managers may escalate unusual cases to the regional office.',
        'Associates are trained to explain the policy to customers.',
        'The policy is reviewed periodically by the legal team.',
        'Signage summarizing the policy is posted near the service desk.',
        'Exceptions require approval from a store manager.',
        'The policy is designed to be simple and fair for customers.',
        'Changes to the policy are announced to store teams in advance.',
        'Customers can read the full policy on request.',
        'Records of exceptions are kept for later review.',
    ],
    'archive': [
        'This page is kept for historical reference only.',
        'Information on this page may no longer be accurate.',
        'Readers should consult the current page for up-to-date details.',
        'The page was preserved when the content was revised.',
        'Archived material is not maintained by the page owner.',
        'Draft material may change before it is published.',
        'Links on this page may point to newer versions of related pages.',
        'The content reflects the situation at the time it was written.',
    ],
    'schema': [
        'The schema is maintained by the data engineering team.',
        'Tables are loaded nightly from the operational systems.',
        'Analysts should use the documented keys when joining tables.',
        'Column names follow the company naming conventions.',
        'Changes to the schema are announced before they take effect.',
        'Historical data is retained according to the data retention policy.',
        'Access to the database is granted by the data platform team.',
        'Documentation for each table is kept alongside the schema.',
    ],
}


def short_name(store_name: str) -> str:
    return store_name.replace('Brightmart ', '')


def name_slug(store_name: str) -> str:
    return short_name(store_name).lower().replace(' ', '-').replace('.', '')


def store_path(store) -> str:
    return f'/regions/{store[1]}/store-{store[0].lower()}-{name_slug(store[2])}.md'


def sales_path(store) -> str:
    return f'/sales/sales-{store[0].lower()}-{name_slug(store[2])}-2025.md'


def sales_title(store) -> str:
    return f'{store[2]} 2025 Sales'


def region_path(region_id: str) -> str:
    return f'/regions/region-{region_id}.md'


def dept_path(dept) -> str:
    return f'/departments/dept-{dept[2]}.md'


def category_path(cat, depts) -> str:
    return f'/departments/{depts[cat[1]][2]}/category-{cat[3]}.md'


def supplier_path(supplier) -> str:
    return f'/suppliers/supplier-{supplier[2]}.md'


def promo_path(promo) -> str:
    return f'/promotions/promo-{promo[2]}.md'


def policy_path(policy) -> str:
    return f'/policies/policy-{policy[2]}.md'


HOME_PATH = '/brightmart-home.md'
SCHEMA_PATH = '/schema/brightmart-sales-database.md'


def load_facts(conn) -> dict:
    """Every table the renderer and the question builder read, keyed by id."""
    cur = conn.cursor()
    f = {
        'regions': {r[0]: r for r in cur.execute('SELECT region_id, name, manager FROM region ORDER BY region_id')},
        'stores': {s[0]: s for s in cur.execute(
            'SELECT store_id, region_id, name, city, state, format, opened_year, sq_ft, has_pharmacy, '
            'has_fuel, status FROM store ORDER BY store_id')},
        'depts': {d[0]: d for d in cur.execute('SELECT dept_id, name, slug, "group" FROM department ORDER BY dept_id')},
        'suppliers': {s[0]: s for s in cur.execute(
            'SELECT supplier_id, name, slug, country FROM supplier ORDER BY supplier_id')},
        'categories': {c[0]: c for c in cur.execute(
            'SELECT cat_id, dept_id, name, slug, supplier_id FROM category ORDER BY cat_id')},
        'promos': {p[0]: p for p in cur.execute(
            'SELECT promo_id, name, slug, dept_id, start, end, discount_pct FROM promotion ORDER BY promo_id')},
        'policies': {p[0]: p for p in cur.execute(
            'SELECT policy_id, name, slug, applies_to FROM policy ORDER BY policy_id')},
        'archives': {a[0]: a for a in cur.execute(
            'SELECT archive_id, slug, subject, stale_key, stale_value, status FROM archive_doc ORDER BY archive_id')},
    }
    f['promo_formats'] = {}
    for pid, fmt in cur.execute('SELECT promo_id, format FROM promotion_format ORDER BY promo_id, format'):
        f['promo_formats'].setdefault(pid, []).append(fmt)
    order = {'supercenter': 0, 'neighborhood': 1, 'express': 2}
    for fmts in f['promo_formats'].values():
        fmts.sort(key=order.get)
    f['policy_facts'] = {}
    for pid, key, value in cur.execute('SELECT policy_id, key, value FROM policy_fact ORDER BY policy_id, key'):
        f['policy_facts'].setdefault(pid, {})[key] = value
    f['store_depts'] = {}
    for sid, did, mgr in cur.execute('SELECT store_id, dept_id, dept_manager FROM store_department ORDER BY store_id, dept_id'):
        f['store_depts'].setdefault(sid, {})[did] = mgr
    f['quarterly'] = {}
    for sid, did, week, sales in cur.execute('SELECT store_id, dept_id, week, sales FROM weekly_sales ORDER BY store_id, dept_id, week'):
        q = next(i for i, (a, b) in enumerate(QUARTER_WEEKS) if a <= week <= b)
        f['quarterly'].setdefault((sid, did), [0.0, 0.0, 0.0, 0.0])[q] += sales
    return f


def quarterly_sales(facts, store_id, dept_id) -> list[str]:
    return [f'{v:,.2f}' for v in facts['quarterly'].get((store_id, dept_id), [0.0] * 4)]


def filler(path: str, doc_type: str) -> str:
    bank = FILLER[doc_type]
    rng = random.Random(f'v4-notes:{path}')
    return ' '.join(rng.sample(bank, 3))


def _render_home(rng, f):
    fm = {'type': 'home', 'title': 'Brightmart Retail Wiki',
          'description': 'Central hub for the Brightmart retail wiki.',
          'tags': ['home', 'synthetic'], 'status': 'stable', 'timestamp': TIMESTAMP}
    body = '# Brightmart Retail Wiki\n\nWelcome to the Brightmart retail wiki.\n\n## Regions\n\n'
    for rid in sorted(f['regions']):
        body += f"- [{f['regions'][rid][1]}]({region_path(rid)})\n"
    body += '\n## Departments\n\n'
    for did in sorted(f['depts']):
        body += f"- [{f['depts'][did][1]}]({dept_path(f['depts'][did])})\n"
    body += '\n## Suppliers\n\n'
    for sid in sorted(f['suppliers']):
        body += f"- [{f['suppliers'][sid][1]}]({supplier_path(f['suppliers'][sid])})\n"
    body += '\n## Promotions\n\n'
    for pid in sorted(f['promos']):
        body += f"- [{f['promos'][pid][1]}]({promo_path(f['promos'][pid])})\n"
    body += '\n## Policies\n\n'
    for pid in sorted(f['policies']):
        body += f"- [{f['policies'][pid][1]}]({policy_path(f['policies'][pid])})\n"
    body += f'\n## Schema\n\n- [Brightmart Sales Database]({SCHEMA_PATH})\n'
    return fm, body


def _render_region(rng, f, region_id):
    region = f['regions'][region_id]
    fm = {'type': 'region', 'title': region[1],
          'description': f'Regional hub for {region[1]}, managed by {region[2]}.',
          'tags': ['region', region_id], 'status': 'stable', 'timestamp': TIMESTAMP,
          'parent': HOME_PATH, 'region_id': region_id, 'manager': region[2]}
    body = f'# {region[1]}\n\n'
    body += rng.choice([
        f'The {region[1]} region is managed by {region[2]}.',
        f'{region[2]} oversees the {region[1]} region.',
        f'Regional director {region[2]} leads the {region[1]} region.',
    ]) + '\n\n## Stores\n\n'
    for sid in sorted(s for s in f['stores'] if f['stores'][s][1] == region_id):
        body += f"- [{f['stores'][sid][2]}]({store_path(f['stores'][sid])})\n"
    return fm, body


def _render_store(rng, f, store_id, variant, sq_ft_override=None):
    s = f['stores'][store_id]
    _, region_id, name, city, state, fmt, opened_year, sq_ft, has_pharmacy, has_fuel, status = s
    sq_ft = sq_ft_override if sq_ft_override is not None else sq_ft
    region_name = f['regions'][region_id][1]
    dept_ids = sorted(f['store_depts'].get(store_id, {}))
    fm = {'type': 'store', 'title': name,
          'description': f'Profile of {name}, a {fmt} store in {city}, {state}.',
          'tags': ['store', store_id, region_id, fmt], 'status': 'stable', 'timestamp': TIMESTAMP,
          'parent': region_path(region_id), 'store_id': store_id, 'region': region_name,
          'city': city, 'state': state, 'store_format': fmt, 'opened_year': opened_year,
          'sq_ft': sq_ft, 'has_pharmacy': bool(has_pharmacy), 'has_fuel': bool(has_fuel),
          'store_status': status, 'departments': [f['depts'][d][1] for d in dept_ids]}
    if variant == 'noparent':
        del fm['region']
        fm['tags'] = ['store', store_id, fmt]
    body = f'# {name}\n\n'
    body += rng.choice([f'{name} is located in {city}, {state}.',
                        f'This store is situated in {city}, {state}.',
                        f'You will find this store in {city}, {state}.']) + '\n\n'
    body += rng.choice([f'{name} is a {fmt} format store.',
                        f'Operating as a {fmt}, {name} offers a curated selection of products.',
                        f'This {fmt} location provides a full range of retail services.']) + '\n\n'
    body += rng.choice([f'{name} opened its doors in {opened_year}.',
                        f'Established in {opened_year}, this store has served the community since.',
                        f'Since {opened_year}, this facility has been operating.']) + '\n\n'
    body += rng.choice([f'{name} occupies {sq_ft:,} square feet of retail space.',
                        f'The sales floor at {name} covers {sq_ft:,} square feet.',
                        f'{name} spans {sq_ft:,} square feet.']) + '\n\n'
    if has_pharmacy:
        body += rng.choice([f'{name} includes a full-service pharmacy.',
                            'Pharmacy services are available on-site.',
                            'A pharmacy operates within this location.']) + '\n\n'
    else:
        body += rng.choice([f'{name} does not operate a pharmacy.',
                            'No pharmacy services are offered at this location.',
                            'This store does not have a pharmacy.']) + '\n\n'
    if has_fuel:
        body += rng.choice(['Fuel services are available at the pumps.',
                            'Customers can fill up at the fuel station.',
                            f'{name} operates a fuel station.']) + '\n\n'
    else:
        body += rng.choice(['No fuel station is available at this location.',
                            'Fuel services are not offered here.',
                            f'{name} does not have a fuel station.']) + '\n\n'
    status_templates = {
        'open': [f'{name} is currently open and operating normally.',
                 'This location is open to customers.',
                 'The store is open and welcoming shoppers.'],
        'closed': [f'{name} is permanently closed.',
                   'This location has closed and ceased operations.',
                   'The store is closed and no longer in operation.'],
        'remodeling': [f'{name} is currently undergoing remodeling.',
                       'Remodeling work is in progress at this location.',
                       'This store is temporarily closed for remodeling.'],
    }
    body += rng.choice(status_templates[status]) + '\n\n## Departments\n\n'
    for did in dept_ids:
        body += f"- [{f['depts'][did][1]}]({dept_path(f['depts'][did])}) – managed by {f['store_depts'][store_id][did]}\n"
    body += f'\n## Sales\n\nQuarterly 2025 sales are reported on the [{sales_title(s)}]({sales_path(s)}) page.\n'
    if variant == 'standard':
        body += f'\n[Back to {region_name}]({region_path(region_id)})\n'
    return fm, body


def _render_sales(rng, f, store_id):
    s = f['stores'][store_id]
    fm = {'type': 'sales_report', 'title': sales_title(s),
          'description': f'Quarterly 2025 sales by department for {s[2]}.',
          'tags': ['sales', store_id], 'status': 'stable', 'timestamp': TIMESTAMP,
          'parent': store_path(s), 'store': s[2], 'fiscal_year': 2025}
    body = f'# {sales_title(s)}\n\n'
    body += rng.choice([f'This report lists 2025 sales by department and quarter for [{s[2]}]({store_path(s)}).',
                        f'Department sales for [{s[2]}]({store_path(s)}) in 2025, summed by quarter.',
                        f'The table below shows how each department at [{s[2]}]({store_path(s)}) sold in 2025.']) + '\n\n'
    if s[10] == 'closed':
        body += 'Reporting stopped when the store closed during the year.\n\n'
    body += '| Department | Q1 | Q2 | Q3 | Q4 |\n|---|---|---|---|---|\n'
    for did in sorted(f['store_depts'].get(store_id, {})):
        q = quarterly_sales(f, store_id, did)
        body += f"| {f['depts'][did][1]} | {q[0]} | {q[1]} | {q[2]} | {q[3]} |\n"
    return fm, body


def _render_department(rng, f, dept_id):
    d = f['depts'][dept_id]
    fm = {'type': 'department', 'title': d[1],
          'description': f'The {d[1]} department focuses on {d[3]} products.',
          'tags': ['department', dept_id, d[3]], 'status': 'stable', 'timestamp': TIMESTAMP,
          'parent': HOME_PATH, 'dept_id': dept_id, 'dept_group': d[3]}
    body = f'# {d[1]}\n\n'
    body += rng.choice([f'The {d[1]} department belongs to the {d[3]} group.',
                        f'Part of the {d[3]} group, {d[1]} brings together related products.',
                        f'This department is part of the {d[3]} group of departments.']) + '\n\n## Categories\n\n'
    for cid in sorted(c for c in f['categories'] if f['categories'][c][1] == dept_id):
        body += f"- [{f['categories'][cid][2]}]({category_path(f['categories'][cid], f['depts'])})\n"
    body += '\n## Available at\n\n'
    for sid in sorted(s for s in f['stores'] if dept_id in f['store_depts'].get(s, {})):
        body += f"- [{f['stores'][sid][2]}]({store_path(f['stores'][sid])})\n"
    return fm, body


def _render_category(rng, f, cat_id, variant):
    c = f['categories'][cat_id]
    d = f['depts'][c[1]]
    sup = f['suppliers'][c[4]]
    fm = {'type': 'category', 'title': c[2],
          'description': f'{c[2]} category offered through Brightmart stores.',
          'tags': ['category', c[3], c[1]], 'status': 'stable', 'timestamp': TIMESTAMP,
          'parent': dept_path(d), 'department': d[1], 'supplier': sup[1]}
    body = f'# {c[2]}\n\n'
    if variant == 'standard':
        body += f'This category is part of the [{d[1]}]({dept_path(d)}) department.\n\n'
    else:
        del fm['department']
        fm['tags'] = ['category', c[3]]
    body += rng.choice([f'{c[2]} is sourced from [{sup[1]}]({supplier_path(sup)}).',
                        f'[{sup[1]}]({supplier_path(sup)}) supplies these products.',
                        f'Our partner [{sup[1]}]({supplier_path(sup)}) provides {c[2]} items.']) + '\n'
    return fm, body


def _country_text(country: str) -> str:
    return 'the United States' if country == 'United States' else country


def _render_supplier(rng, f, supplier_id):
    s = f['suppliers'][supplier_id]
    country = _country_text(s[3])
    fm = {'type': 'supplier', 'title': s[1], 'description': f'Supplier {s[1]} based in {country}.',
          'tags': ['supplier', supplier_id, s[3]], 'status': 'stable', 'timestamp': TIMESTAMP,
          'parent': HOME_PATH, 'country': s[3]}
    body = f'# {s[1]}\n\n'
    body += rng.choice([f'{s[1]} is based in {country}.',
                        f'{s[1]} operates out of {country}.',
                        f'{s[1]} is headquartered in {country}.']) + '\n\n## Categories Supplied\n\n'
    for cid in sorted(c for c in f['categories'] if f['categories'][c][4] == supplier_id):
        body += f"- [{f['categories'][cid][2]}]({category_path(f['categories'][cid], f['depts'])})\n"
    return fm, body


def _render_promotion(rng, f, promo_id, discount_override=None, year_override=None):
    p = f['promos'][promo_id]
    d = f['depts'][p[3]]
    start, end = p[4], p[5]
    if year_override:
        start, end = f'{year_override}{start[4:]}', f'{year_override}{end[4:]}'
    discount = discount_override if discount_override is not None else p[6]
    formats = f['promo_formats'].get(promo_id, [])
    fm = {'type': 'promotion', 'title': p[1], 'description': f'{p[1]} promotion offering {discount}% discount.',
          'tags': ['promotion', promo_id, p[3]], 'status': 'stable', 'timestamp': TIMESTAMP,
          'parent': HOME_PATH, 'department': d[1], 'formats': formats, 'start': start, 'end': end,
          'discount_pct': discount}
    body = f'# {p[1]}\n\n'
    body += f'This promotion applies to the [{d[1]}]({dept_path(d)}) department.\n\n'
    body += f'**Dates:** {start} to {end}\n\n**Discount:** {discount}% off\n\n'
    fmt_str = _format_list(formats)
    body += rng.choice([f'{p[1]} is available at {fmt_str} stores.',
                        f'This promotion applies to {fmt_str} store locations.',
                        f'{fmt_str[0].upper() + fmt_str[1:]} stores take part in {p[1]}.']) + '\n'
    return fm, body


POLICY_SENTENCES = {
    'standard_window_days': ['Under the {policy}, most items can be returned within {v} days of purchase.',
                             'The {policy} allows returns of most items within {v} days.',
                             'Most purchases can be returned within {v} days under the {policy}.'],
    'electronics_window_days': ['Electronics have a {v}-day return window under the {policy}.',
                                'The {policy} provides {v} days for electronics returns.',
                                'Electronics can be returned within {v} days under the {policy}.'],
    'receipt_required': {'yes': ['Under the {policy}, a receipt is required for all returns.',
                                 'The {policy} requires a receipt when returning items.',
                                 'A receipt must be presented to return items under the {policy}.'],
                         'no': ['Under the {policy}, a receipt is not required for returns.',
                                'The {policy} allows returns without a receipt.',
                                'Items can be returned without a receipt under the {policy}.']},
    'weekday_hours': ['Under the {policy}, pharmacies are open {v} on weekdays.',
                      'Weekday pharmacy hours are {v} per the {policy}.',
                      'Monday through Friday, pharmacy hours are {v} under the {policy}.'],
    'sunday_hours': ['Under the {policy}, pharmacies are open {v} on Sundays.',
                     'On Sundays, pharmacy hours are {v} per the {policy}.',
                     'The pharmacy operates {v} on Sundays under the {policy}.'],
    'cents_off_per_gallon': ['The {policy} offers {v} cents off per gallon.',
                             'Customers receive {v} cents off each gallon under the {policy}.',
                             'Each gallon is discounted by {v} cents under the {policy}.'],
    'spend_threshold_usd': ['The {policy} requires a minimum purchase of ${v}.',
                            'Fuel rewards under the {policy} apply after spending ${v}.',
                            'Customers must spend ${v} to qualify under the {policy}.'],
    'match_window_days': ['The {policy} matches a lower competitor price found within {v} days of purchase.',
                          'Under the {policy}, customers have {v} days after purchase to request a price match.',
                          'Price match requests are accepted for {v} days under the {policy}.'],
    'online_competitors_included': {'yes': ['The {policy} also covers prices from online competitors.',
                                            'Online competitor prices are eligible under the {policy}.',
                                            'Under the {policy}, online retailers count as competitors.'],
                                    'no': ['The {policy} does not cover prices from online competitors.',
                                           'Online competitor prices are not eligible under the {policy}.',
                                           'Under the {policy}, only physical store competitors are matched.']},
    'rain_check_valid_days': ['A rain check issued under the {policy} is valid for {v} days.',
                              'Under the {policy}, rain checks can be redeemed within {v} days.',
                              'Rain checks stay valid for {v} days under the {policy}.'],
    'limit_per_customer': ['The {policy} limits each customer to {v} rain checks per visit.',
                           'Customers may receive up to {v} rain checks per visit under the {policy}.',
                           'Under the {policy}, no more than {v} rain checks are issued per visit.'],
    'minimum_deposit_pct': ['The {policy} requires a deposit of {v}% of the purchase price.',
                            'A {v}% deposit starts a layaway plan under the {policy}.',
                            'Under the {policy}, customers pay {v}% up front.'],
    'layaway_period_days': ['Items can be kept on layaway for up to {v} days under the {policy}.',
                            'The {policy} gives customers {v} days to complete payment.',
                            'Under the {policy}, the layaway period lasts {v} days.'],
    'reload_minimum_usd': ['Under the {policy}, gift cards can be reloaded with a minimum of ${v}.',
                           'The {policy} sets a minimum reload amount of ${v}.',
                           'Gift card reloads must be at least ${v} under the {policy}.'],
    'balance_expires': {'never': ['Under the {policy}, gift card balances never expire.',
                                  'The {policy} states that gift card balances do not expire.',
                                  'Gift card balances never expire under the {policy}.']},
    'pickup_window_hours': ['Under the {policy}, orders are held for {v} hours after they are ready.',
                            'The {policy} keeps ready orders for {v} hours.',
                            'Customers have {v} hours to collect a ready order under the {policy}.'],
    'minimum_order_usd': ['The {policy} requires a minimum order of ${v}.',
                          'Curbside orders must total at least ${v} under the {policy}.',
                          'Under the {policy}, the minimum order value is ${v}.'],
    'senior_discount_pct': ['The {policy} gives eligible seniors {v}% off their purchase.',
                            'Under the {policy}, seniors receive a {v}% discount.',
                            'Eligible seniors save {v}% under the {policy}.'],
    'eligible_age': ['Customers aged {v} or older qualify under the {policy}.',
                     'The {policy} applies to shoppers who are at least {v} years old.',
                     'Under the {policy}, the minimum age is {v}.'],
    'discount_day': ['Under the {policy}, the senior discount is offered every {v}.',
                     'The {policy} discount applies on {v}s.',
                     'Seniors can use the {policy} discount each {v}.'],
    'deposit_pct': ['The {policy} requires a {v}% deposit when the order is placed.',
                    'Under the {policy}, special orders need a {v}% deposit.',
                    'A deposit of {v}% is collected for special orders under the {policy}.'],
    'arrival_notice_days': ['Under the {policy}, customers are told the expected arrival within {v} days.',
                            'The {policy} promises an arrival notice within {v} days of ordering.',
                            'Customers receive an arrival estimate within {v} days under the {policy}.'],
}


def policy_sentence_options(key: str, value: str, policy_name: str) -> list[str]:
    options = POLICY_SENTENCES[key]
    if isinstance(options, dict):
        options = options[value]
    return [o.format(policy=policy_name, v=value) for o in options]


def _render_policy(rng, f, policy_id, facts_override=None):
    p = f['policies'][policy_id]
    facts = facts_override if facts_override is not None else f['policy_facts'].get(policy_id, {})
    fm = {'type': 'policy', 'title': p[1], 'description': f'{p[1]} for {p[3]}.',
          'tags': ['policy', policy_id], 'status': 'stable', 'timestamp': TIMESTAMP,
          'parent': HOME_PATH, 'applies_to': p[3]}
    body = f'# {p[1]}\n\n'
    for key in sorted(facts):
        body += rng.choice(policy_sentence_options(key, facts[key], p[1])) + '\n'
    return fm, body


def archive_title(f, archive) -> str:
    subject = archive[2]
    if subject.startswith('POL'):
        return f"{f['policies'][subject][1]} (version 1)"
    if subject.startswith('P'):
        return f"{f['promos'][subject][1]} (2026 draft)"
    return f"{f['stores'][subject][2]} (2022 profile)"


def _render_archive(rng, f, archive_id, variant):
    archive = f['archives'][archive_id]
    _, _, subject, stale_key, stale_value, status = archive
    title = archive_title(f, archive)
    if subject.startswith('POL'):
        facts = dict(f['policy_facts'][subject])
        facts[stale_key] = stale_value
        fm, body = _render_policy(rng, f, subject, facts_override=facts)
        current, note, old_h1 = policy_path(f['policies'][subject]), 'This is version 1 of this policy.', f['policies'][subject][1]
    elif subject.startswith('P'):
        fm, body = _render_promotion(rng, f, subject, discount_override=int(stale_value), year_override='2026')
        current, note, old_h1 = promo_path(f['promos'][subject]), 'This is a 2026 draft version of this promotion.', f['promos'][subject][1]
    else:
        fm, body = _render_store(rng, f, subject, variant, sq_ft_override=int(stale_value))
        current, note, old_h1 = store_path(f['stores'][subject]), 'This is a 2022 profile of this store.', f['stores'][subject][2]
    fm['type'] = 'archive'
    fm['title'] = title
    fm['status'] = status
    fm['parent'] = HOME_PATH
    fm['supersedes'] = current
    body = body.replace(f'# {old_h1}\n\n', f'# {title}\n\n{note}\n\n', 1)
    return fm, body


def _render_schema(rng, f):
    fm = {'type': 'schema', 'title': 'Brightmart Sales Database Schema',
          'description': 'Database schema for the Brightmart retail dataset.',
          'tags': ['schema', 'database'], 'status': 'stable', 'timestamp': TIMESTAMP, 'parent': HOME_PATH}
    tables = [
        ('region', ['`region_id` (PK): TEXT', '`name`: TEXT', '`manager`: TEXT'], []),
        ('store', ['`store_id` (PK): TEXT', '`region_id`: TEXT', '`name`: TEXT', '`city`: TEXT', '`state`: TEXT',
                   '`format`: TEXT', '`opened_year`: INTEGER', '`sq_ft`: INTEGER', '`has_pharmacy`: INTEGER',
                   '`has_fuel`: INTEGER', '`status`: TEXT'], ['`store.region_id` references `region.region_id`.']),
        ('department', ['`dept_id` (PK): TEXT', '`name`: TEXT', '`slug`: TEXT', '`group`: TEXT'], []),
        ('store_department', ['`store_id`: TEXT', '`dept_id`: TEXT', '`dept_manager`: TEXT'],
         ['`store_department.store_id` references `store.store_id`.',
          '`store_department.dept_id` references `department.dept_id`.']),
        ('supplier', ['`supplier_id` (PK): TEXT', '`name`: TEXT', '`slug`: TEXT', '`country`: TEXT'], []),
        ('category', ['`cat_id` (PK): TEXT', '`dept_id`: TEXT', '`name`: TEXT', '`slug`: TEXT', '`supplier_id`: TEXT'],
         ['`category.dept_id` references `department.dept_id`.',
          '`category.supplier_id` references `supplier.supplier_id`.']),
        ('weekly_sales', ['`store_id`: TEXT', '`dept_id`: TEXT', '`week`: INTEGER', '`sales`: REAL'],
         ['`weekly_sales.store_id` references `store.store_id`.',
          '`weekly_sales.dept_id` references `department.dept_id`.']),
        ('promotion', ['`promo_id` (PK): TEXT', '`name`: TEXT', '`slug`: TEXT', '`dept_id`: TEXT', '`start`: TEXT',
                       '`end`: TEXT', '`discount_pct`: INTEGER'], ['`promotion.dept_id` references `department.dept_id`.']),
        ('promotion_format', ['`promo_id`: TEXT', '`format`: TEXT'],
         ['`promotion_format.promo_id` references `promotion.promo_id`.']),
        ('policy', ['`policy_id` (PK): TEXT', '`name`: TEXT', '`slug`: TEXT', '`applies_to`: TEXT'], []),
        ('policy_fact', ['`policy_id`: TEXT', '`key`: TEXT', '`value`: TEXT'],
         ['`policy_fact.policy_id` references `policy.policy_id`.']),
    ]
    body = '# Brightmart Sales Database Schema\n\n'
    for name, cols, refs in tables:
        body += f'### {name}\n\n' + ''.join(f'- {c}\n' for c in cols) + ''.join(f'{r}\n' for r in refs) + '\n'
    return fm, body


def doc_plan(f) -> list[tuple[str, str, str | None]]:
    """(path, doc type, key) for every document, in rendering (RNG) order."""
    plan = [(HOME_PATH, 'home', None)]
    plan += [(region_path(r), 'region', r) for r in sorted(f['regions'])]
    plan += [(store_path(f['stores'][s]), 'store', s) for s in sorted(f['stores'])]
    plan += [(sales_path(f['stores'][s]), 'sales_report', s) for s in sorted(f['stores'])]
    plan += [(dept_path(f['depts'][d]), 'department', d) for d in sorted(f['depts'])]
    plan += [(category_path(f['categories'][c], f['depts']), 'category', c) for c in sorted(f['categories'])]
    plan += [(supplier_path(f['suppliers'][s]), 'supplier', s) for s in sorted(f['suppliers'])]
    plan += [(promo_path(f['promos'][p]), 'promotion', p) for p in sorted(f['promos'])]
    plan += [(policy_path(f['policies'][p]), 'policy', p) for p in sorted(f['policies'])]
    plan += [(f"/archive/{f['archives'][a][1]}.md", 'archive', a) for a in sorted(f['archives'])]
    plan.append((SCHEMA_PATH, 'schema', None))
    return plan


def render(conn, variant: str = 'standard') -> dict[str, str]:
    """Bundle-absolute path -> markdown text for one variant."""
    if variant not in VARIANTS:
        raise ValueError(f'variant must be one of {VARIANTS}')
    f = load_facts(conn)
    rng = random.Random(seed.SEED)
    docs = {}
    for path, doc_type, key in doc_plan(f):
        if doc_type == 'home':
            fm, body = _render_home(rng, f)
        elif doc_type == 'region':
            fm, body = _render_region(rng, f, key)
        elif doc_type == 'store':
            fm, body = _render_store(rng, f, key, variant)
        elif doc_type == 'sales_report':
            fm, body = _render_sales(rng, f, key)
        elif doc_type == 'department':
            fm, body = _render_department(rng, f, key)
        elif doc_type == 'category':
            fm, body = _render_category(rng, f, key, variant)
        elif doc_type == 'supplier':
            fm, body = _render_supplier(rng, f, key)
        elif doc_type == 'promotion':
            fm, body = _render_promotion(rng, f, key)
        elif doc_type == 'policy':
            fm, body = _render_policy(rng, f, key)
        elif doc_type == 'archive':
            fm, body = _render_archive(rng, f, key, variant)
        else:
            fm, body = _render_schema(rng, f)
        body += f'\n## Notes\n\n{filler(path, doc_type)}\n'
        docs[path] = _build_markdown(fm, body)
    return docs


def write(variant: str, out_dir: str | None = None) -> int:
    out_dir = out_dir or OUT_DIRS[variant]
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    conn = seed.connect()
    docs = render(conn, variant)
    conn.close()
    for path, content in docs.items():
        full = os.path.join(out_dir, path.lstrip('/'))
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, 'w', encoding='utf-8', newline='\n') as fh:
            fh.write(content)
    return len(docs)


if __name__ == '__main__':
    for v in VARIANTS:
        print(f'{v}: {write(v)} docs -> {OUT_DIRS[v]}')
