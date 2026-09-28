"""Brightmart v4 question sets (next iteration, test plan T1-T4, T7).

Builds two disjoint sets from the v4 fact database and the rendered standard
corpus: a development set (ids ``bm4-s…``) and a held-out set (ids
``bm4-h-s…``). Each has 40 base questions in each of five strata (S1 direct
fact, S2 two-hop, S3 metadata set, S4 hierarchy set, S5 unanswerable), every
one also written as a paraphrase: 400 records per set.

Gold is computed, never typed: answers and gold documents come from SQL over
`brightmart_v4_seed.connect()` and spans from the rendered documents. Items are
drawn per question kind from candidate lists in a seeded order: the first n
candidates go to the development set, the next n to the held-out set, so no
(kind, entity) pair appears in both. The two sets also use different wordings.

S3 items carry ``condition``: ``numeric`` (numeric/date conditions, the T3
subset) or ``categorical``. S3/S4 items carry ``target_type`` and
``answer_names`` (display names of the gold entities) for answer scoring.

Usage:
    python doc_synthetic/brightmart_v4_questions_build.py
"""

import json
import os
import random
import re
import sys
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import brightmart_v4_seed as seed
import brightmart_v4_render as R

CORPUS = 'synthetic_retail_v4'
SETS = ('dev', 'heldout')
ID_PREFIX = {'dev': 'bm4-', 'heldout': 'bm4-h-'}
OUT = {s: {k: f'doc_synthetic/brightmart_v4_{s}_{k}.jsonl' for k in ('questions', 'qrels', 'gold_spans')}
       for s in SETS}
S3_SIZE = (2, 30)
S4_SIZE = (2, 30)
QWORD = {1: 'first', 2: 'second', 3: 'third', 4: 'fourth'}
FMT_PL = {'supercenter': 'supercenters', 'neighborhood': 'neighborhood stores', 'express': 'express stores'}
LOOKALIKE_STORES = ['Riverside', 'Riverdale', 'Riverbend', 'Cedar Falls', 'Cedar Park', 'Prairie Hub',
                    'Prairie Rose', 'Prairie View', 'Harbor Point', 'Harbor Springs', 'Harbor Lights',
                    'Maple Crossing', 'Maple Grove', 'Sierra Vista', 'Sierra Pines', 'Red River', 'Red Rock']
LOOKALIKE_SUPPLIERS = ['SUP01', 'SUP02', 'SUP04', 'SUP05', 'SUP03', 'SUP21']

# ---------------------------------------------------------------------------
# Wording: kind -> set -> (template, paraphrase). Fields are filled per item.
# ---------------------------------------------------------------------------
W = {
    's1_sqft': {'dev': ('How many square feet does {store} occupy?', "What's the total floor space at {short}, in square feet?"),
                'heldout': ('What is the floor area of {store} in square feet?', 'How large is the {short} location, measured in square feet?')},
    's1_city': {'dev': ('In which city is {store} located?', 'What town is the {short} store in?'),
                'heldout': ('Which city is {store} in?', 'What city is home to the {short} store?')},
    's1_year': {'dev': ('In what year did {store} open?', 'When did {short} first open its doors?'),
                'heldout': ('What year was {store} established?', 'Since what year has the {short} store been operating?')},
    's1_status': {'dev': ('What is the current operating status of {store}?', 'Is {short} open right now, closed, or being remodeled?'),
                  'heldout': ('What is the status of {store} today?', 'Can shoppers visit {short} at the moment, or is it shut or under renovation?')},
    's1_manager': {'dev': ('Who manages the {dept} department at {store}?', 'Who runs {dept} at the {short} store?'),
                   'heldout': ('Who is the {dept} department manager at {store}?', 'Which person is in charge of {dept} at {short}?')},
    's1_sales': {'dev': ('What were {store_poss} {dept} sales in Q{q} 2025?', 'How much did {dept} sell at {short} in the {qword} quarter of 2025?'),
                 'heldout': ('How much revenue did the {dept} department at {store} report for Q{q} 2025?', 'What did {short} take in from {dept} during the {qword} quarter of 2025?')},
    's1_promo_discount': {'dev': ('What discount does the {promo} promotion offer?', 'How much off do shoppers get during {promo}?'),
                          'heldout': ('What percentage discount comes with the {promo} promotion?', 'During {promo}, how big is the price cut?')},
    's1_promo_dates': {'dev': ('When does the {promo} promotion run?', 'What are the start and end dates of {promo}?'),
                       'heldout': ('During which dates is the {promo} promotion in effect?', 'From when to when can customers use {promo}?')},
    's1_supplier_country': {'dev': ('In which country is {supplier} headquartered?', 'Where is {supplier} based?'),
                            'heldout': ('Which country is {supplier} located in?', '{supplier} operates out of which country?')},
    's2_category_supplier_country': {
        'dev': ('In which country is the supplier of {cat} based?', 'What country is the company that provides our {cat} located in?'),
        'heldout': ('Where is the company that supplies {cat} headquartered?', 'The vendor behind {cat} operates out of which country?')},
    's2_store_region_manager': {
        'dev': ('Who is the regional manager responsible for {store}?', 'Which regional director oversees the {short} store?'),
        'heldout': ('Which regional manager oversees {store}?', 'Who leads the region that {short} belongs to?')},
    's2_category_dept_group': {
        'dev': ('Which department group does the department that sells {cat} belong to?', 'Is the department selling {cat} in the food, health or general group?'),
        'heldout': ('What group is the department carrying {cat} part of?', 'The department behind {cat} falls under which group?')},
    's2_promo_dept_group': {
        'dev': ('Which department group does the department running {promo} belong to?', '{promo} is run by a department in which group?'),
        'heldout': ('The {promo} promotion belongs to a department in which group?', 'What group does the department behind {promo} sit in?')},
    's3_format_pharmacy': {'dev': ('Which {fmt_pl} {ph_t}?', 'Among our {fmt_pl}, which ones {ph_p}?'),
                           'heldout': ('Which {fmt_pl} {ph_t}?', 'List the {fmt_pl} that {ph_p}.')},
    's3_format_fuel': {'dev': ('Which {fmt_pl} {fu_t}?', 'Among our {fmt_pl}, which ones {fu_p}?'),
                       'heldout': ('Which {fmt_pl} {fu_t}?', 'List the {fmt_pl} that {fu_p}.')},
    's3_pharmacy_fuel': {'dev': ('{t}', '{p}'), 'heldout': ('{t}', '{p}')},
    's3_region_fuel': {'dev': ('Which {region} stores {fu_t}?', 'In the {region} region, which locations {fu_p}?'),
                       'heldout': ('Which {region} locations {fu_t}?', 'Name the {region} stores {fu_p}.')},
    's3_status': {'dev': ('{t}', '{p}'), 'heldout': ('{t}', '{p}')},
    's3_supplier_country': {'dev': ('{t}', '{p}'), 'heldout': ('{t}', '{p}')},
    's3_dept_group': {'dev': ('Which departments belong to the {group} group?', 'Which departments are part of the {group} group of departments?'),
                      'heldout': ('Which departments are in the {group} group?', 'Name the departments grouped under {group}.')},
    's3_promo_format': {'dev': ('{t}', '{p}'), 'heldout': ('{t}', '{p}')},
    's3_region_open': {'dev': ('Which {region} stores are currently open?', 'Which locations in the {region} region are open for business?'),
                       'heldout': ('Which {region} stores are open today?', 'Name the {region} stores that are operating normally.')},
    's3_region_pharmacy_fuel': {
        'dev': ('Which {region} stores have both a pharmacy and a fuel station?', 'In the {region} region, which locations offer pharmacy services and fuel?'),
        'heldout': ('Which {region} stores offer a pharmacy as well as fuel?', 'Name the {region} stores where you can fill a prescription and buy gas.')},
    's3_region_no_pharmacy': {'dev': ('Which {region} stores have no pharmacy?', 'In the {region} region, which locations lack pharmacy services?'),
                              'heldout': ('Which {region} stores lack a pharmacy?', 'Name the {region} stores without a pharmacy counter.')},
    's3_year_after': {'dev': ('Which stores opened after {y}?', 'Which locations first opened their doors later than {y}?'),
                      'heldout': ('Which stores were established after {y}?', 'Name the stores that opened sometime after {y}.')},
    's3_year_before': {'dev': ('Which stores opened before {y}?', 'Which locations have been operating since before {y}?'),
                       'heldout': ('Which stores were established before {y}?', 'Name the stores that opened prior to {y}.')},
    's3_year_between': {'dev': ('Which stores opened between {y1} and {y2}?', 'Which locations opened in the years {y1} through {y2}?'),
                        'heldout': ('Which stores were established between {y1} and {y2}?', 'Name the stores that opened from {y1} to {y2}, inclusive.')},
    's3_sqft_gt': {'dev': ('Which stores are larger than {x} square feet?', 'Which locations have more than {x} square feet of floor space?'),
                   'heldout': ('Which stores exceed {x} square feet?', 'Name the stores bigger than {x} square feet.')},
    's3_sqft_lt': {'dev': ('Which stores are smaller than {x} square feet?', 'Which locations have less than {x} square feet of space?'),
                   'heldout': ('Which stores are under {x} square feet?', 'Name the stores with a floor area below {x} square feet.')},
    's3_sqft_between': {'dev': ('Which stores are between {x1} and {x2} square feet?', 'Which locations measure from {x1} to {x2} square feet?'),
                        'heldout': ('Which stores have a floor area between {x1} and {x2} square feet?', 'Name the stores sized between {x1} and {x2} square feet.')},
    's3_format_year_after': {'dev': ('Which {fmt_pl} opened after {y}?', 'Which of our {fmt_pl} first opened later than {y}?'),
                             'heldout': ('Which {fmt_pl} were established after {y}?', 'Name the {fmt_pl} that opened sometime after {y}.')},
    's3_pharmacy_year_before': {'dev': ('Which stores with a pharmacy opened before {y}?', 'Which pharmacy locations have been open since before {y}?'),
                                'heldout': ('Which stores that have a pharmacy were established before {y}?', 'Name the stores with a pharmacy that opened prior to {y}.')},
    's3_promo_discount_ge': {'dev': ('Which promotions offer a discount of {d}% or more?', 'Which promotions take at least {d}% off?'),
                             'heldout': ('Which promotions give {d}% off or more?', 'Name the promotions with discounts of {d} percent or higher.')},
    's3_promo_month': {'dev': ('Which promotions run in {month}?', 'Which promotions are active during {month}?'),
                       'heldout': ('Which promotions are in effect in {month}?', 'Name the promotions running at some point in {month}.')},
    's3_promo_start_month': {'dev': ('Which promotions start in {month}?', 'Which promotions kick off in {month}?'),
                             'heldout': ('Which promotions begin in {month}?', 'Name the promotions launching in {month}.')},
    's4_region': {'dev': ('Which stores belong to the {region} region?', 'What locations make up the {region} region?'),
                  'heldout': ('Which stores are part of the {region} region?', 'List every store in the {region} region.')},
    's4_multi_region': {'dev': ('Which stores are in the {r1} or {r2} regions?', 'What locations belong to either the {r1} or the {r2} region?'),
                        'heldout': ('Which stores belong to the {r1} region or the {r2} region?', 'List all stores located in the {r1} and {r2} regions.')},
    's4_department': {'dev': ('Which categories are in the {dept} department?', 'What product categories does the {dept} department cover?'),
                      'heldout': ('Which categories belong to the {dept} department?', 'List the product categories under {dept}.')},
    's4_multi_department': {'dev': ('Which categories are in the {d1} or {d2} departments?', 'What categories fall under either {d1} or {d2}?'),
                            'heldout': ('Which categories belong to the {d1} department or the {d2} department?', 'List the categories of the {d1} and {d2} departments.')},
    's4_group': {'dev': ('Which categories fall under {group}-group departments?', 'What categories are sold by departments in the {group} group?'),
                 'heldout': ('Which categories belong to departments in the {group} group?', 'List the categories carried by {group}-group departments.')},
    's4_region_pharmacy': {'dev': ('Which stores in the {region} region have a pharmacy?', 'Where in the {region} region can customers fill a prescription?'),
                           'heldout': ('Which stores in the {region} region include a pharmacy?', 'List the {region} region stores that offer pharmacy services.')},
    's4_region_format': {'dev': ('Which stores in the {region} region are {fmt_pl}?', 'What {fmt_pl} does the {region} region have?'),
                         'heldout': ('Which {fmt_pl} are in the {region} region?', 'List the {fmt_pl} located in the {region} region.')},
    's4_region_year_after': {'dev': ('Which stores in the {region} region opened after {y}?', 'Which {region} region locations first opened later than {y}?'),
                             'heldout': ('Which stores in the {region} region were established after {y}?', 'List the {region} region stores that opened after {y}.')},
    's4_multi_region_format': {'dev': ('Which {fmt_pl} are in the {r1} or {r2} regions?', 'What {fmt_pl} operate in either the {r1} or the {r2} region?'),
                               'heldout': ('Which {fmt_pl} belong to the {r1} region or the {r2} region?', 'List the {fmt_pl} in the {r1} and {r2} regions.')},
    's4_region_not_supercenter': {'dev': ('Which stores in the {region} region are not supercenters?', 'Which {region} region locations are smaller-format stores rather than supercenters?'),
                                  'heldout': ('Which stores in the {region} region are something other than supercenters?', 'List the non-supercenter stores in the {region} region.')},
    's5_city': {'dev': ('What is the floor area of the Brightmart store in {term}?', 'How big is the {term} Brightmart location?'),
                'heldout': ('In what year did the Brightmart store in {term} open?', 'When did Brightmart open its {term} location?')},
    's5_category': {'dev': ('Who supplies the {term} category?', 'Which vendor provides our {term}?'),
                    'heldout': ('Which department sells {term}?', 'Where in the store would I find {term}?')},
    's5_promo': {'dev': ('What discount does the {term} promotion offer?', 'How much off is {term}?'),
                 'heldout': ('When does the {term} promotion run?', 'What are the dates of {term}?')},
    's5_attribute': {'dev': ('What is the {term} of {store}?', 'Do you know the {term} for {short}?'),
                     'heldout': ('Can you tell me the {term} of {store}?', "What's {short_poss} {term}?")},
    's5_store': {'dev': ('In which state is Brightmart {term} located?', 'Where is the Brightmart {term} store?'),
                 'heldout': ('Who manages the Grocery department at Brightmart {term}?', 'Who runs Grocery at the {term} store?')},
    's5_region': {'dev': ('Who manages the {term} region?', 'Which director leads the {term} region?'),
                  'heldout': ('Which stores belong to the {term} region?', 'What locations are in the {term} region?')},
    's5_policy': {'dev': ('What does the {term} say about refunds?', 'How does the {term} work?'),
                  'heldout': ('Who does the {term} apply to?', 'Which stores follow the {term}?')},
    's5_supplier': {'dev': ('In which country is {term} based?', 'Where is the vendor {term} headquartered?'),
                    'heldout': ('Which categories does {term} supply?', 'What products come from {term}?')},
}

PHARMACY_PHRASES = {  # set -> has_pharmacy -> (template phrase, paraphrase phrase)
    'dev': {1: ('have a pharmacy', 'offer pharmacy services'), 0: ('have no pharmacy', 'operate without a pharmacy')},
    'heldout': {1: ('include a pharmacy', 'come with an in-store pharmacy'), 0: ('lack a pharmacy', "don't offer pharmacy services")},
}
FUEL_PHRASES = {
    'dev': {1: ('have a fuel station', 'sell fuel'), 0: ('have no fuel station', "don't sell fuel")},
    'heldout': {1: ('offer fuel', 'where customers can buy gas'), 0: ('lack a fuel station', 'without gas pumps')},
}
FORMAT_FUEL_PHRASES = {
    'dev': {1: ('have a fuel station', 'sell fuel'), 0: ('have no fuel station', "don't sell fuel")},
    'heldout': {1: ('operate a fuel station', 'sell gas'), 0: ('lack a fuel station', 'have no gas pumps')},
}
PHARMACY_FUEL_TEXT = {
    'dev': {(1, 0): ('Which stores have a pharmacy but no fuel station?', 'Where can I find a pharmacy but no gas pumps?'),
            (0, 1): ('Which stores have a fuel station but no pharmacy?', "Which locations sell fuel but don't have a pharmacy?"),
            (1, 1): ('Which stores have both a pharmacy and a fuel station?', 'Which locations offer both pharmacy services and fuel?'),
            (0, 0): ('Which stores have neither a pharmacy nor a fuel station?', 'Which locations offer neither fuel nor pharmacy services?')},
    'heldout': {(1, 0): ('Which stores include a pharmacy yet lack a fuel station?', "Name the stores with an in-store pharmacy that don't sell fuel."),
                (0, 1): ('Which stores operate a fuel station without a pharmacy?', 'Name the stores that sell gas but have no pharmacy counter.'),
                (1, 1): ('Which stores offer a pharmacy as well as a fuel station?', 'Name the stores where shoppers can fill a prescription and fill up the tank.'),
                (0, 0): ('Which stores lack both a pharmacy and a fuel station?', 'Name the stores with no pharmacy counter and no gas pumps.')},
}
STATUS_TEXT = {
    'dev': {'remodeling': ('Which stores are currently remodeling?', 'Which locations are under renovation right now?'),
            'closed': ('Which stores are permanently closed?', 'Which locations have shut down for good?'),
            'notopen': ('Which stores are not currently open?', "Which locations can't be visited right now?")},
    'heldout': {'remodeling': ('Which stores are undergoing remodeling?', 'Name the stores being remodeled.'),
                'closed': ('Which stores have closed?', 'Name the stores that no longer operate.'),
                'notopen': ('Which stores are not open at the moment?', 'Name the stores that are currently unavailable to shoppers.')},
}
SUPPLIER_COUNTRY_TEXT = {
    'dev': {'in': ('Which suppliers are based in {country}?', 'Which vendors operate out of {country}?'),
            'outside': ('Which suppliers are based outside the United States?', 'Which vendors are located abroad, outside the United States?')},
    'heldout': {'in': ('Which suppliers are headquartered in {country}?', 'Name the suppliers located in {country}.'),
                'outside': ('Which suppliers are headquartered outside the United States?', 'Name the suppliers based in a country other than the United States.')},
}
PROMO_FORMAT_TEXT = {
    'dev': {'at': ('Which promotions run at {fmt} stores?', 'Which promotions can shoppers at {fmt} stores use?'),
            'only': ('Which promotions are limited to supercenter stores?', 'Which promotions are only available at supercenters?')},
    'heldout': {'at': ('Which promotions are available at {fmt} stores?', 'Name the promotions offered in {fmt} locations.'),
                'only': ('Which promotions apply only to supercenters?', 'Name the promotions that supercenters alone take part in.')},
}

# (template, paraphrase, answer format) per policy fact key; a key is used in one set only.
POLICY_Q = {
    'standard_window_days': ('How many days do customers have to return most items under the Returns Policy?', "What's the usual return window for most purchases?", '{v} days'),
    'electronics_window_days': ('How long is the return window for electronics under the Returns Policy?', 'How many days do I get to bring back electronics?', '{v} days'),
    'weekday_hours': ('What are the weekday pharmacy hours under the Pharmacy Hours Policy?', 'When is the pharmacy open Monday through Friday?', '{v}'),
    'sunday_hours': ('What are the Sunday pharmacy hours under the Pharmacy Hours Policy?', 'What time does the pharmacy open and close on Sundays?', '{v}'),
    'cents_off_per_gallon': ('How many cents off per gallon does the Fuel Rewards Policy give?', 'How much is knocked off each gallon with fuel rewards?', '{v} cents'),
    'spend_threshold_usd': ('How much must a customer spend to qualify under the Fuel Rewards Policy?', "What's the minimum spend to earn fuel rewards?", '${v}'),
    'match_window_days': ('Within how many days of purchase can a customer request a price match?', 'How long after buying something can I ask for a price match?', '{v} days'),
    'rain_check_valid_days': ('How long is a rain check valid under the Rain Check Policy?', 'For how many days can a rain check be redeemed?', '{v} days'),
    'limit_per_customer': ('How many rain checks can a customer receive per visit?', "What's the cap on rain checks per shopping trip?", '{v}'),
    'minimum_deposit_pct': ('What deposit does the Layaway Policy require?', 'What percentage do I have to put down to start a layaway?', '{v}%'),
    'layaway_period_days': ('How long can items stay on layaway under the Layaway Policy?', 'How many days do I have to pay off a layaway?', '{v} days'),
    'reload_minimum_usd': ('What is the minimum amount for reloading a gift card?', "What's the smallest reload allowed on a gift card?", '${v}'),
    'pickup_window_hours': ('How long are curbside pickup orders held once they are ready?', 'How many hours do I have to collect a curbside order?', '{v} hours'),
    'minimum_order_usd': ('What is the minimum order value for curbside pickup?', 'How much does a curbside order need to total at minimum?', '${v}'),
    'senior_discount_pct': ('What discount does the Senior Discount Policy give?', 'How much do eligible seniors save with the senior discount?', '{v}%'),
    'eligible_age': ('What is the minimum age to qualify for the senior discount?', 'How old do you need to be to get the senior discount?', '{v}'),
    'discount_day': ('On which day of the week is the senior discount offered?', 'What weekday can seniors use their discount?', '{v}'),
    'deposit_pct': ('What deposit is required for a special order?', 'What percentage down payment do special orders need?', '{v}%'),
    'arrival_notice_days': ('Within how many days are special order customers told the expected arrival?', 'How soon after ordering does a special-order customer get an arrival estimate?', '{v} days'),
}
ARCHIVED_POLICY_KEYS = ['standard_window_days', 'match_window_days', 'layaway_period_days']

ABSENT = {
    's5_city': ['Denver', 'Omaha', 'Memphis', 'Spokane', 'Anchorage', 'Honolulu', 'Louisville', 'Milwaukee', 'Pittsburgh', 'Birmingham'],
    's5_category': ['Frozen Foods', 'Garden Tools', 'Seafood', 'Baby Formula', 'Video Games', 'Jewelry', 'Luggage', 'Party Supplies', 'Scented Candles', 'Swimwear'],
    's5_promo': ['Holiday Toy Drive', 'Memorial Day Blowout', 'Super Saver Sunday', 'Midnight Madness', 'Spring Cleaning Sale', 'Labor Day Deals', 'Valentine Treats', 'Graduation Gifts', 'Lunar New Year Feast', 'Fourth of July Grill Sale'],
    's5_attribute': ['phone number', 'ZIP code', 'email address', 'number of employees', 'parking capacity', 'fax number', 'website address', 'holiday opening hours', 'loyalty program', 'drive-through hours'],
    's5_store': ['Willow Bend', 'Copper Ridge', 'Falcon Point', 'Heron Bay', 'Juniper Flats', 'Quail Run', 'Stag Creek', 'Moss Landing', 'Iron Gate', 'Hollow Pines'],
    's5_region': ['Northwest', 'Gulf Coast', 'New England', 'Alaska', 'Caribbean', 'Appalachia', 'Rust Belt', 'Tri-State', 'Heartland', 'Sun Belt'],
    's5_policy': ['Warranty Policy', 'Bulk Order Policy', 'Price Adjustment Policy', 'Pet Policy', 'Smoking Policy', 'Delivery Policy', 'Coupon Policy', 'Recycling Policy', 'Donation Policy', 'Photography Policy'],
    's5_supplier': ['Zephyr Foods', 'Blackwood Electronics', 'Silverline Apparel', 'Crescent Farms', 'Nimbus Toys', 'Oakridge Hardware', 'Beacon Beverage', 'Lotus Health', 'Granite Sporting', 'Harbor Freight Lines'],
}

# stratum, kind, items per set
PLAN = [
    ('S1', 's1_sqft', 4), ('S1', 's1_city', 4), ('S1', 's1_year', 4), ('S1', 's1_status', 4),
    ('S1', 's1_manager', 4), ('S1', 's1_sales', 4), ('S1', 's1_promo_discount', 4),
    ('S1', 's1_promo_dates', 4), ('S1', 's1_policy', 4), ('S1', 's1_supplier_country', 4),
    ('S2', 's2_category_supplier_country', 10), ('S2', 's2_store_region_manager', 10),
    ('S2', 's2_category_dept_group', 10), ('S2', 's2_promo_dept_group', 10),
    ('S3', 's3_format_pharmacy', 2), ('S3', 's3_format_fuel', 2), ('S3', 's3_pharmacy_fuel', 1),
    ('S3', 's3_region_fuel', 4), ('S3', 's3_status', 1), ('S3', 's3_supplier_country', 1),
    ('S3', 's3_dept_group', 1), ('S3', 's3_promo_format', 2), ('S3', 's3_region_open', 3),
    ('S3', 's3_region_pharmacy_fuel', 2), ('S3', 's3_region_no_pharmacy', 1),
    ('S3', 's3_year_after', 2), ('S3', 's3_year_before', 3), ('S3', 's3_year_between', 2),
    ('S3', 's3_sqft_gt', 2), ('S3', 's3_sqft_lt', 2), ('S3', 's3_sqft_between', 2),
    ('S3', 's3_format_year_after', 2), ('S3', 's3_pharmacy_year_before', 1),
    ('S3', 's3_promo_discount_ge', 1), ('S3', 's3_promo_month', 2), ('S3', 's3_promo_start_month', 1),
    ('S4', 's4_region', 4), ('S4', 's4_multi_region', 5), ('S4', 's4_department', 5),
    ('S4', 's4_multi_department', 5), ('S4', 's4_group', 1), ('S4', 's4_region_pharmacy', 4),
    ('S4', 's4_region_format', 4), ('S4', 's4_region_year_after', 4), ('S4', 's4_multi_region_format', 5),
    ('S4', 's4_region_not_supercenter', 3),
    ('S5', 's5_city', 5), ('S5', 's5_category', 5), ('S5', 's5_promo', 5), ('S5', 's5_attribute', 5),
    ('S5', 's5_store', 5), ('S5', 's5_region', 5), ('S5', 's5_policy', 5), ('S5', 's5_supplier', 5),
]
NUMERIC_KINDS = {'s3_year_after', 's3_year_before', 's3_year_between', 's3_sqft_gt', 's3_sqft_lt',
                 's3_sqft_between', 's3_format_year_after', 's3_pharmacy_year_before',
                 's3_promo_discount_ge', 's3_promo_month', 's3_promo_start_month'}


def _p(path: str) -> str:
    return path.lstrip('/')


def body_lines(doc_text: str) -> list[str]:
    """Body lines outside frontmatter and the Notes section."""
    body = doc_text.split('---\n', 2)[2] if doc_text.startswith('---\n') else doc_text
    body = body.split('\n## Notes')[0]
    return [ln.strip() for ln in body.split('\n') if ln.strip()]


def find_line(doc_text: str, needles, pattern: str | None = None) -> str | None:
    """First non-heading body line containing every needle (or matching `pattern`)."""
    needles = [needles] if isinstance(needles, str) else list(needles)
    for line in body_lines(doc_text):
        if line.startswith('#'):
            continue
        if pattern is not None and not re.search(pattern, line, re.I):
            continue
        if all(n in line for n in needles):
            return line
    return None


def first_line(doc_text: str) -> str:
    return next(ln for ln in body_lines(doc_text) if not ln.startswith('#'))


class Ctx:
    def __init__(self, conn):
        self.conn = conn
        self.f = R.load_facts(conn)
        self.docs = R.render(conn, 'standard')
        self.docs_np = R.render(conn, 'noparent')

    def doc(self, path: str) -> str:
        return self.docs['/' + path.lstrip('/')]

    # entity -> (path, display name)
    def store(self, sid):
        return _p(R.store_path(self.f['stores'][sid])), self.f['stores'][sid][2]

    def entity(self, eid):
        f = self.f
        if eid.startswith('SUP'):
            return _p(R.supplier_path(f['suppliers'][eid])), f['suppliers'][eid][1]
        if eid.startswith('S'):
            return self.store(eid)
        if eid.startswith('P'):
            return _p(R.promo_path(f['promos'][eid])), f['promos'][eid][1]
        if eid.startswith('D'):
            return _p(R.dept_path(f['depts'][eid])), f['depts'][eid][1]
        if eid.startswith('C'):
            return _p(R.category_path(f['categories'][eid], f['depts'])), f['categories'][eid][2]
        raise ValueError(eid)


def _rng(kind: str) -> random.Random:
    return random.Random(f'{seed.SEED}:{kind}')


def _shuffled(kind: str, items: list, priority=None) -> list:
    """Priority items first (shuffled), then the rest (shuffled); deterministic per kind."""
    rng = _rng(kind)
    pri = [x for x in items if priority and priority(x)]
    rest = [x for x in items if not (priority and priority(x))]
    rng.shuffle(pri)
    rng.shuffle(rest)
    return pri + rest


# ---------------------------------------------------------------------------
# Item builders: return None when the candidate is unusable (e.g. set size out of range)
# ---------------------------------------------------------------------------

def _single(ctx, path, answer, span, fields):
    if span is None:
        raise ValueError(f'no span for {path} / {answer}')
    return {'answer': answer, 'gold': [path], 'spans': [{'doc_path': path, 'span': span}], 'fields': fields}


def _store_fields(ctx, sid):
    name = ctx.f['stores'][sid][2]
    poss = lambda n: f"{n}'" if n.endswith('s') else f"{n}'s"
    return {'store': name, 'short': R.short_name(name), 'store_poss': poss(name),
            'short_poss': poss(R.short_name(name))}


def build_s1(ctx, kind, cand):
    f = ctx.f
    if kind in ('s1_sqft', 's1_city', 's1_year', 's1_status'):
        s = f['stores'][cand]
        path = ctx.store(cand)[0]
        text = ctx.doc(path)
        if kind == 's1_sqft':
            return _single(ctx, path, f'{s[7]:,} square feet', find_line(text, f'{s[7]:,}'), _store_fields(ctx, cand))
        if kind == 's1_city':
            return _single(ctx, path, s[3], find_line(text, f'{s[3]}, {s[4]}'), _store_fields(ctx, cand))
        if kind == 's1_year':
            return _single(ctx, path, str(s[6]), find_line(text, str(s[6])), _store_fields(ctx, cand))
        return _single(ctx, path, s[10], find_line(text, [], rf'\b{s[10]}\b'), _store_fields(ctx, cand))
    if kind == 's1_manager':
        sid, did = cand
        path = ctx.store(sid)[0]
        mgr = f['store_depts'][sid][did]
        dname = f['depts'][did][1]
        return _single(ctx, path, mgr, find_line(ctx.doc(path), [f'[{dname}]', mgr]),
                       {**_store_fields(ctx, sid), 'dept': dname})
    if kind == 's1_sales':
        sid, did, q = cand
        s = f['stores'][sid]
        path = _p(R.sales_path(s))
        amount = R.quarterly_sales(f, sid, did)[q - 1]
        dname = f['depts'][did][1]
        return _single(ctx, path, amount, find_line(ctx.doc(path), [f'| {dname} |', amount]),
                       {**_store_fields(ctx, sid), 'dept': dname, 'q': q, 'qword': QWORD[q]})
    if kind in ('s1_promo_discount', 's1_promo_dates'):
        p = f['promos'][cand]
        path = ctx.entity(cand)[0]
        if kind == 's1_promo_discount':
            return _single(ctx, path, f'{p[6]}%', find_line(ctx.doc(path), f'{p[6]}% off'), {'promo': p[1]})
        return _single(ctx, path, f'{p[4]} to {p[5]}', find_line(ctx.doc(path), f'{p[4]} to {p[5]}'), {'promo': p[1]})
    if kind == 's1_policy':
        pid, key = cand
        pol = f['policies'][pid]
        path = _p(R.policy_path(pol))
        value = f['policy_facts'][pid][key]
        options = R.policy_sentence_options(key, value, pol[1])
        span = next((ln for ln in body_lines(ctx.doc(path)) if ln in options), None)
        t, p, fmt = POLICY_Q[key]
        item = _single(ctx, path, fmt.format(v=value), span, {})
        item['text_override'] = (t, p)
        return item
    if kind == 's1_supplier_country':
        s = f['suppliers'][cand]
        path = ctx.entity(cand)[0]
        return _single(ctx, path, s[3], find_line(ctx.doc(path), s[3]), {'supplier': s[1]})
    raise ValueError(kind)


def _two(ctx, paths_needles, answer, fields):
    spans = []
    for path, needles in paths_needles:
        span = find_line(ctx.doc(path), needles)
        if span is None:
            raise ValueError(f'no span for {path} / {needles}')
        spans.append({'doc_path': path, 'span': span})
    return {'answer': answer, 'gold': [p for p, _ in paths_needles], 'spans': spans, 'fields': fields}


def build_s2(ctx, kind, cand):
    f = ctx.f
    if kind == 's2_category_supplier_country':
        c = f['categories'][cand]
        sup = f['suppliers'][c[4]]
        return _two(ctx, [(ctx.entity(cand)[0], sup[1]), (ctx.entity(c[4])[0], sup[3])], sup[3], {'cat': c[2]})
    if kind == 's2_store_region_manager':
        s = f['stores'][cand]
        region = f['regions'][s[1]]
        return _two(ctx, [(ctx.store(cand)[0], f'Back to {region[1]}'), (_p(R.region_path(s[1])), region[2])],
                    region[2], _store_fields(ctx, cand))
    if kind == 's2_category_dept_group':
        c = f['categories'][cand]
        d = f['depts'][c[1]]
        return _two(ctx, [(ctx.entity(cand)[0], f'[{d[1]}]'), (ctx.entity(c[1])[0], f'{d[3]} group')], d[3], {'cat': c[2]})
    if kind == 's2_promo_dept_group':
        p = f['promos'][cand]
        d = f['depts'][p[3]]
        return _two(ctx, [(ctx.entity(cand)[0], f'[{d[1]}]'), (ctx.entity(p[3])[0], f'{d[3]} group')], d[3], {'promo': p[1]})
    raise ValueError(kind)


def _set_item(ctx, sql, params, fields, size, target_type, paths=None):
    rows = [r[0] for r in ctx.conn.execute(sql, params).fetchall()]
    if not (size[0] <= len(rows) <= size[1]):
        return None
    ents = [ctx.entity(e) for e in rows]
    gold = [p for p, _ in ents]
    names = [n for _, n in ents]
    spans = [{'doc_path': p, 'span': first_line(ctx.doc(p))} for p in gold]
    spans.append({'doc_path': None, 'gold_sql': sql, 'gold_sql_params': list(params)})
    for chain in paths or []:
        spans.append({'doc_path': None, 'path': chain})
    return {'answer': ', '.join(names), 'gold': gold, 'spans': spans, 'fields': fields,
            'target_type': target_type, 'answer_names': names}


STORE_SQL = 'SELECT store_id FROM store WHERE {} ORDER BY store_id'
PROMO_SQL = 'SELECT promo_id FROM promotion WHERE {} ORDER BY promo_id'


def _month_bounds(m: int) -> tuple[str, str]:
    last = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1]
    return f'2025-{m:02d}-01', f'2025-{m:02d}-{last:02d}'


def build_s3(ctx, kind, cand, which):
    f = ctx.f
    rname = lambda rid: f['regions'][rid][1]
    if kind == 's3_format_pharmacy':
        fmt, hp = cand
        t, p = PHARMACY_PHRASES[which][hp]
        return _set_item(ctx, STORE_SQL.format('format=? AND has_pharmacy=?'), (fmt, hp),
                         {'fmt_pl': FMT_PL[fmt], 'ph_t': t, 'ph_p': p}, S3_SIZE, 'store')
    if kind == 's3_format_fuel':
        fmt, hf = cand
        t, p = FORMAT_FUEL_PHRASES[which][hf]
        return _set_item(ctx, STORE_SQL.format('format=? AND has_fuel=?'), (fmt, hf),
                         {'fmt_pl': FMT_PL[fmt], 'fu_t': t, 'fu_p': p}, S3_SIZE, 'store')
    if kind == 's3_pharmacy_fuel':
        t, p = PHARMACY_FUEL_TEXT[which][cand]
        return _set_item(ctx, STORE_SQL.format('has_pharmacy=? AND has_fuel=?'), cand, {'t': t, 'p': p}, S3_SIZE, 'store')
    if kind == 's3_region_fuel':
        rid, hf = cand
        t, p = FUEL_PHRASES[which][hf]
        return _set_item(ctx, STORE_SQL.format('region_id=? AND has_fuel=?'), (rid, hf),
                         {'region': rname(rid), 'fu_t': t, 'fu_p': p}, S3_SIZE, 'store')
    if kind == 's3_status':
        t, p = STATUS_TEXT[which][cand]
        if cand == 'notopen':
            return _set_item(ctx, STORE_SQL.format('status!=?'), ('open',), {'t': t, 'p': p}, S3_SIZE, 'store')
        return _set_item(ctx, STORE_SQL.format('status=?'), (cand,), {'t': t, 'p': p}, S3_SIZE, 'store')
    if kind == 's3_supplier_country':
        sql = 'SELECT supplier_id FROM supplier WHERE country{}? ORDER BY supplier_id'
        if cand == 'outside':
            t, p = SUPPLIER_COUNTRY_TEXT[which]['outside']
            return _set_item(ctx, sql.format('!='), ('United States',), {'t': t, 'p': p}, S3_SIZE, 'supplier')
        country = R._country_text(cand)
        t, p = (x.format(country=country) for x in SUPPLIER_COUNTRY_TEXT[which]['in'])
        return _set_item(ctx, sql.format('='), (cand,), {'t': t, 'p': p}, S3_SIZE, 'supplier')
    if kind == 's3_dept_group':
        return _set_item(ctx, 'SELECT dept_id FROM department WHERE "group"=? ORDER BY dept_id', (cand,),
                         {'group': cand}, S3_SIZE, 'department')
    if kind == 's3_promo_format':
        if cand == 'only':
            t, p = PROMO_FORMAT_TEXT[which]['only']
            sql = ('SELECT promo_id FROM promotion WHERE promo_id NOT IN '
                   '(SELECT promo_id FROM promotion_format WHERE format!=?) ORDER BY promo_id')
            return _set_item(ctx, sql, ('supercenter',), {'t': t, 'p': p}, S3_SIZE, 'promotion')
        t, p = (x.format(fmt=cand) for x in PROMO_FORMAT_TEXT[which]['at'])
        return _set_item(ctx, 'SELECT DISTINCT promo_id FROM promotion_format WHERE format=? ORDER BY promo_id',
                         (cand,), {'t': t, 'p': p}, S3_SIZE, 'promotion')
    if kind == 's3_region_open':
        return _set_item(ctx, STORE_SQL.format('region_id=? AND status=?'), (cand, 'open'),
                         {'region': rname(cand)}, S3_SIZE, 'store')
    if kind == 's3_region_pharmacy_fuel':
        return _set_item(ctx, STORE_SQL.format('region_id=? AND has_pharmacy=? AND has_fuel=?'), (cand, 1, 1),
                         {'region': rname(cand)}, S3_SIZE, 'store')
    if kind == 's3_region_no_pharmacy':
        return _set_item(ctx, STORE_SQL.format('region_id=? AND has_pharmacy=?'), (cand, 0),
                         {'region': rname(cand)}, S3_SIZE, 'store')
    if kind == 's3_year_after':
        return _set_item(ctx, STORE_SQL.format('opened_year>?'), (cand,), {'y': cand}, S3_SIZE, 'store')
    if kind == 's3_year_before':
        return _set_item(ctx, STORE_SQL.format('opened_year<?'), (cand,), {'y': cand}, S3_SIZE, 'store')
    if kind == 's3_year_between':
        y1, y2 = cand
        return _set_item(ctx, STORE_SQL.format('opened_year BETWEEN ? AND ?'), (y1, y2), {'y1': y1, 'y2': y2}, S3_SIZE, 'store')
    if kind == 's3_sqft_gt':
        return _set_item(ctx, STORE_SQL.format('sq_ft>?'), (cand,), {'x': f'{cand:,}'}, S3_SIZE, 'store')
    if kind == 's3_sqft_lt':
        return _set_item(ctx, STORE_SQL.format('sq_ft<?'), (cand,), {'x': f'{cand:,}'}, S3_SIZE, 'store')
    if kind == 's3_sqft_between':
        x1, x2 = cand
        return _set_item(ctx, STORE_SQL.format('sq_ft BETWEEN ? AND ?'), (x1, x2),
                         {'x1': f'{x1:,}', 'x2': f'{x2:,}'}, S3_SIZE, 'store')
    if kind == 's3_format_year_after':
        fmt, y = cand
        return _set_item(ctx, STORE_SQL.format('format=? AND opened_year>?'), (fmt, y),
                         {'fmt_pl': FMT_PL[fmt], 'y': y}, S3_SIZE, 'store')
    if kind == 's3_pharmacy_year_before':
        return _set_item(ctx, STORE_SQL.format('has_pharmacy=? AND opened_year<?'), (1, cand), {'y': cand}, S3_SIZE, 'store')
    if kind == 's3_promo_discount_ge':
        return _set_item(ctx, PROMO_SQL.format('discount_pct>=?'), (cand,), {'d': cand}, S3_SIZE, 'promotion')
    if kind == 's3_promo_month':
        first, last = _month_bounds(cand)
        return _set_item(ctx, PROMO_SQL.format('start<=? AND end>=?'), (last, first),
                         {'month': R.MONTHS[cand - 1]}, S3_SIZE, 'promotion')
    if kind == 's3_promo_start_month':
        first, last = _month_bounds(cand)
        return _set_item(ctx, PROMO_SQL.format('start BETWEEN ? AND ?'), (first, last),
                         {'month': R.MONTHS[cand - 1]}, S3_SIZE, 'promotion')
    raise ValueError(kind)


HOME = 'brightmart-home.md'


def build_s4(ctx, kind, cand):
    f = ctx.f
    rname = lambda rid: f['regions'][rid][1]
    rchain = lambda rid: [HOME, _p(R.region_path(rid))]
    dchain = lambda did: [HOME, _p(R.dept_path(f['depts'][did]))]
    in_list = lambda n: ','.join(['?'] * n)
    if kind == 's4_region':
        return _set_item(ctx, STORE_SQL.format('region_id=?'), (cand,), {'region': rname(cand)}, S4_SIZE, 'store', [rchain(cand)])
    if kind == 's4_multi_region':
        r1, r2 = cand
        return _set_item(ctx, STORE_SQL.format(f'region_id IN ({in_list(2)})'), cand,
                         {'r1': rname(r1), 'r2': rname(r2)}, S4_SIZE, 'store', [rchain(r1), rchain(r2)])
    if kind == 's4_department':
        return _set_item(ctx, 'SELECT cat_id FROM category WHERE dept_id=? ORDER BY cat_id', (cand,),
                         {'dept': f['depts'][cand][1]}, S4_SIZE, 'category', [dchain(cand)])
    if kind == 's4_multi_department':
        d1, d2 = cand
        return _set_item(ctx, f'SELECT cat_id FROM category WHERE dept_id IN ({in_list(2)}) ORDER BY cat_id', cand,
                         {'d1': f['depts'][d1][1], 'd2': f['depts'][d2][1]}, S4_SIZE, 'category', [dchain(d1), dchain(d2)])
    if kind == 's4_group':
        dids = sorted(d for d in f['depts'] if f['depts'][d][3] == cand)
        sql = 'SELECT cat_id FROM category WHERE dept_id IN (SELECT dept_id FROM department WHERE "group"=?) ORDER BY cat_id'
        return _set_item(ctx, sql, (cand,), {'group': cand}, S4_SIZE, 'category', [dchain(d) for d in dids])
    if kind == 's4_region_pharmacy':
        return _set_item(ctx, STORE_SQL.format('region_id=? AND has_pharmacy=?'), (cand, 1),
                         {'region': rname(cand)}, S4_SIZE, 'store', [rchain(cand)])
    if kind == 's4_region_format':
        rid, fmt = cand
        return _set_item(ctx, STORE_SQL.format('region_id=? AND format=?'), (rid, fmt),
                         {'region': rname(rid), 'fmt_pl': FMT_PL[fmt]}, S4_SIZE, 'store', [rchain(rid)])
    if kind == 's4_region_year_after':
        rid, y = cand
        return _set_item(ctx, STORE_SQL.format('region_id=? AND opened_year>?'), (rid, y),
                         {'region': rname(rid), 'y': y}, S4_SIZE, 'store', [rchain(rid)])
    if kind == 's4_multi_region_format':
        r1, r2, fmt = cand
        return _set_item(ctx, STORE_SQL.format(f'region_id IN ({in_list(2)}) AND format=?'), (r1, r2, fmt),
                         {'r1': rname(r1), 'r2': rname(r2), 'fmt_pl': FMT_PL[fmt]}, S4_SIZE, 'store',
                         [rchain(r1), rchain(r2)])
    if kind == 's4_region_not_supercenter':
        return _set_item(ctx, STORE_SQL.format('region_id=? AND format!=?'), (cand, 'supercenter'),
                         {'region': rname(cand)}, S4_SIZE, 'store', [rchain(cand)])
    raise ValueError(kind)


def build_s5(ctx, kind, cand):
    term, sid = cand
    for variant_docs in (ctx.docs, ctx.docs_np):
        if any(term.lower() in text.lower() for text in variant_docs.values()):
            raise ValueError(f'S5 term {term!r} occurs in the corpus')
    fields = {'term': term}
    if sid:
        fields.update(_store_fields(ctx, sid))
    return {'answer': None, 'gold': [], 'spans': [{'absent_term': term}], 'fields': fields}


# ---------------------------------------------------------------------------
# Candidate lists
# ---------------------------------------------------------------------------

def candidates(ctx, kind) -> list:
    f = ctx.f
    stores = sorted(f['stores'])
    regions = sorted(f['regions'])
    fmts = ['supercenter', 'neighborhood', 'express']
    lookalike = lambda sid: R.short_name(f['stores'][sid][2]) in LOOKALIKE_STORES
    if kind == 's1_sqft':
        archived = [a[2] for a in f['archives'].values() if a[2].startswith('S') and not a[2].startswith('SUP')]
        return _shuffled(kind, archived)
    if kind == 's1_city':
        return _shuffled(kind, stores, lookalike)
    if kind == 's1_year':
        return _shuffled(kind, stores, lookalike)
    if kind == 's1_status':
        return _shuffled(kind, stores, lambda s: f['stores'][s][10] != 'open')
    if kind == 's1_manager':
        rng = _rng(kind + ':dept')
        return _shuffled(kind, [(s, rng.choice(sorted(f['store_depts'][s]))) for s in stores], lambda c: lookalike(c[0]))
    if kind == 's1_sales':
        rng = _rng(kind + ':dept')
        open_stores = [s for s in stores if f['stores'][s][10] != 'closed']
        return _shuffled(kind, [(s, rng.choice(sorted(f['store_depts'][s])), rng.randint(1, 4)) for s in open_stores])
    if kind in ('s1_promo_discount', 's1_promo_dates'):
        drafted = {a[2] for a in f['archives'].values() if a[2].startswith('P') and not a[2].startswith('POL')}
        return _shuffled(kind, sorted(f['promos']), lambda p: p in drafted and kind == 's1_promo_discount')
    if kind == 's1_policy':
        keys = [(pid, k) for pid in sorted(f['policy_facts']) for k in sorted(f['policy_facts'][pid]) if k in POLICY_Q]
        return _shuffled(kind, keys, lambda c: c[1] in ARCHIVED_POLICY_KEYS)
    if kind == 's1_supplier_country':
        return _shuffled(kind, sorted(f['suppliers']), lambda s: s in LOOKALIKE_SUPPLIERS)
    if kind in ('s2_category_supplier_country', 's2_category_dept_group'):
        return _shuffled(kind, sorted(f['categories']))
    if kind == 's2_store_region_manager':
        return _shuffled(kind, stores, lookalike)
    if kind == 's2_promo_dept_group':
        return _shuffled(kind, sorted(f['promos']))
    if kind in ('s3_format_pharmacy', 's3_format_fuel'):
        return _shuffled(kind, [(fm, v) for fm in fmts for v in (1, 0)])
    if kind == 's3_pharmacy_fuel':
        return _shuffled(kind, [(1, 0), (0, 1), (1, 1), (0, 0)])
    if kind == 's3_region_fuel':
        return _shuffled(kind, [(r, v) for r in regions for v in (1, 0)])
    if kind == 's3_status':
        return _shuffled(kind, ['remodeling', 'closed', 'notopen'])
    if kind == 's3_supplier_country':
        return _shuffled(kind, sorted({s[3] for s in f['suppliers'].values()}) + ['outside'])
    if kind == 's3_dept_group':
        return _shuffled(kind, ['food', 'health', 'general'])
    if kind == 's3_promo_format':
        return _shuffled(kind, fmts + ['only'])
    if kind in ('s3_region_open', 's3_region_pharmacy_fuel', 's3_region_no_pharmacy',
                's4_region', 's4_region_pharmacy', 's4_region_not_supercenter'):
        return _shuffled(kind, regions)
    if kind in ('s3_year_after', 's3_year_before'):
        return _shuffled(kind, list(range(1998, 2022)))
    if kind == 's3_year_between':
        return _shuffled(kind, [(y, y + w) for y in range(1996, 2021) for w in (2, 3, 4)])
    if kind == 's3_sqft_gt':
        return _shuffled(kind, list(range(155000, 200000, 5000)))
    if kind == 's3_sqft_lt':
        return _shuffled(kind, list(range(12000, 21000, 1000)))
    if kind == 's3_sqft_between':
        return _shuffled(kind, [(x, x + 10000) for x in range(35000, 55000, 2500)])
    if kind == 's3_format_year_after':
        return _shuffled(kind, [(fm, y) for fm in fmts for y in range(2000, 2021, 2)])
    if kind == 's3_pharmacy_year_before':
        return _shuffled(kind, list(range(2000, 2021)))
    if kind == 's3_promo_discount_ge':
        return _shuffled(kind, [15, 18, 20, 22, 25])
    if kind in ('s3_promo_month', 's3_promo_start_month'):
        return _shuffled(kind, list(range(1, 13)))
    if kind == 's4_multi_region':
        return _shuffled(kind, list(combinations(regions, 2)))
    if kind == 's4_department':
        return _shuffled(kind, sorted(f['depts']))
    if kind == 's4_multi_department':
        return _shuffled(kind, list(combinations(sorted(f['depts']), 2)))
    if kind == 's4_group':
        return _shuffled(kind, ['food', 'health', 'general'])
    if kind == 's4_region_format':
        return _shuffled(kind, [(r, fm) for r in regions for fm in fmts])
    if kind == 's4_region_year_after':
        return _shuffled(kind, [(r, y) for r in regions for y in range(2000, 2020, 3)])
    if kind == 's4_multi_region_format':
        return _shuffled(kind, [(a, b, fm) for a, b in combinations(regions, 2) for fm in fmts])
    if kind.startswith('s5_'):
        rng = _rng(kind + ':store')
        return [(t, rng.choice(stores) if kind == 's5_attribute' else None) for t in ABSENT[kind]]
    raise ValueError(kind)


def build_item(ctx, stratum, kind, cand, which):
    if stratum == 'S1':
        return build_s1(ctx, kind, cand)
    if stratum == 'S2':
        return build_s2(ctx, kind, cand)
    if stratum == 'S3':
        return build_s3(ctx, kind, cand, which)
    if stratum == 'S4':
        return build_s4(ctx, kind, cand)
    return build_s5(ctx, kind, cand)


def allocate(ctx) -> dict[str, list[dict]]:
    """Per set: list of built items (with stratum, kind, text_t, text_p) in PLAN order."""
    out = {s: [] for s in SETS}
    for stratum, kind, n in PLAN:
        picked = {s: [] for s in SETS}
        for cand in candidates(ctx, kind):
            which = 'dev' if len(picked['dev']) < n else 'heldout'
            if len(picked['heldout']) >= n:
                break
            item = build_item(ctx, stratum, kind, cand, which)
            if item is None:
                continue
            if 'text_override' in item:
                t, p = item.pop('text_override')
            else:
                t, p = (x.format(**item['fields']) for x in W[kind][which])
            item.update({'stratum': stratum, 'kind': kind, 'text_t': t, 'text_p': p})
            picked[which].append(item)
        for s in SETS:
            if len(picked[s]) != n:
                raise ValueError(f'{kind}: only {len(picked[s])} usable candidates for {s}, need {n}')
            out[s].extend(picked[s])
    return out


def records(items: list[dict], which: str):
    questions, qrels, spans = [], [], []
    counters: dict[str, int] = {}
    for it in items:
        st = it['stratum'].lower()
        counters[st] = counters.get(st, 0) + 1
        base = f"{ID_PREFIX[which]}{st}-{counters[st]:02d}"
        for suffix, text, origin in (('t', it['text_t'], 'synthetic_template'), ('p', it['text_p'], 'synthetic_paraphrase')):
            qid = f'{base}-{suffix}'
            q = {'corpus': CORPUS, 'origin': origin, 'query_id': qid, 'reference_answer': it['answer'],
                 'reference_claim': None, 'source_paths': it['gold'], 'stratum': it['stratum'], 'text': text,
                 'kind': it['kind']}
            if it['stratum'] == 'S3':
                q['condition'] = 'numeric' if it['kind'] in NUMERIC_KINDS else 'categorical'
            if 'target_type' in it:
                q['target_type'] = it['target_type']
                q['answer_names'] = it['answer_names']
            questions.append(q)
            qrels += [{'doc_path': p, 'query_id': qid, 'relevance': 1} for p in it['gold']]
            spans += [{**s, 'query_id': qid} for s in it['spans']]
    return questions, qrels, spans


def _write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n')


def build(conn=None) -> dict[str, tuple[list, list, list]]:
    conn = conn or seed.connect()
    ctx = Ctx(conn)
    items = allocate(ctx)
    return {s: records(items[s], s) for s in SETS}


def _sort_key(qid: str):
    base, suffix = qid.rsplit('-', 1)
    return (base, suffix != 't')


if __name__ == '__main__':
    built = build()
    for s in SETS:
        questions, qrels, spans = built[s]
        order = {q['query_id']: i for i, q in enumerate(sorted(questions, key=lambda q: _sort_key(q['query_id'])))}
        _write_jsonl(OUT[s]['questions'], sorted(questions, key=lambda q: order[q['query_id']]))
        _write_jsonl(OUT[s]['qrels'], sorted(qrels, key=lambda r: order[r['query_id']]))
        _write_jsonl(OUT[s]['gold_spans'], sorted(spans, key=lambda r: order[r['query_id']]))
        by = {}
        for q in questions:
            by[q['stratum']] = by.get(q['stratum'], 0) + 1
        print(f'{s}: {len(questions)} questions {dict(sorted(by.items()))}')
