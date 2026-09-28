"""Brightmart v4 fact database (next iteration, test plan T1-T3).

Same schema as the pilot (`brightmart_schema.sql`), about seven times the
entities: 8 regions, 96 stores, 15 departments, 60 categories, 25 suppliers,
20 promotions, 10 policies and 16 archive/draft records. Store attributes are
drawn from one seeded RNG (seed 20260924); `opened_year` is drawn independently
of `format` (T3: decouple store format from opening year).

The in-memory SQLite database is the single source of truth for the v4 corpus
(`brightmart_v4_render.py`) and for every gold answer
(`brightmart_v4_questions_build.py`).
"""

import os
import random
import sqlite3

SEED = 20260924

REGIONS = [
    ('northeast', 'Northeast', 'Dana Whitlock'),
    ('mid-atlantic', 'Mid-Atlantic', 'Owen Castellano'),
    ('southeast', 'Southeast', 'Marcus Ebo'),
    ('midwest', 'Midwest', 'Priya Halvorsen'),
    ('great-plains', 'Great Plains', 'Lorna Kittredge'),
    ('southwest', 'Southwest', 'Rafael Montoya'),
    ('mountain', 'Mountain', 'Tomas Arriaga'),
    ('pacific', 'Pacific', 'Imogen Sato'),
]

# (short name, city, state), 12 per region in REGIONS order.
STORE_SITES = {
    'northeast': [
        ('Riverside', 'Hartford', 'CT'), ('Riverdale', 'Albany', 'NY'),
        ('Harbor Point', 'Portland', 'ME'), ('Beacon Hill', 'Worcester', 'MA'),
        ('Granite Falls', 'Manchester', 'NH'), ('Maple Crossing', 'Burlington', 'VT'),
        ('Seaside Commons', 'Providence', 'RI'), ('Stonebridge', 'Syracuse', 'NY'),
        ('Elm Terrace', 'New Haven', 'CT'), ('Lighthouse Square', 'Bangor', 'ME'),
        ('Birch Meadow', 'Springfield', 'MA'), ('Kingsbury', 'Rochester', 'NY'),
    ],
    'mid-atlantic': [
        ('Liberty Square', 'Allentown', 'PA'), ('Chestnut Grove', 'Trenton', 'NJ'),
        ('Bayview', 'Annapolis', 'MD'), ('Brandywine', 'Wilmington', 'DE'),
        ('Blue Ridge', 'Roanoke', 'VA'), ('Keystone Plaza', 'Harrisburg', 'PA'),
        ('Shore Point', 'Toms River', 'NJ'), ('Fox Chase', 'Frederick', 'MD'),
        ('Colonial Heights', 'Richmond', 'VA'), ('Laurel Park', 'Scranton', 'PA'),
        ('Tidewater', 'Norfolk', 'VA'), ('Millbrook', 'Dover', 'DE'),
    ],
    'southeast': [
        ('Magnolia Row', 'Savannah', 'GA'), ('Palmetto Cross', 'Columbia', 'SC'),
        ('Pinehurst', 'Raleigh', 'NC'), ('Coral Bay', 'Tampa', 'FL'),
        ('Live Oak', 'Tallahassee', 'FL'), ('Peachtree Commons', 'Macon', 'GA'),
        ('Cypress Landing', 'Charleston', 'SC'), ('Smoky Hollow', 'Knoxville', 'TN'),
        ('Azalea Park', 'Mobile', 'AL'), ('Sandhill', 'Fayetteville', 'NC'),
        ('Riverbend', 'Chattanooga', 'TN'), ('Sunrise Plaza', 'Orlando', 'FL'),
    ],
    'midwest': [
        ('Cedar Falls', 'Cedar Falls', 'IA'), ('Lakeshore', 'Madison', 'WI'),
        ('Northstar', 'Duluth', 'MN'), ('Prairie View', 'Peoria', 'IL'),
        ('Buckeye Commons', 'Columbus', 'OH'), ('Harbor Springs', 'Traverse City', 'MI'),
        ('Hoosier Crossing', 'Fort Wayne', 'IN'), ('Maple Grove', 'St. Cloud', 'MN'),
        ('Willow Creek', 'Green Bay', 'WI'), ('Oak Hollow', 'Dayton', 'OH'),
        ('Fox River', 'Aurora', 'IL'), ('Sugar Creek', 'Des Moines', 'IA'),
    ],
    'great-plains': [
        ('Prairie Hub', 'Wichita', 'KS'), ('Prairie Rose', 'Lincoln', 'NE'),
        ('Red River', 'Fargo', 'ND'), ('Sooner Plaza', 'Norman', 'OK'),
        ('Black Hills', 'Rapid City', 'SD'), ('Sunflower Commons', 'Topeka', 'KS'),
        ('Cottonwood', 'Grand Island', 'NE'), ('Wheatfield', 'Salina', 'KS'),
        ('Cimarron', 'Tulsa', 'OK'), ('Badlands Gate', 'Sioux Falls', 'SD'),
        ('Buffalo Ridge', 'Bismarck', 'ND'), ('Flint Hills', 'Manhattan', 'KS'),
    ],
    'southwest': [
        ('Desert Bloom', 'Tucson', 'AZ'), ('Mesa Verde', 'Albuquerque', 'NM'),
        ('Saguaro Point', 'Phoenix', 'AZ'), ('Lone Star Plaza', 'Austin', 'TX'),
        ('Rio Grande', 'El Paso', 'TX'), ('Canyon Ridge', 'Flagstaff', 'AZ'),
        ('Adobe Springs', 'Santa Fe', 'NM'), ('Bluebonnet', 'Waco', 'TX'),
        ('Painted Sky', 'Las Cruces', 'NM'), ('Copper Creek', 'Yuma', 'AZ'),
        ('Pecos Crossing', 'Lubbock', 'TX'), ('Sierra Vista', 'Sierra Vista', 'AZ'),
    ],
    'mountain': [
        ('Cedar Park', 'Boise', 'ID'), ('Summit Ridge', 'Reno', 'NV'),
        ('Aspen Grove', 'Fort Collins', 'CO'), ('Wasatch Commons', 'Provo', 'UT'),
        ('Big Sky', 'Billings', 'MT'), ('Teton Plaza', 'Casper', 'WY'),
        ('Silver Peak', 'Carson City', 'NV'), ('Pikes View', 'Colorado Springs', 'CO'),
        ('Snake River', 'Idaho Falls', 'ID'), ('Red Rock', 'St. George', 'UT'),
        ('Glacier Point', 'Missoula', 'MT'), ('Elk Meadow', 'Cheyenne', 'WY'),
    ],
    'pacific': [
        ('Evergreen', 'Tacoma', 'WA'), ('Cascade View', 'Bend', 'OR'),
        ('Redwood Crossing', 'Eureka', 'CA'), ('Bayshore', 'Oakland', 'CA'),
        ('Rainier Commons', 'Olympia', 'WA'), ('Willamette', 'Salem', 'OR'),
        ('Sunset Cliffs', 'San Diego', 'CA'), ('Puget Landing', 'Everett', 'WA'),
        ('Rogue Valley', 'Medford', 'OR'), ('Sierra Pines', 'Fresno', 'CA'),
        ('Harbor Lights', 'Long Beach', 'CA'), ('Orchard Hill', 'Yakima', 'WA'),
    ],
}

DEPARTMENTS = [
    ('D01', 'Grocery', 'grocery', 'food'),
    ('D02', 'Bakery', 'bakery', 'food'),
    ('D03', 'Deli', 'deli', 'food'),
    ('D04', 'Beverages', 'beverages', 'food'),
    ('D05', 'Pharmacy', 'pharmacy', 'health'),
    ('D06', 'Beauty', 'beauty', 'health'),
    ('D07', 'Wellness', 'wellness', 'health'),
    ('D08', 'Electronics', 'electronics', 'general'),
    ('D09', 'Home & Garden', 'home-garden', 'general'),
    ('D10', 'Apparel', 'apparel', 'general'),
    ('D11', 'Toys', 'toys', 'general'),
    ('D12', 'Automotive', 'automotive', 'general'),
    ('D13', 'Sporting Goods', 'sporting-goods', 'general'),
    ('D14', 'Office Supplies', 'office-supplies', 'general'),
    ('D15', 'Pet Care', 'pet-care', 'general'),
]

SUPPLIERS = [
    ('SUP01', 'Northwind Provisions', 'northwind-provisions', 'United States'),
    ('SUP02', 'Northgate Provisions', 'northgate-provisions', 'Canada'),
    ('SUP03', 'Kestrel Electronics', 'kestrel-electronics', 'Taiwan'),
    ('SUP04', 'Greenfield Farms Co-op', 'greenfield-farms-coop', 'United States'),
    ('SUP05', 'Greenfield Home Supply', 'greenfield-home-supply', 'Canada'),
    ('SUP06', 'Alder & Finch Textiles', 'alder-finch-textiles', 'Portugal'),
    ('SUP07', 'Copperline Auto Parts', 'copperline-auto-parts', 'Mexico'),
    ('SUP08', 'Bluewater Beverage Company', 'bluewater-beverage-company', 'United States'),
    ('SUP09', 'Harbor & Vine Foods', 'harbor-vine-foods', 'Italy'),
    ('SUP10', 'Solstice Wellness Labs', 'solstice-wellness-labs', 'Germany'),
    ('SUP11', 'Maplewood Paper Goods', 'maplewood-paper-goods', 'Canada'),
    ('SUP12', 'Tessera Cosmetics', 'tessera-cosmetics', 'France'),
    ('SUP13', 'Ironpeak Outdoor', 'ironpeak-outdoor', 'United States'),
    ('SUP14', 'Lumen Home Goods', 'lumen-home-goods', 'Vietnam'),
    ('SUP15', 'Pinwheel Toy Works', 'pinwheel-toy-works', 'Denmark'),
    ('SUP16', 'Orchard Crest Bakery Supply', 'orchard-crest-bakery-supply', 'United States'),
    ('SUP17', 'Delta Pharma Distributors', 'delta-pharma-distributors', 'India'),
    ('SUP18', 'Quillmark Office', 'quillmark-office', 'Japan'),
    ('SUP19', 'Stonehaven Dairy', 'stonehaven-dairy', 'Netherlands'),
    ('SUP20', 'Pawprint Pet Supply', 'pawprint-pet-supply', 'United States'),
    ('SUP21', 'Velocity Cycle Company', 'velocity-cycle-company', 'Taiwan'),
    ('SUP22', 'Cobalt Audio Works', 'cobalt-audio-works', 'South Korea'),
    ('SUP23', 'Redfern Meats', 'redfern-meats', 'Australia'),
    ('SUP24', 'Summit Trail Apparel', 'summit-trail-apparel', 'Vietnam'),
    ('SUP25', 'Riverstone Tea Traders', 'riverstone-tea-traders', 'India'),
]

# (cat_id, dept_id, name, slug, supplier_id), 4 per department.
CATEGORIES = [
    ('C01', 'D01', 'Fresh Produce', 'fresh-produce', 'SUP04'),
    ('C02', 'D01', 'Dairy', 'dairy', 'SUP19'),
    ('C03', 'D01', 'Canned Goods', 'canned-goods', 'SUP01'),
    ('C04', 'D01', 'Breakfast Cereal', 'breakfast-cereal', 'SUP01'),
    ('C05', 'D02', 'Artisan Bread', 'artisan-bread', 'SUP04'),
    ('C06', 'D02', 'Pastries', 'pastries', 'SUP16'),
    ('C07', 'D02', 'Birthday Cakes', 'birthday-cakes', 'SUP16'),
    ('C08', 'D02', 'Bagels', 'bagels', 'SUP02'),
    ('C09', 'D03', 'Sliced Meats', 'sliced-meats', 'SUP23'),
    ('C10', 'D03', 'Specialty Cheese', 'specialty-cheese', 'SUP09'),
    ('C11', 'D03', 'Prepared Salads', 'prepared-salads', 'SUP04'),
    ('C12', 'D03', 'Rotisserie Chicken', 'rotisserie-chicken', 'SUP23'),
    ('C13', 'D04', 'Sparkling Water', 'sparkling-water', 'SUP08'),
    ('C14', 'D04', 'Coffee Beans', 'coffee-beans', 'SUP09'),
    ('C15', 'D04', 'Fruit Juice', 'fruit-juice', 'SUP08'),
    ('C16', 'D04', 'Loose Leaf Tea', 'loose-leaf-tea', 'SUP25'),
    ('C17', 'D05', 'OTC Medicines', 'otc-medicines', 'SUP17'),
    ('C18', 'D05', 'Vitamins', 'vitamins', 'SUP10'),
    ('C19', 'D05', 'First Aid', 'first-aid', 'SUP17'),
    ('C20', 'D05', 'Allergy Relief', 'allergy-relief', 'SUP01'),
    ('C21', 'D06', 'Skin Care', 'skin-care', 'SUP12'),
    ('C22', 'D06', 'Hair Care', 'hair-care', 'SUP10'),
    ('C23', 'D06', 'Cosmetics', 'cosmetics', 'SUP12'),
    ('C24', 'D06', 'Fragrances', 'fragrances', 'SUP12'),
    ('C25', 'D07', 'Fitness Nutrition', 'fitness-nutrition', 'SUP10'),
    ('C26', 'D07', 'Sleep Aids', 'sleep-aids', 'SUP17'),
    ('C27', 'D07', 'Herbal Supplements', 'herbal-supplements', 'SUP25'),
    ('C28', 'D07', 'Massage Tools', 'massage-tools', 'SUP14'),
    ('C29', 'D08', 'Televisions', 'televisions', 'SUP03'),
    ('C30', 'D08', 'Phones', 'phones', 'SUP03'),
    ('C31', 'D08', 'Headphones', 'headphones', 'SUP22'),
    ('C32', 'D08', 'Laptops', 'laptops', 'SUP22'),
    ('C33', 'D09', 'Patio Furniture', 'patio-furniture', 'SUP05'),
    ('C34', 'D09', 'Kitchenware', 'kitchenware', 'SUP14'),
    ('C35', 'D09', 'Bedding', 'bedding', 'SUP14'),
    ('C36', 'D09', 'Houseplants', 'houseplants', 'SUP04'),
    ('C37', 'D10', 'Kids Apparel', 'kids-apparel', 'SUP06'),
    ('C38', 'D10', 'Outerwear', 'outerwear', 'SUP24'),
    ('C39', 'D10', 'Footwear', 'footwear', 'SUP24'),
    ('C40', 'D10', 'Activewear', 'activewear', 'SUP06'),
    ('C41', 'D11', 'Board Games', 'board-games', 'SUP15'),
    ('C42', 'D11', 'Building Blocks', 'building-blocks', 'SUP15'),
    ('C43', 'D11', 'Puzzles', 'puzzles', 'SUP11'),
    ('C44', 'D11', 'Plush Toys', 'plush-toys', 'SUP15'),
    ('C45', 'D12', 'Motor Oil', 'motor-oil', 'SUP07'),
    ('C46', 'D12', 'Car Batteries', 'car-batteries', 'SUP07'),
    ('C47', 'D12', 'Wiper Blades', 'wiper-blades', 'SUP07'),
    ('C48', 'D12', 'Tire Care', 'tire-care', 'SUP13'),
    ('C49', 'D13', 'Camping Gear', 'camping-gear', 'SUP13'),
    ('C50', 'D13', 'Fishing Tackle', 'fishing-tackle', 'SUP13'),
    ('C51', 'D13', 'Bicycles', 'bicycles', 'SUP21'),
    ('C52', 'D13', 'Yoga Mats', 'yoga-mats', 'SUP21'),
    ('C53', 'D14', 'Notebooks', 'notebooks', 'SUP11'),
    ('C54', 'D14', 'Printer Ink', 'printer-ink', 'SUP18'),
    ('C55', 'D14', 'Desk Organizers', 'desk-organizers', 'SUP18'),
    ('C56', 'D14', 'Backpacks', 'backpacks', 'SUP24'),
    ('C57', 'D15', 'Dog Food', 'dog-food', 'SUP20'),
    ('C58', 'D15', 'Cat Litter', 'cat-litter', 'SUP20'),
    ('C59', 'D15', 'Pet Toys', 'pet-toys', 'SUP20'),
    ('C60', 'D15', 'Aquarium Supplies', 'aquarium-supplies', 'SUP05'),
]

PROMOTIONS = [
    ('P01', 'Spring Garden Days', 'spring-garden-days', 'D09', '2025-03-15', '2025-04-15', 20),
    ('P02', 'Back to Class', 'back-to-class', 'D10', '2025-08-01', '2025-08-31', 15),
    ('P03', 'Big Screen Weekend', 'big-screen-weekend', 'D08', '2025-11-28', '2025-11-30', 25),
    ('P04', 'Fresh Friday', 'fresh-friday', 'D01', '2025-01-03', '2025-12-26', 10),
    ('P05', 'Winter Care', 'winter-care', 'D05', '2025-12-01', '2025-12-31', 12),
    ('P06', 'Holiday Bake Fest', 'holiday-bake-fest', 'D02', '2025-12-05', '2025-12-24', 18),
    ('P07', 'Summer Sip', 'summer-sip', 'D04', '2025-06-01', '2025-08-31', 8),
    ('P08', 'Deli Days', 'deli-days', 'D03', '2025-04-01', '2025-04-30', 10),
    ('P09', 'Glow Up Week', 'glow-up-week', 'D06', '2025-02-10', '2025-02-16', 30),
    ('P10', 'New Year Reset', 'new-year-reset', 'D07', '2025-01-01', '2025-01-31', 20),
    ('P11', 'Game Night Sale', 'game-night-sale', 'D11', '2025-10-15', '2025-11-15', 22),
    ('P12', 'Road Trip Ready', 'road-trip-ready', 'D12', '2025-05-20', '2025-06-10', 15),
    ('P13', 'Trail Season', 'trail-season', 'D13', '2025-04-15', '2025-05-31', 25),
    ('P14', 'Office Refresh', 'office-refresh', 'D14', '2025-07-15', '2025-08-15', 35),
    ('P15', 'Pet Pals Month', 'pet-pals-month', 'D15', '2025-09-01', '2025-09-30', 12),
    ('P16', 'Harvest Market', 'harvest-market', 'D01', '2025-10-01', '2025-10-31', 15),
    ('P17', 'Cozy Home Event', 'cozy-home-event', 'D09', '2025-11-01', '2025-11-20', 25),
    ('P18', 'Tech Upgrade Days', 'tech-upgrade-days', 'D08', '2025-03-01', '2025-03-10', 18),
    ('P19', 'Cold and Flu Defense', 'cold-and-flu-defense', 'D05', '2025-10-15', '2025-12-15', 10),
    ('P20', 'Fall Fashion Preview', 'fall-fashion-preview', 'D10', '2025-09-10', '2025-09-30', 20),
]

_ALL = ('supercenter', 'neighborhood', 'express')
_SN = ('supercenter', 'neighborhood')
_S = ('supercenter',)
PROMOTION_FORMATS_BY_PROMO = {
    'P01': _SN, 'P02': _S, 'P03': _S, 'P04': _ALL, 'P05': _SN, 'P06': _ALL, 'P07': _ALL,
    'P08': _SN, 'P09': _SN, 'P10': _S, 'P11': _S, 'P12': _S, 'P13': _S, 'P14': _S,
    'P15': _SN, 'P16': _ALL, 'P17': _SN, 'P18': _S, 'P19': _SN, 'P20': _S,
}

POLICIES = [
    ('POL01', 'Returns Policy', 'returns-policy', 'all stores'),
    ('POL02', 'Pharmacy Hours Policy', 'pharmacy-hours-policy', 'stores with pharmacy'),
    ('POL03', 'Fuel Rewards Policy', 'fuel-rewards-policy', 'stores with fuel'),
    ('POL04', 'Price Match Policy', 'price-match-policy', 'all stores'),
    ('POL05', 'Rain Check Policy', 'rain-check-policy', 'all stores'),
    ('POL06', 'Layaway Policy', 'layaway-policy', 'supercenter stores'),
    ('POL07', 'Gift Card Policy', 'gift-card-policy', 'all stores'),
    ('POL08', 'Curbside Pickup Policy', 'curbside-pickup-policy', 'supercenter and neighborhood stores'),
    ('POL09', 'Senior Discount Policy', 'senior-discount-policy', 'all stores'),
    ('POL10', 'Special Order Policy', 'special-order-policy', 'all stores'),
]

POLICY_FACTS = [
    ('POL01', 'standard_window_days', '30'),
    ('POL01', 'electronics_window_days', '15'),
    ('POL01', 'receipt_required', 'yes'),
    ('POL02', 'weekday_hours', '08:00-21:00'),
    ('POL02', 'sunday_hours', '10:00-18:00'),
    ('POL03', 'cents_off_per_gallon', '10'),
    ('POL03', 'spend_threshold_usd', '100'),
    ('POL04', 'match_window_days', '14'),
    ('POL04', 'online_competitors_included', 'no'),
    ('POL05', 'rain_check_valid_days', '45'),
    ('POL05', 'limit_per_customer', '4'),
    ('POL06', 'minimum_deposit_pct', '10'),
    ('POL06', 'layaway_period_days', '60'),
    ('POL07', 'reload_minimum_usd', '10'),
    ('POL07', 'balance_expires', 'never'),
    ('POL08', 'pickup_window_hours', '48'),
    ('POL08', 'minimum_order_usd', '35'),
    ('POL09', 'senior_discount_pct', '5'),
    ('POL09', 'eligible_age', '60'),
    ('POL09', 'discount_day', 'Tuesday'),
    ('POL10', 'deposit_pct', '25'),
    ('POL10', 'arrival_notice_days', '3'),
]

# Archive/draft pages: near-duplicates carrying one stale value.
# Store subjects are resolved to the region's third store below (one per region).
ARCHIVE_PROMOS = [
    ('A09', 'draft-spring-garden-days-2026', 'P01', 'discount_pct', '30', 'draft'),
    ('A10', 'draft-big-screen-weekend-2026', 'P03', 'discount_pct', '35', 'draft'),
    ('A11', 'draft-holiday-bake-fest-2026', 'P06', 'discount_pct', '25', 'draft'),
    ('A12', 'draft-game-night-sale-2026', 'P11', 'discount_pct', '30', 'draft'),
    ('A13', 'draft-office-refresh-2026', 'P14', 'discount_pct', '40', 'draft'),
]
ARCHIVE_POLICIES = [
    ('A14', 'archived-returns-policy-v1', 'POL01', 'standard_window_days', '60', 'deprecated'),
    ('A15', 'archived-price-match-policy-v1', 'POL04', 'match_window_days', '7', 'deprecated'),
    ('A16', 'archived-layaway-policy-v1', 'POL06', 'layaway_period_days', '90', 'deprecated'),
]

FIRST_NAMES = [
    'Alice', 'Bob', 'Carol', 'David', 'Emma', 'Frank', 'Grace', 'Henry', 'Iris', 'Jack',
    'Karen', 'Leo', 'Mona', 'Nathan', 'Olivia', 'Paul', 'Quinn', 'Rachel', 'Samuel', 'Tina',
    'Uma', 'Victor', 'Wendy', 'Xavier', 'Yuki', 'Zara', 'Adrian', 'Bella', 'Carlos', 'Diane',
]
LAST_NAMES = [
    'Chen', 'Martinez', 'Ramirez', 'Kim', 'Johnson', 'Miller', 'Lee', 'Brown', 'Wilson',
    'Anderson', 'Thomas', 'Garcia', 'Rodriguez', 'Taylor', 'Moore', 'Jackson', 'White', 'Davis',
    'Martin', 'Harris', 'Patel', 'Lewis', 'Clark', 'Young', 'Tanaka', 'Ahmed', 'Scott', 'Green',
]

DEPT_WEIGHT = {
    'D01': 1.0, 'D02': 0.14, 'D03': 0.18, 'D04': 0.3, 'D05': 0.45, 'D06': 0.2, 'D07': 0.12,
    'D08': 0.6, 'D09': 0.35, 'D10': 0.3, 'D11': 0.2, 'D12': 0.15, 'D13': 0.18, 'D14': 0.1,
    'D15': 0.16,
}
BASE_BY_FORMAT = {'supercenter': 42000, 'neighborhood': 15000, 'express': 6000}
SQFT_RANGE = {'supercenter': (150000, 200000), 'neighborhood': (35000, 60000), 'express': (10000, 20000)}
P_PHARMACY = {'supercenter': 0.8, 'neighborhood': 0.5, 'express': 0.2}
P_FUEL = {'supercenter': 0.75, 'neighborhood': 0.3, 'express': 0.5}
FORMATS_PER_REGION = ['supercenter'] * 4 + ['neighborhood'] * 5 + ['express'] * 3
CLOSED_LAST_WEEK = 30


def store_departments(store_format: str, has_pharmacy: int) -> list[str]:
    """Departments a store carries; the Pharmacy department exists iff has_pharmacy."""
    if store_format == 'supercenter':
        depts = [d[0] for d in DEPARTMENTS]
    elif store_format == 'neighborhood':
        depts = ['D01', 'D02', 'D03', 'D04', 'D05', 'D06', 'D07', 'D09']
    else:
        depts = ['D01', 'D02', 'D04', 'D05']
    return [d for d in depts if d != 'D05' or has_pharmacy]


def build_stores(rng: random.Random) -> list[tuple]:
    rows = []
    n = 0
    for region_id, _, _ in REGIONS:
        formats = list(FORMATS_PER_REGION)
        rng.shuffle(formats)
        for (short, city, state), fmt in zip(STORE_SITES[region_id], formats):
            n += 1
            lo, hi = SQFT_RANGE[fmt]
            sq_ft = rng.randrange(lo, hi + 1, 500)
            opened_year = rng.randint(1995, 2024)
            has_pharmacy = int(rng.random() < P_PHARMACY[fmt])
            has_fuel = int(rng.random() < P_FUEL[fmt])
            r = rng.random()
            status = 'open' if r < 0.84 else ('remodeling' if r < 0.95 else 'closed')
            rows.append((f'S{n:03d}', region_id, f'Brightmart {short}', city, state, fmt,
                         opened_year, sq_ft, has_pharmacy, has_fuel, status))
    return rows


def build_archive_stores(stores: list[tuple], rng: random.Random) -> list[tuple]:
    """One archived 2022 profile per region (its third store) with a stale floor area."""
    rows = []
    for i, (region_id, _, _) in enumerate(REGIONS, start=1):
        store = [s for s in stores if s[1] == region_id][2]
        slug = store[2].replace('Brightmart ', '').lower().replace(' ', '-')
        stale = int(round(store[7] * rng.uniform(0.78, 0.92) / 500)) * 500
        rows.append((f'A{i:02d}', f'archived-{slug}-profile-2022', store[0], 'sq_ft', str(stale), 'deprecated'))
    return rows


def _read_schema() -> str:
    path = os.path.join(os.path.dirname(__file__), 'brightmart_schema.sql')
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def connect() -> sqlite3.Connection:
    """Create and populate the in-memory Brightmart v4 database."""
    rng = random.Random(SEED)
    conn = sqlite3.connect(':memory:')
    conn.execute('PRAGMA foreign_keys=ON')
    conn.executescript(_read_schema())

    conn.executemany('INSERT INTO region VALUES (?, ?, ?)', REGIONS)
    stores = build_stores(rng)
    conn.executemany('INSERT INTO store VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', stores)
    conn.executemany('INSERT INTO department VALUES (?, ?, ?, ?)', DEPARTMENTS)

    store_dept_rows = []
    for s in stores:
        for dept_id in store_departments(s[5], s[8]):
            manager = f'{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}'
            store_dept_rows.append((s[0], dept_id, manager))
    conn.executemany('INSERT INTO store_department VALUES (?, ?, ?)', store_dept_rows)

    conn.executemany('INSERT INTO supplier VALUES (?, ?, ?, ?)', SUPPLIERS)
    conn.executemany('INSERT INTO category VALUES (?, ?, ?, ?, ?)', CATEGORIES)
    conn.executemany('INSERT INTO promotion VALUES (?, ?, ?, ?, ?, ?, ?)', PROMOTIONS)
    conn.executemany('INSERT INTO promotion_format VALUES (?, ?)',
                     [(p, f) for p, fmts in sorted(PROMOTION_FORMATS_BY_PROMO.items()) for f in fmts])
    conn.executemany('INSERT INTO policy VALUES (?, ?, ?, ?)', POLICIES)
    conn.executemany('INSERT INTO policy_fact VALUES (?, ?, ?)', POLICY_FACTS)
    archives = build_archive_stores(stores, rng) + ARCHIVE_PROMOS + ARCHIVE_POLICIES
    conn.executemany('INSERT INTO archive_doc VALUES (?, ?, ?, ?, ?, ?)', archives)

    sales_rows = []
    for s in stores:
        weeks = CLOSED_LAST_WEEK if s[10] == 'closed' else 52
        for dept_id in store_departments(s[5], s[8]):
            base = BASE_BY_FORMAT[s[5]] * DEPT_WEIGHT[dept_id]
            for week in range(1, weeks + 1):
                sales_rows.append((s[0], dept_id, week, round(base * rng.uniform(0.8, 1.2), 2)))
    conn.executemany('INSERT INTO weekly_sales VALUES (?, ?, ?, ?)', sales_rows)
    conn.commit()
    return conn


if __name__ == '__main__':
    c = connect()
    for table in ('region', 'store', 'department', 'store_department', 'supplier', 'category',
                  'promotion', 'promotion_format', 'policy', 'policy_fact', 'archive_doc', 'weekly_sales'):
        print(f'{table}: {c.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]}')
