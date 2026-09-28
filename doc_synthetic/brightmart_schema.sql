CREATE TABLE region (
    region_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    manager TEXT NOT NULL
);

CREATE TABLE store (
    store_id TEXT PRIMARY KEY,
    region_id TEXT NOT NULL,
    name TEXT NOT NULL,
    city TEXT NOT NULL,
    state TEXT NOT NULL,
    format TEXT NOT NULL,
    opened_year INTEGER NOT NULL,
    sq_ft INTEGER NOT NULL,
    has_pharmacy INTEGER NOT NULL,
    has_fuel INTEGER NOT NULL,
    status TEXT NOT NULL,
    FOREIGN KEY (region_id) REFERENCES region(region_id),
    CHECK (format IN ('supercenter', 'neighborhood', 'express')),
    CHECK (status IN ('open', 'closed', 'remodeling')),
    CHECK (has_pharmacy IN (0, 1)),
    CHECK (has_fuel IN (0, 1))
);

CREATE TABLE department (
    dept_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    slug TEXT NOT NULL,
    "group" TEXT NOT NULL
);

CREATE TABLE store_department (
    store_id TEXT NOT NULL,
    dept_id TEXT NOT NULL,
    dept_manager TEXT NOT NULL,
    PRIMARY KEY (store_id, dept_id),
    FOREIGN KEY (store_id) REFERENCES store(store_id),
    FOREIGN KEY (dept_id) REFERENCES department(dept_id)
);

CREATE TABLE supplier (
    supplier_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    slug TEXT NOT NULL,
    country TEXT NOT NULL
);

CREATE TABLE category (
    cat_id TEXT PRIMARY KEY,
    dept_id TEXT NOT NULL,
    name TEXT NOT NULL,
    slug TEXT NOT NULL,
    supplier_id TEXT NOT NULL,
    FOREIGN KEY (dept_id) REFERENCES department(dept_id),
    FOREIGN KEY (supplier_id) REFERENCES supplier(supplier_id)
);

CREATE TABLE weekly_sales (
    store_id TEXT NOT NULL,
    dept_id TEXT NOT NULL,
    week INTEGER NOT NULL,
    sales REAL NOT NULL,
    PRIMARY KEY (store_id, dept_id, week),
    FOREIGN KEY (store_id) REFERENCES store(store_id),
    FOREIGN KEY (dept_id) REFERENCES department(dept_id)
);

CREATE TABLE promotion (
    promo_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    slug TEXT NOT NULL,
    dept_id TEXT NOT NULL,
    start TEXT NOT NULL,
    end TEXT NOT NULL,
    discount_pct INTEGER NOT NULL,
    FOREIGN KEY (dept_id) REFERENCES department(dept_id)
);

CREATE TABLE promotion_format (
    promo_id TEXT NOT NULL,
    format TEXT NOT NULL,
    PRIMARY KEY (promo_id, format),
    FOREIGN KEY (promo_id) REFERENCES promotion(promo_id)
);

CREATE TABLE policy (
    policy_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    slug TEXT NOT NULL,
    applies_to TEXT NOT NULL
);

CREATE TABLE policy_fact (
    policy_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    PRIMARY KEY (policy_id, key),
    FOREIGN KEY (policy_id) REFERENCES policy(policy_id)
);

CREATE TABLE archive_doc (
    archive_id TEXT PRIMARY KEY,
    slug TEXT NOT NULL,
    subject TEXT NOT NULL,
    stale_key TEXT NOT NULL,
    stale_value TEXT NOT NULL,
    status TEXT NOT NULL
);
