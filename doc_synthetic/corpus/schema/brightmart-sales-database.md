---
type: schema
title: Brightmart Sales Database Schema
description: Database schema for the Brightmart synthetic retail dataset.
tags:
- schema
- database
status: stable
timestamp: '2026-01-15'
parent: /brightmart-home.md
---
# Brightmart Sales Database Schema

### region

- `region_id` (PK): TEXT
- `name`: TEXT
- `manager`: TEXT

### store

- `store_id` (PK): TEXT
- `region_id`: TEXT
- `name`: TEXT
- `city`: TEXT
- `state`: TEXT
- `format`: TEXT
- `opened_year`: INTEGER
- `sq_ft`: INTEGER
- `has_pharmacy`: INTEGER
- `has_fuel`: INTEGER
- `status`: TEXT
`store.region_id` references `region.region_id`.

### department

- `dept_id` (PK): TEXT
- `name`: TEXT
- `slug`: TEXT
- `group`: TEXT

### store_department

- `store_id`: TEXT
- `dept_id`: TEXT
- `dept_manager`: TEXT
`store_department.store_id` references `store.store_id`.
`store_department.dept_id` references `department.dept_id`.

### supplier

- `supplier_id` (PK): TEXT
- `name`: TEXT
- `slug`: TEXT
- `country`: TEXT

### category

- `cat_id` (PK): TEXT
- `dept_id`: TEXT
- `name`: TEXT
- `slug`: TEXT
- `supplier_id`: TEXT
`category.dept_id` references `department.dept_id`.
`category.supplier_id` references `supplier.supplier_id`.

### weekly_sales

- `store_id`: TEXT
- `dept_id`: TEXT
- `week`: INTEGER
- `sales`: REAL
`weekly_sales.store_id` references `store.store_id`.
`weekly_sales.dept_id` references `department.dept_id`.

### promotion

- `promo_id` (PK): TEXT
- `name`: TEXT
- `slug`: TEXT
- `dept_id`: TEXT
- `start`: TEXT
- `end`: TEXT
- `discount_pct`: INTEGER
`promotion.dept_id` references `department.dept_id`.

### promotion_format

- `promo_id`: TEXT
- `format`: TEXT
`promotion_format.promo_id` references `promotion.promo_id`.

### policy

- `policy_id` (PK): TEXT
- `name`: TEXT
- `slug`: TEXT
- `applies_to`: TEXT

### policy_fact

- `policy_id`: TEXT
- `key`: TEXT
- `value`: TEXT
`policy_fact.policy_id` references `policy.policy_id`.

### archive_doc

- `archive_id` (PK): TEXT
- `slug`: TEXT
- `subject`: TEXT
- `stale_key`: TEXT
- `stale_value`: TEXT
- `status`: TEXT


## Notes

Analysts access this database to understand store performance patterns, compare regions meaningfully, and identify trends that inform inventory and staffing decisions throughout the enterprise. The structured data enables pattern discovery across departments and weeks, supporting questions about which products drive customer traffic and which timing considerations matter most for profitability. Strategic decisions about promotions, store operations, and merchandising flow from insights these analytical queries reveal about customer behavior and profitability across the entire organization. Data quality and completeness ensure confident decision-making about resource allocation and strategic priorities.
