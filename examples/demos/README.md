# syntab demos

Example **Specs** plus their generated **output datasets**, from basic to
advanced. Each Spec is run with:

```bash
syntab generate --spec <spec> --out output/<name> --format csv
# or --format json / jsonl / parquet / sql
```

`output/` already contains the generated data.

## 01 — Basic single table (`01_basic_single_table.yaml`)
A single `employees` table mixing built-in generators: `sequence` (PK),
`faker.name` / `faker.email` (PII), `choice` with `weights`, `float` with
`decimals`, `datetime` range, and `bool` with probability `p`.

## 02 — Basic relational (`02_basic_relational.yaml`)
Two tables, `customers` → `orders`, joined by a foreign key
(`customer_id` → `customers.id`). Demonstrates referential integrity (every
`orders.customer_id` exists in `customers.id`).

## 03 — Advanced custom generators (`03_advanced_custom_generators.yaml`)
A `sellers` → `listings` relationship where `listings` uses:
- `demo_generators:SkuGenerator` — a **stateful `BaseGenerator` subclass**
  producing `LIST-00001`, `LIST-00002`, … (unique per run).
- `demo_generators:local_title` — a **`@register_generator` function** that
  derives the title's language tag from the **parent** seller's country.
- `demo_generators:region_phone` — another function reading the joined parent's
  region to pick a country code.

Custom generators see the in-progress `row` and `row["__parents__"][alias]`, and
are referenced by **import path** (`module:Attr`). See `demo_generators.py`.

## 04 — Advanced business-rule DSL (`04_advanced_business_rules.yaml`)
A `customers` → `orders` relationship with constraints expressed in the safe
DSL, both **within-table** and **cross-join**:

```yaml
rules:
  - "discount <= total * 0.3"          # within-table
  - "end_date >= start_date"           # within-table (depends_on start_date)
  - "region == customer.region"        # cross-join (aliased parent)
  - "total <= customer.credit_limit"   # cross-join
  - "if region == 'EU' then tax > 0"   # conditional
```

Enforced via the hybrid strategy (rejection sampling + FK-parent re-picking),
`fallback: raise`.

## 05 — Profiling round-trip (`05_profiled_roundtrip.json`)
A Spec in the format a **future profiler** would emit: columns carry a `profile`
(numeric `distribution`, categorical `values`/frequencies, `string_pattern`)
and `generator: auto`. `syntab` infers a concrete generator + parameters from
the stats, so a profiled dataset can be re-synthesized compatibly. Output is a
single JSON file (`--format json`).
