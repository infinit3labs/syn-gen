# syntab

Spec-driven synthetic tabular and relational data for testing, fixtures, and
mock environments.

syntab uses a versionable **Spec** as the contract for generated data. Write a
Spec by hand for deterministic test data, or profile an existing dataset and
round-trip the resulting `generator: auto` specification. The engine preserves
table relationships, dependencies, business rules, and useful distributional
properties without copying source rows into the generated output.

> syntab is early-stage software. Review generated data and profiled Specs
> before using them outside a development or test environment.

## Contents

- [Features](#features)
- [Install](#install)
- [Quick start](#quick-start)
- [Library](#library)
- [CLI](#cli)
- [Profiling and evaluation](#profiling-an-existing-dataset)
- [Disclosure and privacy](#disclosure-posture-read-before-committing-a-profiled-spec)
- [Examples and documentation](#examples-and-documentation)
- [Development](#development)
- [License](#license)

## Features
- **Relational**: foreign keys with referential integrity (`relationships` + `fk`/`alias`).
- **Custom generators**: both `@register_generator` functions *and* `BaseGenerator`
  subclasses; referenced by name (`custom:name`) or import path (`module:Attr`).
- **Column dependencies**: any generator sees the in-progress `row` and `depends_on`
  orders columns within a table.
- **Business-rule DSL**: a safe, hand-written expression language for within-table
  and cross-join constraints (`region == user.region`, `if region == 'EU' then total > 0`).
  Columns whose names are not bare identifiers are written in backticks:
  `` `ZIP code` == '02139' ``, `` `Timely response?` == 'Yes' ``.
- **Conditional generation**: sample a column from the observed
  P(column | determinant) instead of independently, so relationships between
  columns survive a profile/generate round trip.
- **Hybrid enforcement**: rejection sampling + FK-parent re-picking, with
  configurable `fallback` (`raise` | `drop` | `null`).
- **Outputs**: CSV / JSON / JSONL / Parquet / SQL. Both a **library** and a **CLI**.

## Install

For local development, clone the repository and install it in editable mode:

```bash
git clone https://github.com/infinit3labs/syn-gen.git
cd syn-gen
pip install -e .
pip install -e ".[parquet,dev]"   # optional parquet + dev deps
pip install -e ".[discovery]"     # key / foreign-key / dependency discovery
                                  # (AGPL-3.0-only, Linux + macOS wheels only)
```

The core package does not require the optional discovery extra. Install only
the extras needed for your workflow; `parquet` adds PyArrow and `discovery`
adds Desbordante-based dependency discovery.

## Quick start

Generate a small relational dataset from the committed example Spec:

```bash
syntab generate \
  --spec examples/demos/02_basic_relational.yaml \
  --out ./output/basic-relational \
  --format csv
```

Validate the generated files against the same Spec:

```bash
syntab check \
  --spec examples/demos/02_basic_relational.yaml \
  --data ./output/basic-relational
```

The `output/` directory is ignored by Git, so local runs do not add generated
datasets to a checkout.

## Library
```python
from syntab import Spec, GenerationEngine, register_generator

@register_generator("order_region")
def order_region(row, rng, faker, params):
    return row["__parents__"]["user"]["region"]

from syntab import from_file
spec = from_file("spec.yaml")
engine = GenerationEngine(spec)
engine.run().write("./out", fmt="csv")
```

## CLI
```bash
syntab generate --spec spec.yaml --out ./out --format csv
syntab validate  --spec spec.yaml
syntab list-generators
syntab init > spec.yaml
syntab profile data.csv --out spec.yaml --name my_table --sample 5000
syntab profile data.csv --out spec.yaml --min-cell-count 5 --redact-categoricals
```

## Profiling an existing dataset
`syntab profile` analyses a real CSV/Parquet/JSON/XLSX file and emits a
compatible **Spec** with `generator: auto` and embedded `profile` statistics
(numeric distribution, categorical frequencies, datetime ranges, string length).
The generation engine then re-synthesizes a statistically similar dataset:

```bash
syntab profile your_data.parquet --out spec.yaml --sample 5000
syntab generate --spec spec.yaml --out ./synthetic --format csv
```

For the *shape* of a profiled spec, see the two committed examples:
[`examples/spec_profiled.json`](examples/spec_profiled.json) and
[`examples/demos/05_profiled_roundtrip.json`](examples/demos/05_profiled_roundtrip.json).
Both are hand-authored with synthetic values, so they show the format without
embedding anything real.

### Reproducing the CFPB walkthrough

The worked example referenced below profiles the public CFPB
consumer-finance-complaints dataset. **Neither the source extract nor the spec
profiled from it is committed** — both are gitignored, because a profiled spec
records the real distinct values and exact frequencies of every categorical
column. See the header comment in [`.gitignore`](.gitignore) for the rationale;
treat a profiled spec as derived from the source data, not as source code.
To regenerate them locally:

```bash
# 1. fetch the dataset yourself (gitignored: data/ and *.parquet)
mkdir -p data
#    e.g. from HuggingFace: CFPB consumer-finance-complaints
#    -> data/consumer_complaints.parquet

# 2. profile it (gitignored: examples/profiled_*)
syntab profile data/consumer_complaints.parquet \
    --out examples/profiled_consumer_complaints.yaml --sample 5000

# 3. synthesize from the profile
syntab generate --spec examples/profiled_consumer_complaints.yaml \
    --out ./synthetic --format parquet

# 4. compare real vs synthetic (gitignored: validation_report.json)
syntab compare --real data/consumer_complaints.parquet \
    --synthetic ./synthetic/consumer_complaints.parquet \
    --out examples/validation_report.json
```

## Evaluating synthetic data

Three reports, answering three different questions. See
**[docs/quality.md](docs/quality.md)** for the metric definitions and the
SDMetrics dependency decision.

| Command | Question | Output | Needs real data? |
|---|---|---|---|
| `syntab quality` | How closely does the synthetic data **resemble** the real data? | Graded, 0..1 | yes |
| `syntab diagnose` | Is the synthetic data structurally **valid**? | Pass/fail | yes |
| `syntab check` | Does the output honour its **Spec**? | Pass/fail | no |

Fidelity is a matter of degree, so `quality` scores it rather than judging it.
Structural validity is not — a category that does not exist in the source is
simply wrong — so `diagnose` is pass/fail. Spec conformance is a third
question, which is why it needs no source data and is not folded into either.

```bash
syntab quality  --real data.parquet --synthetic out.csv --verbose
syntab quality  --real data.parquet --synthetic out.csv --min-score 0.8   # CI gate
syntab diagnose --real data.parquet --synthetic out.csv --spec spec.yaml
syntab check    --spec spec.yaml --data out/
```

```python
from syntab import quality_report, diagnostic_report
report = quality_report(real_df, synthetic_df, spec=spec)
print(report.overall_score)
print(report.to_text(verbose=True))
```

`quality` scores five properties — Column Shapes, **Column Pair Trends**,
Coverage, Boundary Adherence and Missing Value Similarity — each a named
SDMetrics metric (`github.com/sdv-dev/SDMetrics`). Column Pair Trends is
the one that matters most here: a rule-based generator samples each column
independently by construction, so it is exactly the kind of generator that can
reproduce every marginal perfectly while destroying every relationship between
columns. A marginals-only check cannot see that; this one can.

### Per-column distances (`compare`)

`syntab compare` remains as the quick per-column gate. It is **marginals only**
— it says nothing about the relationships between columns, so a dataset can
pass it with every correlation destroyed:

  * numeric / datetime → Kolmogorov–Smirnov statistic D (empirical-CDF supremum)
  * categorical / boolean → Total Variation Distance (0.5 · Σ|p − q|)
  * text → structural distance (length statistics + character-distribution TVD)

Each column gets a `pass` / `warn` / `fail` against configurable thresholds,
and an overall verdict. Text columns are included in that verdict: length
distribution and character frequency are structural properties, not semantic
ones, and they are what a broken text generator gets wrong.

```bash
# Python
from syntab import validate
report = validate(real_df, synthetic_df, spec=spec)   # spec optional (column types)
print(report.to_text())

# CLI
syntab compare --real data.csv --synthetic out.csv --out report.json
```

A real run on the profiled CFPB data (step 4 above, which writes
`examples/validation_report.json` locally — gitignored, not committed) shows
categorical columns matching with TVD ≈ 0.01–0.02 and numeric/datetime
distances within tolerance — the synthetic data closely matches the original.
To enable this, numeric synthesis samples from the detected distribution
rather than uniform. The profiler auto-selects, per numeric column:

  * `uniform` — flat range;
  * `normal` — bell-shaped (Gaussian around `mean`/`std`);
  * `lognormal` / `exponential` / `pareto` — right-skewed positive data;
  * `empirical` — any other shape, captured as a histogram and sampled by
    inverse-CDF so the synthetic column reproduces the real shape closely.

A round-trip (`profile` → `generate` → `compare`) therefore matches real
numeric distributions with low KS distance.

## Checking conformance to a Spec
`quality` and `compare` check synthetic data against *real data*; `check`
certifies that a generated (or real) dataset actually **obeys its Spec** — primary-key and
per-column uniqueness, not-null / `null_rate` constraints, foreign-key
integrity, and every business rule (in-table *and* cross-join via parent
aliases):

```bash
syntab check --spec spec.yaml --data out_dir        # dir of <table>.csv/.parquet/.jsonl
syntab check --spec spec.yaml --data out.json       # combined {table: rows}
# exit 0 = CONFORMS, exit 1 = NON-CONFORMING
```

Programmatic:

```python
from syntab import validate_against_spec
report = validate_against_spec(frames_dict, spec)   # table_name -> DataFrame
print(report.overall_ok, report.to_text())
```

A non-conforming row (e.g. a broken FK) is reported per table with a count of
violations, and any cross-join rule that references the broken key cascades
into a failure on the child table as well.

## Profiling multiple tables
`syntab profile` accepts **one or more** datasets. With several, it discovers
foreign-key relationships from *inclusion dependencies* (SPIDER), not from
column names:

```bash
pip install "syntab[discovery]"
syntab profile users.csv orders.csv --out spec.yaml
```

On the real CFPB extract, normalized into the star schema its own values
describe, the previous name-based rule (`<parent>_id` matching a table name)
finds **0** of the 6 foreign keys that are there, because no CFPB column is
named after a table. SPIDER finds all 6 in about four seconds.

Keys come from discovered unique column combinations (HyUCC) rather than from
names; the name is only a tie-breaker between candidates the data cannot
separate. Functional dependencies (HyFD for exact ones, Pyro for approximate
ones) are available with `--discover-fds`.

**Discovered dependencies steer generation.** A column determined by another is
generated by sampling from the observed conditional distribution rather than
independently from its own marginal, so a hierarchy in the source survives the
round trip. On the 208,398-row CFPB extract:

| Column pair | Independent sampling | Conditional |
|---|---|---|
| `Product \| Sub-product` | 0.162 | **0.979** |
| `Issue \| Sub-issue` | 0.159 | **0.942** |
| Overall quality score | 0.834 | **0.867** |

The dependency graph is a maximum-weight spanning forest over the discovered
dependencies (Chow & Liu 1968), giving each column at most one determinant.
This is a **disclosure** change as well as a fidelity one — a conditional table
is a contingency table of the source — so read
[docs/disclosure.md §3.3](docs/disclosure.md) and use `--no-condition-on-fds`
if a cross-tabulation must not leave the boundary. Details in
[docs/discovery.md](docs/discovery.md).

Every inferred rule records which algorithm produced it, over how many rows,
and how strong the evidence was -- and `--merge-into` lets you re-profile
without losing hand edits.

**Desbordante, which provides the algorithms, is AGPL-3.0-only and ships no
Windows wheel**, which is why it is an optional extra. Profiling works without
it, falling back to the name-based rules and saying so in the spec.

See **[docs/discovery.md](docs/discovery.md)** for the algorithms, the
measures, the cost and the guards.

## PII / anonymization
A profiled dataset can contain personally identifiable information. syntab
never copies real values into generated data, but for PII columns it goes
further: the profiler **strips any captured values/patterns** and substitutes a
faker generator, so the spec itself never stores real PII.

Mark PII when profiling (auto-detected for email/phone/SSN-like columns):

```bash
syntab profile data.csv --pii email,name --pii-strategy faker --out spec.yaml
```

Or set it directly in the Spec via `pii: true` + `pii_strategy`:

  * `faker`  — replace with a synthetic faker value (default; provider chosen by
    column name, e.g. `email` → `faker.email`, `name` → `faker.name`);
  * `mask`   — keep the generated value but hide it (emails → `a*****@domain`,
    strings → first/last char kept, numerics rounded to the nearest 10);
  * `redact` — constant placeholder (`"REDACTED"` for text);
  * `hash`   — a deterministic 12-char SHA-256 token (stable per input value).

`pii: true` alone (no strategy) is just metadata; set `pii_strategy` to activate
anonymization.

## SQL output
`syntab generate --format sql` writes a loadable SQL script. When given the
Spec (as the CLI always is), the DDL is **typed**:
columns map to `INTEGER` / `REAL` / `TEXT` / `BOOLEAN` / `TIMESTAMP` / `DATE`,
honouring `primary_key`, per-column `unique`, `nullable: false` → `NOT NULL`,
and emitting `FOREIGN KEY` constraints from relationships (parent tables are
written first). Without a spec, columns fall back to generic `TEXT`.

## Streaming (large datasets)
By default the engine materializes every table in memory. For large datasets,
`--stream` writes each table to disk as soon as it is generated (csv or jsonl
directory output) and frees parent tables from memory once every table that
depends on them has been generated — so peak memory stays bounded to the
current table plus the parents it still needs, instead of the whole dataset:

```bash
syntab generate --spec spec.yaml --out big_out/ --stream          # csv
syntab generate --spec spec.yaml --out big_out/ --stream --format jsonl
syntab generate --spec spec.yaml --out big_out/ --stream --chunk-size 10000
```

Programmatic sinks (`formats.CSVSink`, `formats.JSONLSink`, or any object with
a `write_table(name, rows)` method) can be passed to
`GenerationEngine.run(stream_sink=..., chunk_size=10000)`. A supplied
`progress(table, emitted, total)` callback runs after each successful write;
generation errors propagate and previously written chunks remain on disk.

## Conditional generation

Two mechanisms, for two different shapes of problem.

### `conditional` — sample Y given X

For a dependency with many determinant values, `conditional` stores the
observed P(dependent | determinant) and draws from the row it applies to. This
is what `syntab profile --discover-fds` writes for a discovered dependency, and
it is what preserves a hierarchy across a profile/generate round trip:

```yaml
- name: Product
  dtype: str
  generator: conditional
  params:
    on: ["Sub-product"]
    keys:    [["Checking account"], ["Auto loan"], [null]]
    values:  [["Bank account or service"], ["Consumer loan"], ["Mortgage", "Consumer loan"]]
    weights: [[1.0], [1.0], [0.7, 0.3]]
    default: {values: ["Bank account or service", "Mortgage"], weights: [0.6, 0.4]}
```

`params.on` is the generation dependency, so the engine orders the determinant
first without needing it restated in `depends_on`. An unseen key falls back to
`default`; a `conditional` column carries NULL as a value inside its per-key
distributions rather than through `constraints.null_rate`.

### `when` — pick a generator branch by rule

Columns can instead select a generator branch using the same safe rule DSL used
by business rules. The first matching `when` branch wins; otherwise the default
generator is used. References in branch conditions automatically become column
dependencies, so the referenced columns are generated first:

```yaml
- name: overdraft_limit
  dtype: float
  generator: float
  params: {min: 10, max: 1000}
  when:
    - if: "account_type == 'savings'"
      generator: const
      params: {value: 0}
- name: account_type
  dtype: str
  generator: choice
  params: {values: [checking, savings]}
```

After an in-table rule fails, the engine now re-samples only the referenced
columns and their derived dependents, preserving unrelated values. Rules that
reference parent aliases, FK columns, or parent-dependent custom generators
still trigger a full safe retry with parent selection. This avoids relying on
rejection sampling for exact conditional values while retaining the existing
fallback behavior.

## Composite keys and relationship cardinality
Tables support composite primary keys and composite unique constraints:

```yaml
primary_key: [tenant_id, account_id]
unique_constraints:
  - [tenant_id, external_reference]
```

Composite foreign keys use a list of child columns and explicit parent columns:

```yaml
relationships:
  - from: [tenant_id, account_id]
    to: accounts
    to_columns: [tenant_id, account_id]
```

Relationships also control parent assignment:

  * `allocation: uniform` — existing random parent selection;
  * `allocation: balanced` — child counts differ by at most one;
  * `allocation: weighted` + `weight_column` — parent row weights control selection;
  * `cardinality: one_to_one` — each parent is selected at most once;
  * `min_children` / `max_children` — enforce per-parent child-count bounds.

These constraints are enforced during generation, checked by `syntab check`,
and emitted in typed SQL DDL.

### Hierarchies (self-referencing relationships)

A table can reference itself to model trees such as `employee.manager_id`:

```yaml
relationships:
  - from: manager_id
    to: employees.id
    alias: manager
    root_fraction: 0.1
```

The engine generates parents before children: each foreign key points at a
row already emitted, and the first row is always a root (NULL foreign key).
Raise `root_fraction` to grow the number of roots. Self-referencing FK columns
must be nullable, and conformance verifies the resulting tree.

A self-reference can also cap the tree height with `max_depth` so hierarchies
stay realistic (e.g. an org chart with at most N reporting levels):

```yaml
relationships:
  - from: manager_id
    to: employees.id
    max_depth: 3
```

When every existing node has reached `max_depth`, further rows become
additional roots. Note that `min_children` / `max_children` are only honoured
by `balanced` / `weighted` allocations (or `cardinality: one_to_one`); setting
them with `allocation: uniform` is rejected at validation time.

### Many-to-many relationships

Declare a junction table between two parents instead of wiring it by hand:

```yaml
many_to_many:
  - name: enrollments
    table_a: students
    table_b: courses
    column_a: student_id
    column_b: course_id
    row_count: 12
    min_per_a: 2
    max_per_a: 3
    min_per_b: 2
    max_per_b: 4
    unique_pairs: true
```

`syntab` lowers this into a `enrollments` table with two foreign keys and two
relationships. Links are generated so each `(student_id, course_id)` pair is
unique (by default) while honouring the per-parent degree bounds
(`min_per_a`/`max_per_a`, `min_per_b`/`max_per_b`). Both parents must have a
scalar primary key; infeasible budgets are rejected at validation time. The
junction appears in `syntab check` output and in typed SQL DDL.

### Ordered children within a parent

For event-style child tables, mark a relationship `ordered` and name an
`order_column`. After generation each child row receives a `1..n` value within
its parent group, so events come out ordered per parent:

```yaml
relationships:
  - from: session_id
    to: sessions.id
    ordered: true
    order_column: event_seq
```

`syntab check` verifies the `order_column` is strictly increasing inside each
parent group.

## Vectorized generation (large datasets)

For tables with only independent, vectorizable columns (numeric / categorical /
boolean / datetime / uuid / sequence / const, plus plain scalar FKs), set
`vectorized=True` to generate every column in one column-wise pass instead of
row-by-row Python loops. On the analytics-style tables this is typically
**4–10× faster** and scales to millions of rows.

```python
GenerationEngine(spec).run(vectorized=True)
```

```bash
syntab generate --spec spec.yaml --out big/ --vectorized
```

Tables that need row-by-row semantics are **automatically falls back** to the
proven row engine, so correctness is never sacrificed:

  * business rules (`rules`);
  * column dependencies (`depends_on`) or conditional generators (`when`);
  * Faker / text / regex columns and `pii_strategy: faker`;
  * self-referencing relationships (the parent pool grows as rows are emitted);
  * many-to-many junctions (links are precomputed);
  * uniqueness constraints or composite primary keys.

## Disclosure posture (read before committing a profiled spec)
**A profiled spec is derived from your source data. Treat it as data, not as
source code, and review it before committing or sharing it.**

`syntab profile` embeds real source values in the spec it writes: the distinct
values and exact frequencies of every categorical column, the true min/max of
every numeric column, real first/last timestamps, and a character pattern taken
from real strings. That is what makes the profiling round trip work — but it
means the spec is derived data, and the repository `.gitignore` treats it that
way.

Every `syntab profile` run now prints a disclosure summary to stderr saying how
many real values from how many columns landed in the file, which columns were
flagged PII and by which signal, and whether the controls below were applied.

Two controls, both standard statistical-disclosure-control techniques:

```bash
# generalize categorical values occurring fewer than K times in the SOURCE
# into an "__other__" bucket. Rare values are the identifying ones.
syntab profile data.parquet --out spec.yaml            # K=5 by default
syntab profile data.parquet --out spec.yaml --min-cell-count 0   # opt out

# replace categorical labels with opaque tokens, keeping cardinality and the
# frequency vector — so the spec still generates and still conforms.
syntab profile data.parquet --out spec.yaml --redact-categoricals
```

`--min-cell-count` defaults to `5` (on). Rare categorical values are the ones
that identify people, and a spec now records not just which values exist but
which ones **co-occur** — on the reference dataset the same `K=5` suppresses 61
conditional cells against 2 marginal values. Getting this default wrong in the
protective direction costs fidelity and says so in the summary; getting it
wrong in the other direction writes identifying values into a file that gets
committed and shared. Pass `--min-cell-count 0` to opt out explicitly and
profile at full fidelity. The summary reports how many values fall below the
threshold either way.

Synthetic data is **not** automatically anonymous — a generator that faithfully
reproduces a two-member category reproduces the fact that those two people
exist. See **[docs/disclosure.md](docs/disclosure.md)** for what the spec
contains field by field, what the controls do, the known residual risks
(numeric extremes, histogram bin edges, id-like columns), and references.

## Spec format
See `examples/spec.yaml` (hand-authored) and `examples/spec_profiled.json`
(profiler-style, with `profile` stats and `generator: auto`).

## Examples and documentation

The repository is intentionally organized around the public workflow:

| Path | Purpose |
|---|---|
| `src/syntab/` | Installable Python package and CLI implementation |
| `tests/` | Automated behavior and regression tests |
| `examples/demos/` | Small, progressive Specs covering common features |
| `examples/mock_specs/` | End-to-end relational and business-rule demos |
| `docs/discovery.md` | Optional dependency and relationship discovery |
| `docs/quality.md` | Fidelity, validity, and conformance reports |
| `docs/disclosure.md` | What profiling records and how to review it safely |

Start with the [basic demo](examples/demos/01_basic_single_table.yaml), then
read the [demo guide](examples/demos/README.md). The longer sections below are
reference material for advanced Specs and production-shaped test fixtures.

## Mock specs (rules, constraints, key consistency)
`examples/mock_specs/` demonstrates the business-rule DSL and constraint system
end to end, with a verification runner that asserts every declared constraint
holds on the generated data:

```bash
python examples/mock_specs/run_demo.py
```

  * `ecommerce.yaml` — three tables (`customers` → `orders` → `order_items`):
    primary keys + uniqueness, `depends_on`, a computed column, **custom
    generators** (`BaseGenerator` + function, via import path), FK referential
    integrity, cross-join rules (`region == customer.region`,
    `total <= customer.credit_limit`) and **keys kept consistent across all
    three tables** (each order item reuses its parent order's `customer_id`).
  * `accounts.yaml` — single table exercising the DSL: `if/then`, `and`/`or`/
    `not`, and helper functions (`days_between`, `coalesce`, `abs`, `min`).


## Development

Install the development dependencies and run the full test suite:

```bash
python -m pip install -e ".[parquet,dev]"
pytest
```

Run a focused test while iterating:

```bash
pytest tests/test_engine.py -q
```

Please keep real datasets, profiler output, credentials, and generated output
out of commits. The repository's [`.gitignore`](.gitignore) intentionally
ignores common tabular formats and derived profiler artifacts; see the
[disclosure guide](docs/disclosure.md) before sharing a profiled Spec.

Contribution expectations and the local verification commands are in
[CONTRIBUTING.md](CONTRIBUTING.md).

## License

syntab is released under the [MIT License](LICENSE), allowing use,
modification, distribution, and commercial use subject to its terms.
