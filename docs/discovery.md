# Dependency discovery

`syntab profile` works out a dataset's structure — its key, its foreign keys,
which columns determine which — using published data-profiling algorithms
rather than rules about column names. This page says which algorithms, what
they cost, and what they will and will not tell you.

Everything here needs the optional extra:

```bash
pip install 'syntab[discovery]'
```

Without it, profiling still works and falls back to the name-based rules that
preceded this. A spec always records which of the two produced each rule, so
you can tell them apart without re-running anything.

---

## Why not column names

The previous foreign-key rule looked for a column named `<parent>_id` whose
stem matched a table name. On the real CFPB consumer-complaints extract
(208,398 rows), normalized into the star schema its own values describe, that
rule finds **zero** of the six foreign keys that are there — because not one
CFPB column is named after a table. They are named `Company`, `State`,
`ZIP code`, `Product`, `Issue`. That is not an unusual dataset; it is what a
source keyed on natural values looks like.

SPIDER finds all six, in about four seconds:

```
complaints.Company    -> companies.name
complaints.Issue      -> issues.issue
complaints.Product    -> products.product
complaints.State      -> states.code
complaints.ZIP code   -> zip_codes.zip
zip_codes.state       -> states.code      (dimension referencing a dimension)
```

---

## The algorithms

| What | Algorithm | Reference |
|---|---|---|
| Exact functional dependencies | HyFD | Papenbrock & Naumann, SIGMOD 2016 |
| Approximate functional dependencies | Pyro | Kruse & Naumann, PVLDB 11(7), 2018 |
| Unique column combinations (keys) | HyUCC | Papenbrock & Naumann, BTW 2017 |
| Inclusion dependencies (foreign keys) | SPIDER | Bauckmann, Leser, Naumann & Tietz, ICDE 2007 |

All four come from [Desbordante](https://github.com/desbordante/desbordante-core).
Two measures are computed on top of them:

- **g1 error** (Kivinen & Mannila, TCS 149(1), 1995) — the fraction of ordered
  tuple pairs that violate a dependency. This is what `--fd-error` bounds.
- **mu-prime** (Piatetsky-Shapiro & Matheus, KDD-93) — a
  cardinality-corrected dependency strength. This is what `--fd-min-mu`
  bounds, and it exists for a specific reason; see below.

BINDER is not used: desbordante 2.4.1 does not expose it through the Python
bindings.

### Licence

**Desbordante is AGPL-3.0-only.** That is why it is an optional extra and not
a core dependency: making an AGPL library a hard runtime dependency of syntab
would be a distribution decision — the network-use clause in particular —
rather than a profiling one. Installing the extra attaches that obligation to
your deployment. Decide accordingly.

It is also a compiled C++ extension: Linux (manylinux_2_28 x86_64/aarch64) and
macOS wheels only, **no Windows wheel**, about 13 MB installed.

---

## Keys

The primary key is chosen from HyUCC's unique column combinations, filtered to
the NOT NULL ones. If exactly one candidate survives, that is the key and the
column name is never consulted. If several survive, the data has said they are
indistinguishable, and *then* the name breaks the tie — recorded in the
provenance as `HyUCC+name-tiebreak` so you can see it happened.

Composite candidate keys are found too. A table whose only key is `(day, slot)`
used to get `primary_key: null`; it now gets the composite key. Composite
unique combinations that are not the key are written to `unique_constraints`,
capped at three. Single-column uniqueness deliberately stays in
`constraints.unique`, because populating `unique_constraints` switches off the
engine's vectorized generation path.

---

## Foreign keys

SPIDER produces inclusion dependencies — statements that one column's values
are contained in another's. Most of them are not foreign keys. Four rules,
following the feature set in Rostin, Albrecht, Bauckmann, Naumann & Leser,
"A Machine Learning Approach to Foreign Key Discovery" (WebDB 2009), decide:

1. **The referenced column is the parent's primary key.** This does nearly all
   the work. It rejects the reverse inclusions — a dimension table's values
   are trivially contained in the fact column they were built from — and the
   accidental ones between two unconstrained columns.
2. **The referencing column is not the child's own primary key.** A mutual
   inclusion between two keys is a one-to-one correspondence with no direction
   the data can settle. Asserting one at random is worse than asserting none.
   This declines some real 1:1 foreign keys; that is a deliberate trade.
3. **Both sides hold the same kind of value.** SPIDER converts values itself,
   so without this the string `"1"` is contained in a column of integer `1`s.
4. **The tables differ.** A column contained in its own table's key is a
   self-reference, which the engine supports only with a nullable key,
   `root_fraction` and `max_depth` set. Inferring one without them produces a
   spec that fails validation, so it is left to you.

When a column is contained in several keys, the one with the highest
**coverage** — distinct child values over distinct parent key values — wins.
Column name breaks a remaining tie and nothing more.

Edges that would make the table graph cyclic are dropped, because the engine
generates parents before children.

`--ind-error` relaxes rule 0: by default a foreign key must hold exactly. Raise
it for an extract with a few orphaned rows.

---

## Functional dependencies

Off by default. Turn them on with `--discover-fds`:

```bash
syntab profile data.parquet --out spec.yaml --discover-fds
```

They are recorded under `metadata.discovery.functional_dependencies` with the
algorithm, the g1 error and the mu-prime strength for each. On the CFPB source
this surfaces the real semantics of the dataset:

```
['ZIP code'] -> State          Pyro   err=0.00007   mu'=0.985
['Issue']    -> Product        Pyro   err=0.00092   mu'=0.967
['Sub-product', 'Company'] -> Product Pyro err=0.00016 mu'=0.755
[]           -> Submitted via  HyFD   err=0                     (constant column)
```

Note that `ZIP code -> State` is **not exact** — real data is dirty, and exact
FD discovery alone finds essentially none of the dependencies that describe a
real dataset. That is what Pyro and `--fd-error` are for.

### Why mu-prime exists

Under the g1 bound alone, a near-key column determines everything. On the CFPB
source, `Date received` has 1,133 distinct values across 208,398 rows, and at
`--fd-error 0.001` it comes back "determining" thirteen unrelated columns —
purely because few tuple pairs share a date. mu-prime subtracts out the score a
random determinant of the same cardinality would get:

| Dependency | pdep | mu-prime | verdict |
|---|---|---|---|
| `ZIP code -> State` | 0.986 | **0.985** | real |
| `Issue -> Product` | 0.971 | **0.967** | real |
| `Sub-product -> Product` | 0.765 | **0.728** | real |
| `Company -> Timely response?` | 0.962 | 0.409 | mostly cardinality |
| `Date received -> Product` | 0.220 | 0.102 | artefact |
| `State -> Product` | 0.140 | 0.006 | artefact |

`--fd-min-mu` defaults to 0.5.

### What functional dependencies do *not* do

They do not currently affect generation. They are a report for a human reading
the spec. Two things block wiring them in, both outside the profiler:

- `ColumnSpec.depends_on` is only an ordering edge to the engine; no generator
  reads the depended-on values for an `auto` column.
- The engine feature that *could* act on a dependency is a `when` clause, and
  the rule DSL's identifier grammar is `[A-Za-z_][A-Za-z0-9_]*`, which cannot
  name `ZIP code`, `Sub-product` or `Timely response?`.

This is also why `metadata.suggested_depends_on` — the old Pearson-correlation
hints — was removed rather than extended.

---

## Cost

Measured on the CFPB source, 208,398 rows x 18 columns, 16 threads:

| | Time |
|---|---|
| `profile`, no discovery (previous behaviour) | 2.4 s |
| `profile`, keys discovered (new default) | 3.3 s |
| `profile --discover-fds` | 20.7 s |
| `profile` over the 6-table star schema, name gate | 2.6 s |
| `profile` over the 6-table star schema, SPIDER | 6.6 s |

Key discovery costs about a second and changes the evidence behind every key,
so it is on by default. Functional-dependency discovery costs six times that
and produces a report, so it is not.

### Guards

- **Free-text and all-null columns** are excluded from the FD search. The CFPB
  narrative column has 202,516 distinct values in 208,398 rows; it participates
  in no interesting dependency and dominates the runtime.
- **`max_lhs`** bounds the lattice depth (default 3). The FD search space is
  exponential in the column count.
- **Sampling for candidate generation, validation on the full data** above
  50,000 rows or 25 columns. This is HyFD's own internal structure applied at
  the integration level, and it matters: on the CFPB source, HyFD reports 191
  exact FDs from a 5,000-row sample and 29 from the full data. About 85% of a
  sample-only answer is an artefact of the sample. Note the direction of the
  trade — validation can only reject, so sampling costs recall, not precision.
- **Candidate pruning for inclusion dependencies.** Only columns that are a
  primary key, or that could be contained in one (same value kind, no more
  distinct values), are handed to SPIDER. Both conditions are necessary for
  containment, so nothing real is discarded.

---

## Provenance, and re-profiling without losing your edits

Every inferred rule carries its provenance:

```yaml
primary_key: Complaint ID
key_provenance:
  algorithm: HyUCC
  citation: Papenbrock & Naumann, BTW 2017
  measure: exact
  confidence: 1.0
  support: 208398
  validated_on: full
  fingerprint: 8916c10d499dc192
```

`fingerprint` is a digest of the value the profiler itself wrote. On a
re-profile with `--merge-into`, a rule whose current value no longer matches
its fingerprint was changed by a human, so the change is kept and marked
`human_edited: true` — permanently, so it is not re-decided later.

```bash
# first pass
syntab profile data/*.parquet --out spec.yaml
# ... edit spec.yaml by hand ...
# second pass, keeping your edits
syntab profile data/*.parquet --out spec.yaml --merge-into spec.yaml
```

Without `--merge-into`, a re-profile overwrites the file, exactly as before.

**Scope.** The merge covers what discovery writes: the primary key and the
relationships. Column-level edits — a corrected dtype, a `pii` flag, a
hand-tuned categorical distribution — are still overwritten, because those
fields carry no provenance to compare against.
