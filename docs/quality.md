# Evaluating synthetic data

syntab reports on a generated dataset with **three separate artefacts**. They
answer three different questions, and keeping them apart is the point.

| Command | Question | Output | Needs real data? |
|---|---|---|---|
| `syntab quality` | How closely does the synthetic data **resemble** the real data? | Graded, 0..1 | yes |
| `syntab diagnose` | Is the synthetic data structurally **valid**? | Pass/fail | yes |
| `syntab check` | Does the output honour its **Spec**? | Pass/fail | no |

Fidelity is a matter of degree, so `quality` scores it rather than judging it.
Structural validity is not a matter of degree — a category that does not exist
in the source is simply wrong — so `diagnose` is pass/fail. Spec conformance is
a third question again: it compares the output against the contract it was
generated from, which is why it needs no source dataset and why it is not
folded into either of the others.

`syntab compare` still exists and is unchanged in scope: per-column
pass/warn/fail, **marginals only**.

## Why this exists

The evaluation path used to be `compare` alone. It checked per-column marginal
distributions and nothing else — no pairwise trends, no coverage, no boundary
adherence, no category adherence, no missingness comparison, no structural
match.

The consequence is easy to state and easy to demonstrate: take a real dataset,
shuffle each column independently, and every marginal is preserved *exactly*
while every relationship between columns is destroyed. `compare` reported
`OVERALL: PASS`. That is the canonical failure mode in the synthetic data
literature, and a **rule-based generator is the kind most exposed to it**,
because it samples each column independently by construction. Nothing in the
old evaluation path could see it.

`Column Pair Trends` is the property that catches it, and
`tests/test_quality.py::test_destroyed_correlations_pass_marginals_but_fail_pair_trends`
pins the gap directly rather than leaving it implied.

## The metrics

Every metric is a named, published metric from
[SDMetrics](https://github.com/sdv-dev/SDMetrics) (DataCebo, MIT licensed), the
field-standard synthetic-data metrics suite. Each function in
`syntab.quality` is named after its SDMetrics counterpart.

### `syntab quality` — graded

| Property | Metrics | Applies to |
|---|---|---|
| Column Shapes | `KSComplement`, `TVComplement` | numeric/datetime, categorical/boolean |
| Column Pair Trends | `CorrelationSimilarity`, `ContingencySimilarity` | continuous pairs, all other pairs |
| Coverage | `RangeCoverage`, `CategoryCoverage` | numeric/datetime, categorical/boolean |
| Boundary Adherence | `BoundaryAdherence` | numeric/datetime |
| Missing Value Similarity | `MissingValueSimilarity` | every column |

A property's score is the mean of its member scores. The **Overall Score is the
mean of Column Shapes and Column Pair Trends only** — exactly what SDMetrics'
Quality Score aggregates, since those are the only two properties in its
`QualityReport`.

The other three are graded and reported but do not feed the headline number,
and that is deliberate. Averaging all five dilutes the one property that
detects the failure this report exists to catch. Independently shuffling every
column leaves Coverage, Boundary Adherence and Missing Value Similarity at
exactly `1.0` by construction — a permutation of a column has the same range,
the same categories and the same null rate — so only Column Pair Trends moves:

```
  Column Shapes               1.0000  * KSComplement, TVComplement
  Column Pair Trends          0.5471  * ContingencySimilarity, CorrelationSimilarity
  Coverage                    1.0000    CategoryCoverage, RangeCoverage
  Boundary Adherence          1.0000    BoundaryAdherence
  Missing Value Similarity    1.0000    MissingValueSimilarity
```

A five-way mean scores that dataset **0.9094** and buries the defect. The
two-property Quality Score reports **0.7735**, which reads like the problem it
is. `overall_score_all_properties` exposes the five-way mean for anyone who
wants it.

Sub-scores stay visible — `--verbose` gives the per-column breakdown — because
an aggregate that cannot be decomposed is not much use for diagnosis.

### `syntab diagnose` — pass/fail

| Property | Metrics |
|---|---|
| Data Structure | `TableStructure` |
| Data Validity | `BoundaryAdherence`, `CategoryAdherence`, `KeyUniqueness` |

Each metric is expected to be exactly `1.0`; `--tolerance` widens that. Key
columns get `KeyUniqueness` **instead of** `BoundaryAdherence`, following
SDMetrics: synthetic keys are not supposed to fall inside the real key range,
since reusing real identifiers would be a disclosure problem rather than a
success.

### Free text

Free text has no SDMetrics `sdtype` — SDV excludes it from scoring. syntab
scores it anyway, because a rule-based generator emits text and a defect there
is otherwise invisible. Both text metrics are standard metrics over a derived
distribution rather than new metrics:

- `KSComplement` over the distribution of value **lengths**
- `TVComplement` over the **character-unigram** distribution

They are marked `(derived)` in the report so they can be told apart from the
SDMetrics-native scores. The column's score is the **worse** of the two:
averaging them would let a generator that produces the right characters at one
fixed length hide behind its character score, and that is precisely the defect
worth catching.

## The dependency decision: why not just depend on SDMetrics?

SDMetrics is pip-installable, MIT licensed, and the obvious default. Using an
established library directly is normally the right call, so this decision needs
a reason rather than a preference. It was investigated before anything was
built. The findings, against `sdmetrics 0.29.0` (the latest release):

**1. It pins `pandas<3.0.0`, and syntab runs pandas 3.x.** This is the
disqualifying one. Installing SDMetrics into syntab's environment downgraded
pandas from 3.0.5 to 2.3.3. Adding it to `dependencies` would silently
downgrade the core data dependency of the project for everyone installing
syntab. The metrics do in fact *run* on pandas 3.0.5 when force-installed, but
shipping a library against a dependency constraint we know we violate is not a
position that can be maintained — the next resolver change makes it someone
else's broken install.

**2. Dependency weight.** SDMetrics pulls `scikit-learn`, `scipy`, `plotly`,
`copulas`, `tqdm`, and their transitive dependencies — roughly 70 MB. syntab's
entire current dependency set is `pydantic`, `pyyaml`, `faker`, `pandas`,
`click`. Most of that weight exists for ML-detection and copula-based metrics
that do not apply to a rule-based, non-ML generator.

**3. Metadata coupling — but only at the report layer.** The `QualityReport`
and `DiagnosticReport` classes require an SDV metadata dict and raise without
one. syntab already has its own `Spec` with the same information, so adopting
SDV metadata purely to satisfy the report constructors would be a translation
layer serving no one. The *individual metric classes*, by contrast, work
directly on bare pandas Series and DataFrames with no metadata at all — which
is what made a faithful native implementation straightforward.

**4. Maturity signals.** The single-table report classes are already deprecated
in 0.29.0, and the package still ships `Development Status :: 2 - Pre-Alpha`.

**Decision: implement the metrics natively, with SDMetrics as the normative
reference definition, and cite it.** The definitions were taken from the
SDMetrics source, not reimplemented from memory or from prose descriptions.

The cost of that decision is that "these are the same metrics" becomes a claim
in a docstring rather than something the build enforces. So it is enforced:
`tests/test_quality_sdmetrics_parity.py` runs both implementations over the
same data and asserts they agree, skipping when SDMetrics is not installed.
**All 13 parity checks pass against `sdmetrics 0.29.0`.** SDMetrics stays an
optional development dependency and never enters the install path:

```bash
pip install sdmetrics
pytest tests/test_quality_sdmetrics_parity.py
```

If SDMetrics ever changes a definition, that test says so.

## Usage

```bash
# graded fidelity; exits 0 unless --min-score is given
syntab quality --real data/source.parquet --synthetic out/t.csv --verbose

# gate it in CI
syntab quality --real data/source.parquet --synthetic out/t.csv --min-score 0.8

# structural validity; exits non-zero on any failure
syntab diagnose --real data/source.parquet --synthetic out/t.csv --spec spec.yaml

# spec conformance; no real data needed
syntab check --spec spec.yaml --data out/
```

Passing `--spec` is worth doing for both `quality` and `diagnose`: a column
kind declared in the Spec beats the cardinality heuristic used to infer one
from the data, which is unreliable on small frames and on high-cardinality
categoricals.

### Sampling

`Column Pair Trends` is quadratic in the column count, and the
character-distribution metrics have to read every character of a free-text
column. Both are distributional summaries, so `quality` samples 50,000 rows by
default (`--sample`, `--sample 0` to disable). SDMetrics exposes the same
control as `num_rows_subsample`.

There is also a default budget of 1,000 column pairs. That evaluates every
pair through 45 non-text columns, so ordinary-width tables retain their prior
behavior. Wider tables score a deterministic sample of eligible pairs,
selected from the report seed; the text and JSON reports disclose the total
eligible count, scored count, and truncation. Library callers can pass
`max_pair_trends=None` when they explicitly need the unbounded calculation.

## Text generation

Un-suppressing the text metrics surfaced a real defect, which is fixed in the
same branch. A profiled free-text column resolved to the `string` generator
with `length` set to the **maximum** observed length, and `string` filled that
width from a uniform `[a-zA-Z0-9]` alphabet. Every value came out at the
longest length in the source, made of alphanumeric noise with no spaces and no
punctuation.

`string` now accepts, in precedence order:

| Params | Behaviour |
|---|---|
| `length_edges` + `length_counts` | histogram of observed lengths, inverse-CDF sampled |
| `length_values` + `length_weights` | exact discrete length distribution |
| `min_length` / `max_length` | uniform over the observed range |
| `length` | a single fixed width (hand-authored specs; unchanged) |
| `char_values` + `char_weights` | observed character distribution |
| `alphabet` | uniform alphabet (unchanged default) |

`syntab.generators.empirical_text_params(values)` derives the length and
character distributions from a source column in exactly this form, and
`syntab.fit_text_params(spec, frames)` applies it across a profiled spec:

```python
import syntab
spec = syntab.DatasetProfiler(df, name="t").profile()
spec = syntab.fit_text_params(spec, {"t": df})   # opt-in
```

This is a **stopgap**. `ColumnProfile` records a text column's length as a
`(min, max)` pair and nothing about its characters, so a profiled spec cannot
by itself tell the generator what the source text looked like. The profiler is
the right place to record those statistics — and no Spec schema change is
needed for it to do so, because `ColumnSpec.params` is already free-form.

### Disclosure

Both helpers write **source-derived statistics** into the spec, in the same way
that profiling a categorical column writes its real values and frequencies —
see [disclosure.md](disclosure.md). A character-frequency vector over a corpus
is weak evidence about any single record, but a length distribution over a
small group is not nothing. That is a large part of why `fit_text_params` is
opt-in rather than something profiling does on its own. Treat a fitted spec as
derived from the source data, and review it before sharing it.
