# Disclosure posture

**A profiled spec is derived from your source data. Treat it as data, not as
source code: review it before you commit it to version control or share it
outside the boundary the source data lives in.**

`syntab profile` reads a real dataset and writes a spec file. That spec is
designed to be committed, edited by hand and shared — and to make the profiling
round trip work it has to carry real statistics about the source. This document
says exactly what it carries, why that matters, and what the controls do.

---

## 1. What a profiled spec actually contains

| Profile field | What is written | Derived from real data? |
|---|---|---|
| `profile.categorical.values` | every distinct value of the column, as a literal string, with its frequency to 4 d.p. | **Yes — verbatim source values** |
| `profile.numeric.min` / `.max` | the true minimum and maximum | **Yes — two real observations** |
| `profile.numeric.mean` / `.std` | aggregate moments | Yes, aggregated |
| `profile.numeric.hist_edges` | empirical histogram bin edges | **Yes — computed from real values** |
| `profile.datetime_range` | the earliest and latest timestamp | **Yes — two real observations** |
| `profile.string_pattern` | a representative value with letters → `?` and digits → `#`; **punctuation is kept literally** | Partly — shape of a real value |
| `profile.length` | the true min/max string length | Yes, aggregated |
| `constraints.null_rate` | the exact null fraction | Yes, aggregated |
| `row_count`, `metadata.profiling` | true source row count and sample size | Yes, aggregated |
| `primary_key`, `relationships` | inferred structure | Schema, not values |
| `params` of a `conditional` generator | the observed joint frequency of every (determinant, dependent) pair | **Yes — a contingency table of the source** |

The categorical row is the one that matters. Everything else is an aggregate or
a pair of extremes; that row is a verbatim copy of a column's value domain.

Widening the categorical threshold (`--max-categorical`,
`--max-categorical-ratio`) captures more real categoricals *and* copies more
real values into the file. It is a fidelity/disclosure trade-off in both
directions.

The conditional row is newer and is the larger change of the two. A marginal
says *which values occur and how often*. A conditional says *which combinations
occur and in what proportion* — it is a two-way contingency table of the
source, written into the spec. See §3.3.

## 2. Why it matters

**Rare values are the identifying ones.** This is the k-anonymity framing
(Samarati & Sweeney, 1998; Sweeney, 2002): a record is safe to release only if
it is indistinguishable from at least k−1 others on the quasi-identifiers. A
category with one member *is* that member. A spec that publishes
`"Rare condition 17": 0.0025` over 400 rows has published the existence of one
person and something true about them.

**Minimum cell size is the standard response.** Statistical agencies suppress
or coarsen table cells below a threshold before release. Five is the most
common floor in published practice; CMS uses a stricter 11 for Medicare claims;
some agencies use 3. There is no principled universal K — it is a policy
decision about acceptable risk.

**Suppression and generalization are the two standard moves.** Suppression
removes the cell. Generalization coarsens it until the cell is big enough —
rolling rare categories into an "other" bucket is the conventional form.
syntab implements generalization, because it preserves the total frequency mass
that the generator reproduces.

**Synthetic data is not automatically anonymous.** This is worth stating flatly
because the opposite is widely assumed. A generator that faithfully reproduces
a two-member category reproduces the fact that those two people exist and what
distinguishes them; a generator that samples uniformly between a real min and a
real max reproduces two real observations exactly. Regulators and standards
bodies treat synthetic output as requiring its own disclosure assessment rather
than as de-identified by construction. Fidelity and disclosure risk trade
against each other, and the controls below are knobs on that trade — not
proofs of anything.

## 3. The controls

### `--min-cell-count K`

Minimum cell-size suppression. Categorical values occurring fewer than K times
**in the source data** are generalized into a single `__other__` bucket. Total
frequency mass is preserved, so generation is unaffected in aggregate.

```bash
syntab profile data.parquet --out spec.yaml --min-cell-count 5
```

```yaml
# before                              # after, K=5
values:                               values:
  alpha:   0.6116                       alpha:     0.6116
  beta:    0.367                        beta:      0.367
  gamma:   0.0122                       __other__: 0.0214
  delta:   0.0061                     min_cell_count: 5
  epsilon: 0.0031                     suppressed_values: 3
```

Two details that matter:

* **Counts come from the full column, not the profiling sample.** "How many
  people share this value" is a fact about the source; a value seen 7 times in
  a 400-row draw of 20,000 rows may occur 300 times in reality. Suppressing on
  sample counts would delete real, non-identifying categories.
* **Secondary (complementary) suppression is applied.** If the `__other__`
  bucket is itself below K it discloses its own members just as badly as the
  cells it absorbed — a bucket of one is the value it hid. syntab keeps
  absorbing the smallest surviving category until the bucket clears K.

**The default is `0`, i.e. off.** This is deliberate and it is a decision you
may want to override in your own workflows. Enabling it by default would
silently change the statistical content of every profile, and how much
disclosure risk is acceptable belongs to whoever holds the data. Instead, the
disclosure summary reports how many embedded values fall below the recommended
threshold of 5 **whether or not you passed the flag**, so the decision is put
in front of you rather than made for you.

### `--redact-categoricals`

Replaces categorical value labels with opaque tokens (`value_001`, `value_002`,
… in descending frequency order) while preserving cardinality and the frequency
vector exactly.

```bash
syntab profile data.parquet --out spec.yaml --redact-categoricals
```

That vector is all the generation engine consumes — `infer.resolve_generator`
turns `profile.categorical.values` into `choice` values and weights, and
nothing downstream reads the labels — so **a redacted spec still generates and
the generated data still conforms to it.**

Use it when the spec will be committed or shared and the *labels themselves*
are the sensitive part: diagnoses, internal account classes, the specific
product a customer complained about.

Placeholders rather than hashes, deliberately: hashing a low-cardinality
categorical is not de-identification. The domain is small and usually
guessable, so an unsalted digest is recovered by hashing the candidate list and
matching — the failure demonstrated at scale on the 2014 NYC taxi release,
where MD5-hashed medallion numbers were recovered by brute force over the known
medallion format. A keyed hash would fix that but needs key management this
tool has no business inventing, and nothing downstream joins on these labels.

The one thing you lose: `syntab compare` will report redacted categorical
columns as mismatched, because the labels genuinely do not match. Distribution
shape, conformance and generation are unaffected.

### 3.3 `--condition-on-fds` — the fidelity/disclosure trade, made explicit

This is the control that most changes the shape of what a spec discloses, so
it gets its own discussion rather than a line in a table.

`--discover-fds` finds dependencies between columns. With
`--condition-on-fds` (the default once `--discover-fds` is on), a column on the
receiving end of one is generated by drawing from the **observed conditional
distribution** given its determinant, instead of independently from its own
marginal. The spec then carries, for each such edge, the observed frequency of
every `(determinant value, dependent value)` pair.

**Why it exists.** Without it, generated data reproduces every marginal and
destroys every relationship. On the 208,398-row CFPB source, `Product` and
`Sub-product` score 0.98 and 0.96 against the real data on their marginals and
**0.16** as a pair: the product hierarchy is measured, written into the spec,
and thrown away at generation time. Conditioning is what stops that.

**What it costs.** A conditional table is a two-way contingency table of the
source. Compare what the two forms say about the same two columns:

```
marginal only          "Sub-product = Checking account occurs 12% of the time"
                       "Product     = Bank account or service occurs 21%"

conditional            "of the rows whose Sub-product is Checking account,
                        99.4% have Product = Bank account or service and
                         0.6% have Product = Consumer loan"
```

The second is strictly more information about the source. In particular, a
*rare combination* is more identifying than either of its rare values alone —
which is the k-anonymity argument in §2 applied to a cell of a cross-tabulation
rather than a cell of a one-way table. A conditional cell is a smaller group
than a marginal cell **by construction**, so minimum cell size matters more
here, not less.

**What is done about it.**

* `--min-cell-count K` is applied *per determinant group*, using the
  conditional cell count, not the marginal one. A dependent value seen fewer
  than K times **within a determinant group** is generalized into
  `__other__` even if that value is common overall.
* `--redact-categoricals` translates both sides into the same placeholder
  vocabulary the marginals use, so a redacted spec carries no real label in its
  conditional tables either. It still carries the co-occurrence *structure*
  between the categories those placeholders stand for; the disclosure notice
  says so.
* PII columns are never a determinant and never conditional, so no conditional
  table can be built over a column whose profile was dropped.
* Only single-column determinants are used (`profiler.MAX_CONDITIONAL_LHS`).
  A two-column determinant records the joint frequency of every observed
  triple, which approaches publishing the source cross-tabulation itself.
* The disclosure notice reports the number of conditional tables, keys and
  cells, and the number of cells suppressed.

**Turning it off.** `--no-condition-on-fds` restores independent per-column
sampling. You keep the FD *report* and lose the relationships in the generated
data. That is a real cost, and it is the right choice when the spec is going
somewhere the marginals can go but a cross-tabulation cannot.

### `--pii` / `--pii-strategy`, and automatic PII detection

A column flagged as PII has its profile dropped entirely and a synthetic
generator substituted, so no real value reaches the spec.

Two **independent** signals flag a column, and the spec records which fired in
`pii_detected_by`:

| Signal | Meaning | Basis |
|---|---|---|
| `explicit` | you named it in `--pii` | your call |
| `column-name` | the column *name* matches a known identifier | HIPAA Safe Harbor (45 CFR 164.514(b)(2)) identifier list, cross-checked against NIST SP 800-122 §2.2 |
| `value-pattern` | the column *values* match an identifier format | email / SSN / phone regexes |

They warrant different follow-up. A `value-pattern` hit is evidence about the
data. A `column-name` hit is an inference from a label and can be wrong in
either direction — `notes` holding names will not be caught, and a
`product_name` column would be a false positive if it were not qualified out.

Name matching is on **name tokens**, not substrings, so `home_address` and
`addressLine1` both match while `unaddressed` and `zippy` do not.

## 4. Known limits — read this part

### Optional disclosure budgets

The profiler can enforce simple release budgets over the disclosure report:

```bash
syntab profile data.csv --out spec.yaml \
  --redact-categoricals --min-cell-count 5 \
  --max-unredacted-values 0 --max-rare-values 0
```

When a configured limit is exceeded, profiling fails before writing the spec.
The limits cover unredacted categorical labels, remaining rare categorical
values, and conditional-table cells (`--max-conditional-cells`). These are
governance checks over the artefact's measured disclosure surface, not a
differential-privacy budget or a formal anonymity guarantee. Differential
privacy requires calibrated noise and a separate accounting model, which
syntab does not currently provide.

These are residual risks the controls above do **not** cover.

1. **Id-like columns are never auto-flagged.** A trailing `id`/`uuid`/`guid`
   token marks a structural key, and anonymizing a key breaks primary- and
   foreign-key integrity. But `patient_id` and `mrn_id` genuinely are direct
   identifiers. The disclosure summary lists them under "Id-like columns NOT
   auto-flagged — your call". Use `--pii patient_id` if it is one, and accept
   that the key relationship will not survive.
2. **Numeric min/max are two real observations.** They are not aggregated away.
   An outlier maximum is frequently a single identifiable individual. Neither
   flag touches numeric profiles.
3. **Empirical histograms embed real bin edges** computed from real values, at
   up to 50 bins.
4. **`string_pattern` keeps punctuation verbatim.** Letters become `?` and
   digits `#`, but separators, `@`, `-` and so on are copied through, which
   leaks the structure of a real value.
5. **Frequencies are computed on the profiling sample**, not the full column,
   even though *suppression decisions* use full-column counts. So the emitted
   frequency is an estimate while the suppression is exact. Making frequencies
   exact is a separate change.
6. **PII name detection is a heuristic over a published list.** It will miss
   identifiers whose columns are named unconventionally. It is a floor, not a
   guarantee.
7. **Conditional distributions are not noised.** The construction in §3.3 is
   the sampling half of a Bayesian-network synthesizer, and the published form
   of that idea — PrivBayes (Zhang et al., SIGMOD 2014) — adds calibrated noise
   to each conditional to obtain a differential-privacy guarantee. syntab
   stores the observed conditional **exactly**. There is therefore no formal
   privacy guarantee attached to it, and a conditional cell with one member
   discloses that member's combination of values. `--min-cell-count` is the
   mitigation and it is a policy control, not a proof.
8. **None of this constitutes de-identification** under HIPAA, GDPR or anything
   else. It reduces obvious disclosure; it does not certify a release.

## 5. Suggested workflow

```bash
# 1. profile with the controls on, and READ the notice it prints to stderr
syntab profile data.parquet --out spec.yaml \
    --min-cell-count 5 \
    --redact-categoricals \
    --pii patient_id,mrn

# ... or, if the spec is going somewhere a cross-tabulation must not,
#     keep the dependency report and drop the conditional tables:
syntab profile data.parquet --out spec.yaml \
    --min-cell-count 5 --discover-fds --no-condition-on-fds

# 2. read the spec itself before it goes anywhere
$EDITOR spec.yaml

# 3. only then commit it
git add spec.yaml
```

The repository `.gitignore` blocks `examples/profiled_*`, `validation_report.json`
and every common tabular data format by default. That is a backstop, not a
review.

## 6. References

* Samarati, P. & Sweeney, L. (1998). *Protecting Privacy when Disclosing
  Information: k-Anonymity and Its Enforcement through Generalization and
  Suppression.*
* Sweeney, L. (2002). *k-Anonymity: A Model for Protecting Privacy.*
  International Journal of Uncertainty, Fuzziness and Knowledge-Based Systems.
* HIPAA Privacy Rule, de-identification standard, 45 CFR §164.514(b)(2)
  (Safe Harbor — the 18-identifier list).
* NIST SP 800-122, *Guide to Protecting the Confidentiality of Personally
  Identifiable Information (PII)*, §2.2 and Appendix A.
* Hundepool et al., *Statistical Disclosure Control* (Wiley) — the standard
  reference for cell suppression and generalization.
* Zhang, J., Cormode, G., Procopiuc, C. M., Srivastava, D. & Xiao, X. (2014).
  *PrivBayes: Private Data Release via Bayesian Networks.* SIGMOD; ACM TODS
  42(4), 2017. The conditional-sampling construction in §3.3, with the
  differential-privacy noise that syntab does **not** add.
