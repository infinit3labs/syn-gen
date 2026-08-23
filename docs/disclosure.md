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

The categorical row is the one that matters. Everything else is an aggregate or
a pair of extremes; that row is a verbatim copy of a column's value domain.

Widening the categorical threshold (`--max-categorical`,
`--max-categorical-ratio`) captures more real categoricals *and* copies more
real values into the file. It is a fidelity/disclosure trade-off in both
directions.

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
7. **None of this constitutes de-identification** under HIPAA, GDPR or anything
   else. It reduces obvious disclosure; it does not certify a release.

## 5. Suggested workflow

```bash
# 1. profile with the controls on, and READ the notice it prints to stderr
syntab profile data.parquet --out spec.yaml \
    --min-cell-count 5 \
    --redact-categoricals \
    --pii patient_id,mrn

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
