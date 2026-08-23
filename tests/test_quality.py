"""The graded quality report and the structural diagnostic report.

The headline test is ``test_destroyed_correlations_pass_marginals_but_fail_pair_trends``:
it builds the canonical failure mode -- perfect per-column marginals, every
relationship between columns destroyed -- and pins the fact that the
marginals-only validator calls it a pass while the quality report does not.
That gap is the reason this module exists, so it is tested directly rather
than implied.
"""
import math

import numpy as np
import pandas as pd
import pytest

from syntab.quality import (
    DEFAULT_SUBSAMPLE,
    _as_numeric,
    boundary_adherence,
    category_adherence,
    category_coverage,
    contingency_similarity,
    correlation_similarity,
    diagnostic_report,
    infer_kind,
    key_uniqueness,
    ks_complement,
    missing_value_similarity,
    quality_report,
    range_coverage,
    table_structure,
    text_character_complement,
    text_length_complement,
    tv_complement,
)
from syntab.spec import (
    CategoricalProfile,
    ColumnProfile,
    ColumnSpec,
    Spec,
    SpecMetadata,
    TableSpec,
)
from syntab.validator import validate

SEED = 7


def _correlated(n=3000, seed=SEED):
    """A frame where every column is strongly tied to every other."""
    rng = np.random.default_rng(seed)
    base = rng.normal(50, 10, n)
    # amount is a deterministic-ish function of base -> strong correlation
    amount = base * 3 + rng.normal(0, 2, n)
    # tier is a function of base -> strong contingency
    tier = np.where(base < 45, "low", np.where(base < 55, "mid", "high"))
    region = np.where(base < 50, "EU", "US")
    return pd.DataFrame({"base": base, "amount": amount, "tier": tier,
                         "region": region})


def _shuffle_independently(df, seed=99):
    """Preserve every marginal exactly; destroy every joint relationship."""
    rng = np.random.default_rng(seed)
    out = {}
    for i, col in enumerate(df.columns):
        values = df[col].to_numpy().copy()
        rng.shuffle(values)
        out[col] = values
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# The headline: marginals-only is not enough
# ---------------------------------------------------------------------------

def test_destroyed_correlations_pass_marginals_but_fail_pair_trends():
    real = _correlated()
    synth = _shuffle_independently(real)

    # Independently shuffling each column is a permutation of that column, so
    # every marginal is IDENTICAL, not merely close.
    for col in real.columns:
        assert sorted(real[col].astype(str)) == sorted(synth[col].astype(str))

    # The old marginals-only validator is perfectly happy with this.
    assert validate(real, synth).overall_pass is True

    report = quality_report(real, synth)
    shapes = report.get_property("Column Shapes")
    trends = report.get_property("Column Pair Trends")

    # Marginals are perfect...
    assert shapes.score > 0.99, report.to_text(verbose=True)
    # ...and every relationship is gone.
    assert trends.score < 0.75, report.to_text(verbose=True)
    # so the overall score is dragged down rather than reported as a pass.
    assert report.overall_score < shapes.score


def test_faithful_synthetic_scores_high_on_every_property():
    real = _correlated(seed=1)
    synth = _correlated(seed=2)  # same generative process, different draw
    report = quality_report(real, synth)
    assert report.overall_score > 0.9, report.to_text(verbose=True)
    for p in report.properties:
        assert p.score > 0.85, f"{p.name}={p.score}\n{report.to_text(verbose=True)}"


def test_key_columns_are_excluded_from_the_graded_properties():
    """A surrogate key has no distribution or relationships worth scoring.

    Including one adds noise in both directions: a synthetic key range that
    legitimately does not overlap the real one drags Column Shapes down, and
    pairing the key against every other column pulls Column Pair Trends
    toward the middle. SDMetrics excludes the `id` sdtype for the same
    reason; key integrity is KeyUniqueness's job in the diagnostic report.
    """
    real = _correlated(seed=21)
    real.insert(0, "id", range(len(real)))
    synth = _correlated(seed=22)
    synth.insert(0, "id", range(10_000, 10_000 + len(synth)))

    spec = Spec(
        metadata=SpecMetadata(name="t"),
        tables=[TableSpec(name="t", row_count=len(real), primary_key="id",
                          columns=[ColumnSpec(name="id", dtype="int",
                                              generator="sequence")])],
    )
    report = quality_report(real, synth, spec=spec)
    assert report.excluded_key_columns == ["id"]
    scored = {r.column for p in report.properties for r in p.results}
    assert "id" not in scored
    assert not any("id" in c for c in scored if "|" in c)
    assert "Key columns excluded" in report.to_text()

    # Without a spec there is nothing declaring it a key, so it is scored.
    assert quality_report(real, synth).excluded_key_columns == []


def test_pair_trends_property_is_actually_populated():
    """Guard against the property silently computing nothing."""
    real = _correlated(seed=3)
    trends = quality_report(real, _correlated(seed=4)).get_property("Column Pair Trends")
    # 4 columns -> 6 unordered pairs
    assert len(trends.results) == 6
    assert {r.metric for r in trends.results} == {
        "CorrelationSimilarity", "ContingencySimilarity"}


def test_pair_trends_are_bounded_and_disclose_wide_table_truncation():
    """Wide tables score a deterministic subset instead of all column pairs."""
    real = pd.DataFrame({f"c{i}": np.arange(40) + i for i in range(20)})
    report = quality_report(real, real, max_pair_trends=25)

    trends = report.get_property("Column Pair Trends")
    assert len(trends.results) == 25
    assert report.pair_trends_total == 190
    assert report.pair_trends_scored == 25
    assert report.pair_trends_truncated is True
    assert "25 of 190" in report.to_text()
    assert report.to_dict()["pair_trends"] == {
        "total": 190, "scored": 25, "truncated": True,
    }


# ---------------------------------------------------------------------------
# Individual metric definitions
# ---------------------------------------------------------------------------

def test_ks_complement_is_one_for_identical_columns():
    s = pd.Series(np.linspace(0, 100, 500))
    assert ks_complement(s, s) == pytest.approx(1.0)


def test_ks_complement_falls_for_shifted_distribution():
    a = pd.Series(np.linspace(0, 100, 500))
    b = pd.Series(np.linspace(200, 300, 500))
    assert ks_complement(a, b) == pytest.approx(0.0)


def test_tv_complement_matches_hand_computed_tvd():
    real = pd.Series(["a"] * 50 + ["b"] * 50)
    synth = pd.Series(["a"] * 70 + ["b"] * 30)
    # TVD = 0.5 * (|0.5-0.7| + |0.5-0.3|) = 0.2
    assert tv_complement(real, synth) == pytest.approx(0.8)


def test_datetime_metrics_ignore_unparseable_values():
    """NaT must not enter the metrics as a timestamp near the year 1677.

    Found on the real CFPB source, whose date columns hold more than one
    format. astype("int64") maps NaT to INT64_MIN rather than to something
    dropna() removes, which put RangeCoverage at exactly 0.0 -- the real
    range started at INT64_MIN, so nothing could cover it -- and KSComplement
    at 0.6044, roughly the fraction of rows that happened to parse.
    """
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    real = pd.Series([d.strftime("%Y-%m-%d") for d in dates] + ["not a date"] * 10)
    synth = pd.Series([d.strftime("%Y-%m-%d") for d in dates] + ["not a date"] * 10)
    assert ks_complement(real, synth) == pytest.approx(1.0)
    assert range_coverage(real, synth) == pytest.approx(1.0)
    assert boundary_adherence(real, synth) == pytest.approx(1.0)


def test_mixed_date_formats_in_one_column_still_parse():
    """The real CFPB source holds both 07/29/2017 and 09-09-2015."""
    real = pd.Series(["07/29/2017", "09-09-2015", "01/02/2016", "03-04-2017"] * 25)
    assert infer_kind(real) == "datetime"
    parsed = _as_numeric(real)
    assert parsed.notna().all(), "a minority date format was coerced to NaT"
    assert ks_complement(real, real) == pytest.approx(1.0)


def test_range_coverage_penalises_a_narrow_synthetic_range():
    real = pd.Series(np.arange(0, 101, dtype=float))
    full = pd.Series([0.0, 100.0])
    half = pd.Series([50.0, 100.0])
    assert range_coverage(real, full) == pytest.approx(1.0)
    assert range_coverage(real, half) == pytest.approx(0.5)


def test_category_coverage_counts_missing_categories():
    real = pd.Series(list("abcd") * 10)
    synth = pd.Series(list("ab") * 10)
    assert category_coverage(real, synth) == pytest.approx(0.5)


def test_boundary_adherence_counts_out_of_range_values():
    real = pd.Series([0.0, 10.0])
    synth = pd.Series([5.0, 5.0, 5.0, 999.0])
    assert boundary_adherence(real, synth) == pytest.approx(0.75)


def test_category_adherence_catches_an_invented_category():
    real = pd.Series(["x", "y"] * 10)
    synth = pd.Series(["x", "y", "x", "INVENTED"])
    assert category_adherence(real, synth) == pytest.approx(0.75)


def test_category_adherence_does_not_punish_faithful_nulls():
    """A null is a value the source produces, when the source produces nulls.

    Found on the real CFPB data: `Tags` is 82.9% null and was scoring 0.1714
    -- exactly its non-null fraction -- because the real value set was built
    with dropna() and so could never contain NaN. The column was being
    reported as structurally broken while it faithfully reproduced the source.
    """
    real = pd.Series(["x", "y", None, None] * 10)
    synth = pd.Series(["x", "y", None, None] * 10)
    assert category_adherence(real, synth) == pytest.approx(1.0)


def test_category_adherence_still_punishes_nulls_the_source_never_has():
    """The other half: NULL into a never-null column IS an invented value."""
    real = pd.Series(["x", "y"] * 10)
    synth = pd.Series(["x", "y", None, None])
    assert category_adherence(real, synth) == pytest.approx(0.5)


def test_missing_value_similarity_compares_null_rates():
    real = pd.Series([1.0, None, None, None])       # 0.75 null
    synth = pd.Series([1.0, 2.0, 3.0, None])        # 0.25 null
    assert missing_value_similarity(real, synth) == pytest.approx(0.5)


def test_key_uniqueness_catches_duplicates_and_nulls():
    assert key_uniqueness(pd.DataFrame({"id": [1, 2, 3, 4]})) == pytest.approx(1.0)
    assert key_uniqueness(pd.DataFrame({"id": [1, 1, 2, 3]})) == pytest.approx(0.75)


def test_table_structure_is_jaccard_over_name_dtype_pairs():
    a = pd.DataFrame({"x": [1], "y": [1]})
    assert table_structure(a, a) == pytest.approx(1.0)
    b = pd.DataFrame({"x": [1], "z": [1]})
    # intersection {x}, union {x, y, z} -> 1/3
    assert table_structure(a, b) == pytest.approx(1 / 3)


def test_correlation_similarity_detects_a_broken_relationship():
    rng = np.random.default_rng(SEED)
    x = rng.normal(0, 1, 2000)
    real = pd.DataFrame({"a": x, "b": x * 2 + rng.normal(0, 0.01, 2000)})
    kept = real.copy()
    broken = pd.DataFrame({"a": x, "b": rng.permutation(real["b"].to_numpy())})
    assert correlation_similarity(real, kept) == pytest.approx(1.0)
    assert correlation_similarity(real, broken) < 0.6


def test_contingency_similarity_detects_a_broken_joint_distribution():
    rng = np.random.default_rng(SEED)
    a = rng.choice(["p", "q"], 2000)
    real = pd.DataFrame({"a": a, "b": np.where(a == "p", "yes", "no")})
    broken = pd.DataFrame({"a": a, "b": rng.permutation(real["b"].to_numpy())})
    assert contingency_similarity(real, real) == pytest.approx(1.0)
    assert contingency_similarity(real, broken) < 0.7


def test_contingency_similarity_discretizes_continuous_columns():
    rng = np.random.default_rng(SEED)
    x = rng.normal(0, 1, 2000)
    real = pd.DataFrame({"num": x, "cat": np.where(x < 0, "lo", "hi")})
    broken = pd.DataFrame({"num": x, "cat": rng.permutation(real["cat"].to_numpy())})
    assert contingency_similarity(real, real, ["num"]) == pytest.approx(1.0)
    assert contingency_similarity(real, broken, ["num"]) < 0.7


# ---------------------------------------------------------------------------
# Text metrics
# ---------------------------------------------------------------------------

def test_text_length_complement_catches_fixed_maximum_length():
    """The exact defect: every value emitted at the maximum observed length."""
    rng = np.random.default_rng(SEED)
    lengths = rng.integers(5, 60, 2000)
    real = pd.Series(["x" * int(n) for n in lengths])
    at_max = pd.Series(["x" * 59] * 2000)
    assert text_length_complement(real, real) == pytest.approx(1.0)
    assert text_length_complement(real, at_max) < 0.1


def test_text_character_complement_catches_a_uniform_alphabet():
    rng = np.random.default_rng(SEED)
    alnum = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    prose = pd.Series(["the quick brown fox jumps over the lazy dog "] * 500)
    uniform = pd.Series(["".join(rng.choice(list(alnum), 40)) for _ in range(500)])
    assert text_character_complement(prose, prose) == pytest.approx(1.0)
    assert text_character_complement(prose, uniform) < 0.5


def test_text_column_shape_uses_the_worse_of_the_two_metrics():
    """Right characters at a fixed length must not average away the defect."""
    rng = np.random.default_rng(SEED)
    words = ["the quick brown fox", "a lazy dog sleeps", "hello there world"]
    real = pd.DataFrame({"note": [rng.choice(words) + " " * int(rng.integers(0, 30))
                                  for _ in range(1500)]})
    # Same characters, one single length.
    fixed = pd.DataFrame({"note": ["the quick brown fox" + " " * 30] * 1500})
    result = next(r for r in quality_report(real, fixed)
                  .get_property("Column Shapes").results
                  if r.metric.startswith("TextSimilarity"))
    chars = result.details["TVComplement_over_characters"]
    lengths = result.details["KSComplement_over_lengths"]
    assert result.score == pytest.approx(min(chars, lengths))
    assert result.score < chars  # the length defect is not averaged away


# ---------------------------------------------------------------------------
# Diagnostic report
# ---------------------------------------------------------------------------

def _spec_with_key():
    """A profiled-shaped spec: primary key + a column declared categorical."""
    return Spec(
        metadata=SpecMetadata(name="t"),
        tables=[TableSpec(name="t", row_count=4, primary_key="id", columns=[
            ColumnSpec(name="id", dtype="int", generator="sequence"),
            ColumnSpec(name="cat", dtype="str", generator="choice",
                       profile=ColumnProfile(
                           categorical=CategoricalProfile(
                               values={"a": 0.5, "b": 0.5}))),
        ])],
    )


def test_diagnostic_passes_on_structurally_valid_data():
    real = pd.DataFrame({"id": [1, 2, 3, 4], "cat": ["a", "b", "a", "b"]})
    synth = pd.DataFrame({"id": [5, 6, 7, 8], "cat": ["a", "a", "b", "b"]})
    report = diagnostic_report(real, synth, _spec_with_key())
    assert report.overall_ok, report.to_text(verbose=True)


def test_key_columns_get_key_uniqueness_not_boundary_adherence():
    """Synthetic keys outside the real key range is correct, not a failure.

    Reusing real identifiers would be a disclosure problem; range is simply
    the wrong question to ask of a key. SDMetrics' DataValidity property
    substitutes KeyUniqueness for BoundaryAdherence on key columns and so
    does this one.
    """
    real = pd.DataFrame({"id": [1, 2, 3, 4], "cat": ["a", "b", "a", "b"]})
    synth = pd.DataFrame({"id": [900, 901, 902, 903], "cat": ["a", "b", "a", "b"]})
    report = diagnostic_report(real, synth, _spec_with_key())
    metrics = {(r.column, r.metric) for p in report.properties for r in p.results}
    assert ("id", "KeyUniqueness") in metrics
    assert ("id", "BoundaryAdherence") not in metrics
    assert report.overall_ok, report.to_text(verbose=True)


def test_spec_declared_kind_beats_the_cardinality_heuristic():
    """A declared categorical is scored as one even when inference disagrees."""
    real = pd.DataFrame({"id": [1, 2, 3, 4], "cat": ["a", "b", "a", "b"]})
    # 2 distinct over 4 rows trips the inference heuristic into "text".
    assert infer_kind(real["cat"]) == "text"
    synth = pd.DataFrame({"id": [5, 6, 7, 8], "cat": ["a", "b", "a", "ZZZ"]})
    report = diagnostic_report(real, synth, _spec_with_key())
    assert any(r.metric == "CategoryAdherence" for r in report.failures())


def test_diagnostic_fails_on_an_invented_category():
    real = pd.DataFrame({"id": [1, 2, 3, 4], "cat": ["a", "b", "a", "b"]})
    synth = pd.DataFrame({"id": [5, 6, 7, 8], "cat": ["a", "b", "a", "ZZZ"]})
    report = diagnostic_report(real, synth, _spec_with_key())
    assert not report.overall_ok
    assert any(r.metric == "CategoryAdherence" for r in report.failures())


def test_diagnostic_fails_on_a_duplicated_key():
    real = pd.DataFrame({"id": [1, 2, 3, 4], "cat": ["a", "b", "a", "b"]})
    synth = pd.DataFrame({"id": [5, 5, 7, 8], "cat": ["a", "b", "a", "b"]})
    report = diagnostic_report(real, synth, _spec_with_key())
    assert not report.overall_ok
    assert any(r.metric == "KeyUniqueness" for r in report.failures())


def test_diagnostic_fails_on_out_of_range_values():
    real = pd.DataFrame({"n": [0.0, 1.0, 2.0]})
    synth = pd.DataFrame({"n": [0.0, 1.0, 99.0]})
    report = diagnostic_report(real, synth)
    assert not report.overall_ok
    assert any(r.metric == "BoundaryAdherence" for r in report.failures())


def test_diagnostic_fails_on_a_missing_column():
    real = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    synth = pd.DataFrame({"a": [1, 2]})
    report = diagnostic_report(real, synth)
    assert not report.overall_ok
    assert any(r.metric == "TableStructure" for r in report.failures())


def test_diagnostic_tolerance_can_loosen_the_bar():
    real = pd.DataFrame({"n": [0.0] * 99 + [1.0]})
    synth = pd.DataFrame({"n": [0.5] * 99 + [99.0]})  # 99% adherent
    assert not diagnostic_report(real, synth).overall_ok
    assert diagnostic_report(real, synth, tolerance=0.05).overall_ok


# ---------------------------------------------------------------------------
# Report plumbing
# ---------------------------------------------------------------------------

def test_quality_report_serializes_and_renders():
    real = _correlated(seed=5)
    report = quality_report(real, _correlated(seed=6))
    d = report.to_dict()
    assert 0.0 <= d["overall_score"] <= 1.0
    assert {p["name"] for p in d["properties"]} == {
        "Column Shapes", "Column Pair Trends", "Coverage",
        "Boundary Adherence", "Missing Value Similarity"}
    assert report.to_text()
    assert report.to_text(verbose=True)


def test_diagnostic_report_serializes_and_renders():
    real = pd.DataFrame({"id": [1, 2, 3], "cat": ["a", "b", "a"]})
    report = diagnostic_report(real, real, _spec_with_key())
    d = report.to_dict()
    assert d["overall_ok"] is True
    assert {p["name"] for p in d["properties"]} == {"Data Structure", "Data Validity"}
    assert report.to_text()
    assert report.to_text(verbose=True)


def test_overall_score_is_the_mean_of_the_two_sdmetrics_properties():
    """The headline score is SDMetrics' Quality Score, not a five-way mean.

    Averaging all five properties dilutes the only one that detects destroyed
    relationships -- see test_five_way_mean_would_bury_destroyed_relationships.
    """
    real = _correlated(seed=8)
    report = quality_report(real, _correlated(seed=9))
    scored = [p for p in report.properties if p.contributes]
    assert {p.name for p in scored} == {"Column Shapes", "Column Pair Trends"}
    assert report.overall_score == pytest.approx(np.mean([p.score for p in scored]))
    assert report.overall_score_all_properties == pytest.approx(
        np.mean([p.score for p in report.properties]))


def test_five_way_mean_would_bury_destroyed_relationships():
    """Why the headline score is not the mean of all five properties.

    Independently shuffling every column leaves Coverage, Boundary Adherence
    and Missing Value Similarity at exactly 1.0 by construction -- a
    permutation of a column has the same range, categories and null rate. Only
    Column Pair Trends moves. Averaging all five therefore reports a dataset
    with every relationship destroyed as a respectable score.
    """
    real = _correlated(seed=12)
    report = quality_report(real, _shuffle_independently(real))

    for name in ("Coverage", "Boundary Adherence", "Missing Value Similarity"):
        assert report.get_property(name).score == pytest.approx(1.0)

    assert report.overall_score_all_properties > 0.85   # the diluted number
    assert report.overall_score < 0.8                   # the honest one
    assert report.overall_score < report.overall_score_all_properties - 0.1


def test_infer_kind_classifies_the_five_kinds():
    assert infer_kind(pd.Series([1, 2, 3])) == "numeric"
    assert infer_kind(pd.Series([True, False])) == "boolean"
    assert infer_kind(pd.Series(["a", "b"] * 50)) == "categorical"
    assert infer_kind(pd.to_datetime(pd.Series(["2020-01-01", "2021-06-05"]))) == "datetime"
    assert infer_kind(pd.Series([f"free text value number {i}" for i in range(200)])) == "text"


def test_subsampling_does_not_change_the_verdict():
    real = _correlated(n=4000, seed=11)
    synth = _shuffle_independently(real)
    full = quality_report(real, synth, sample=None)
    sampled = quality_report(real, synth, sample=1000)
    assert full.get_property("Column Pair Trends").score < 0.75
    assert sampled.get_property("Column Pair Trends").score < 0.75
