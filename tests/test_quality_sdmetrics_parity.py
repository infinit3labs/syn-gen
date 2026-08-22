"""Cross-check syntab's metrics against SDMetrics itself.

syntab implements its quality metrics natively rather than depending on
SDMetrics -- see ``docs/quality.md`` for why (the short version: SDMetrics
pins ``pandas<3.0.0`` and syntab runs pandas 3.x). The cost of that decision
is that "these are the same metrics" becomes a claim in a docstring instead
of something the build enforces.

This module buys the claim back. When SDMetrics happens to be importable it
runs both implementations over the same data and asserts they agree. It is
skipped otherwise, so SDMetrics stays an optional development dependency and
never enters the install path:

    pip install sdmetrics    # then run this file

If SDMetrics ever changes a definition out from under us, this is the test
that says so.
"""
import numpy as np
import pandas as pd
import pytest

sdmetrics = pytest.importorskip(
    "sdmetrics", reason="SDMetrics is an optional dev dependency; see docs/quality.md")

from sdmetrics.column_pairs import (  # noqa: E402
    ContingencySimilarity as SDMContingencySimilarity,
    CorrelationSimilarity as SDMCorrelationSimilarity,
)
from sdmetrics.single_column import (  # noqa: E402
    BoundaryAdherence as SDMBoundaryAdherence,
    CategoryAdherence as SDMCategoryAdherence,
    CategoryCoverage as SDMCategoryCoverage,
    KSComplement as SDMKSComplement,
    KeyUniqueness as SDMKeyUniqueness,
    MissingValueSimilarity as SDMMissingValueSimilarity,
    RangeCoverage as SDMRangeCoverage,
    TVComplement as SDMTVComplement,
)
from sdmetrics.single_table import TableStructure as SDMTableStructure  # noqa: E402

from syntab import quality  # noqa: E402

SEED = 13

# SDMetrics regularizes synthetic-only categories with a 1e-6 count before
# normalizing, so discrete metrics agree to about that order rather than
# exactly. The continuous metrics are exact.
DISCRETE_TOL = 1e-4
EXACT_TOL = 1e-9


@pytest.fixture(scope="module")
def frames():
    rng = np.random.default_rng(SEED)
    n = 2000
    base = rng.normal(50, 10, n)
    real = pd.DataFrame({
        "num": base,
        "num2": base * 2 + rng.normal(0, 5, n),
        "cat": rng.choice(["a", "b", "c", "d"], n, p=[0.4, 0.3, 0.2, 0.1]),
        "cat2": rng.choice(["p", "q"], n, p=[0.7, 0.3]),
    })
    synth = pd.DataFrame({
        "num": rng.normal(52, 11, n),
        "num2": rng.normal(100, 25, n),
        "cat": rng.choice(["a", "b", "c", "e"], n, p=[0.3, 0.3, 0.3, 0.1]),
        "cat2": rng.choice(["p", "q"], n, p=[0.5, 0.5]),
    })
    return real, synth


def test_ks_complement_matches_sdmetrics(frames):
    real, synth = frames
    for col in ("num", "num2"):
        assert quality.ks_complement(real[col], synth[col]) == pytest.approx(
            SDMKSComplement.compute(real[col], synth[col]), abs=EXACT_TOL)


def test_tv_complement_matches_sdmetrics(frames):
    real, synth = frames
    for col in ("cat", "cat2"):
        assert quality.tv_complement(real[col], synth[col]) == pytest.approx(
            SDMTVComplement.compute(real[col], synth[col]), abs=DISCRETE_TOL)


def test_datetime_metrics_match_sdmetrics():
    """Datetime columns, where the NaT-to-INT64_MIN bug lived."""
    dates = pd.date_range("2020-01-01", periods=500, freq="h")
    real = pd.Series(dates)
    synth = pd.Series(pd.date_range("2020-01-02", periods=500, freq="h"))
    assert quality.ks_complement(real, synth) == pytest.approx(
        SDMKSComplement.compute(real, synth), abs=EXACT_TOL)
    assert quality.range_coverage(real, synth) == pytest.approx(
        SDMRangeCoverage.compute(real, synth), abs=1e-6)
    assert quality.boundary_adherence(real, synth) == pytest.approx(
        SDMBoundaryAdherence.compute(real, synth), abs=EXACT_TOL)


def test_range_coverage_matches_sdmetrics(frames):
    real, synth = frames
    for col in ("num", "num2"):
        assert quality.range_coverage(real[col], synth[col]) == pytest.approx(
            SDMRangeCoverage.compute(real[col], synth[col]), abs=EXACT_TOL)


def test_category_coverage_matches_sdmetrics(frames):
    real, synth = frames
    for col in ("cat", "cat2"):
        assert quality.category_coverage(real[col], synth[col]) == pytest.approx(
            SDMCategoryCoverage.compute(real[col], synth[col]), abs=EXACT_TOL)


def test_boundary_adherence_matches_sdmetrics(frames):
    real, synth = frames
    for col in ("num", "num2"):
        assert quality.boundary_adherence(real[col], synth[col]) == pytest.approx(
            SDMBoundaryAdherence.compute(real[col], synth[col]), abs=EXACT_TOL)


def test_category_adherence_matches_sdmetrics(frames):
    real, synth = frames
    for col in ("cat", "cat2"):
        assert quality.category_adherence(real[col], synth[col]) == pytest.approx(
            SDMCategoryAdherence.compute(real[col], synth[col]), abs=EXACT_TOL)


@pytest.mark.parametrize("real_values,synth_values", [
    (["a", "b", "a", "b"], ["a", "b", "a", "ZZ"]),          # invented category
    (["a", "b", None, None], ["a", "b", None, None]),        # nulls both sides
    (["a", "b", "a", "b"], ["a", "b", None, None]),          # nulls only synth
    (["a", "b", None, "b"], ["a", "ZZ", None, None]),        # both at once
])
def test_category_adherence_null_handling_matches_sdmetrics(real_values, synth_values):
    """The null cases specifically.

    The first parametrization is the only one the original fixture covered,
    which is how a null-handling divergence got past this file once already.
    """
    real = pd.Series(real_values)
    synth = pd.Series(synth_values)
    assert quality.category_adherence(real, synth) == pytest.approx(
        SDMCategoryAdherence.compute(real, synth), abs=EXACT_TOL)


def test_missing_value_similarity_matches_sdmetrics():
    real = pd.Series([1.0, 2.0, None, None, 5.0])
    synth = pd.Series([1.0, None, 3.0, 4.0, 5.0])
    assert quality.missing_value_similarity(real, synth) == pytest.approx(
        SDMMissingValueSimilarity.compute(real, synth), abs=EXACT_TOL)


def test_key_uniqueness_matches_sdmetrics():
    real = pd.Series([1, 2, 3, 4, 5])
    synth = pd.Series([1, 1, 3, None, 5])
    assert quality.key_uniqueness(synth.to_frame()) == pytest.approx(
        SDMKeyUniqueness.compute(real, synth), abs=EXACT_TOL)


def test_table_structure_matches_sdmetrics(frames):
    real, synth = frames
    assert quality.table_structure(real, synth) == pytest.approx(
        SDMTableStructure.compute(real, synth), abs=EXACT_TOL)
    dropped = synth.drop(columns=["cat2"])
    assert quality.table_structure(real, dropped) == pytest.approx(
        SDMTableStructure.compute(real, dropped), abs=EXACT_TOL)


def test_correlation_similarity_matches_sdmetrics(frames):
    real, synth = frames
    cols = ["num", "num2"]
    assert quality.correlation_similarity(real[cols], synth[cols]) == pytest.approx(
        SDMCorrelationSimilarity.compute(real[cols], synth[cols]), abs=1e-6)


def test_contingency_similarity_matches_sdmetrics(frames):
    real, synth = frames
    cols = ["cat", "cat2"]
    assert quality.contingency_similarity(real[cols], synth[cols]) == pytest.approx(
        SDMContingencySimilarity.compute(real[cols], synth[cols]), abs=DISCRETE_TOL)


def test_contingency_similarity_with_discretization_matches_sdmetrics(frames):
    real, synth = frames
    cols = ["num", "cat"]
    assert quality.contingency_similarity(
        real[cols], synth[cols], ["num"]) == pytest.approx(
        SDMContingencySimilarity.compute(
            real[cols], synth[cols], continuous_column_names=["num"],
            num_discrete_bins=quality.NUM_DISCRETE_BINS), abs=DISCRETE_TOL)


def test_column_shapes_property_matches_a_hand_built_sdmetrics_equivalent(frames):
    """The aggregation, not just the individual metrics.

    SDMetrics' Column Shapes property is the mean of KSComplement over
    numerical/datetime columns and TVComplement over categorical/boolean
    ones. Rebuild that by hand from SDMetrics primitives and check syntab's
    property score lands on the same number.
    """
    real, synth = frames
    expected = np.mean([
        SDMKSComplement.compute(real["num"], synth["num"]),
        SDMKSComplement.compute(real["num2"], synth["num2"]),
        SDMTVComplement.compute(real["cat"], synth["cat"]),
        SDMTVComplement.compute(real["cat2"], synth["cat2"]),
    ])
    shapes = quality.quality_report(real, synth).get_property("Column Shapes")
    assert shapes.score == pytest.approx(expected, abs=DISCRETE_TOL)
