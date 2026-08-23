"""Tests for minimum cell-size suppression (--min-cell-count).

Rare categorical values are the identifying ones -- a category with one member
is that member (k-anonymity, Samarati & Sweeney 1998). A profiled spec
publishes the distinct values and their exact frequencies, so it publishes
exactly that. The standard responses are suppression and generalization; this
implements generalization into a residual "__other__" bucket, which preserves
the total frequency mass the generator reproduces.
"""
import pandas as pd
import pytest

from syntab.conformance import validate_against_spec
from syntab.engine import GenerationEngine
from syntab.loaders import from_file, to_file
from syntab.profiler import (
    OTHER_BUCKET_LABEL,
    RECOMMENDED_MIN_CELL_COUNT,
    DatasetProfiler,
    _suppress_small_cells,
)

SEED = 1


def _frame():
    """Two safe categories and three that identify almost nobody."""
    cat = (["alpha"] * 200 + ["beta"] * 120
           + ["gamma"] * 4 + ["delta"] * 2 + ["epsilon"] * 1)
    return pd.DataFrame({"id": range(1, len(cat) + 1), "cat": cat})


def _cat(df, **kw):
    spec = DatasetProfiler(df, name="t", sample=None, seed=SEED, **kw).profile()
    return {c.name: c for c in spec.tables[0].columns}["cat"].profile.categorical


# ---------------------------------------------------------------------------
# the helper, in isolation
# ---------------------------------------------------------------------------

def test_helper_rolls_up_only_the_cells_below_k():
    """Two cells below K whose combined count already clears it.

    No secondary suppression is needed here, so the surviving categories are
    left exactly as they were.
    """
    freqs = {"a": 0.5, "b": 0.3, "c": 0.15, "d": 0.03, "e": 0.02}
    counts = pd.Series({"a": 500, "b": 300, "c": 150, "d": 3, "e": 4})
    out, n = _suppress_small_cells(freqs, counts, 5)
    assert n == 2
    assert set(out) == {"a", "b", "c", OTHER_BUCKET_LABEL}
    assert out[OTHER_BUCKET_LABEL] == pytest.approx(0.05)


def test_helper_is_a_no_op_when_nothing_is_below_k():
    freqs = {"a": 0.6, "b": 0.4}
    counts = pd.Series({"a": 60, "b": 40})
    out, n = _suppress_small_cells(freqs, counts, 5)
    assert out == freqs and n == 0


def test_helper_applies_secondary_suppression():
    """A bucket of one is the value it hid.

    Only `d` is below K, so a naive implementation renames it "__other__" and
    protects nothing -- the bucket still has exactly one member. The standard
    fix is to keep absorbing the smallest surviving cell until the bucket
    itself clears K.
    """
    freqs = {"a": 0.5, "b": 0.3, "c": 0.16, "d": 0.04}
    counts = pd.Series({"a": 500, "b": 300, "c": 160, "d": 4})
    out, n = _suppress_small_cells(freqs, counts, 5)
    assert n == 2, "the smallest survivor must be absorbed too"
    assert OTHER_BUCKET_LABEL in out and "c" not in out and "d" not in out
    assert out[OTHER_BUCKET_LABEL] == pytest.approx(0.20)


def test_helper_can_generalize_a_column_completely():
    freqs = {"a": 0.34, "b": 0.33, "c": 0.33}
    counts = pd.Series({"a": 2, "b": 2, "c": 2})
    out, n = _suppress_small_cells(freqs, counts, 10)
    assert n == 3
    assert list(out) == [OTHER_BUCKET_LABEL]
    assert out[OTHER_BUCKET_LABEL] == pytest.approx(1.0)


def test_helper_preserves_total_mass():
    freqs = {"a": 0.5, "b": 0.2, "c": 0.2, "d": 0.06, "e": 0.04}
    counts = pd.Series({"a": 50, "b": 20, "c": 20, "d": 6, "e": 4})
    out, _ = _suppress_small_cells(freqs, counts, 8)
    assert sum(out.values()) == pytest.approx(sum(freqs.values()))


# ---------------------------------------------------------------------------
# through the profiler
# ---------------------------------------------------------------------------

def test_zero_turns_suppression_off():
    """Renamed from test_suppression_is_off_by_default.

    The behaviour asserted is unchanged and is still worth asserting -- it is
    now reached by passing 0 rather than by passing nothing, because the
    default is K=5. tests/test_min_cell_count_default.py covers the default.
    """
    cat = _cat(_frame(), min_cell_count=0)
    assert cat.min_cell_count is None
    assert cat.suppressed_values == 0
    assert OTHER_BUCKET_LABEL not in cat.values
    assert "epsilon" in cat.values, "the single-member category is published"


def test_rare_values_are_counted_even_when_suppression_is_off():
    """The whole point: a spec that needs review says so on its face.

    Pinned to K=0 so the rare values still exist to be counted; the count is
    reported whether or not suppression ran, which is what this asserts.
    """
    cat = _cat(_frame(), min_cell_count=0)
    # gamma (4), delta (2), epsilon (1) are all below the recommended 5
    assert cat.rare_value_count == 3


def test_min_cell_count_generalizes_the_rare_values():
    cat = _cat(_frame(), min_cell_count=RECOMMENDED_MIN_CELL_COUNT)
    assert cat.min_cell_count == 5
    assert cat.suppressed_values == 3
    assert set(cat.values) == {"alpha", "beta", OTHER_BUCKET_LABEL}
    for gone in ("gamma", "delta", "epsilon"):
        assert gone not in cat.values
    assert cat.rare_value_count == 0
    assert sum(cat.values.values()) == pytest.approx(1.0, abs=1e-3)


def test_suppressed_values_do_not_reach_the_written_spec(tmp_path):
    out = tmp_path / "spec.yaml"
    spec = DatasetProfiler(_frame(), name="t", sample=None, seed=SEED,
                           min_cell_count=5).profile()
    to_file(spec, out)
    text = out.read_text(encoding="utf-8")
    for gone in ("gamma", "delta", "epsilon"):
        assert gone not in text
    assert OTHER_BUCKET_LABEL in text


def test_a_high_k_collapses_the_column_entirely():
    cat = _cat(_frame(), min_cell_count=1000)
    assert list(cat.values) == [OTHER_BUCKET_LABEL]
    assert cat.values[OTHER_BUCKET_LABEL] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# cell counts come from the full column, not the sample
# ---------------------------------------------------------------------------

def test_cell_counts_are_measured_on_the_full_column_not_the_sample():
    """A value common in the source can be scarce in a sample draw.

    Suppressing on sample counts would delete real, non-identifying categories
    -- the same class of mistake as deciding uniqueness on a sample (d3f78eb).
    The test asserts its own precondition, so it exercises the full-column path
    rather than passing by accident.
    """
    n_full, n_sample, k = 20_000, 400, 10
    values = ["common"] * 18_500 + ["mid"] * 1_200 + ["scarce"] * 300
    assert len(values) == n_full
    df = pd.DataFrame({"cat": values})

    # precondition: 'scarce' is below K in the sample and far above it in full
    drawn = df.sample(n=n_sample, random_state=SEED)["cat"].value_counts()
    assert int(drawn["scarce"]) < k, "precondition: scarce looks rare in the sample"
    assert int(df["cat"].value_counts()["scarce"]) == 300 >= k

    spec = DatasetProfiler(df, name="t", sample=n_sample, seed=SEED,
                           min_cell_count=k).profile()
    cat = {c.name: c for c in spec.tables[0].columns}["cat"].profile.categorical
    assert "scarce" in cat.values, "a source-common value was suppressed on sample counts"
    assert cat.suppressed_values == 0


# ---------------------------------------------------------------------------
# composition and round trip
# ---------------------------------------------------------------------------

def test_suppression_and_redaction_compose():
    cat = _cat(_frame(), min_cell_count=5, redact_categoricals=True)
    assert cat.redacted is True and cat.suppressed_values == 3
    assert all(k.startswith("value_") for k in cat.values)
    assert len(cat.values) == 3  # alpha, beta, other


def test_roundtrip_profile_suppressed_generate_validate():
    df = _frame()
    spec = DatasetProfiler(df, name="t", sample=None, seed=SEED,
                           min_cell_count=5).profile()
    frames = GenerationEngine(spec).run().to_frames()
    assert len(frames["t"]) == len(df)
    assert validate_against_spec(frames, spec).overall_ok
    assert not frames["t"]["cat"].isin(["gamma", "delta", "epsilon"]).any()


def test_profile_set_threads_min_cell_count_through():
    spec = DatasetProfiler.profile_set({"t": _frame()}, sample=None, seed=SEED,
                                       min_cell_count=5)
    cat = {c.name: c for c in spec.tables[0].columns}["cat"].profile.categorical
    assert cat.suppressed_values == 3


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_exposes_min_cell_count(tmp_path):
    from click.testing import CliRunner
    from syntab.cli import cli

    data = tmp_path / "d.csv"
    _frame().to_csv(data, index=False)
    runner = CliRunner()

    # K=0: the explicit opt-out, which is now what "no suppression" requires
    plain = tmp_path / "plain.yaml"
    res = runner.invoke(cli, ["profile", str(data), "--out", str(plain),
                              "--min-cell-count", "0"])
    assert res.exit_code == 0, res.output
    assert "epsilon" in plain.read_text(encoding="utf-8")

    supp = tmp_path / "supp.yaml"
    res = runner.invoke(cli, ["profile", str(data), "--out", str(supp),
                              "--min-cell-count", "5"])
    assert res.exit_code == 0, res.output
    text = supp.read_text(encoding="utf-8")
    assert "epsilon" not in text and "gamma" not in text
    assert OTHER_BUCKET_LABEL in text
    assert from_file(supp).tables[0].columns[1].profile.categorical.suppressed_values == 3
