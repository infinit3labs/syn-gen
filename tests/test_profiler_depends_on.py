"""Regression tests for `_infer_depends_on`.

It was a double loop calling `df[[a, b]].dropna().corr()` once per pair --
O(C^2) pandas round-trips for a quantity one `df[cols].corr()` produces in a
single pass. These tests pin both the call count and the resulting values.

NOTE: the output of this method, `metadata.suggested_depends_on`, is consumed
by nothing in the package. These tests fix its current behaviour so that
whatever is decided about it later is a deliberate change.
"""
import numpy as np
import pandas as pd

from syntab.profiler import DatasetProfiler

SEED = 7


def _correlated_frame(n_cols=20, n_rows=400):
    """Numeric columns in correlated pairs, plus noise."""
    rng = np.random.default_rng(SEED)
    data = {}
    for i in range(0, n_cols, 2):
        base = rng.normal(0, 1, n_rows)
        data[f"c{i}"] = base
        data[f"c{i + 1}"] = base * 2.0 + rng.normal(0, 0.01, n_rows)
    return pd.DataFrame(data)


def _suggested(df):
    spec = DatasetProfiler.profile_set({"t": df}, sample=None, seed=SEED)
    return spec.tables[0].metadata.get("suggested_depends_on")


def _reference_pairwise(df, cols, threshold=0.95):
    """The previous implementation, kept as an oracle."""
    sugg = {}
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            try:
                r = df[[a, b]].dropna().corr().iloc[0, 1]
            except Exception:
                r = 0
            if r is not None and abs(float(r)) > threshold:
                sugg.setdefault(a, []).append(b)
                sugg.setdefault(b, []).append(a)
    return sugg or None


# --------------------------------------------------------------------------
# the fix: one correlation call, not C(C-1)/2
# --------------------------------------------------------------------------

def test_correlation_is_computed_in_a_single_pass(monkeypatch):
    original = pd.DataFrame.corr
    calls = []

    def counting_corr(self, *args, **kwargs):
        calls.append(self.shape)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "corr", counting_corr)

    df = _correlated_frame(n_cols=20)
    DatasetProfiler.profile_set({"t": df}, sample=None, seed=SEED)

    # 20 columns is 190 pairs under the old implementation.
    assert len(calls) == 1, f"expected one corr() call, got {len(calls)}"
    assert calls[0][1] == 20, "the single call should cover every numeric column"


def test_call_count_does_not_grow_with_column_count(monkeypatch):
    original = pd.DataFrame.corr
    calls = []
    monkeypatch.setattr(
        pd.DataFrame, "corr",
        lambda self, *a, **kw: (calls.append(1), original(self, *a, **kw))[1],
    )
    for n_cols in (4, 12, 40):
        calls.clear()
        DatasetProfiler.profile_set(
            {"t": _correlated_frame(n_cols=n_cols)}, sample=None, seed=SEED
        )
        assert len(calls) == 1, f"{n_cols} columns produced {len(calls)} calls"


# --------------------------------------------------------------------------
# equivalence with the implementation it replaces
# --------------------------------------------------------------------------

def test_results_match_the_pairwise_reference():
    df = _correlated_frame(n_cols=12)
    expected = _reference_pairwise(df, list(df.columns))
    assert _suggested(df) == expected


def test_results_match_the_reference_with_missing_values():
    """DataFrame.corr uses pairwise-complete rows, as the per-pair dropna did."""
    rng = np.random.default_rng(SEED)
    df = _correlated_frame(n_cols=8, n_rows=500)
    for col in df.columns:
        df.loc[rng.random(len(df)) < 0.1, col] = np.nan
    expected = _reference_pairwise(df, list(df.columns))
    assert _suggested(df) == expected


# --------------------------------------------------------------------------
# behaviour
# --------------------------------------------------------------------------

def test_a_strongly_correlated_pair_is_reported_both_ways():
    base = np.arange(200, dtype="float64")
    df = pd.DataFrame({"a": base, "b": base * 3 + 1, "noise": np.sin(base)})
    sugg = _suggested(df)
    assert sugg["a"] == ["b"]
    assert sugg["b"] == ["a"]
    assert "noise" not in sugg


def test_uncorrelated_columns_are_not_reported():
    rng = np.random.default_rng(SEED)
    df = pd.DataFrame({"a": rng.normal(0, 1, 500), "b": rng.normal(0, 1, 500)})
    assert _suggested(df) is None


def test_a_constant_column_yields_no_suggestion():
    """corr() against a zero-variance column is NaN, not a correlation."""
    df = pd.DataFrame({"a": np.arange(100.0), "k": np.ones(100)})
    assert _suggested(df) is None


def test_single_numeric_column_is_handled():
    df = pd.DataFrame({"a": np.arange(50.0), "label": ["x"] * 50})
    assert _suggested(df) is None


def test_negative_correlation_counts():
    base = np.arange(200, dtype="float64")
    df = pd.DataFrame({"a": base, "b": -2.0 * base})
    assert _suggested(df) == {"a": ["b"], "b": ["a"]}
