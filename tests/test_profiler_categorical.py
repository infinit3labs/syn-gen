"""Regression tests for the categorical-vs-free-text decision.

The cutoff was hardcoded at `nunique <= 50` (and unreachable from the CLI,
which never passed `max_categorical` at all). A column with 120 distinct
product codes across a million rows is unambiguously categorical, but 120 > 50,
so it fell through to the free-text branch and was re-synthesized as random
characters with the right length and nothing else.
"""
import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from syntab.cli import cli
from syntab.loaders import from_file
from syntab.profiler import (
    DEFAULT_MAX_CATEGORICAL,
    DEFAULT_MAX_CATEGORICAL_RATIO,
    DEFAULT_MAX_CATEGORICAL_RATIO_CAP,
    DatasetProfiler,
)

SEED = 5


def _codes_frame(n_codes, n_rows):
    rng = np.random.default_rng(SEED)
    return pd.DataFrame({
        "code": rng.choice([f"CODE{i:04d}" for i in range(n_codes)], size=n_rows),
    })


def _profile(df, **kw):
    kw.setdefault("sample", None)
    # These tests are about which columns become categorical and how many
    # values survive the --max-categorical thresholds. Minimum cell-size
    # suppression removes values for an unrelated reason and would confound
    # every count below, so it is pinned off unless a test asks for it.
    kw.setdefault("min_cell_count", 0)
    return DatasetProfiler(df, name="t", seed=SEED, **kw).profile()


def _categorical(spec):
    return spec.tables[0].columns[0].profile.categorical


# --------------------------------------------------------------------------
# the absolute threshold is now configurable
# --------------------------------------------------------------------------

def test_default_threshold_is_unchanged():
    assert DEFAULT_MAX_CATEGORICAL == 50


def test_a_column_over_the_default_absolute_threshold_is_free_text_by_default():
    df = _codes_frame(n_codes=120, n_rows=400)   # ratio 0.3 -> over both rules
    assert _categorical(_profile(df)) is None


def test_raising_max_categorical_captures_it():
    df = _codes_frame(n_codes=120, n_rows=400)
    cat = _categorical(_profile(df, max_categorical=200))
    assert cat is not None
    # every value present in the data is captured (a 400-row draw over 120
    # codes does not necessarily contain all of them)
    assert len(cat.values) == df["code"].nunique()
    assert abs(sum(cat.values.values()) - 1.0) < 1e-6


def test_lowering_max_categorical_pushes_a_column_to_free_text():
    # 20 codes over 100 rows: ratio 0.2, so the ratio rule does not rescue it
    df = _codes_frame(n_codes=20, n_rows=100)
    assert _categorical(_profile(df)) is not None
    assert _categorical(_profile(df, max_categorical=10)) is None


# --------------------------------------------------------------------------
# the new ratio rule
# --------------------------------------------------------------------------

def test_ratio_rule_captures_a_high_cardinality_categorical_at_scale():
    """120 codes over 100k rows: ratio 0.0012, well under the 5% default."""
    df = _codes_frame(n_codes=120, n_rows=100_000)
    cat = _categorical(_profile(df))
    assert cat is not None
    assert len(cat.values) == 120


def test_ratio_rule_respects_its_hard_cap():
    """Even a tiny ratio may not write an unbounded number of real values."""
    df = _codes_frame(n_codes=DEFAULT_MAX_CATEGORICAL_RATIO_CAP + 200,
                      n_rows=40_000)
    assert _categorical(_profile(df)) is None


def test_ratio_cap_is_configurable():
    n_codes = DEFAULT_MAX_CATEGORICAL_RATIO_CAP + 200
    df = _codes_frame(n_codes=n_codes, n_rows=40_000)
    cat = _categorical(_profile(df, max_categorical_ratio_cap=n_codes))
    assert cat is not None
    assert len(cat.values) == df["code"].nunique()


def test_ratio_rule_does_not_fire_on_near_unique_columns():
    df = _codes_frame(n_codes=300, n_rows=400)   # ratio ~0.6
    assert _categorical(_profile(df)) is None


def test_ratio_threshold_is_configurable():
    df = _codes_frame(n_codes=120, n_rows=1_000)   # ratio 0.12
    assert _categorical(_profile(df)) is None
    assert _categorical(_profile(df, max_categorical_ratio=0.2)) is not None


def test_default_ratio_is_documented_value():
    assert DEFAULT_MAX_CATEGORICAL_RATIO == 0.05


# --------------------------------------------------------------------------
# small frames keep their old behaviour
# --------------------------------------------------------------------------

def test_small_frame_of_nearly_distinct_values_is_still_free_text():
    df = pd.DataFrame({"code": ["a", "b", "c", "d", "e", "a"]})
    assert _categorical(_profile(df)) is None


def test_small_low_cardinality_column_is_still_categorical():
    df = pd.DataFrame({"status": ["open", "closed", "open", "open", "closed"]})
    cat = _categorical(_profile(df))
    assert cat is not None
    assert set(cat.values) == {"open", "closed"}


# --------------------------------------------------------------------------
# the thresholds are reachable from the CLI (they were not, at all)
# --------------------------------------------------------------------------

def test_cli_exposes_max_categorical(tmp_path):
    data = tmp_path / "d.csv"
    _codes_frame(n_codes=120, n_rows=400).to_csv(data, index=False)

    runner = CliRunner()
    default_out = tmp_path / "default.yaml"
    res = runner.invoke(cli, ["profile", str(data), "--out", str(default_out),
                              "--sample", "400", "--min-cell-count", "0"])
    assert res.exit_code == 0, res.output
    assert from_file(default_out).tables[0].columns[0].profile.categorical is None

    wide_out = tmp_path / "wide.yaml"
    res = runner.invoke(cli, ["profile", str(data), "--out", str(wide_out),
                              "--sample", "400", "--max-categorical", "200",
                              # see _profile(): suppression would eat the
                              # rare codes this assertion counts
                              "--min-cell-count", "0"])
    assert res.exit_code == 0, res.output
    cat = from_file(wide_out).tables[0].columns[0].profile.categorical
    assert cat is not None
    assert len(cat.values) == _codes_frame(n_codes=120, n_rows=400)["code"].nunique()


def test_cli_exposes_max_categorical_ratio(tmp_path):
    data = tmp_path / "d.csv"
    _codes_frame(n_codes=120, n_rows=1_000).to_csv(data, index=False)
    runner = CliRunner()
    out = tmp_path / "s.yaml"
    res = runner.invoke(cli, ["profile", str(data), "--out", str(out),
                              "--sample", "1000",
                              "--max-categorical-ratio", "0.2"])
    assert res.exit_code == 0, res.output
    assert from_file(out).tables[0].columns[0].profile.categorical is not None


def test_profile_set_threads_the_thresholds_through():
    df = _codes_frame(n_codes=120, n_rows=400)
    spec = DatasetProfiler.profile_set({"t": df}, sample=None, seed=SEED,
                                       max_categorical=200)
    assert spec.tables[0].columns[0].profile.categorical is not None
