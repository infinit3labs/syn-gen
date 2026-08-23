"""The default value of --min-cell-count.

Minimum cell-size suppression generalizes the categorical values that occur too
few times in the source to be safely republished. Those are exactly the values
that identify individuals, so the protective setting is the one that should
apply when nobody has expressed a preference.

This file pins the default ON, at RECOMMENDED_MIN_CELL_COUNT, so it cannot
drift back to off silently -- and pins ``--min-cell-count 0`` as a working
explicit opt-out, so the data holder who wants full fidelity can still say so.
"""
import pandas as pd
from click.testing import CliRunner

from syntab.cli import cli
from syntab.profiler import (
    DEFAULT_MIN_CELL_COUNT,
    OTHER_BUCKET_LABEL,
    RECOMMENDED_MIN_CELL_COUNT,
    DatasetProfiler,
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


def _profile_csv(tmp_path, *extra):
    src = tmp_path / "d.csv"
    _frame().to_csv(src, index=False)
    out = tmp_path / "spec.yaml"
    res = CliRunner().invoke(
        cli, ["profile", str(src), "--out", str(out), "--sample", "0", *extra])
    assert res.exit_code == 0, res.output
    return out.read_text(encoding="utf-8"), res


# ---------------------------------------------------------------------------
# the default itself
# ---------------------------------------------------------------------------

def test_the_default_is_the_recommended_threshold():
    """The default and the recommendation are the same number, by construction.

    If they ever diverge the disclosure summary starts recommending something
    the tool does not do, which is worse than either setting on its own.
    """
    assert DEFAULT_MIN_CELL_COUNT == RECOMMENDED_MIN_CELL_COUNT
    assert DEFAULT_MIN_CELL_COUNT == 5


def test_suppression_applies_without_being_asked_for():
    cat = _cat(_frame())
    assert cat.min_cell_count == RECOMMENDED_MIN_CELL_COUNT
    assert cat.suppressed_values == 3
    assert OTHER_BUCKET_LABEL in cat.values
    for identifying in ("gamma", "delta", "epsilon"):
        assert identifying not in cat.values, f"{identifying} was republished"
    assert cat.rare_value_count == 0


# ---------------------------------------------------------------------------
# the opt-out still works
# ---------------------------------------------------------------------------

def test_zero_is_still_an_explicit_opt_out():
    cat = _cat(_frame(), min_cell_count=0)
    assert cat.min_cell_count is None
    assert cat.suppressed_values == 0
    assert OTHER_BUCKET_LABEL not in cat.values
    assert "epsilon" in cat.values, "the single-member category is published"
    assert cat.rare_value_count == 3


def test_cli_zero_writes_the_rare_values_into_the_spec(tmp_path):
    text, _ = _profile_csv(tmp_path, "--min-cell-count", "0")
    for identifying in ("gamma", "delta", "epsilon"):
        assert identifying in text


def test_cli_default_keeps_the_rare_values_out_of_the_spec(tmp_path):
    text, _ = _profile_csv(tmp_path)
    for identifying in ("gamma", "delta", "epsilon"):
        assert identifying not in text
    assert OTHER_BUCKET_LABEL in text


# ---------------------------------------------------------------------------
# what the operator is told
# ---------------------------------------------------------------------------

def test_help_text_shows_the_new_default():
    out = CliRunner().invoke(cli, ["profile", "--help"]).output
    collapsed = " ".join(out.split())
    assert "--min-cell-count" in collapsed
    assert "[default: 5]" in collapsed


def test_disclosure_summary_reports_suppression_as_applied_by_default(tmp_path):
    _, res = _profile_csv(tmp_path)
    collapsed = " ".join(res.output.split())
    assert "--min-cell-count APPLIED at K=5" in collapsed


def test_disclosure_summary_reports_the_opt_out_as_not_applied(tmp_path):
    _, res = _profile_csv(tmp_path, "--min-cell-count", "0")
    collapsed = " ".join(res.output.split())
    assert "--min-cell-count not applied" in collapsed
