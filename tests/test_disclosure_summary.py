"""Tests for the post-profile disclosure summary.

The failure this addresses is not the profiler's behaviour -- embedding real
values is what makes the round trip work -- it is that nothing told the user an
assessment was needed, so the assessment did not happen. These tests assert
that the numbers are right and that the notice actually reaches the user.
"""
import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from syntab.cli import cli
from syntab.disclosure import DisclosureBudget, DisclosureReport
from syntab.profiler import RECOMMENDED_MIN_CELL_COUNT, DatasetProfiler

SEED = 3


def _clinic(n=400):
    rng = np.random.default_rng(SEED)
    common = ["Type 2 diabetes"] * 200 + ["Hypertension"] * 120 + ["Asthma"] * 60
    rare = [f"Rare condition {i}" for i in range(20)]
    return pd.DataFrame({
        "patient_id": range(1, n + 1),
        "patient_name": [f"Person {i}" for i in range(n)],
        "email": [f"p{i}@example.org" for i in range(n)],
        "diagnosis": common + rare,
        "site": list(rng.choice(["North", "South"], size=n)),
        "cost": rng.lognormal(5, 1.0, n),
    })


def _report(df=None, **kw):
    spec = DatasetProfiler(df if df is not None else _clinic(),
                           name="clinic", sample=None, seed=SEED, **kw).profile()
    return DisclosureReport.from_spec(spec)


# ---------------------------------------------------------------------------
# counting
# ---------------------------------------------------------------------------

def test_counts_the_real_values_embedded_and_the_columns_they_came_from():
    rep = _report()
    # diagnosis (23 distinct) + site (2 distinct)
    assert len(rep.open_categoricals) == 2
    assert rep.embedded_values == 25
    assert rep.n_columns == 6
    assert rep.needs_review is True


def test_counts_rare_values_as_the_disclosure_risk():
    rep = _report()
    # the 20 one-off conditions are each below the recommended threshold
    assert rep.rare_values == 20


def test_counts_other_kinds_of_embedded_real_values():
    rep = _report()
    assert rep.n_numeric_ranges >= 1, "cost embeds a real min/max"


def test_reports_pii_columns_and_which_signal_fired():
    rep = _report()
    by_col = {p.column: p for p in rep.pii}
    assert set(by_col) == {"patient_name", "email"}
    assert by_col["patient_name"].signals == ["column-name"]
    assert by_col["email"].signals == ["column-name", "value-pattern"]
    assert by_col["patient_name"].generator == "faker.name"


def test_reports_id_like_columns_that_were_not_auto_flagged():
    rep = _report()
    assert rep.key_candidates == ["patient_id"]


def test_reports_that_redaction_was_applied():
    rep = _report(redact_categoricals=True)
    assert rep.redaction_applied is True
    assert rep.embedded_values == 0, "no real labels are in the spec"
    assert len(rep.redacted_categoricals) == 2


def test_reports_that_suppression_was_applied():
    rep = _report(min_cell_count=RECOMMENDED_MIN_CELL_COUNT)
    assert rep.suppression_applied is True
    assert rep.min_cell_count == RECOMMENDED_MIN_CELL_COUNT
    assert rep.suppressed_values == 20
    assert rep.rare_values == 0


def test_a_spec_with_nothing_embedded_says_so():
    """A uuid column carries no profile at all, so there is nothing to review."""
    df = pd.DataFrame({"row_uuid": [
        "3f2504e0-4f89-11d3-9a0c-0305e82c3301",
        "3f2504e0-4f89-11d3-9a0c-0305e82c3302",
        "3f2504e0-4f89-11d3-9a0c-0305e82c3303",
    ]})
    rep = _report(df)
    assert rep.embedded_values == 0
    assert rep.needs_review is False
    assert "No real source values are embedded" in rep.to_text()


def test_a_pii_only_spec_still_needs_review():
    """Nothing real is embedded, but a human should still confirm the call."""
    rep = _report(pd.DataFrame({"patient_name": ["a", "b", "c"]}))
    assert rep.embedded_values == 0
    assert rep.needs_review is True
    assert len(rep.pii) == 1


def test_multi_table_specs_qualify_column_names():
    spec = DatasetProfiler.profile_set(
        {"clinic": _clinic(), "sites": pd.DataFrame({"id": [1, 2], "x": [1.0, 2.0]})},
        sample=None, seed=SEED)
    rep = DisclosureReport.from_spec(spec)
    assert any(c.qualified == "clinic.diagnosis" for c in rep.categoricals)
    assert rep.key_candidates == ["clinic.patient_id"]


# ---------------------------------------------------------------------------
# the rendered notice
# ---------------------------------------------------------------------------

def test_notice_states_the_headline_numbers():
    text = _report().to_text(spec_path="spec.yaml")
    assert "DISCLOSURE NOTICE" in text
    assert "spec.yaml" in text
    assert "25 distinct categorical value(s) from 2 of 6 column(s)" in text
    assert "PII columns flagged: 2 of 6" in text
    assert "patient_name" in text and "column-name" in text
    assert "Id-like columns NOT auto-flagged" in text and "patient_id" in text


def test_notice_names_the_controls_and_whether_they_ran():
    off = _report().to_text()
    assert "--redact-categoricals   not applied" in off
    assert "--min-cell-count        not applied" in off
    assert f"recommended: {RECOMMENDED_MIN_CELL_COUNT}" in off
    assert "20 embedded value(s) occur fewer than 5 times in the source." in off

    on = _report(min_cell_count=5, redact_categoricals=True).to_text()
    assert "--redact-categoricals   APPLIED" in on
    assert "--min-cell-count        APPLIED at K=5" in on


def test_notice_says_the_spec_is_derived_data_and_must_be_reviewed():
    text = _report().to_text()
    assert "DERIVED FROM SOURCE DATA" in text
    assert "before committing it to version control" in text
    assert "not automatically" in text and "anonymous" in text


def test_disclosure_budget_reports_only_exceeded_limits():
    rep = _report()
    budget = DisclosureBudget(max_unredacted_values=10, max_rare_values=0)

    violations = rep.check_budget(budget)

    assert [v.metric for v in violations] == [
        "unredacted categorical values", "rare categorical values"
    ]
    assert violations[0].actual == 25
    assert violations[0].limit == 10
    assert violations[1].actual == 20
    assert violations[1].limit == 0


def test_disclosure_budget_accepts_redaction_and_suppression():
    rep = _report(min_cell_count=RECOMMENDED_MIN_CELL_COUNT,
                  redact_categoricals=True)

    assert rep.check_budget(DisclosureBudget(
        max_unredacted_values=0, max_rare_values=0,
        max_conditional_cells=0)) == []


def test_disclosure_budget_rejects_negative_limits():
    with pytest.raises(ValueError, match="non-negative"):
        DisclosureBudget(max_rare_values=-1)


# ---------------------------------------------------------------------------
# it actually reaches the user
# ---------------------------------------------------------------------------

def test_profile_command_emits_the_notice(tmp_path):
    data = tmp_path / "d.csv"
    _clinic().to_csv(data, index=False)
    res = CliRunner().invoke(
        cli, ["profile", str(data), "--out", str(tmp_path / "s.yaml")])
    assert res.exit_code == 0, res.output
    combined = res.output + (res.stderr or "")
    assert "DISCLOSURE NOTICE" in combined
    assert "patient_name" in combined
    assert "--min-cell-count" in combined


def test_notice_goes_to_stderr_so_it_survives_redirection(tmp_path):
    """stdout may be piped or captured; a notice nobody sees is not a notice."""
    data = tmp_path / "d.csv"
    _clinic().to_csv(data, index=False)
    res = CliRunner().invoke(
        cli, ["profile", str(data), "--out", str(tmp_path / "s.yaml")])
    assert res.exit_code == 0
    assert "DISCLOSURE NOTICE" in res.stderr
    assert "DISCLOSURE NOTICE" not in res.stdout


def test_profile_cli_enforces_disclosure_budget_before_writing(tmp_path):
    data = tmp_path / "d.csv"
    _clinic().to_csv(data, index=False)
    out = tmp_path / "s.yaml"
    res = CliRunner().invoke(cli, [
        "profile", str(data), "--out", str(out),
        "--max-unredacted-values", "10",
    ])

    assert res.exit_code != 0
    assert "disclosure budget exceeded" in res.output
    assert not out.exists()
