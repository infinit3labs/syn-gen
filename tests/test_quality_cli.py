"""The three evaluation commands, and the distinction between them.

`syntab quality` scores, `syntab diagnose` judges structure, `syntab check`
judges the contract. The exit codes differ deliberately: a quality score is a
measurement, so the command exits 0 whatever the number unless a threshold is
asked for; a structural failure is a failure, so it exits non-zero.
"""
import json
import re

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from syntab.cli import cli

SEED = 41


@pytest.fixture
def paths(tmp_path):
    rng = np.random.default_rng(SEED)
    n = 800
    base = rng.normal(50, 10, n)
    real = pd.DataFrame({
        "amount": base * 3,
        "score": base + rng.normal(0, 1, n),
        "tier": np.where(base < 50, "low", "high"),
    })
    good = real.sample(frac=1.0, random_state=1).reset_index(drop=True)
    # Independently shuffled: identical marginals, no relationships left.
    broken = pd.DataFrame({c: rng.permutation(real[c].to_numpy())
                           for c in real.columns})

    rp, gp, bp = (tmp_path / "real.csv", tmp_path / "good.csv",
                  tmp_path / "broken.csv")
    real.to_csv(rp, index=False)
    good.to_csv(gp, index=False)
    broken.to_csv(bp, index=False)
    return str(rp), str(gp), str(bp)


def _score(output):
    m = re.search(r"Overall Score:\s+([0-9.]+)", output)
    assert m, output
    return float(m.group(1))


def test_quality_reports_a_score_and_the_properties(paths):
    real, good, _ = paths
    result = CliRunner().invoke(cli, ["quality", "--real", real, "--synthetic", good])
    assert result.exit_code == 0, result.output
    assert "QUALITY REPORT" in result.output
    for prop in ("Column Shapes", "Column Pair Trends", "Coverage",
                 "Boundary Adherence", "Missing Value Similarity"):
        assert prop in result.output
    assert _score(result.output) > 0.9


def test_quality_scores_destroyed_relationships_lower(paths):
    real, good, broken = paths
    run = CliRunner().invoke
    good_score = _score(run(cli, ["quality", "--real", real, "--synthetic", good]).output)
    broken_score = _score(run(cli, ["quality", "--real", real, "--synthetic", broken]).output)
    assert broken_score < good_score - 0.1


def test_quality_exits_zero_even_on_a_bad_score(paths):
    """A score is a measurement, not a verdict."""
    real, _, broken = paths
    result = CliRunner().invoke(cli, ["quality", "--real", real, "--synthetic", broken])
    assert result.exit_code == 0


def test_quality_min_score_turns_it_into_a_gate(paths):
    real, _, broken = paths
    result = CliRunner().invoke(
        cli, ["quality", "--real", real, "--synthetic", broken, "--min-score", "0.99"])
    assert result.exit_code == 1
    assert "below" in result.output


def test_quality_writes_json(paths, tmp_path):
    real, good, _ = paths
    out = tmp_path / "q.json"
    result = CliRunner().invoke(
        cli, ["quality", "--real", real, "--synthetic", good, "--out", str(out)])
    assert result.exit_code == 0
    payload = json.loads(out.read_text())
    assert 0.0 <= payload["overall_score"] <= 1.0
    assert len(payload["properties"]) == 5


def test_quality_verbose_shows_the_per_column_breakdown(paths):
    real, good, _ = paths
    result = CliRunner().invoke(
        cli, ["quality", "--real", real, "--synthetic", good, "--verbose"])
    assert "KSComplement" in result.output
    assert "CorrelationSimilarity" in result.output


def test_diagnose_passes_on_valid_data_and_exits_zero(paths):
    real, good, _ = paths
    result = CliRunner().invoke(cli, ["diagnose", "--real", real, "--synthetic", good])
    assert result.exit_code == 0, result.output
    assert "DIAGNOSTIC REPORT" in result.output
    assert "VALID" in result.output


def test_diagnose_exits_non_zero_on_structural_failure(tmp_path):
    real = pd.DataFrame({"n": [0.0, 1.0, 2.0], "c": ["a", "b", "a"]})
    synth = pd.DataFrame({"n": [0.0, 1.0, 999.0], "c": ["a", "b", "a"]})
    rp, sp = tmp_path / "r.csv", tmp_path / "s.csv"
    real.to_csv(rp, index=False)
    synth.to_csv(sp, index=False)
    result = CliRunner().invoke(
        cli, ["diagnose", "--real", str(rp), "--synthetic", str(sp)])
    assert result.exit_code == 1
    assert "INVALID" in result.output
    assert "BoundaryAdherence" in result.output


def test_the_three_commands_are_all_registered_and_described():
    result = CliRunner().invoke(cli, ["--help"])
    for name in ("quality", "diagnose", "check", "compare"):
        assert name in result.output


def test_compare_help_says_it_is_marginals_only():
    """The limitation should be discoverable from the tool, not just the docs."""
    result = CliRunner().invoke(cli, ["compare", "--help"])
    assert "MARGINALS ONLY" in result.output
    assert "quality" in result.output
