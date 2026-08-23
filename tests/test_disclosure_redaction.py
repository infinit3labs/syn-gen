"""Round-trip tests for --redact-categoricals.

A profiled spec is the one artefact this tool asks users to commit to version
control, and for every categorical column it records the real distinct values
and their exact frequencies. Redaction replaces the *labels* with opaque
tokens while keeping the cardinality and the frequency vector, which is all
the generation engine actually consumes -- `infer.resolve_generator` turns
`profile.categorical.values` into `choice` values + weights and nothing else
reads the labels.

The test that matters is the full round trip: a redacted spec must still
generate, and the generated data must still conform to it.
"""
import pandas as pd
import pytest

from syntab.conformance import validate_against_spec
from syntab.engine import GenerationEngine
from syntab.loaders import from_file, to_file
from syntab.profiler import DatasetProfiler

SEED = 11


def _frame(n=400):
    """A categorical column whose labels are the sensitive part."""
    diagnoses = (["Type 2 diabetes"] * 200 + ["Hypertension"] * 120
                 + ["Asthma"] * 50 + ["Sickle cell anaemia"] * 20
                 + ["Huntington disease"] * 10)
    assert len(diagnoses) == n
    return pd.DataFrame({
        "id": range(1, n + 1),
        "diagnosis": diagnoses,
        "site": (["north"] * 250 + ["south"] * 150),
    })


def _profile(df, **kw):
    return DatasetProfiler(df, name="clinic", sample=None, seed=SEED, **kw).profile()


def _cat(spec, col):
    return {c.name: c for c in spec.tables[0].columns}[col].profile.categorical


# ---------------------------------------------------------------------------
# what redaction does and does not change
# ---------------------------------------------------------------------------

def test_redaction_is_off_by_default():
    cat = _cat(_profile(_frame()), "diagnosis")
    assert cat.redacted is False
    assert "Type 2 diabetes" in cat.values


def test_redaction_removes_the_real_labels():
    cat = _cat(_profile(_frame(), redact_categoricals=True), "diagnosis")
    assert cat.redacted is True
    for real in ("Type 2 diabetes", "Hypertension", "Asthma",
                 "Sickle cell anaemia", "Huntington disease"):
        assert real not in cat.values


def test_redaction_preserves_cardinality_and_frequencies_exactly():
    plain = _cat(_profile(_frame()), "diagnosis")
    redacted = _cat(_profile(_frame(), redact_categoricals=True), "diagnosis")
    assert len(redacted.values) == len(plain.values) == 5
    # the frequency vector is the generation-relevant content, and it is intact
    assert list(redacted.values.values()) == list(plain.values.values())


def test_tokens_are_ordered_by_descending_frequency():
    cat = _cat(_profile(_frame(), redact_categoricals=True), "diagnosis")
    keys = list(cat.values)
    assert keys == sorted(keys), "tokens should sort in emitted order"
    freqs = list(cat.values.values())
    assert freqs == sorted(freqs, reverse=True)
    assert keys[0] == "value_001"          # the modal category
    assert cat.values["value_001"] == pytest.approx(200 / 400)


def test_no_real_value_survives_into_the_written_spec_file(tmp_path):
    out = tmp_path / "spec.yaml"
    to_file(_profile(_frame(), redact_categoricals=True), out)
    text = out.read_text(encoding="utf-8")
    for real in ("Type 2 diabetes", "Hypertension", "Asthma",
                 "Sickle cell anaemia", "Huntington disease",
                 "north", "south"):
        assert real not in text, f"{real!r} leaked into the spec file"
    assert "value_001" in text
    assert "redacted: true" in text


def test_every_categorical_column_is_redacted_not_just_the_first():
    spec = _profile(_frame(), redact_categoricals=True)
    for col in ("diagnosis", "site"):
        cat = _cat(spec, col)
        assert cat.redacted is True
        assert all(k.startswith("value_") for k in cat.values)


# ---------------------------------------------------------------------------
# the round trip: profile (redacted) -> generate -> validate
# ---------------------------------------------------------------------------

def test_full_roundtrip_profile_redacted_generate_validate():
    df = _frame()
    spec = _profile(df, redact_categoricals=True)

    frames = GenerationEngine(spec).run().to_frames()
    out = frames["clinic"]

    # generation worked and produced the contracted number of rows
    assert len(out) == len(df)

    # the generated data conforms to the spec it came from
    report = validate_against_spec(frames, spec)
    assert report.overall_ok, report.to_text()

    # the synthetic column reproduces the recorded frequency shape
    cat = _cat(spec, "diagnosis")
    assert set(out["diagnosis"]) <= set(cat.values)
    observed = out["diagnosis"].value_counts(normalize=True)
    for token, expected in cat.values.items():
        assert observed.get(token, 0.0) == pytest.approx(expected, abs=0.08)

    # and no real label reached the output
    for real in ("Type 2 diabetes", "Sickle cell anaemia", "north"):
        assert not (out.astype(str) == real).any().any()


def test_roundtrip_through_a_file_on_disk(tmp_path):
    """The spec has to survive serialization, since that is the shared artefact."""
    out = tmp_path / "spec.yaml"
    to_file(_profile(_frame(), redact_categoricals=True), out)
    reloaded = from_file(out)
    assert _cat(reloaded, "diagnosis").redacted is True
    frames = GenerationEngine(reloaded).run().to_frames()
    assert validate_against_spec(frames, reloaded).overall_ok


def test_profile_set_threads_the_flag_through():
    spec = DatasetProfiler.profile_set(
        {"clinic": _frame()}, sample=None, seed=SEED, redact_categoricals=True)
    cat = {c.name: c for c in spec.tables[0].columns}["diagnosis"].profile.categorical
    assert cat.redacted is True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_exposes_redact_categoricals(tmp_path):
    from click.testing import CliRunner
    from syntab.cli import cli

    data = tmp_path / "d.csv"
    _frame().to_csv(data, index=False)
    runner = CliRunner()

    plain = tmp_path / "plain.yaml"
    res = runner.invoke(cli, ["profile", str(data), "--out", str(plain)])
    assert res.exit_code == 0, res.output
    assert "Type 2 diabetes" in plain.read_text(encoding="utf-8")

    red = tmp_path / "red.yaml"
    res = runner.invoke(cli, ["profile", str(data), "--out", str(red),
                              "--redact-categoricals"])
    assert res.exit_code == 0, res.output
    text = red.read_text(encoding="utf-8")
    assert "Type 2 diabetes" not in text
    assert "value_001" in text

    # and the redacted spec is still generatable from the CLI
    gen_out = tmp_path / "gen"
    res = runner.invoke(cli, ["generate", "--spec", str(red),
                              "--out", str(gen_out), "--format", "csv"])
    assert res.exit_code == 0, res.output
