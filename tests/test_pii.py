"""Tests for PII handling (#4): flagging, anonymization, and spec safety."""
import pandas as pd

from syntab.engine import GenerationEngine
from syntab.profiler import DatasetProfiler
from syntab.spec import (
    ColumnSpec, Settings, Spec, SpecMetadata, TableSpec,
)


def _gen(spec: Spec) -> dict:
    return GenerationEngine(spec).run().to_frames()


def test_profiler_auto_detects_email_pii():
    df = pd.DataFrame({"user_email": ["a@b.com", "c@d.com", "e@f.com"]})
    spec = DatasetProfiler(df, name="t", sample=None).profile()
    col = spec.tables[0].columns[0]
    assert col.pii is True
    assert col.pii_strategy == "faker"
    assert col.generator == "faker.email"
    # no real values / pattern leaked into the spec
    assert col.profile is None
    assert col.params == {}


def test_profiler_explicit_pii_column():
    df = pd.DataFrame({"name": ["Alice", "Bob", "Carol"]})
    spec = DatasetProfiler(
        df, name="t", sample=None, pii_columns=["name"], pii_strategy="mask"
    ).profile()
    col = spec.tables[0].columns[0]
    assert col.pii is True
    assert col.pii_strategy == "mask"
    assert col.generator == "faker.name"  # real values stripped


def test_engine_pii_faker_overrides_real_values():
    spec = Spec(
        metadata=SpecMetadata(name="t"),
        settings=Settings(seed=1, fallback="raise", max_rule_attempts=100),
        tables=[TableSpec(name="t", row_count=20, columns=[
            ColumnSpec(name="name", dtype="str", generator="choice",
                       params={"values": ["Alice", "Bob", "Carol"]},
                       pii=True, pii_strategy="faker"),
        ])],
    )
    vals = _gen(spec)["t"]["name"].tolist()
    assert "Alice" not in vals and "Bob" not in vals and "Carol" not in vals
    assert len(vals) == 20


def test_engine_pii_mask_email():
    spec = Spec(
        metadata=SpecMetadata(name="t"),
        settings=Settings(seed=2, fallback="raise", max_rule_attempts=100),
        tables=[TableSpec(name="t", row_count=10, columns=[
            ColumnSpec(name="email", dtype="str", generator="faker.email",
                       pii_strategy="mask"),
        ])],
    )
    vals = _gen(spec)["t"]["email"].tolist()
    for v in vals:
        local, _, domain = v.partition("@")
        assert domain  # domain preserved
        assert "@" in v
        assert local[1:] == "*" * (len(local) - 1)


def test_engine_pii_redact():
    spec = Spec(
        metadata=SpecMetadata(name="t"),
        settings=Settings(seed=3, fallback="raise", max_rule_attempts=100),
        tables=[TableSpec(name="t", row_count=5, columns=[
            ColumnSpec(name="secret", dtype="str", generator="string",
                       params={"length": 8}, pii_strategy="redact"),
        ])],
    )
    vals = _gen(spec)["t"]["secret"].tolist()
    assert all(v == "REDACTED" for v in vals)


def test_engine_pii_hash_is_deterministic_token():
    spec = Spec(
        metadata=SpecMetadata(name="t"),
        settings=Settings(seed=4, fallback="raise", max_rule_attempts=100),
        tables=[TableSpec(name="t", row_count=5, columns=[
            ColumnSpec(name="id_token", dtype="str", generator="int",
                       params={"min": 1, "max": 1000}, pii_strategy="hash"),
        ])],
    )
    vals = _gen(spec)["t"]["id_token"].tolist()
    assert all(len(v) == 12 and all(c in "0123456789abcdef" for c in v) for v in vals)


def test_end_to_end_pii_profile_generate_no_leak():
    df = pd.DataFrame({
        "id": range(1, 51),
        "email": [f"user{i}@real-domain.com" for i in range(1, 51)],
        "name": [f"RealName{i}" for i in range(111, 161)],
    })
    spec = DatasetProfiler(
        df, name="people", sample=None, pii_columns=["email", "name"]
    ).profile()
    frames = GenerationEngine(spec).run().to_frames()
    out = frames["people"]
    # real PII must not appear in the synthetic output
    assert not out["email"].str.contains("real-domain.com", regex=False).any()
    assert not out["name"].str.startswith("RealName").any()
    # and the spec itself must not contain the real values
    cols = {c.name: c for c in spec.tables[0].columns}
    assert cols["email"].profile is None
