"""Regression tests for Spec serialization (loaders.to_file / spec_to_dict).

Before the fix, ``to_file`` dumped the model verbatim, so every optional field
was written out as ``null``/``{}``/``[]``. An 18-column profiled table came out
at ~390 lines, of which only a small fraction carried information.
"""
import numpy as np
import pandas as pd
import pytest
import yaml

from syntab.loaders import from_file, spec_to_dict, to_file
from syntab.profiler import DatasetProfiler
from syntab.spec import (
    ColumnSpec,
    RelationshipSpec,
    Settings,
    Spec,
    SpecMetadata,
    TableSpec,
    WhenClause,
)


def _wide_profiled_spec():
    """A profiled spec shaped like the CFPB example: many mostly-empty columns."""
    rng = np.random.default_rng(0)
    n = 200
    data = {"id": list(range(1, n + 1))}
    for i in range(6):
        data[f"cat_{i}"] = list(rng.choice(["a", "b", "c"], size=n))
    for i in range(6):
        data[f"num_{i}"] = rng.normal(50, 10, n)
    for i in range(5):
        data[f"txt_{i}"] = [f"free text value {j} {i}" for j in range(n)]
    return DatasetProfiler(pd.DataFrame(data), name="wide", sample=None).profile()


def test_yaml_has_no_null_or_empty_placeholders(tmp_path):
    out = tmp_path / "spec.yaml"
    to_file(_wide_profiled_spec(), out)
    text = out.read_text(encoding="utf-8")

    assert ": null" not in text, "None-valued keys must be omitted"
    assert ": {}" not in text, "empty mappings must be omitted"
    assert ": []" not in text, "empty sequences must be omitted"
    # nothing should survive that only ever holds a default
    for noise_key in ("depends_on:", "when:", "pii: false",
                      "pii_strategy:", "owner:", "license:"):
        assert noise_key not in text, f"{noise_key!r} should not be emitted"
    # the only description in the file is the metadata one the profiler sets;
    # the 17 null column-level descriptions are gone
    assert text.count("description:") == 1


def test_emitted_spec_is_dramatically_smaller_than_a_verbatim_dump(tmp_path):
    spec = _wide_profiled_spec()
    out = tmp_path / "spec.yaml"
    to_file(spec, out)

    lean = len(out.read_text(encoding="utf-8").splitlines())
    verbatim = len(
        yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=False).splitlines()
    )
    # 18 columns previously produced ~390 lines; the trimmed form is a small
    # fraction of that. Keep the bound loose so the test tracks the property,
    # not an exact byte count.
    assert lean < verbatim / 2, f"lean={lean} verbatim={verbatim}"


def test_roundtrip_is_lossless(tmp_path):
    """exclude_defaults is only safe if pydantic re-applies what we dropped."""
    spec = _wide_profiled_spec()
    for suffix in ("yaml", "json"):
        out = tmp_path / f"spec.{suffix}"
        to_file(spec, out)
        assert from_file(out).model_dump() == spec.model_dump()


def test_roundtrip_is_lossless_for_a_rich_hand_authored_spec(tmp_path):
    spec = Spec(
        metadata=SpecMetadata(name="rich", owner="data-team", tags=["a", "b"]),
        settings=Settings(seed=3, fallback="drop", max_rule_attempts=7),
        tables=[
            TableSpec(
                name="users", row_count=10, primary_key="id",
                columns=[ColumnSpec(name="id", dtype="int", generator="sequence",
                                    constraints={"unique": True})],
            ),
            TableSpec(
                name="orders", row_count=20, primary_key="id",
                columns=[
                    ColumnSpec(name="id", dtype="int", generator="sequence"),
                    ColumnSpec(name="user_id", dtype="int", generator="fk"),
                    ColumnSpec(
                        name="tier", dtype="str", generator="const",
                        params={"value": "std"}, depends_on=["user_id"],
                        when=[WhenClause(**{"if": "user_id > 5",
                                            "generator": "const",
                                            "params": {"value": "vip"}})],
                    ),
                ],
                relationships=[RelationshipSpec(
                    **{"from": "user_id", "to": "users.id", "alias": "user"}
                )],
                rules=["tier != ''"],
            ),
        ],
    )
    out = tmp_path / "rich.yaml"
    to_file(spec, out)
    assert from_file(out).model_dump() == spec.model_dump()


def test_public_alias_keys_are_used(tmp_path):
    """Emitted YAML must use the documented keys, not internal field names."""
    spec = Spec(
        metadata=SpecMetadata(name="a"),
        tables=[
            TableSpec(name="p", row_count=1, primary_key="id",
                      columns=[ColumnSpec(name="id", dtype="int")]),
            TableSpec(
                name="c", row_count=1,
                columns=[
                    ColumnSpec(name="p_id", dtype="int", generator="fk"),
                    ColumnSpec(name="x", dtype="str",
                               when=[WhenClause(**{"if": "p_id > 0",
                                                   "generator": "const"})]),
                ],
                relationships=[RelationshipSpec(**{"from": "p_id", "to": "p.id"})],
            ),
        ],
    )
    out = tmp_path / "alias.yaml"
    to_file(spec, out)
    text = out.read_text(encoding="utf-8")
    assert "from: p_id" in text
    assert "from_" not in text
    assert "if: p_id > 0" in text
    assert "condition:" not in text


def test_spec_to_dict_always_declares_spec_version():
    assert spec_to_dict(_wide_profiled_spec())["spec_version"] == "1.0"


def test_unsupported_format_still_rejected(tmp_path):
    with pytest.raises(ValueError):
        to_file(_wide_profiled_spec(), tmp_path / "x.toml")
