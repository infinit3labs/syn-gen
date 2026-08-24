"""Tests for the INF-225 reproducibility / scale / merge work.

The ticket asks for five things:

  1. same input + options + seed -> byte-equivalent Spec files (YAML/JSON)
  2. bounded-memory resource controls (--max-rows, --columns)
  3. full config snapshot in metadata.profiling (controls + syntab_version
     + source fingerprint)
  4. column-level merge drift detection (added/removed/changed/preserved)
  5. (handled in cli tests + docs) the CLI flags exist and the help text
     names the contract

The "byte-equivalent" test is the contract: the existing
``test_same_input_and_controls_produce_the_same_profile`` checked
``spec_to_dict`` equality; this file extends that to literal file bytes.
"""
import json
import os
from pathlib import Path

import pandas as pd
import pytest

from syntab import __version__ as SYNTAB_VERSION
from syntab.loaders import from_file, spec_to_dict, to_file
from syntab.profiler import (
    DatasetProfiler,
    MergeReport,
    _read_dataframe_bounded,
    merge_preserving_edits,
)


# ---------------------------------------------------------------------------
# 1. Byte-equivalent Specs across runs
# ---------------------------------------------------------------------------

def _two_column_frame():
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6, 7, 8],
        "status": ["open", "closed", "open", "closed",
                   "pending", "open", "closed", "open"],
    })


def test_same_input_produces_byte_equivalent_yaml(tmp_path):
    """A re-run with the same input, options, and seed must write a
    byte-identical file. This is the literal contract the ticket names."""
    df = _two_column_frame()
    spec_a = DatasetProfiler(df, name="t", sample=4, seed=7).profile()
    spec_b = DatasetProfiler(df, name="t", sample=4, seed=7).profile()

    p_a = tmp_path / "a.yaml"
    p_b = tmp_path / "b.yaml"
    to_file(spec_a, p_a)
    to_file(spec_b, p_b)

    assert p_a.read_bytes() == p_b.read_bytes(), \
        "YAML serialization is not deterministic; a re-run produced different bytes"


def test_same_input_produces_byte_equivalent_json(tmp_path):
    df = _two_column_frame()
    spec_a = DatasetProfiler(df, name="t", sample=4, seed=7).profile()
    spec_b = DatasetProfiler(df, name="t", sample=4, seed=7).profile()

    p_a = tmp_path / "a.json"
    p_b = tmp_path / "b.json"
    to_file(spec_a, p_a, fmt="json")
    to_file(spec_b, p_b, fmt="json")

    assert p_a.read_bytes() == p_b.read_bytes(), \
        "JSON serialization is not deterministic; a re-run produced different bytes"


def test_yaml_and_json_round_trip_to_the_same_spec(tmp_path):
    """Writing YAML and JSON of the same spec, then loading both, gives
    the same Spec back. This is what the round-trip guarantee is FOR."""
    df = _two_column_frame()
    spec = DatasetProfiler(df, name="t", sample=4, seed=7).profile()

    p_y = tmp_path / "s.yaml"
    p_j = tmp_path / "s.json"
    to_file(spec, p_y)
    to_file(spec, p_j, fmt="json")

    re_y = from_file(p_y)
    re_j = from_file(p_j)
    assert spec_to_dict(re_y) == spec_to_dict(re_j)


# ---------------------------------------------------------------------------
# 2. Bounded-memory read
# ---------------------------------------------------------------------------

def test_bounded_read_caps_csv_rows(tmp_path):
    p = tmp_path / "wide.csv"
    p.write_text("id,val\n" + "\n".join(f"{i},{i*2}" for i in range(20)))
    df = _read_dataframe_bounded(str(p), max_rows=5)
    assert len(df) == 5
    assert list(df.columns) == ["id", "val"]


def test_bounded_read_subsets_columns_before_capping(tmp_path):
    p = tmp_path / "wide.csv"
    p.write_text("id,val,other\n" + "\n".join(f"{i},{i*2},{i*3}" for i in range(20)))
    df = _read_dataframe_bounded(str(p), max_rows=5, columns=["val"])
    assert len(df) == 5
    assert list(df.columns) == ["val"]


def test_bounded_read_handles_jsonl(tmp_path):
    p = tmp_path / "wide.jsonl"
    p.write_text("\n".join(json.dumps({"i": i, "v": i * 2}) for i in range(10)))
    df = _read_dataframe_bounded(str(p), max_rows=4)
    assert len(df) == 4


def test_bounded_read_falls_back_to_whole_load_when_uncapped(tmp_path):
    """The default-from_file path (no max_rows/columns) should still read
    the file whole in one call, preserving the previous behavior."""
    p = tmp_path / "wide.csv"
    p.write_text("id,val\n" + "\n".join(f"{i},{i*2}" for i in range(5)))
    df = _read_dataframe_bounded(str(p))
    assert len(df) == 5
    assert list(df.columns) == ["id", "val"]


def test_from_file_with_max_rows_caps_a_csv_profile(tmp_path):
    """End-to-end: --max-rows bounds what the profiler sees."""
    p = tmp_path / "wide.csv"
    rows = ["k,v"] + [f"{i},{i*2}" for i in range(50)]
    p.write_text("\n".join(rows))

    full = DatasetProfiler.from_file(str(p), name="t").profile()
    capped = DatasetProfiler.from_file(str(p), name="t", max_rows=10).profile()

    full_meta = full.tables[0].metadata["profiling"]
    capped_meta = capped.tables[0].metadata["profiling"]

    assert full_meta["source_row_count"] == 50
    assert capped_meta["source_row_count"] == 10
    assert capped_meta["controls"]["max_rows"] == 10


def test_from_file_with_columns_subsets_a_csv_profile(tmp_path):
    p = tmp_path / "wide.csv"
    rows = ["a,b,c"] + [f"{i},{i*2},{i*3}" for i in range(10)]
    p.write_text("\n".join(rows))

    spec = DatasetProfiler.from_file(
        str(p), name="t", columns=["a", "c"]
    ).profile()
    col_names = {c.name for c in spec.tables[0].columns}
    assert col_names == {"a", "c"}


def test_max_rows_zero_is_rejected():
    with pytest.raises(ValueError, match="max_rows"):
        DatasetProfiler(_two_column_frame(), max_rows=0)


def test_max_rows_must_be_an_integer():
    with pytest.raises(ValueError, match="max_rows"):
        DatasetProfiler(_two_column_frame(), max_rows="lots")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 3. Config snapshot + syntab version + source fingerprint
# ---------------------------------------------------------------------------

def test_profiled_spec_records_syntab_version():
    spec = DatasetProfiler(_two_column_frame(), name="t", sample=None).profile()
    assert spec.metadata.syntab_version == SYNTAB_VERSION


def test_profiled_spec_records_non_default_controls_only():
    """Default-options profile: the controls block is absent (or empty).
    A non-default run: only the changed controls appear, never the defaults.
    This is what keeps the spec readable."""
    default_spec = DatasetProfiler(
        _two_column_frame(), name="t", sample=None
    ).profile()
    custom_spec = DatasetProfiler(
        _two_column_frame(), name="t", sample=None,
        min_cell_count=10, redact_categoricals=True,
    ).profile()

    assert default_spec.tables[0].metadata["profiling"].get("controls", {}) == {}

    ctrls = custom_spec.tables[0].metadata["profiling"]["controls"]
    assert ctrls.get("min_cell_count") == 10
    assert ctrls.get("redact_categoricals") is True
    # The defaults that the user did not change are NOT in the dict.
    assert "max_categorical" not in ctrls
    assert "pii_strategy" not in ctrls
    assert "discover" not in ctrls


def test_profiled_spec_records_source_fingerprint(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text("a,b\n1,2\n3,4\n5,6\n")

    spec = DatasetProfiler.from_file(str(p), name="t").profile()
    src = spec.tables[0].metadata["profiling"]["source"]
    assert src["source"] == str(p)
    assert src["profiled_rows"] == 3
    assert src["source_size_bytes"] == p.stat().st_size
    assert src["source_mtime"] == int(p.stat().st_mtime)


def test_in_memory_profile_records_only_profiled_rows():
    """No file path -> no file-stat fields, but profiled_rows is still there.
    The point is that the source field is always present so a reviewer
    knows what was profiled even when there is no file to stat."""
    spec = DatasetProfiler(
        _two_column_frame(), name="t", source="in-memory", sample=None
    ).profile()
    src = spec.tables[0].metadata["profiling"]["source"]
    assert src["profiled_rows"] == 8
    assert "source_size_bytes" not in src
    assert "source_mtime" not in src


# ---------------------------------------------------------------------------
# 4. Column-level merge drift detection
# ---------------------------------------------------------------------------

def _two_table_profiled_set():
    users = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
    orders = pd.DataFrame({
        "id": [10, 11, 12],
        "user_id": [1, 2, 1],
        "amount": [1.0, 2.0, 3.0],
    })
    return DatasetProfiler.profile_set(
        {"users": users, "orders": orders},
        sample=None, discover=False,
    )


def test_merge_report_appears_on_metadata_after_merge():
    a = _two_table_profiled_set()
    b = _two_table_profiled_set()
    merged = merge_preserving_edits(a, b)
    assert merged.metadata.merge is not None
    assert "tables_kept" in merged.metadata.merge


def test_merge_no_drift_marks_all_columns_preserved():
    a = _two_table_profiled_set()
    b = _two_table_profiled_set()
    mr = MergeReport()
    merge_preserving_edits(a, b, report=mr)
    # Every column that was in both is in the preserved list.
    assert all(
        any(c.name in s for s in mr.columns_preserved)
        for t in b.tables
        for c in t.columns
    )


def test_merge_added_column_survives_from_fresh_profile():
    a = _two_table_profiled_set()
    # b has an extra column on orders.
    b_users = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
    b_orders = pd.DataFrame({
        "id": [10, 11, 12],
        "user_id": [1, 2, 1],
        "amount": [1.0, 2.0, 3.0],
        "note": ["x", "y", "z"],
    })
    b = DatasetProfiler.profile_set(
        {"users": b_users, "orders": b_orders},
        sample=None, discover=False,
    )
    mr = MergeReport()
    merged = merge_preserving_edits(a, b, report=mr)

    # The new column is in the merged orders table.
    orders = next(t for t in merged.tables if t.name == "orders")
    col_names = {c.name for c in orders.columns}
    assert "note" in col_names
    # And the report recorded it as added.
    added_cols = {c for _, c in mr.columns_added}
    assert "note" in added_cols


def test_merge_removed_column_is_kept_not_silently_dropped():
    """A column the old spec had but the new profile did not see is kept,
    so a column a person wrote by hand does not get deleted by a
    re-profile that no longer sees the source."""
    a_users = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
    a_orders = pd.DataFrame({
        "id": [10, 11, 12],
        "user_id": [1, 2, 1],
        "amount": [1.0, 2.0, 3.0],
        "note": ["x", "y", "z"],
    })
    a = DatasetProfiler.profile_set(
        {"users": a_users, "orders": a_orders},
        sample=None, discover=False,
    )
    # b drops the `note` column from orders.
    b_users = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
    b_orders = pd.DataFrame({
        "id": [10, 11, 12],
        "user_id": [1, 2, 1],
        "amount": [1.0, 2.0, 3.0],
    })
    b = DatasetProfiler.profile_set(
        {"users": b_users, "orders": b_orders},
        sample=None, discover=False,
    )
    mr = MergeReport()
    merged = merge_preserving_edits(a, b, report=mr)

    orders = next(t for t in merged.tables if t.name == "orders")
    col_names = {c.name for c in orders.columns}
    assert "note" in col_names
    removed = {c for _, c in mr.columns_removed}
    assert "note" in removed


def test_merge_changed_column_records_drift():
    a = _two_table_profiled_set()
    # Re-profile with a different amount dtype (str -> int).
    b_users = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
    b_orders = pd.DataFrame({
        "id": [10, 11, 12],
        "user_id": [1, 2, 1],
        "amount": ["1.0", "2.0", "3.0"],
    })
    b = DatasetProfiler.profile_set(
        {"users": b_users, "orders": b_orders},
        sample=None, discover=False,
    )
    mr = MergeReport()
    merge_preserving_edits(a, b, report=mr)
    drifted = {c for _, c in mr.columns_changed_reinferred}
    assert "amount" in drifted


def test_merge_added_table_records_in_report():
    """A table the new profile added is in the report's tables_added list."""
    a = _two_table_profiled_set()
    # b adds a third table (line_items) on top of the two from a.
    users_df = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
    orders_df = pd.DataFrame({
        "id": [10, 11, 12], "user_id": [1, 2, 1], "amount": [1.0, 2.0, 3.0],
    })
    line_items = pd.DataFrame({
        "id": [100, 101], "order_id": [10, 11], "sku": ["A", "B"],
    })
    b = DatasetProfiler.profile_set(
        {"users": users_df, "orders": orders_df, "line_items": line_items},
        sample=None, discover=False,
    )
    mr = MergeReport()
    merge_preserving_edits(a, b, report=mr)
    assert "line_items" in mr.tables_added


def test_merge_repeated_runs_converge_to_columns_preserved():
    """A no-drift merge then a no-drift merge converges: by the second
    pass every column that did not change is in columns_preserved."""
    a = _two_table_profiled_set()
    first = merge_preserving_edits(a, _two_table_profiled_set())
    mr = MergeReport()
    merge_preserving_edits(first, _two_table_profiled_set(), report=mr)
    assert all(
        any(c.name in s for s in mr.columns_preserved)
        for t in first.tables
        for c in t.columns
    )


def test_profiled_column_carries_a_fingerprint():
    """Every column the profiler writes gets a provenance fingerprint, the
    same scheme key_provenance/relationship provenance already use -- this
    is what makes column-level drift detection on re-profile possible."""
    spec = DatasetProfiler(_two_column_frame(), name="t", sample=None).profile()
    for col in spec.tables[0].columns:
        assert col.provenance is not None
        assert col.provenance.algorithm == "profile-column"
        assert col.provenance.fingerprint


def test_merge_preserves_hand_edited_column():
    """A column edited by hand after profiling survives a re-profile: the
    edit is kept (not silently overwritten by the fresh inference), and the
    merge report records it as columns_changed_kept."""
    df = _two_column_frame()
    a = DatasetProfiler(df, name="t", sample=None, seed=7).profile()

    status_col = next(c for c in a.tables[0].columns if c.name == "status")
    original_fingerprint = status_col.provenance.fingerprint
    # Hand-edit: flip a constraint, as a person correcting the profiler.
    status_col.constraints["nullable"] = not status_col.constraints.get(
        "nullable", False
    )

    b = DatasetProfiler(df, name="t", sample=None, seed=7).profile()
    mr = MergeReport()
    merged = merge_preserving_edits(a, b, report=mr)

    merged_status = next(c for c in merged.tables[0].columns if c.name == "status")
    assert merged_status.constraints["nullable"] == status_col.constraints["nullable"]
    assert merged_status.provenance.human_edited is True
    # The fingerprint is stamped stale, not silently refreshed.
    assert merged_status.provenance.fingerprint == original_fingerprint
    assert ("t", "status") in mr.columns_changed_kept
    assert ("t", "status") not in mr.columns_changed_reinferred


def test_merge_hand_authored_column_with_no_provenance_is_kept():
    """A column with no provenance at all (hand-authored, or written before
    this fingerprinting existed) is treated as human-authored and kept on
    drift, same as the no-provenance rule for the primary key."""
    from syntab.spec import ColumnSpec

    df = _two_column_frame()
    a = DatasetProfiler(df, name="t", sample=None, seed=7).profile()
    status_col = next(c for c in a.tables[0].columns if c.name == "status")
    status_col.provenance = None
    status_col.constraints["nullable"] = not status_col.constraints.get(
        "nullable", False
    )

    b = DatasetProfiler(df, name="t", sample=None, seed=7).profile()
    mr = MergeReport()
    merged = merge_preserving_edits(a, b, report=mr)
    merged_status = next(c for c in merged.tables[0].columns if c.name == "status")
    assert merged_status.constraints["nullable"] == status_col.constraints["nullable"]
    assert ("t", "status") in mr.columns_changed_kept


# ---------------------------------------------------------------------------
# Follow-up fixes found in code review of the initial INF-225 PR
# ---------------------------------------------------------------------------

def test_profile_set_records_real_source_per_table(tmp_path):
    """Multi-table profiling records each table's own file identity, not
    'unknown' -- the ``sources`` mapping threads the real path through."""
    users_p = tmp_path / "users.csv"
    orders_p = tmp_path / "orders.csv"
    users_p.write_text("id,name\n1,a\n2,b\n")
    orders_p.write_text("id,user_id,amount\n10,1,1.0\n11,2,2.0\n")

    dfs = {}
    sources = {}
    for tname, p in (("users", users_p), ("orders", orders_p)):
        prof = DatasetProfiler.from_file(str(p), name=tname)
        dfs[prof.name] = prof.full
        sources[prof.name] = prof.source

    spec = DatasetProfiler.profile_set(
        dfs, sample=None, discover=False, sources=sources
    )
    for t in spec.tables:
        src = t.metadata["profiling"]["source"]
        assert src["source"] == sources[t.name]
        assert src["source_path"] == sources[t.name]
        assert "source_size_bytes" in src
        assert "source_mtime" in src


def test_profile_set_without_sources_falls_back_to_unknown():
    """Callers with only in-memory frames (no ``sources`` mapping) keep the
    previous behavior instead of erroring."""
    spec = DatasetProfiler.profile_set(
        {"users": pd.DataFrame({"id": [1, 2], "name": ["a", "b"]})},
        sample=None, discover=False,
    )
    assert spec.tables[0].metadata["profiling"]["source"]["source"] == "unknown"


def test_read_dataframe_bounded_handles_jsonl_without_bounds(tmp_path):
    """JSONL profiles the same way whether or not --max-rows/--columns is
    set -- the default (whole-file) dispatch used to raise on .jsonl."""
    p = tmp_path / "wide.jsonl"
    p.write_text("\n".join(json.dumps({"i": i, "v": i * 2}) for i in range(5)))
    df = _read_dataframe_bounded(str(p))
    assert len(df) == 5
    assert list(df.columns) == ["i", "v"]


def test_read_dataframe_bounded_parses_tsv_with_tab_separator(tmp_path):
    p = tmp_path / "data.tsv"
    p.write_text("id\tval\n1\t10\n2\t20\n")
    df = _read_dataframe_bounded(str(p))
    assert list(df.columns) == ["id", "val"]
    assert len(df) == 2


def test_read_dataframe_bounded_parses_tsv_when_capped(tmp_path):
    p = tmp_path / "data.tsv"
    p.write_text("id\tval\n" + "\n".join(f"{i}\t{i * 2}" for i in range(10)))
    df = _read_dataframe_bounded(str(p), max_rows=3)
    assert list(df.columns) == ["id", "val"]
    assert len(df) == 3


def test_read_dataframe_bounded_rejects_unknown_columns_streamed(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text("id,val\n1,10\n2,20\n")
    with pytest.raises(ValueError, match="val1"):
        _read_dataframe_bounded(str(p), columns=["id", "val1"])


def test_read_dataframe_bounded_rejects_unknown_columns_whole_load(tmp_path):
    p = tmp_path / "data.json"
    p.write_text(json.dumps([{"id": 1, "val": 10}, {"id": 2, "val": 20}]))
    with pytest.raises(ValueError, match="val1"):
        _read_dataframe_bounded(str(p), columns=["id", "val1"])


def test_controls_never_duplicates_sample_or_seed():
    """sample/seed are always recorded via requested_sample/seed and must
    never also appear in `controls` -- including at CLI-realistic non-None
    values (sample=5000, seed=42), which is exactly the case the original
    'controls omitted at defaults' contract failed to cover."""
    spec = DatasetProfiler(
        _two_column_frame(), name="t", sample=5000, seed=42,
    ).profile()
    ctrls = spec.tables[0].metadata["profiling"].get("controls", {})
    assert "sample" not in ctrls
    assert "seed" not in ctrls
    profiling_meta = spec.tables[0].metadata["profiling"]
    assert profiling_meta["requested_sample"] == 5000
    assert profiling_meta["seed"] == 42
