import pandas as pd

from syntab.profiler import DatasetProfiler


def _df():
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "age": [20, 30, 40, 50, 60],
            "status": ["active", "inactive", "active", "active", "inactive"],
            "created": ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01", "2020-05-01"],
            "note": [None, None, None, None, None],  # fully-null -> skipped
        }
    )


def test_profiler_basic():
    # min_cell_count=0 is pinned explicitly, not inherited.
    #
    # This test is about dtype inference, generator selection and key
    # detection. It is not about disclosure control. Its 5-row frame has
    # status = active x3, inactive x2, so with the default K=5 both cells fall
    # below the threshold and the column collapses to a single '__other__'
    # bucket -- which would make the categorical assertions below fail for a
    # reason that has nothing to do with what they are checking.
    #
    # Pinning K=0 here keeps the test measuring what it was written to measure
    # and stops it silently re-breaking whenever the disclosure default moves.
    # The default itself is covered in tests/test_min_cell_count_default.py.
    prof = DatasetProfiler(_df(), name="t", sample=None, min_cell_count=0)
    spec = prof.profile()
    tbl = spec.tables[0]
    by_name = {c.name: c for c in tbl.columns}
    assert tbl.row_count == 5
    # fully-null 'note' skipped
    assert "note" not in by_name
    # id -> unique int -> sequence + primary key
    assert by_name["id"].dtype == "int"
    assert by_name["id"].generator == "sequence"
    assert tbl.primary_key == "id"
    # age -> numeric
    assert by_name["age"].dtype == "int"
    assert by_name["age"].profile.numeric.min == 20
    assert by_name["age"].profile.numeric.max == 60
    # status -> categorical
    assert by_name["status"].dtype == "str"
    vals = by_name["status"].profile.categorical.values
    assert set(vals) == {"active", "inactive"}
    assert abs(sum(vals.values()) - 1.0) < 1e-6
    # created -> datetime
    assert by_name["created"].dtype == "datetime"
    assert by_name["created"].profile.datetime_range is not None


def test_profiler_from_csv(tmp_path):
    p = tmp_path / "d.csv"
    _df().to_csv(p, index=False)
    spec = DatasetProfiler.from_file(str(p), name="csv_t", sample=None).profile()
    assert spec.metadata.source.startswith("profiled:")
    assert len(spec.tables[0].columns) == 4  # note dropped


def _users():
    return pd.DataFrame({"id": [1, 2, 3, 4, 5], "name": ["a", "b", "c", "d", "e"]})


def _orders():
    return pd.DataFrame(
        {"id": [10, 11, 12], "user_id": [1, 2, 3], "total": [100, 200, 300]}
    )


def test_profiler_set_infers_relationships():
    spec = DatasetProfiler.profile_set(
        {"users": _users(), "orders": _orders()}, sample=None
    )
    tables = {t.name: t for t in spec.tables}
    assert set(tables) == {"users", "orders"}
    # orders.user_id -> users.id inferred
    rels = tables["orders"].relationships
    assert len(rels) == 1
    rel = rels[0]
    assert rel.from_ == "user_id"
    assert rel.to == "users.id"
    assert rel.alias == "users"


def test_profiler_set_no_false_positive():
    # a numeric column that is NOT a foreign key should not be linked
    spec = DatasetProfiler.profile_set(
        {"a": pd.DataFrame({"id": [1, 2, 3]}),
         "b": pd.DataFrame({"x": [1, 2, 3]})},
        sample=None,
    )
    assert spec.tables[1].relationships == []

