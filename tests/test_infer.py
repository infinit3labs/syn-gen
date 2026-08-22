from syntab.infer import resolve_generator
from syntab.spec import ColumnSpec, ColumnProfile, NumericProfile, CategoricalProfile


def test_infer_categorical():
    col = ColumnSpec(
        name="c", dtype="str", generator="auto",
        profile=ColumnProfile(categorical=CategoricalProfile(values={"a": 0.7, "b": 0.3})),
    )
    gen, params = resolve_generator(col)
    assert gen == "choice"
    assert params["values"] == ["a", "b"]
    assert params["weights"] == [0.7, 0.3]


def test_infer_numeric_normal():
    col = ColumnSpec(
        name="n", dtype="float", generator="auto",
        profile=ColumnProfile(numeric=NumericProfile(min=0, max=100, mean=50, std=5, distribution="normal")),
    )
    gen, params = resolve_generator(col)
    assert gen == "float"
    assert params["min"] >= 0
    assert params["max"] <= 100


def test_infer_datetime():
    from datetime import datetime
    col = ColumnSpec(
        name="d", dtype="datetime", generator="auto",
        profile=ColumnProfile(datetime_range=(datetime(2020, 1, 1), datetime(2021, 1, 1))),
    )
    gen, params = resolve_generator(col)
    assert gen == "datetime"
    assert "start" in params


def test_explicit_generator_passthrough():
    col = ColumnSpec(name="x", dtype="int", generator="int", params={"min": 1, "max": 2})
    gen, params = resolve_generator(col)
    assert gen == "int"
    assert params == {"min": 1, "max": 2}
