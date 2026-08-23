"""Tests for streaming/large-scale generation (#6)."""
from pathlib import Path

import pandas as pd

from syntab import formats
from syntab.conformance import validate_against_spec
from syntab.engine import GenerationEngine
from syntab.spec import (
    ColumnSpec, RelationshipSpec, Settings, Spec, SpecMetadata, TableSpec,
)


def _spec():
    return Spec(
        metadata=SpecMetadata(name="shop"),
        settings=Settings(seed=7, fallback="raise", max_rule_attempts=200),
        tables=[
            TableSpec(
                name="customers", row_count=20, primary_key="id",
                columns=[
                    ColumnSpec(name="id", dtype="int", generator="sequence",
                               constraints={"unique": True}),
                    ColumnSpec(name="region", dtype="str", generator="choice",
                               params={"values": ["EU", "US"], "weights": [0.5, 0.5]}),
                    ColumnSpec(name="credit_limit", dtype="float", generator="float",
                               params={"min": 500, "max": 20000, "decimals": 2}),
                ],
            ),
            TableSpec(
                name="orders", row_count=60, primary_key="id",
                columns=[
                    ColumnSpec(name="id", dtype="int", generator="sequence",
                               constraints={"unique": True}),
                    ColumnSpec(name="customer_id", dtype="int", generator="fk",
                               params={"ref": "customers.id"}),
                    ColumnSpec(name="region", dtype="str", generator="choice",
                               params={"values": ["EU", "US"], "weights": [0.5, 0.5]}),
                    ColumnSpec(name="total", dtype="float", generator="float",
                               params={"min": 10, "max": 1000, "decimals": 2}),
                ],
                relationships=[RelationshipSpec(
                    **{"from": "customer_id", "to": "customers.id", "alias": "customer"}
                )],
                rules=["region == customer.region", "total <= customer.credit_limit"],
            ),
            TableSpec(
                name="order_items", row_count=120, primary_key="id",
                columns=[
                    ColumnSpec(name="id", dtype="int", generator="sequence",
                               constraints={"unique": True}),
                    ColumnSpec(name="order_id", dtype="int", generator="fk",
                               params={"ref": "orders.id"}),
                    ColumnSpec(name="quantity", dtype="int", generator="int",
                               params={"min": 1, "max": 10}),
                ],
                relationships=[RelationshipSpec(
                    **{"from": "order_id", "to": "orders.id", "alias": "order"}
                )],
                rules=["quantity > 0"],
            ),
        ],
    )


class ListSink(formats.RowSink):
    """Records the order in which tables are written."""

    def __init__(self):
        self.order: list = []

    def write_table(self, name, rows):
        self.order.append((name, len(rows)))


class RowsSink(formats.RowSink):
    def __init__(self):
        self.calls = []

    def write_table(self, name, rows):
        self.calls.append((name, list(rows)))


def test_stream_writes_parents_before_children_and_no_inmemory_accumulation():
    sink = ListSink()
    result = GenerationEngine(_spec()).run(stream_sink=sink)
    # tables written once each, parents first
    assert [n for n, _ in sink.order] == ["customers", "orders", "order_items"]
    assert dict(sink.order) == {"customers": 20, "orders": 60, "order_items": 120}
    # streaming mode must not accumulate rows in the returned result
    assert result.tables == {}


def test_stream_csv_output_matches_in_memory(tmp_path):
    spec = _spec()

    mem_dir = tmp_path / "mem"
    GenerationEngine(spec).run().write(str(mem_dir), "csv")

    stream_dir = tmp_path / "stream"
    sink = formats.CSVSink(str(stream_dir))
    GenerationEngine(spec).run(stream_sink=sink)
    sink.close()

    for table in ("customers", "orders", "order_items"):
        a = pd.read_csv(mem_dir / f"{table}.csv")
        b = pd.read_csv(stream_dir / f"{table}.csv")
        pd.testing.assert_frame_equal(a, b)


def test_stream_jsonl_output(tmp_path):
    spec = _spec()
    stream_dir = tmp_path / "jsonl"
    sink = formats.JSONLSink(str(stream_dir))
    GenerationEngine(spec).run(stream_sink=sink)
    sink.close()

    df = pd.read_json(stream_dir / "orders.jsonl", lines=True)
    assert len(df) == 60
    assert set(df.columns) == {"id", "customer_id", "region", "total"}


def test_streamed_output_conforms_to_spec(tmp_path):
    spec = _spec()
    stream_dir = tmp_path / "out"
    sink = formats.CSVSink(str(stream_dir))
    GenerationEngine(spec).run(stream_sink=sink)
    sink.close()

    frames = {
        t: pd.read_csv(stream_dir / f"{t}.csv")
        for t in ("customers", "orders", "order_items")
    }
    report = validate_against_spec(frames, spec)
    assert report.overall_ok, report.to_text()


def test_stream_chunking_preserves_relational_rows_and_seeded_output():
    spec = _spec()
    unchunked = RowsSink()
    chunked = RowsSink()

    GenerationEngine(spec).run(stream_sink=unchunked)
    GenerationEngine(spec).run(stream_sink=chunked, chunk_size=7)

    assert [len(rows) for name, rows in chunked.calls if name == "customers"] == [7, 7, 6]
    assert [len(rows) for name, rows in chunked.calls if name == "orders"] == [7] * 8 + [4]
    assert [len(rows) for name, rows in chunked.calls if name == "order_items"] == [7] * 17 + [1]
    assert [row for _, rows in unchunked.calls for row in rows] == [
        row for _, rows in chunked.calls for row in rows
    ]
    parents = {row["id"] for name, rows in unchunked.calls if name == "customers" for row in rows}
    assert all(
        row["customer_id"] in parents
        for name, rows in chunked.calls if name == "orders"
        for row in rows
    )


def test_stream_progress_reports_each_chunk():
    events = []
    sink = RowsSink()

    GenerationEngine(_spec()).run(
        stream_sink=sink,
        chunk_size=25,
        progress=lambda table, emitted, total: events.append((table, emitted, total)),
    )

    assert events == [
        ("customers", 20, 20),
        ("orders", 25, 60), ("orders", 50, 60), ("orders", 60, 60),
        ("order_items", 25, 120), ("order_items", 50, 120),
        ("order_items", 75, 120), ("order_items", 100, 120),
        ("order_items", 120, 120),
    ]


def test_stream_failure_preserves_completed_chunks():
    spec = _spec()
    spec.tables[0].row_count = 5
    spec.tables[0].rules = ["id <= 1"]
    sink = RowsSink()

    import pytest
    with pytest.raises(Exception, match="Could not satisfy rules"):
        GenerationEngine(spec).run(stream_sink=sink, chunk_size=1)

    assert [(name, len(rows)) for name, rows in sink.calls] == [("customers", 1)]
