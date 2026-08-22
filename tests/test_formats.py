import json
from pathlib import Path

import pandas as pd

from syntab import GenerationEngine
from syntab import formats
from syntab.spec import (
    Spec, SpecMetadata, TableSpec, ColumnSpec,
)


def _small_spec():
    return Spec(
        metadata=SpecMetadata(name="t"),
        settings=__import__("syntab.spec", fromlist=["Settings"]).Settings(seed=1),
        tables=[
            TableSpec(
                name="people", row_count=10, primary_key="id",
                columns=[
                    ColumnSpec(name="id", dtype="int", generator="sequence", constraints={"unique": True}),
                    ColumnSpec(name="name", dtype="str", generator="faker.name"),
                    ColumnSpec(name="score", dtype="float", generator="float", params={"min": 0, "max": 1, "decimals": 2}),
                ],
            )
        ],
    )


def _data():
    res = GenerationEngine(_small_spec()).run()
    return res.tables


def test_csv_roundtrip(tmp_path):
    tables = _data()
    out = tmp_path / "out"
    formats.write(tables, str(out), "csv")
    df = pd.read_csv(out / "people.csv")
    assert len(df) == 10
    assert list(df.columns) == ["id", "name", "score"]


def test_json_roundtrip(tmp_path):
    tables = _data()
    out = tmp_path / "out.json"
    formats.write(tables, str(out), "json")
    data = json.loads(out.read_text())
    assert len(data["people"]) == 10


def test_jsonl_roundtrip(tmp_path):
    tables = _data()
    out = tmp_path / "out.jsonl"
    formats.write(tables, str(out), "jsonl")
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 10


def test_parquet_roundtrip(tmp_path):
    tables = _data()
    out = tmp_path / "out"
    formats.write(tables, str(out), "parquet")
    df = pd.read_parquet(out / "people.parquet")
    assert len(df) == 10


def test_sql_output(tmp_path):
    tables = _data()
    out = tmp_path / "out.sql"
    formats.write(tables, str(out), "sql")
    text = out.read_text()
    assert "CREATE TABLE" in text
    assert "INSERT INTO" in text


def _relational_spec():
    Settings = __import__("syntab.spec", fromlist=["Settings"]).Settings
    return Spec(
        metadata=SpecMetadata(name="shop"),
        settings=Settings(seed=1),
        tables=[
            TableSpec(
                name="customers", row_count=5, primary_key="id",
                columns=[
                    ColumnSpec(name="id", dtype="int", generator="sequence",
                               constraints={"unique": True, "nullable": False}),
                    ColumnSpec(name="email", dtype="str", generator="faker.email",
                               constraints={"unique": True, "nullable": False}),
                    ColumnSpec(name="balance", dtype="float", generator="float",
                               params={"min": 0, "max": 100, "decimals": 2}),
                    ColumnSpec(name="active", dtype="bool", generator="bool"),
                ],
            ),
            TableSpec(
                name="orders", row_count=8, primary_key="oid",
                columns=[
                    ColumnSpec(name="oid", dtype="int", generator="sequence",
                               constraints={"unique": True}),
                    ColumnSpec(name="customer_id", dtype="int", generator="fk",
                               params={"ref": "customers.id"},
                               constraints={"nullable": False}),
                    ColumnSpec(name="total", dtype="float", generator="float",
                               params={"min": 1, "max": 50, "decimals": 2}),
                ],
                relationships=[__import__("syntab.spec", fromlist=["RelationshipSpec"])
                               .RelationshipSpec(**{"from": "customer_id",
                                                    "to": "customers.id",
                                                    "alias": "customer"})],
            ),
        ],
    )


def test_sql_typed_ddl_with_constraints_and_fk(tmp_path):
    spec = _relational_spec()
    tables = GenerationEngine(spec).run().tables
    out = tmp_path / "shop.sql"
    formats.write(tables, str(out), "sql", spec=spec)
    text = out.read_text()

    # parents emitted before children
    assert text.index('CREATE TABLE IF NOT EXISTS "customers"') < \
        text.index('CREATE TABLE IF NOT EXISTS "orders"')

    # typed columns
    assert '"id" INTEGER' in text
    assert '"email" TEXT' in text
    assert '"balance" REAL' in text
    assert '"active" BOOLEAN' in text
    assert '"total" REAL' in text

    # constraints
    assert "PRIMARY KEY" in text
    assert "UNIQUE" in text
    assert "NOT NULL" in text

    # foreign key
    assert 'FOREIGN KEY ("customer_id") REFERENCES "customers" ("id")' in text

    # data still written
    assert text.count("INSERT INTO") == 5 + 8


def test_sql_fallback_without_spec_is_text(tmp_path):
    # Without a spec, columns fall back to generic TEXT (backward compatible).
    tables = _data()
    out = tmp_path / "out.sql"
    formats.write(tables, str(out), "sql")  # no spec
    text = out.read_text()
    assert '"id" TEXT' in text
    assert "PRIMARY KEY" not in text
