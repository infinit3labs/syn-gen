"""Tests for composite keys and relationship cardinality controls."""

import pandas as pd
import pytest

from syntab import GenerationEngine
from syntab.conformance import validate_against_spec
from syntab.formats import write
from syntab.spec import (
    ColumnSpec,
    RelationshipSpec,
    Settings,
    Spec,
    SpecMetadata,
    TableSpec,
)


def _composite_spec():
    return Spec(
        metadata=SpecMetadata(name="composite"),
        settings=Settings(seed=31, fallback="raise", max_rule_attempts=50),
        tables=[
            TableSpec(
                name="accounts",
                row_count=6,
                primary_key=["tenant_id", "account_id"],
                columns=[
                    ColumnSpec(name="tenant_id", dtype="int", generator="choice",
                               params={"values": [10, 20, 30]}),
                    ColumnSpec(name="account_id", dtype="int", generator="sequence"),
                    ColumnSpec(name="tier", dtype="str", generator="choice",
                               params={"values": ["basic", "gold"]}),
                ],
            ),
            TableSpec(
                name="transactions",
                row_count=18,
                primary_key="id",
                columns=[
                    ColumnSpec(name="id", dtype="int", generator="sequence"),
                    ColumnSpec(name="tenant_id", dtype="int", generator="fk"),
                    ColumnSpec(name="account_id", dtype="int", generator="fk"),
                    ColumnSpec(name="amount", dtype="float", generator="float",
                               params={"min": 1, "max": 100}),
                ],
                relationships=[RelationshipSpec(
                    **{
                        "from": ["tenant_id", "account_id"],
                        "to": "accounts",
                        "to_columns": ["tenant_id", "account_id"],
                        "alias": "account",
                        "allocation": "balanced",
                    }
                )],
                unique_constraints=[["tenant_id", "account_id", "id"]],
            ),
        ],
    )


def test_composite_pk_fk_and_unique_constraint():
    spec = _composite_spec()
    frames = GenerationEngine(spec).run().to_frames()
    report = validate_against_spec(frames, spec)
    assert report.overall_ok, report.to_text()

    accounts = frames["accounts"]
    transactions = frames["transactions"]
    assert not accounts.duplicated(["tenant_id", "account_id"]).any()
    assert not transactions.duplicated(["tenant_id", "account_id", "id"]).any()

    pairs = set(zip(accounts.tenant_id, accounts.account_id))
    assert set(zip(transactions.tenant_id, transactions.account_id)).issubset(pairs)


def test_composite_sql_constraints(tmp_path):
    spec = _composite_spec()
    out = tmp_path / "composite.sql"
    write(GenerationEngine(spec).run().tables, str(out), "sql", spec=spec)
    text = out.read_text()
    assert 'PRIMARY KEY ("tenant_id", "account_id")' in text
    assert 'UNIQUE ("tenant_id", "account_id", "id")' in text
    assert ('FOREIGN KEY ("tenant_id", "account_id") '
            'REFERENCES "accounts" ("tenant_id", "account_id")') in text


def _cardinality_spec(allocation="balanced", cardinality="many_to_one"):
    return Spec(
        metadata=SpecMetadata(name="cardinality"),
        settings=Settings(seed=9, fallback="raise", max_rule_attempts=20),
        tables=[
            TableSpec(
                name="parents", row_count=5, primary_key="id",
                columns=[
                    ColumnSpec(name="id", dtype="int", generator="sequence"),
                    ColumnSpec(name="weight", dtype="float", generator="const",
                               params={"value": 1}),
                ],
            ),
            TableSpec(
                name="children", row_count=10, primary_key="id",
                columns=[
                    ColumnSpec(name="id", dtype="int", generator="sequence"),
                    ColumnSpec(name="parent_id", dtype="int", generator="fk"),
                ],
                relationships=[RelationshipSpec(
                    **{
                        "from": "parent_id",
                        "to": "parents.id",
                        "alias": "parent",
                        "allocation": allocation,
                        "cardinality": cardinality,
                        "weight_column": "weight" if allocation == "weighted" else None,
                    }
                )],
            ),
        ],
    )


def test_balanced_allocation_distributes_children_evenly():
    frames = GenerationEngine(_cardinality_spec()).run().to_frames()
    counts = frames["children"]["parent_id"].value_counts()
    assert set(counts.index) == {1, 2, 3, 4, 5}
    assert counts.max() - counts.min() <= 1


def test_one_to_one_allocation_never_reuses_parent():
    spec = _cardinality_spec(allocation="uniform", cardinality="one_to_one")
    spec.tables[1].row_count = 5
    frames = GenerationEngine(spec).run().to_frames()
    assert frames["children"]["parent_id"].is_unique


def test_min_max_children_are_enforced():
    spec = _cardinality_spec(allocation="balanced")
    spec.tables[1].relationships[0].min_children = 2
    spec.tables[1].relationships[0].max_children = 2
    spec.tables[1].row_count = 10
    frames = GenerationEngine(spec).run().to_frames()
    counts = frames["children"]["parent_id"].value_counts()
    assert set(counts.tolist()) == {2}


def test_weighted_allocation_uses_parent_weight_column():
    spec = _cardinality_spec(allocation="weighted")
    parents = spec.tables[0]
    parents.columns[1].generator = "sequence"
    spec.tables[1].row_count = 1000
    frames = GenerationEngine(spec).run().to_frames()
    counts = frames["children"]["parent_id"].value_counts()
    assert counts.loc[5] > counts.loc[1]


def test_impossible_one_to_one_fails_early():
    spec = _cardinality_spec(allocation="uniform", cardinality="one_to_one")
    spec.tables[1].row_count = 6
    with pytest.raises(Exception, match="one_to_one"):
        GenerationEngine(spec).run()
