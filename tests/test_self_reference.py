"""Tests for self-referencing (hierarchical) relationships."""

import pytest

from syntab import GenerationEngine, SpecError
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


def _employees_spec(root_fraction=0.0):
    return Spec(
        metadata=SpecMetadata(name="hierarchy"),
        settings=Settings(seed=7, fallback="raise", max_rule_attempts=20),
        tables=[
            TableSpec(
                name="employees",
                row_count=20,
                primary_key="id",
                columns=[
                    ColumnSpec(name="id", dtype="int", generator="sequence"),
                    ColumnSpec(name="manager_id", dtype="int", generator="fk",
                               constraints={"nullable": True}),
                    ColumnSpec(name="name", dtype="str", generator="faker.name"),
                ],
                relationships=[RelationshipSpec(
                    **{
                        "from": "manager_id",
                        "to": "employees.id",
                        "alias": "manager",
                        "root_fraction": root_fraction,
                    }
                )],
            ),
        ],
    )


def test_self_reference_is_referentially_valid():
    spec = _employees_spec()
    frames = GenerationEngine(spec).run().to_frames()
    report = validate_against_spec(frames, spec)
    assert report.overall_ok, report.to_text()

    emp = frames["employees"]
    ids = set(emp["id"])
    managers = set(emp["manager_id"].dropna())
    assert managers.issubset(ids)
    # the first row is always a root
    assert emp["manager_id"].isna().any()


def test_root_fraction_increases_roots():
    few = GenerationEngine(_employees_spec(root_fraction=0.0)).run().to_frames()
    many = GenerationEngine(_employees_spec(root_fraction=0.6)).run().to_frames()
    roots_few = int(few["employees"]["manager_id"].isna().sum())
    roots_many = int(many["employees"]["manager_id"].isna().sum())
    assert roots_many > roots_few


def test_non_nullable_self_ref_fk_rejected():
    spec = _employees_spec()
    spec.tables[0].columns[1].constraints = {"nullable": False}
    with pytest.raises(SpecError, match="nullable"):
        GenerationEngine(spec).run()


def test_composite_self_reference():
    spec = Spec(
        metadata=SpecMetadata(name="graph"),
        settings=Settings(seed=3, fallback="raise", max_rule_attempts=20),
        tables=[
            TableSpec(
                name="nodes",
                row_count=15,
                primary_key=["graph_id", "node_id"],
                columns=[
                    ColumnSpec(name="graph_id", dtype="int", generator="choice",
                               params={"values": [1]}),
                    ColumnSpec(name="node_id", dtype="int", generator="sequence"),
                    ColumnSpec(name="parent_graph_id", dtype="int", generator="fk",
                               constraints={"nullable": True}),
                    ColumnSpec(name="parent_node_id", dtype="int", generator="fk",
                               constraints={"nullable": True}),
                ],
                relationships=[RelationshipSpec(
                    **{
                        "from": ["parent_graph_id", "parent_node_id"],
                        "to": "nodes",
                        "to_columns": ["graph_id", "node_id"],
                        "alias": "parent",
                    }
                )],
            ),
        ],
    )
    frames = GenerationEngine(spec).run().to_frames()
    report = validate_against_spec(frames, spec)
    assert report.overall_ok, report.to_text()


def test_self_reference_sql(tmp_path):
    spec = _employees_spec()
    out = tmp_path / "self.sql"
    write(GenerationEngine(spec).run().tables, str(out), "sql", spec=spec)
    text = out.read_text()
    assert ('FOREIGN KEY ("manager_id") REFERENCES "employees" ("id")') in text
