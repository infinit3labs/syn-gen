"""Tests for the vectorized (column-wise) generation fast path."""

from syntab import GenerationEngine, SpecError
from syntab.conformance import validate_against_spec
from syntab.engine import _can_vectorize
from syntab.spec import (
    ColumnSpec,
    RelationshipSpec,
    Settings,
    Spec,
    SpecMetadata,
    TableSpec,
)


def _vectorizable_spec(seed=1):
    users = TableSpec(
        name="users", row_count=20, primary_key="id",
        columns=[ColumnSpec(name="id", dtype="int", generator="sequence")],
    )
    events = TableSpec(
        name="events", row_count=500, primary_key="id",
        columns=[
            ColumnSpec(name="id", dtype="int", generator="sequence"),
            ColumnSpec(name="user_id", dtype="int", generator="fk"),
            ColumnSpec(name="amount", dtype="float", generator="float",
                       params={"min": 1, "max": 100}),
            ColumnSpec(name="cat", dtype="str", generator="choice",
                       params={"values": ["a", "b", "c"], "weights": [0.5, 0.3, 0.2]}),
            ColumnSpec(name="flag", dtype="bool", generator="bool"),
            ColumnSpec(name="ts", dtype="datetime", generator="datetime",
                       params={"start": "2020-01-01", "end": "2021-01-01"}),
            ColumnSpec(name="uid", dtype="uuid", generator="uuid"),
        ],
        relationships=[RelationshipSpec(
            **{"from": "user_id", "to": "users.id", "allocation": "balanced"}
        )],
    )
    return Spec(
        metadata=SpecMetadata(name="v"),
        settings=Settings(seed=seed, max_rule_attempts=20),
        tables=[users, events],
    )


def test_vectorized_is_conforming():
    spec = _vectorizable_spec()
    frames = GenerationEngine(spec).run(vectorized=True).to_frames()
    report = validate_against_spec(frames, spec)
    assert report.overall_ok, report.to_text()


def test_vectorized_matches_row_by_row_structure():
    spec = _vectorizable_spec()
    fv = GenerationEngine(spec).run(vectorized=True).to_frames()
    fr = GenerationEngine(spec).run(vectorized=False).to_frames()
    assert set(fv["events"].columns) == set(fr["events"].columns)
    assert len(fv["events"]) == len(fr["events"])
    assert set(fv["events"]["user_id"]).issubset(set(fv["users"]["id"]))


def test_vectorized_one_to_one_fk():
    spec = _vectorizable_spec()
    spec.tables[1].row_count = 20
    spec.tables[1].relationships[0].cardinality = "one_to_one"
    frames = GenerationEngine(spec).run(vectorized=True).to_frames()
    assert frames["events"]["user_id"].is_unique
    assert validate_against_spec(frames, spec).overall_ok


def test_vectorized_standalone_fk_generator():
    spec = _vectorizable_spec()
    events = spec.tables[1]
    events.relationships = []
    events.columns[1].params = {"ref": "users.id"}

    frames = GenerationEngine(spec).run(vectorized=True).to_frames()

    assert set(frames["events"]["user_id"]).issubset(set(frames["users"]["id"]))


def test_rule_table_falls_back_gracefully():
    spec = _vectorizable_spec()
    spec.tables[1].rules = ["amount > 0"]
    # A rule disables the vectorized path but generation still succeeds.
    frames = GenerationEngine(spec).run(vectorized=True).to_frames()
    assert validate_against_spec(frames, spec).overall_ok


def test_can_vectorize_classifies_tables():
    spec = _vectorizable_spec()
    assert _can_vectorize(spec.tables[1]) is True

    with_rules = _vectorizable_spec()
    with_rules.tables[1].rules = ["amount > 0"]
    assert _can_vectorize(with_rules.tables[1]) is False

    with_faker = _vectorizable_spec()
    with_faker.tables[1].columns[1].generator = "faker.name"
    assert _can_vectorize(with_faker.tables[1]) is False

    with_unique = _vectorizable_spec()
    with_unique.tables[1].columns[1].constraints = {"unique": True}
    assert _can_vectorize(with_unique.tables[1]) is False
