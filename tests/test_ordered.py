"""Tests for ordered generation within a parent group."""

from syntab import GenerationEngine
from syntab.conformance import validate_against_spec
from syntab.spec import (
    ColumnSpec,
    RelationshipSpec,
    Settings,
    Spec,
    SpecMetadata,
    TableSpec,
)


def _spec(ordered):
    sessions = TableSpec(
        name="sessions", row_count=4, primary_key="id",
        columns=[ColumnSpec(name="id", dtype="int", generator="sequence")],
    )
    events = TableSpec(
        name="events", row_count=20, primary_key="id",
        columns=[
            ColumnSpec(name="id", dtype="int", generator="sequence"),
            ColumnSpec(name="session_id", dtype="int", generator="fk"),
            ColumnSpec(name="event_seq", dtype="int", generator="const",
                       params={"value": 7}),
        ],
        relationships=[RelationshipSpec(
            **{
                "from": "session_id",
                "to": "sessions.id",
                "alias": "session",
                "ordered": ordered,
                "order_column": "event_seq" if ordered else None,
            }
        )],
    )
    return Spec(
        metadata=SpecMetadata(name="events"),
        settings=Settings(seed=4, fallback="raise", max_rule_attempts=20),
        tables=[sessions, events],
    )


def test_ordered_children_are_numbered_within_each_parent():
    frames = GenerationEngine(_spec(ordered=True)).run().to_frames()
    report = validate_against_spec(frames, _spec(ordered=True))
    assert report.overall_ok, report.to_text()

    ev = frames["events"]
    for _, sub in ev.groupby("session_id"):
        seq = list(sub["event_seq"])
        assert seq == list(range(1, len(seq) + 1))


def test_without_ordered_flag_sequence_is_not_grouped():
    frames = GenerationEngine(_spec(ordered=False)).run().to_frames()
    ev = frames["events"]
    # At least one session's event_seq is not a clean 1..n run.
    all_clean = all(
        list(sub["event_seq"]) == list(range(1, len(sub) + 1))
        for _, sub in ev.groupby("session_id")
    )
    assert not all_clean
