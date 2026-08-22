"""Tests for many-to-many (junction table) relationships."""

import pytest

from syntab import GenerationEngine, SpecError
from syntab.conformance import validate_against_spec
from syntab.formats import write
from syntab.spec import (
    ColumnSpec,
    ManyToManySpec,
    RelationshipSpec,
    Settings,
    Spec,
    SpecMetadata,
    TableSpec,
)


def _base_spec(m2m: ManyToManySpec):
    students = TableSpec(
        name="students", row_count=5, primary_key="id",
        columns=[ColumnSpec(name="id", dtype="int", generator="sequence")],
    )
    courses = TableSpec(
        name="courses", row_count=4, primary_key="id",
        columns=[ColumnSpec(name="id", dtype="int", generator="sequence")],
    )
    return Spec(
        metadata=SpecMetadata(name="school"),
        settings=Settings(seed=11, fallback="raise", max_rule_attempts=20),
        tables=[students, courses],
        many_to_many=[m2m],
    )


def test_m2m_generates_unique_pairs_within_degree_bounds():
    spec = _base_spec(ManyToManySpec(
        name="enrollments", table_a="students", table_b="courses",
        column_a="student_id", column_b="course_id",
        row_count=12, min_per_a=2, max_per_a=3, min_per_b=2, max_per_b=4,
    ))
    frames = GenerationEngine(spec).run().to_frames()
    report = validate_against_spec(frames, spec)
    assert report.overall_ok, report.to_text()

    enr = frames["enrollments"]
    assert not enr.duplicated(["student_id", "course_id"]).any()
    per_student = enr["student_id"].value_counts()
    per_course = enr["course_id"].value_counts()
    assert per_student.between(2, 3).all()
    assert per_course.between(2, 4).all()
    assert set(enr["student_id"]).issubset(set(frames["students"]["id"]))
    assert set(enr["course_id"]).issubset(set(frames["courses"]["id"]))


def test_m2m_non_unique_pairs():
    spec = _base_spec(ManyToManySpec(
        name="enrollments", table_a="students", table_b="courses",
        column_a="student_id", column_b="course_id",
        row_count=20, unique_pairs=False,
    ))
    frames = GenerationEngine(spec).run().to_frames()
    report = validate_against_spec(frames, spec)
    assert report.overall_ok, report.to_text()
    assert "enrollments" in frames
    assert frames["enrollments"]["id"].is_unique


def test_m2m_infeasible_row_count_raises():
    # 25 unique pairs impossible with 5 x 4 = 20 combinations.
    spec = _base_spec(ManyToManySpec(
        name="enrollments", table_a="students", table_b="courses",
        column_a="student_id", column_b="course_id",
        row_count=25, max_per_a=4, max_per_b=5, unique_pairs=True,
    ))
    with pytest.raises(SpecError, match="feasible maximum"):
        GenerationEngine(spec).run()


def test_m2m_requires_scalar_primary_keys():
    courses = TableSpec(
        name="courses", row_count=4, primary_key=["id", "track"],
        columns=[
            ColumnSpec(name="id", dtype="int", generator="sequence"),
            ColumnSpec(name="track", dtype="int", generator="sequence"),
        ],
    )
    students = TableSpec(
        name="students", row_count=5, primary_key="id",
        columns=[ColumnSpec(name="id", dtype="int", generator="sequence")],
    )
    spec = Spec(
        metadata=SpecMetadata(name="school"),
        settings=Settings(seed=1),
        tables=[students, courses],
        many_to_many=[ManyToManySpec(
            name="enrollments", table_a="students", table_b="courses",
            column_a="student_id", column_b="course_id",
            row_count=5,
        )],
    )
    with pytest.raises(SpecError, match="scalar"):
        GenerationEngine(spec).run()


def test_m2m_sql(tmp_path):
    spec = _base_spec(ManyToManySpec(
        name="enrollments", table_a="students", table_b="courses",
        column_a="student_id", column_b="course_id",
        row_count=10,
    ))
    out = tmp_path / "m2m.sql"
    write(GenerationEngine(spec).run().tables, str(out), "sql", spec=spec)
    text = out.read_text()
    assert 'FOREIGN KEY ("student_id") REFERENCES "students" ("id")' in text
    assert 'FOREIGN KEY ("course_id") REFERENCES "courses" ("id")' in text
    assert 'PRIMARY KEY ("student_id", "course_id")' in text
