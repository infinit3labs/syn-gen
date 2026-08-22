"""Polish and special-generation tests: bounded self-ref depth, min/max
allocation validation, and empty-table handling."""

import pytest

from syntab import GenerationEngine, SpecError
from syntab.conformance import validate_against_spec
from syntab.spec import (
    ColumnSpec,
    RelationshipSpec,
    Settings,
    Spec,
    SpecMetadata,
    TableSpec,
)


def _employees_spec(max_depth=None, seed=7):
    rel = {"from": "manager_id", "to": "employees.id", "alias": "manager"}
    if max_depth is not None:
        rel["max_depth"] = max_depth
    return Spec(
        metadata=SpecMetadata(name="org"),
        settings=Settings(seed=seed, fallback="raise", max_rule_attempts=20),
        tables=[
            TableSpec(
                name="employees",
                row_count=50,
                primary_key="id",
                columns=[
                    ColumnSpec(name="id", dtype="int", generator="sequence"),
                    ColumnSpec(name="manager_id", dtype="int", generator="fk",
                               constraints={"nullable": True}),
                ],
                relationships=[RelationshipSpec(**rel)],
            ),
        ],
    )


def _tree_depth(frames):
    emp = frames["employees"]
    mgr = emp["manager_id"].fillna(-1).astype(int)
    parent = dict(zip(emp["id"].astype(int), mgr))
    memo = {}

    def depth(i):
        if i is None:
            return 0
        if i in memo:
            return memo[i]
        cur, d, seen = i, 0, set()
        while cur != -1 and cur in parent and parent[cur] != -1 and cur not in seen:
            seen.add(cur)
            d += 1
            cur = parent[cur]
        memo[i] = d
        return d

    return max(depth(i) for i in emp["id"])


def test_self_ref_max_depth_is_respected():
    frames = GenerationEngine(_employees_spec(max_depth=2)).run().to_frames()
    assert validate_against_spec(frames, _employees_spec(max_depth=2)).overall_ok
    assert _tree_depth(frames) <= 2


def test_self_ref_max_depth_one_is_flat():
    frames = GenerationEngine(_employees_spec(max_depth=1)).run().to_frames()
    assert _tree_depth(frames) <= 1


def test_min_max_children_with_uniform_raises():
    spec = _employees_spec()
    spec.tables[0].relationships[0].allocation = "uniform"
    spec.tables[0].relationships[0].min_children = 2
    with pytest.raises(SpecError, match="min/max_children"):
        GenerationEngine(spec).run()


def test_empty_table_generates_zero_rows():
    spec = _employees_spec()
    spec.tables[0].row_count = 0
    frames = GenerationEngine(spec).run().to_frames()
    assert len(frames["employees"]) == 0
    assert validate_against_spec(frames, spec).overall_ok


def test_max_depth_validation_rejects_zero():
    spec = _employees_spec(max_depth=0)
    with pytest.raises(SpecError, match="max_depth"):
        GenerationEngine(spec).run()
