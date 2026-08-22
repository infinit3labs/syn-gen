"""Tests for conditional generators and targeted rule retries."""

from syntab import GenerationEngine, register_generator
from syntab.spec import (
    ColumnSpec,
    Settings,
    Spec,
    SpecMetadata,
    TableSpec,
    WhenClause,
)


def test_when_generator_adds_implicit_dependency_and_enforces_branch():
    spec = Spec(
        metadata=SpecMetadata(name="accounts"),
        settings=Settings(seed=12, fallback="raise", max_rule_attempts=20),
        tables=[TableSpec(
            name="accounts",
            row_count=40,
            columns=[
                # Deliberately declared before account_type: the conditional
                # expression must add an implicit generation dependency.
                ColumnSpec(
                    name="overdraft_limit",
                    dtype="float",
                    generator="float",
                    params={"min": 10, "max": 1000, "decimals": 2},
                    when=[WhenClause(
                        condition="account_type == 'savings'",
                        generator="const",
                        params={"value": 0},
                    )],
                ),
                ColumnSpec(
                    name="account_type",
                    dtype="str",
                    generator="choice",
                    params={"values": ["checking", "savings"], "weights": [0.6, 0.4]},
                ),
            ],
            rules=["if account_type == 'savings' then overdraft_limit == 0"],
        )],
    )

    frame = GenerationEngine(spec).run().to_frames()["accounts"]
    savings = frame[frame["account_type"] == "savings"]
    checking = frame[frame["account_type"] == "checking"]
    assert len(savings) > 0
    assert (savings["overdraft_limit"] == 0).all()
    assert (checking["overdraft_limit"] >= 10).all()


_retry_calls = {"a": 0, "unrelated": 0}


@register_generator("test_retry_a")
def _retry_a(row, rng, faker, params):
    _retry_calls["a"] += 1
    return 0 if _retry_calls["a"] == 1 else 1


@register_generator("test_retry_unrelated")
def _retry_unrelated(row, rng, faker, params):
    _retry_calls["unrelated"] += 1
    return "stable"


def test_targeted_retry_preserves_unrelated_columns():
    _retry_calls.update(a=0, unrelated=0)
    spec = Spec(
        metadata=SpecMetadata(name="retry"),
        settings=Settings(seed=1, fallback="raise", max_rule_attempts=3),
        tables=[TableSpec(
            name="retry",
            row_count=1,
            columns=[
                ColumnSpec(name="a", dtype="int", generator="custom:test_retry_a"),
                ColumnSpec(name="unrelated", dtype="str", generator="custom:test_retry_unrelated"),
            ],
            rules=["a == 1"],
        )],
    )

    row = GenerationEngine(spec).run().tables["retry"][0]
    assert row["a"] == 1
    assert row["unrelated"] == "stable"
    assert _retry_calls["a"] == 2
    # The unrelated column is not referenced by the failed rule.
    assert _retry_calls["unrelated"] == 1
