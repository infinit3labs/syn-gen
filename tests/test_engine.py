import pytest

from syntab import GenerationEngine, register_generator
from syntab.spec import (
    Spec, SpecMetadata, Settings, TableSpec, ColumnSpec, RelationshipSpec,
)


@register_generator("order_region")
def order_region(row, rng, faker, params):
    return row["__parents__"]["user"]["region"]


def build_shop(users_n=50, orders_n=200, fallback="raise", max_attempts=200):
    return Spec(
        metadata=SpecMetadata(name="shop"),
        settings=Settings(seed=42, fallback=fallback, max_rule_attempts=max_attempts),
        tables=[
            TableSpec(
                name="users", row_count=users_n, primary_key="id",
                columns=[
                    ColumnSpec(name="id", dtype="int", generator="sequence", constraints={"unique": True}),
                    ColumnSpec(name="region", dtype="str", generator="choice",
                               params={"values": ["EU", "US", "APAC"], "weights": [0.4, 0.4, 0.2]}),
                    ColumnSpec(name="credit_limit", dtype="float", generator="float",
                               params={"min": 100, "max": 10000, "decimals": 2}),
                ],
            ),
            TableSpec(
                name="orders", row_count=orders_n, primary_key="id",
                columns=[
                    ColumnSpec(name="id", dtype="int", generator="sequence", constraints={"unique": True}),
                    ColumnSpec(name="user_id", dtype="int", generator="fk", params={"ref": "users.id"}),
                    ColumnSpec(name="region", dtype="str", generator="custom:order_region", depends_on=["user_id"]),
                    ColumnSpec(name="total", dtype="float", generator="float",
                               params={"min": 1, "max": 500, "decimals": 2}),
                    ColumnSpec(name="discount", dtype="float", generator="float",
                               params={"min": 0, "max": 50, "decimals": 2}, depends_on=["total"]),
                ],
                relationships=[RelationshipSpec(**{"from": "user_id", "to": "users.id", "alias": "user"})],
                rules=[
                    "discount <= total * 0.5",
                    "region == user.region",
                    "total <= user.credit_limit",
                    "if region == 'EU' then total > 0",
                ],
            ),
        ],
    )


def test_full_generation_fk_and_rules():
    spec = build_shop()
    res = GenerationEngine(spec).run()
    users = res.tables["users"]
    orders = res.tables["orders"]
    assert len(users) == 50
    assert len(orders) == 200

    user_ids = {u["id"] for u in users}
    user_by_id = {u["id"]: u for u in users}

    for o in orders:
        assert o["user_id"] in user_ids
    for o in orders:
        u = user_by_id[o["user_id"]]
        assert o["region"] == u["region"]
        assert o["discount"] <= o["total"] * 0.5
        assert o["total"] <= u["credit_limit"]
        if o["region"] == "EU":
            assert o["total"] > 0


def test_uniqueness():
    spec = build_shop()
    res = GenerationEngine(spec).run()
    ids = [u["id"] for u in res.tables["users"]]
    assert len(ids) == len(set(ids))


def test_deterministic_seed():
    r1 = GenerationEngine(build_shop()).run().tables["orders"][0]
    r2 = GenerationEngine(build_shop()).run().tables["orders"][0]
    assert r1 == r2


def test_fallback_drop():
    spec = build_shop(fallback="drop", max_attempts=5)
    spec.tables[1].rules.append("total > 1000000")
    res = GenerationEngine(spec).run()
    assert len(res.tables["orders"]) == 0


def test_fallback_null():
    spec = build_shop(fallback="null", max_attempts=5)
    spec.tables[1].rules.append("total > 1000000")
    res = GenerationEngine(spec).run()
    assert len(res.tables["orders"]) == 200
    assert res.tables["orders"][0]["total"] is None


def test_fallback_raise():
    spec = build_shop(fallback="raise", max_attempts=3)
    spec.tables[1].rules.append("total > 1000000")
    with pytest.raises(Exception):
        GenerationEngine(spec).run()


def test_profiled_auto_spec():
    from pathlib import Path
    from syntab.loaders import from_file
    path = Path(__file__).parent.parent / "examples" / "spec_profiled.json"
    spec = from_file(path)
    res = GenerationEngine(spec).run()
    orders = res.tables["orders"]
    assert len(orders) == 500
    cats = [o["category"] for o in orders]
    assert set(cats) <= {"books", "music", "films"}
