from datetime import datetime

from syntab.rules import Rule, compile_rules


def test_comparison_and_arithmetic():
    r = Rule("discount <= total * 0.5")
    assert r.check({"discount": 1, "total": 10}, {})
    assert not r.check({"discount": 10, "total": 10}, {})


def test_arithmetic_precedence():
    assert Rule("1 + 2 * 3 == 7").check({}, {})
    assert Rule("(1 + 2) * 3 == 9").check({}, {})


def test_cross_join_alias():
    r = Rule("region == user.region")
    assert r.check({"region": "EU"}, {"user": {"region": "EU"}})
    assert not r.check({"region": "EU"}, {"user": {"region": "US"}})
    assert r.parent_aliases() == ["user"]


def test_if_then_else():
    r = Rule("if region == 'EU' then total > 0 else total >= 0")
    assert r.check({"region": "EU", "total": 5}, {})
    assert not r.check({"region": "EU", "total": 0}, {})
    assert r.check({"region": "US", "total": 0}, {})


def test_logical_ops():
    r = Rule("a > 0 and b > 0")
    assert r.check({"a": 1, "b": 2}, {})
    assert not r.check({"a": 1, "b": -1}, {})
    r2 = Rule("a > 0 or b > 0")
    assert r2.check({"a": -1, "b": 1}, {})


def test_functions():
    r = Rule("coalesce(a, 'x') == 'x'")
    assert r.check({"a": None}, {})
    r2 = Rule("abs(-5) == 5")
    assert r2.check({}, {})
    r3 = Rule("min(3, 1, 2) == 1")
    assert r3.check({}, {})


def test_days_between():
    r = Rule("days_between(d2, d1) == 10")
    d1 = datetime(2024, 1, 1)
    d2 = datetime(2024, 1, 11)
    assert r.check({"d1": d1, "d2": d2}, {})


def test_syntax_error():
    import pytest
    with pytest.raises(Exception):
        compile_rules(["this is not valid @@@"])


def test_compiled_rule_list():
    rules = compile_rules(["a > 0", "b < 10"])
    assert all(rl.check({"a": 1, "b": 5}, {}) for rl in rules)
