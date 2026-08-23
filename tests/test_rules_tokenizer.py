"""Regression tests for the rule-DSL tokenizer.

`true`, `false` and `null` were regex alternatives placed ahead of the
identifier rule, so the engine matched a *prefix* of a longer word:

    nullable == true   ->   NULL, IDENT("able"), OP("=="), BOOL("true")

Any column whose name starts with one of those three words was silently
mis-parsed. The fix is the standard lexer shape: match the whole word with one
identifier rule, then classify it against a keyword/literal table.
"""
import pytest

from syntab.rules import Rule, compile_rules, tokenize


def _kinds(expr):
    return [(t.kind, t.value) for t in tokenize(expr)]


# --------------------------------------------------------------------------
# the reported bug
# --------------------------------------------------------------------------

def test_nullable_tokenizes_as_one_identifier():
    assert _kinds("nullable") == [("IDENT", "nullable")]


def test_nullable_equals_true_parses_as_written():
    assert _kinds("nullable == true") == [
        ("IDENT", "nullable"), ("OP", "=="), ("BOOL", "true"),
    ]


@pytest.mark.parametrize("name", [
    "nullable", "nulls", "null_rate", "nullify", "nulled",
    "trueup", "true_positive", "trues", "truer",
    "falsework", "false_positive", "falsehood", "falses",
    "andover", "orbit", "notes", "iframe", "then_value", "elsewhere",
])
def test_identifiers_beginning_with_a_literal_or_keyword(name):
    assert _kinds(name) == [("IDENT", name)]


def test_the_bare_literals_are_still_literals():
    assert _kinds("true") == [("BOOL", "true")]
    assert _kinds("false") == [("BOOL", "false")]
    assert _kinds("null") == [("NULL", "null")]


def test_keywords_are_still_keywords():
    for kw in ("if", "then", "else", "and", "or", "not"):
        assert _kinds(kw) == [("KW", kw)]


# --------------------------------------------------------------------------
# evaluation, end to end
# --------------------------------------------------------------------------

def test_rule_on_a_column_named_nullable_evaluates_correctly():
    rule = Rule("nullable == true")
    assert rule.check({"nullable": True}, {})
    assert not rule.check({"nullable": False}, {})


def test_rule_on_a_column_named_null_rate():
    rule = Rule("null_rate < 0.5")
    assert rule.check({"null_rate": 0.1}, {})
    assert not rule.check({"null_rate": 0.9}, {})


def test_column_refs_report_the_whole_name():
    assert Rule("nullable == true").column_refs() == ["nullable"]
    assert Rule("null_rate + trueup > 1").column_refs() == ["null_rate", "trueup"]


def test_parent_alias_named_notes_still_resolves():
    rule = Rule("notes.length > 0")
    assert rule.check({}, {"notes": {"length": 3}})
    assert rule.parent_aliases() == ["notes"]


def test_null_comparison_still_works():
    assert Rule("a == null").check({"a": None}, {})
    assert not Rule("a == null").check({"a": 1}, {})


def test_boolean_literals_still_work_in_expressions():
    assert Rule("if a > 0 then true else false").check({"a": 1}, {})
    assert not Rule("if a > 0 then true else false").check({"a": -1}, {})


def test_compile_rules_over_literal_prefixed_names():
    rules = compile_rules([
        "nullable == true",
        "null_rate <= 1",
        "not (falsehood == true)",
    ])
    row = {"nullable": True, "null_rate": 0.2, "falsehood": False}
    assert all(r.check(row, {}) for r in rules)


def test_unknown_character_still_rejected():
    with pytest.raises(ValueError):
        tokenize("a @ b")
