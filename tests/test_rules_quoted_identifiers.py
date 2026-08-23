"""Quoted identifiers in the rule DSL.

The unquoted identifier grammar is ``[A-Za-z_][A-Za-z0-9_]*``, so a rule could
not name a real column called ``ZIP code``, ``Timely response?`` or
``Sub-product``. That is not a cosmetic limit: every column of the CFPB source
this project profiles is named that way, so the whole ``when``-clause and
``rules`` half of the spec was unreachable for real profiled data.

Backticks are the quoting character rather than double quotes because ``"..."``
already means a string literal in this DSL and redefining it would silently
change the meaning of every existing spec. A literal backtick inside a quoted
identifier is written by doubling it, as in SQL.

These tests also re-assert the previously-fixed prefix-tokenizer bug from
``test_rules_tokenizer.py`` through the new alternative, because adding a
regex alternative ahead of the identifier rule is exactly the change that
caused it the first time.
"""
import pytest

from syntab.rules import Rule, quote_identifier, tokenize


def _kinds(expr):
    return [(t.kind, t.value) for t in tokenize(expr)]


# --------------------------------------------------------------------------
# tokenizing
# --------------------------------------------------------------------------

def test_quoted_identifier_is_one_token():
    assert _kinds("`ZIP code`") == [("QIDENT", "ZIP code")]


@pytest.mark.parametrize("name", [
    "ZIP code", "Timely response?", "Sub-product", "Consumer disputed?",
    "Date Sent to Company", "Company Public Response", "Unnamed: 18",
    "100% match", "a.b", "col(with)parens", "tab\tsep", "üñïçø∂é",
])
def test_awkward_column_names_round_trip(name):
    assert _kinds(f"`{name}`") == [("QIDENT", name)]


def test_a_doubled_backtick_is_a_literal_backtick():
    assert _kinds("`we``ird`") == [("QIDENT", "we`ird")]


def test_quoted_identifier_in_a_comparison():
    assert _kinds("`ZIP code` == '02139'") == [
        ("QIDENT", "ZIP code"), ("OP", "=="), ("STRING", "02139"),
    ]


def test_an_unterminated_quote_is_an_error():
    with pytest.raises(ValueError):
        tokenize("`ZIP code")


def test_an_empty_quoted_identifier_names_the_empty_column():
    """pandas will hand you a column named ``''``; the DSL can name it."""
    assert _kinds("`` == 1") == [("QIDENT", ""), ("OP", "=="), ("NUMBER", "1")]
    assert Rule("`` == 1").check({"": 1}, {}) is True


# --------------------------------------------------------------------------
# keywords and literals are never keywords when quoted
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["true", "false", "null", "if", "then",
                                  "else", "and", "or", "not"])
def test_quoting_turns_a_keyword_into_a_plain_column_name(name):
    assert _kinds(f"`{name}`") == [("QIDENT", name)]
    assert Rule(f"`{name}` == 1").check({name: 1}, {}) is True


# --------------------------------------------------------------------------
# the prefix-tokenizer bug must not come back through the new alternative
# --------------------------------------------------------------------------

def test_unquoted_literals_still_tokenize_as_before():
    assert _kinds("nullable == true") == [
        ("IDENT", "nullable"), ("OP", "=="), ("BOOL", "true"),
    ]
    assert _kinds("true") == [("BOOL", "true")]
    assert _kinds("false") == [("BOOL", "false")]
    assert _kinds("null") == [("NULL", "null")]
    assert _kinds("nulls_allowed") == [("IDENT", "nulls_allowed")]
    assert _kinds("trueup") == [("IDENT", "trueup")]
    assert _kinds("falsework") == [("IDENT", "falsework")]


def test_mixed_quoted_and_unquoted_in_one_rule():
    assert _kinds("nullable == true and `ZIP code` != null") == [
        ("IDENT", "nullable"), ("OP", "=="), ("BOOL", "true"),
        ("KW", "and"),
        ("QIDENT", "ZIP code"), ("OP", "!="), ("NULL", "null"),
    ]


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------

def test_quoted_identifier_evaluates_against_the_row():
    rule = Rule("`Timely response?` == 'Yes'")
    assert rule.check({"Timely response?": "Yes"}, {}) is True
    assert rule.check({"Timely response?": "No"}, {}) is False


def test_quoted_identifier_with_a_dot_is_not_a_parent_path():
    """``a.b`` quoted is a column literally named ``a.b``, not ``b`` of parent ``a``."""
    rule = Rule("`a.b` == 1")
    assert rule.check({"a.b": 1}, {"a": {"b": 2}}) is True
    assert rule.check({"a.b": 2}, {"a": {"b": 1}}) is False


def test_quoted_identifier_as_a_parent_attribute():
    rule = Rule("user.`home state` == 'MA'")
    assert rule.check({}, {"user": {"home state": "MA"}}) is True
    assert rule.check({}, {"user": {"home state": "NY"}}) is False


def test_quoted_parent_alias():
    rule = Rule("`parent table`.region == 'EU'")
    assert rule.check({}, {"parent table": {"region": "EU"}}) is True


def test_quoted_identifiers_in_an_if_expression():
    rule = Rule("if `Sub-product` == 'Checking account' "
                "then `Product` == 'Bank account or service' else true")
    assert rule.check({"Sub-product": "Checking account",
                       "Product": "Bank account or service"}, {}) is True
    assert rule.check({"Sub-product": "Checking account",
                       "Product": "Mortgage"}, {}) is False


def test_a_quoted_name_is_not_a_function_call():
    """``abs`` quoted names a column; it must not resolve to the function."""
    with pytest.raises(ValueError):
        Rule("`abs`(1) == 1")


# --------------------------------------------------------------------------
# introspection used by the engine
# --------------------------------------------------------------------------

def test_a_quoted_keyword_is_a_column_not_a_keyword():
    """The parser must discriminate on token KIND, not on the text.

    Before quoting existed, a token spelled ``if`` could only be the keyword,
    so the parser tested ``token.value`` alone. ```if` == 1`` then parsed as
    the start of a malformed if-expression.
    """
    assert Rule("`if` == 1").check({"if": 1}, {}) is True
    assert Rule("`not` == 1").check({"not": 1}, {}) is True
    assert Rule("`and` == `or`").check({"and": 2, "or": 2}, {}) is True
    assert Rule("if `if` == 1 then `then` == 2 else `else` == 3").check(
        {"if": 1, "then": 2, "else": 9}, {}) is True


def test_column_refs_reports_quoted_names():
    rule = Rule("`ZIP code` == '02139' and `Timely response?` == 'Yes'")
    assert rule.column_refs() == ["Timely response?", "ZIP code"]


def test_parent_aliases_still_excludes_the_quoted_column():
    rule = Rule("user.`home state` == `home state`")
    assert rule.parent_aliases() == ["user"]
    assert rule.column_refs() == ["home state"]


# --------------------------------------------------------------------------
# quote_identifier: the helper anything generating rule text should use
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["col", "user_id", "_x", "a1", "nullable",
                                  "trueup", "falsework"])
def test_quote_identifier_leaves_plain_names_alone(name):
    assert quote_identifier(name) == name


@pytest.mark.parametrize("name", ["ZIP code", "Timely response?", "Sub-product",
                                  "1st", "", "true", "false", "null", "if",
                                  "and", "or", "not", "then", "else"])
def test_quote_identifier_quotes_what_needs_quoting(name):
    quoted = quote_identifier(name)
    assert quoted.startswith("`") and quoted.endswith("`")
    assert _kinds(quoted) == [("QIDENT", name)]


def test_quote_identifier_escapes_backticks():
    assert quote_identifier("we`ird") == "`we``ird`"
    assert _kinds(quote_identifier("we`ird")) == [("QIDENT", "we`ird")]


@pytest.mark.parametrize("name", [
    "ZIP code", "Timely response?", "Sub-product", "we`ird", "a.b",
    "nullable", "true", "col", "Unnamed: 18",
])
def test_quote_identifier_round_trips_through_the_parser(name):
    rule = Rule(f"{quote_identifier(name)} == 1")
    assert rule.column_refs() == [name]
    assert rule.check({name: 1}, {}) is True


# --------------------------------------------------------------------------
# load-bearing, not decorative: the engine paths that were unreachable
# --------------------------------------------------------------------------

def test_a_when_clause_can_now_name_a_real_cfpb_column():
    """The ``when`` path was unusable for profiled CFPB data, for this reason."""
    from syntab.engine import GenerationEngine
    from syntab.spec import (ColumnSpec, Settings, Spec, SpecMetadata,
                             TableSpec, WhenClause)

    spec = Spec(
        metadata=SpecMetadata(name="complaints"),
        settings=Settings(seed=4),
        tables=[TableSpec(
            name="complaints", row_count=200,
            columns=[
                ColumnSpec(
                    name="Company Response to Consumer", dtype="str",
                    generator="choice",
                    params={"values": ["Closed with explanation",
                                       "Closed with monetary relief"]},
                    when=[WhenClause(
                        condition="`Timely response?` == 'No'",
                        generator="const",
                        params={"value": "Untimely"},
                    )],
                ),
                ColumnSpec(
                    name="Timely response?", dtype="str", generator="choice",
                    params={"values": ["Yes", "No"], "weights": [0.6, 0.4]},
                ),
            ],
            rules=["if `Timely response?` == 'No' then "
                   "`Company Response to Consumer` == 'Untimely'"],
        )],
    )
    frame = GenerationEngine(spec).run().to_frames()["complaints"]
    untimely = frame[frame["Timely response?"] == "No"]
    assert len(untimely) > 0
    assert (untimely["Company Response to Consumer"] == "Untimely").all()


def test_an_unknown_quoted_column_in_a_when_clause_is_rejected():
    from syntab.engine import GenerationEngine
    from syntab.spec import (ColumnSpec, Spec, SpecError, SpecMetadata,
                             TableSpec, WhenClause)

    spec = Spec(
        metadata=SpecMetadata(name="t"),
        tables=[TableSpec(name="t", row_count=5, columns=[
            ColumnSpec(name="a", dtype="int", generator="int",
                       when=[WhenClause(condition="`no such column` == 1",
                                        generator="const",
                                        params={"value": 0})]),
        ])],
    )
    with pytest.raises(SpecError, match="no such column"):
        GenerationEngine(spec)
