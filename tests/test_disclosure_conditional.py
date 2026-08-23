"""The disclosure notice has to account for conditional distributions.

A conditional table embeds strictly more of the source than the two marginals
it replaces: not just which values exist and how often, but which combinations
co-occur and in what proportion. A fidelity win that quietly widens the
privacy surface without saying so is the exact failure the disclosure module
was written to prevent, so it has to count this too.
"""
import pytest

from syntab.disclosure import DisclosureReport
from syntab.profiler import OTHER_BUCKET_LABEL, RECOMMENDED_MIN_CELL_COUNT
from syntab.spec import (
    CategoricalProfile,
    ColumnProfile,
    ColumnSpec,
    Spec,
    SpecMetadata,
    TableSpec,
)


def cat(values, **kw):
    return ColumnProfile(categorical=CategoricalProfile(values=values, **kw))


def spec_with_conditional(keys=None, values=None, redacted=False,
                          min_cell_count=None):
    keys = keys if keys is not None else [["checking"], ["savings"], [None]]
    values = values if values is not None else [
        ["bank account"], ["bank account"], ["bank account", "mortgage"]]
    return Spec(
        metadata=SpecMetadata(name="complaints"),
        tables=[TableSpec(
            name="complaints", row_count=100,
            columns=[
                ColumnSpec(
                    name="sub_product", dtype="str",
                    profile=cat({"checking": 0.6, "savings": 0.4},
                                redacted=redacted,
                                min_cell_count=min_cell_count),
                ),
                ColumnSpec(
                    name="product", dtype="str", generator="conditional",
                    depends_on=["sub_product"],
                    params={"on": ["sub_product"], "keys": keys,
                            "values": values,
                            "weights": [[1.0]] * (len(keys) - 1) + [[0.5, 0.5]],
                            "default": {"values": ["bank account", "mortgage"],
                                        "weights": [0.7, 0.3]}},
                    profile=cat({"bank account": 0.7, "mortgage": 0.3},
                                redacted=redacted,
                                min_cell_count=min_cell_count),
                ),
            ],
        )],
    )


def report(**kw):
    return DisclosureReport.from_spec(spec_with_conditional(**kw))


# --------------------------------------------------------------------------
# it is counted at all
# --------------------------------------------------------------------------

def test_a_conditional_column_is_recorded():
    rep = report()
    assert len(rep.conditionals) == 1
    c = rep.conditionals[0]
    assert c.determinant == ["sub_product"]
    assert c.dependent == "product"


def test_the_size_of_the_embedded_table_is_counted():
    rep = report()
    c = rep.conditionals[0]
    assert c.n_keys == 3
    assert c.n_cells == 4


def test_a_spec_with_no_conditionals_reports_none():
    spec = spec_with_conditional()
    spec.tables[0].columns[1].generator = None
    spec.tables[0].columns[1].params = {}
    rep = DisclosureReport.from_spec(spec)
    assert rep.conditionals == []
    assert "conditional" not in rep.to_text().lower()


# --------------------------------------------------------------------------
# it is stated, in the notice, in terms of what it means
# --------------------------------------------------------------------------

def test_the_notice_says_the_joint_structure_is_embedded():
    text = report().to_text()
    assert "Conditional" in text
    lowered = text.lower()
    assert "co-occur" in lowered
    assert "sub_product -> product" in lowered


def test_the_notice_says_this_is_more_than_the_marginals():
    lowered = report().to_text().lower()
    assert "more" in lowered and "marginal" in lowered


def test_the_notice_reports_the_cell_count():
    text = report().to_text()
    assert "4" in text


def test_a_conditional_spec_always_needs_review():
    rep = report()
    assert rep.needs_review


def test_conditional_structure_needs_review_even_when_redacted():
    """Opaque labels are not the same as no structure."""
    spec = spec_with_conditional(redacted=True)
    for col in spec.tables[0].columns:
        col.profile.categorical.values = {"value_001": 0.6, "value_002": 0.4}
    rep = DisclosureReport.from_spec(spec)
    assert rep.embedded_values == 0        # no real labels
    assert rep.conditionals                # but structure is still there
    assert rep.needs_review
    lowered = rep.to_text().lower()
    assert "conditional" in lowered
    assert "redact" in lowered


# --------------------------------------------------------------------------
# the controls are reported against it
# --------------------------------------------------------------------------

def test_suppressed_conditional_cells_are_counted():
    rep = report(
        values=[["bank account"], [OTHER_BUCKET_LABEL],
                ["bank account", OTHER_BUCKET_LABEL]],
        min_cell_count=RECOMMENDED_MIN_CELL_COUNT,
    )
    assert rep.conditionals[0].suppressed_cells == 2
    assert "suppressed" in rep.to_text().lower()


def test_an_unsuppressed_conditional_says_so():
    rep = report()
    assert rep.conditionals[0].suppressed_cells == 0
    text = rep.to_text()
    assert "--min-cell-count" in text


def test_a_null_key_is_not_mistaken_for_a_value():
    rep = report()
    assert rep.conditionals[0].n_keys == 3


def test_totals_add_up_across_several_conditional_columns():
    spec = spec_with_conditional()
    spec.tables[0].columns.append(ColumnSpec(
        name="issue", dtype="str", generator="conditional",
        params={"on": ["product"], "keys": [["a"], ["b"]],
                "values": [["x", "y", "z"], ["w"]],
                "weights": [[0.3, 0.3, 0.4], [1.0]]},
        profile=cat({"x": 0.5, "y": 0.5}),
    ))
    rep = DisclosureReport.from_spec(spec)
    assert len(rep.conditionals) == 2
    assert rep.conditional_cells == 8
    assert rep.conditional_keys == 5


def test_a_malformed_conditional_does_not_crash_the_notice():
    """The notice must render for any spec, including a hand-broken one."""
    spec = spec_with_conditional()
    spec.tables[0].columns[1].params = {"on": ["sub_product"]}
    rep = DisclosureReport.from_spec(spec)
    assert rep.to_text()
