"""Regression tests for primary-key detection.

`_detect_pk` matched `low.endswith("id")`, which is true of `paid`, `void`,
`valid`, `bid`, `grid`, `rapid` and `humid`. It also carried a hardcoded
`"complaint id"` -- the name of a column in one example dataset -- which is
the tell that the suffix rule was not working.
"""
import ast
import inspect

import numpy as np
import pandas as pd
import pytest

from syntab import profiler as profiler_mod
from syntab.conformance import validate_against_spec
from syntab.profiler import DatasetProfiler, _id_base, _is_id_like, _name_tokens


# --------------------------------------------------------------------------
# name tokenization
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "id", "ID", "Id",
    "user_id", "userId", "UserId", "UserID", "USER_ID",
    "Complaint ID", "complaint id", "Complaint Id",
    "order-id", "order.id",
    "customer_uuid", "uuid", "UUID", "guid", "GUID",
])
def test_id_like_names_are_recognised(name):
    assert _is_id_like(name), name


@pytest.mark.parametrize("name", [
    "paid", "void", "valid", "bid", "grid", "rapid", "humid", "squid",
    "amount_paid", "is_valid", "avoid", "overdraft",
    "id_number", "identity", "identifier", "idle",
    "", "   ", "total", "region",
])
def test_non_id_names_are_not_mistaken_for_keys(name):
    assert not _is_id_like(name), name


@pytest.mark.parametrize("name,expected", [
    ("Complaint ID", ["Complaint", "ID"]),
    ("userId", ["user", "Id"]),
    ("UserID", ["User", "ID"]),
    ("user_id", ["user", "id"]),
    ("HTTPResponseCode", ["HTTP", "Response", "Code"]),
])
def test_name_tokenization(name, expected):
    assert _name_tokens(name) == expected


@pytest.mark.parametrize("name,expected", [
    ("user_id", "user"),
    ("userId", "user"),
    ("User ID", "user"),
    ("order_line_id", "order_line"),
    ("id", None),
    ("paid", None),
    ("total", None),
])
def test_id_base_extraction(name, expected):
    assert _id_base(name) == expected


# --------------------------------------------------------------------------
# primary-key selection
# --------------------------------------------------------------------------

def _spec_for(df, **kw):
    return DatasetProfiler(df, name="t", sample=None, **kw).profile()


def test_paid_is_not_promoted_to_primary_key_ahead_of_the_real_key():
    # `paid` comes first, is unique, and ends in the letters "id".
    df = pd.DataFrame({
        "paid": [1.5, 2.5, 3.5, 4.5],
        "record_id": [10, 11, 12, 13],
        "region": ["EU", "US", "EU", "US"],
    })
    assert _spec_for(df).tables[0].primary_key == "record_id"


@pytest.mark.parametrize("name", ["paid", "void", "valid", "grid"])
def test_no_lookalike_column_is_chosen_over_a_real_id(name):
    df = pd.DataFrame({name: [1.5, 2.5, 3.5], "thing_id": [7, 8, 9]})
    assert _spec_for(df).tables[0].primary_key == "thing_id"


def test_complaint_id_is_detected_without_a_hardcoded_special_case():
    df = pd.DataFrame({
        "Product": ["debt", "mortgage", "debt", "card"],
        "Complaint ID": [1290534, 1290535, 1290536, 1290537],
    })
    assert _spec_for(df).tables[0].primary_key == "Complaint ID"


def test_profiler_carries_no_dataset_specific_column_names():
    """No string *literal* in the profiler may name a column of one dataset.

    `_detect_pk` used to compare against the literal "complaint id", taken
    from the CFPB example. Comments and docstrings may of course still discuss
    it, so this inspects string constants only, minus docstrings.
    """
    tree = ast.parse(inspect.getsource(profiler_mod))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef,
                             ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) and \
                    isinstance(body[0].value, ast.Constant) and \
                    isinstance(body[0].value.value, str):
                docstrings.add(id(body[0].value))

    literals = [
        n.value for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
        and id(n) not in docstrings
    ]
    offenders = [s for s in literals if "complaint" in s.lower()]
    assert offenders == [], f"dataset-specific literal(s) in profiler: {offenders}"


def test_primary_key_must_be_not_null():
    """UNIQUE alone is not enough -- a PK is UNIQUE and NOT NULL."""
    df = pd.DataFrame({"record_id": [1.0, 2.0, np.nan, 4.0]})
    spec = _spec_for(df)
    assert spec.tables[0].columns[0].constraints.get("unique") is True
    assert spec.tables[0].primary_key is None
    # and the spec now certifies its own source data
    assert validate_against_spec({"t": df}, spec).overall_ok


def test_no_unique_column_means_no_primary_key():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    assert _spec_for(df).tables[0].primary_key is None


def test_uuid_column_can_be_the_primary_key():
    df = pd.DataFrame({
        "row_uuid": [
            "3f2504e0-4f89-11d3-9a0c-0305e82c3301",
            "3f2504e0-4f89-11d3-9a0c-0305e82c3302",
            "3f2504e0-4f89-11d3-9a0c-0305e82c3303",
        ],
        "amount": [1.0, 2.0, 3.0],
    })
    assert _spec_for(df).tables[0].primary_key == "row_uuid"
