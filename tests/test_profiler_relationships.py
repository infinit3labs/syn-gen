"""Regression tests for foreign-key inference in `profile_set`.

`_infer_relationships` compared `{str(v) for v in ...}` on both sides. A
nullable foreign key is held by pandas in a float64 column, so a child value
of 1 became "1.0" while the int64 parent's 1 became "1", the containment test
failed, and the relationship was silently dropped. A nullable FK is the
ordinary case, so this lost most real relationships.
"""
import numpy as np
import pandas as pd

from syntab.profiler import DatasetProfiler


def _rels(tables, **kw):
    kw.setdefault("sample", None)
    spec = DatasetProfiler.profile_set(tables, **kw)
    return {t.name: t.relationships for t in spec.tables}


def _users():
    return pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})


# --------------------------------------------------------------------------
# the reported bug: nullable FK stored as float64
# --------------------------------------------------------------------------

def test_nullable_float_fk_matches_an_int_parent_key():
    orders = pd.DataFrame({
        "id": [10, 11, 12, 13],
        "user_id": [1.0, 2.0, 3.0, np.nan],   # nullable -> pandas gives float64
        "total": [5.0, 6.0, 7.0, 8.0],
    })
    # Precondition: this is the dtype pandas actually chooses.
    assert orders["user_id"].dtype == np.dtype("float64")
    assert _users()["id"].dtype == np.dtype("int64")

    rels = _rels({"users": _users(), "orders": orders})
    assert len(rels["orders"]) == 1
    rel = rels["orders"][0]
    assert rel.from_ == "user_id"
    assert rel.to == "users.id"


def test_all_null_fk_column_is_not_linked():
    orders = pd.DataFrame({"id": [10, 11], "user_id": [np.nan, np.nan]})
    assert _rels({"users": _users(), "orders": orders})["orders"] == []


def test_nullable_int_extension_dtype_also_matches():
    orders = pd.DataFrame({
        "id": [10, 11, 12],
        "user_id": pd.array([1, 2, None], dtype="Int64"),
    })
    rels = _rels({"users": _users(), "orders": orders})
    assert len(rels["orders"]) == 1


# --------------------------------------------------------------------------
# type awareness: only compare comparable columns
# --------------------------------------------------------------------------

def test_string_keys_match_string_keys():
    users = pd.DataFrame({"id": ["u1", "u2", "u3"], "name": ["a", "b", "c"]})
    orders = pd.DataFrame({"id": [1, 2, 3], "user_id": ["u1", "u2", "u3"]})
    assert len(_rels({"users": users, "orders": orders})["orders"]) == 1


def test_a_string_child_is_not_matched_to_a_numeric_parent():
    """Stringifying both sides used to make these look joinable."""
    users = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
    orders = pd.DataFrame({"id": [10, 11, 12], "user_id": ["1", "2", "3"]})
    assert _rels({"users": users, "orders": orders})["orders"] == []


def test_a_bool_child_is_not_matched_to_a_numeric_parent():
    flags = pd.DataFrame({"id": [0, 1], "label": ["no", "yes"]})
    rows = pd.DataFrame({"id": [1, 2, 3], "flag_id": [True, False, True]})
    assert _rels({"flags": flags, "rows": rows})["rows"] == []


def test_non_integral_floats_are_compared_as_floats():
    parent = pd.DataFrame({"id": [1.5, 2.5, 3.5], "x": ["a", "b", "c"]})
    child = pd.DataFrame({"id": [1, 2], "parent_id": [1.5, 2.5]})
    assert len(_rels({"parent": parent, "child": child})["child"]) == 1


# --------------------------------------------------------------------------
# name handling
# --------------------------------------------------------------------------

def test_camelcase_foreign_key_is_recognised():
    orders = pd.DataFrame({"id": [10, 11, 12], "userId": [1.0, 2.0, 3.0]})
    rels = _rels({"users": _users(), "orders": orders})
    assert len(rels["orders"]) == 1
    assert rels["orders"][0].from_ == "userId"


def test_plural_child_column_matches_singular_table():
    person = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
    visit = pd.DataFrame({"id": [10, 11], "persons_id": [1.0, 2.0]})
    assert len(_rels({"person": person, "visit": visit})["visit"]) == 1


def test_bare_id_column_is_never_treated_as_a_foreign_key():
    a = pd.DataFrame({"id": [1, 2, 3]})
    b = pd.DataFrame({"id": [1, 2, 3]})
    rels = _rels({"a": a, "b": b})
    assert rels["a"] == [] and rels["b"] == []


def test_orphan_values_still_prevent_a_relationship():
    orders = pd.DataFrame({"id": [10, 11], "user_id": [1.0, 999.0]})
    assert _rels({"users": _users(), "orders": orders})["orders"] == []


def test_self_referencing_column_is_not_linked_to_its_own_table():
    users = pd.DataFrame({"id": [1, 2, 3], "users_id": [1.0, 1.0, 2.0]})
    assert _rels({"users": users})["users"] == []


def test_parent_without_a_primary_key_is_skipped():
    # `things` has no unique column, so no primary key to point at.
    things = pd.DataFrame({"id": [1, 1, 2], "v": ["a", "b", "c"]})
    child = pd.DataFrame({"k": [10, 11], "things_id": [1.0, 2.0]})
    assert _rels({"things": things, "child": child})["child"] == []
