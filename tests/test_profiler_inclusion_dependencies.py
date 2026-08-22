"""Foreign keys come from inclusion dependencies, not from column names.

Complements tests/test_profiler_relationships.py, which pins the behaviour of
the name-based rule. That rule still exists -- it is the fallback when the
optional discovery extra is absent -- and every one of its tests still passes
unchanged. What is new here is that recall no longer depends on it.
"""
import pandas as pd
import pytest

from syntab import discovery as D
from syntab.profiler import DatasetProfiler

requires_discovery = pytest.mark.skipif(
    not D.is_available(), reason="desbordante (optional [discovery] extra) not installed"
)


def _rels(tables, **kw):
    kw.setdefault("sample", None)
    spec = DatasetProfiler.profile_set(tables, **kw)
    return {t.name: t.relationships for t in spec.tables}


def _pairs(rels):
    return {(child, r.from_, r.to)
            for child, rs in rels.items() for r in rs}


# --------------------------------------------------------------------------
# the recall the name gate could not reach
# --------------------------------------------------------------------------

def _natural_key_schema():
    """A schema keyed on natural values. No column is named `<table>_id`."""
    return {
        "complaints": pd.DataFrame({
            "Complaint ID": [1, 2, 3, 4],
            "Company": ["Acme", "Beta", "Acme", "Beta"],
            "State": ["CA", "NY", "CA", "TX"],
        }),
        "companies": pd.DataFrame({"name": ["Acme", "Beta", "Gamma"]}),
        "states": pd.DataFrame({"code": ["CA", "NY", "TX", "WA"]}),
    }


def test_the_name_gate_finds_nothing_in_a_naturally_keyed_schema():
    """The baseline this change exists to move. Not an aspiration: a fact."""
    rels = _rels(_natural_key_schema(), discover=False)
    assert _pairs(rels) == set()


@requires_discovery
def test_inclusion_dependencies_find_every_foreign_key_the_name_gate_missed():
    rels = _rels(_natural_key_schema(), discover=True)
    assert _pairs(rels) == {
        ("complaints", "Company", "companies.name"),
        ("complaints", "State", "states.code"),
    }


@requires_discovery
def test_a_discovered_foreign_key_records_how_it_was_found():
    rels = _rels(_natural_key_schema(), discover=True)
    rel = next(r for r in rels["complaints"] if r.from_ == "Company")
    p = rel.provenance
    assert p.algorithm == "SPIDER"
    assert p.citation == "Bauckmann, Leser, Naumann & Tietz, ICDE 2007"
    assert p.measure == "exact"
    assert p.confidence == 1.0
    assert p.support == 4
    assert p.validated_on == "full"
    assert p.fingerprint


def test_the_fallback_path_does_not_claim_an_algorithm_it_did_not_run():
    users = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
    orders = pd.DataFrame({"id": [10, 11], "user_id": [1.0, 2.0]})
    rels = _rels({"users": users, "orders": orders}, discover=False)
    assert rels["orders"][0].provenance.algorithm == "name-heuristic"
    assert rels["orders"][0].provenance.citation is None


@requires_discovery
def test_a_transitive_foreign_key_between_two_dimensions_is_found():
    """zip_codes.state -> states.code: a dimension referencing a dimension."""
    tables = {
        "facts": pd.DataFrame({"id": [1, 2, 3],
                               "zip": ["90210", "10001", "90210"]}),
        "zip_codes": pd.DataFrame({"zip": ["90210", "10001", "73301"],
                                   "state": ["CA", "NY", "TX"]}),
        "states": pd.DataFrame({"code": ["CA", "NY", "TX"]}),
    }
    pairs = _pairs(_rels(tables, discover=True))
    assert ("zip_codes", "state", "states.code") in pairs
    assert ("facts", "zip", "zip_codes.zip") in pairs


# --------------------------------------------------------------------------
# which inclusion dependencies are NOT foreign keys
# --------------------------------------------------------------------------

@requires_discovery
def test_the_reverse_inclusion_into_a_fact_column_is_not_a_foreign_key():
    """A dimension's values are trivially contained in the column they came
    from. SPIDER reports both directions; only the one referencing a key is a
    foreign key."""
    rels = _rels(_natural_key_schema(), discover=True)
    assert rels["companies"] == []
    assert rels["states"] == []


@requires_discovery
def test_two_mutually_contained_primary_keys_produce_no_relationship():
    """A one-to-one correspondence with no direction the data can settle."""
    a = pd.DataFrame({"id": [1, 2, 3], "x": ["p", "q", "r"]})
    b = pd.DataFrame({"id": [1, 2, 3], "y": ["s", "t", "u"]})
    assert _pairs(_rels({"a": a, "b": b}, discover=True)) == set()


@requires_discovery
def test_a_self_reference_is_not_inferred():
    """Deliberate: the engine needs a nullable key and a root fraction for a
    hierarchy to terminate, and inferring one without them yields a spec that
    does not validate. Called out in the PR rather than guessed at."""
    users = pd.DataFrame({"id": [1, 2, 3], "manager": [1.0, 1.0, 2.0]})
    assert _rels({"users": users}, discover=True)["users"] == []


@requires_discovery
def test_a_column_contained_in_a_non_key_is_not_a_foreign_key():
    parent = pd.DataFrame({"pk": [1, 2, 3], "tag": ["x", "x", "y"]})
    child = pd.DataFrame({"id": [7, 8], "tag": ["x", "y"]})
    pairs = _pairs(_rels({"parent": parent, "child": child}, discover=True))
    assert ("child", "tag", "parent.tag") not in pairs


# --------------------------------------------------------------------------
# choosing between candidate parents
# --------------------------------------------------------------------------

@requires_discovery
def test_the_tighter_covering_key_wins_when_two_parents_contain_the_column():
    """Coverage -- distinct child values over distinct parent key values -- is
    one of the ranking features in Rostin et al., WebDB 2009."""
    tables = {
        "small": pd.DataFrame({"code": ["a", "b", "c"]}),
        "big": pd.DataFrame({"code": [f"x{i}" for i in range(200)]
                             + ["a", "b", "c"]}),
        "facts": pd.DataFrame({"id": [1, 2, 3], "code": ["a", "b", "c"]}),
    }
    rels = _rels(tables, discover=True)
    assert len(rels["facts"]) == 1
    assert rels["facts"][0].to == "small.code"


@requires_discovery
def test_only_one_parent_is_chosen_per_child_column():
    tables = {
        "p1": pd.DataFrame({"k": ["a", "b"]}),
        "p2": pd.DataFrame({"k": ["a", "b"]}),
        "c": pd.DataFrame({"id": [1, 2], "k": ["a", "b"]}),
    }
    rels = _rels(tables, discover=True)
    assert len(rels["c"]) == 1


@requires_discovery
def test_a_cycle_between_two_tables_is_broken():
    """The engine generates parents before children and rejects a cyclic
    spec, so at most one direction of a mutual reference may survive."""
    a = pd.DataFrame({"id": [1, 2, 3], "b_ref": ["x", "y", "z"]})
    b = pd.DataFrame({"id": ["x", "y", "z"], "a_ref": [1, 2, 3]})
    rels = _rels({"a": a, "b": b}, discover=True)
    edges = {(child, r.to.split(".")[0])
             for child, rs in rels.items() for r in rs}
    assert not ({("a", "b"), ("b", "a")} <= edges)


@requires_discovery
def test_a_generated_spec_with_discovered_foreign_keys_still_generates():
    from syntab import GenerationEngine

    spec = DatasetProfiler.profile_set(_natural_key_schema(), sample=None,
                                       discover=True)
    out = GenerationEngine(spec).run().tables
    parents = {r["name"] for r in out["companies"]}
    assert {r["Company"] for r in out["complaints"]} <= parents
