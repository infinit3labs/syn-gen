"""Primary keys come from discovered unique column combinations.

Complements tests/test_profiler_keys.py, which pins the name-tokenization
rules. Those rules still exist and are still tested there; what changes here
is their STATUS. Uniqueness evidence is now HyUCC's, over the whole table, and
the column name is consulted only to break a tie between candidates the data
cannot separate.
"""
import pandas as pd
import pytest

from syntab import discovery as D
from syntab.conformance import validate_against_spec
from syntab.profiler import DatasetProfiler

requires_discovery = pytest.mark.skipif(
    not D.is_available(), reason="desbordante (optional [discovery] extra) not installed"
)


def _table(df, **kw):
    kw.setdefault("sample", None)
    return DatasetProfiler(df, name="t", **kw).profile().tables[0]


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------

@requires_discovery
def test_the_key_records_which_algorithm_found_it_and_over_how_many_rows():
    df = pd.DataFrame({"only_key": range(300), "g": ["a", "b"] * 150})
    t = _table(df)
    assert t.primary_key == "only_key"
    p = t.key_provenance
    assert p is not None
    assert p.algorithm == "HyUCC"
    assert p.citation == "Papenbrock & Naumann, BTW 2017"
    assert p.support == 300
    assert p.validated_on == "full"
    assert p.confidence == 1.0
    assert p.human_edited is False
    assert p.fingerprint


def test_without_discovery_the_provenance_does_not_claim_an_algorithm():
    df = pd.DataFrame({"only_key": range(50), "g": ["a", "b"] * 25})
    t = _table(df, discover=False)
    assert t.primary_key == "only_key"
    assert t.key_provenance.algorithm == "name-heuristic"
    assert t.key_provenance.citation is None


def test_requesting_discovery_explicitly_when_it_is_missing_is_an_error():
    if D.is_available():
        pytest.skip("desbordante is installed; the raising path needs it absent")
    with pytest.raises(D.DiscoveryUnavailable):
        DatasetProfiler(pd.DataFrame({"a": [1]}), discover=True)


# --------------------------------------------------------------------------
# the name is a tie-breaker, and only that
# --------------------------------------------------------------------------

@requires_discovery
def test_a_single_candidate_key_is_chosen_without_consulting_the_name():
    """No tie, so no name rule -- the provenance has to say so."""
    df = pd.DataFrame({"wholly_unconventional": range(40),
                       "g": ["a", "b"] * 20})
    t = _table(df)
    assert t.primary_key == "wholly_unconventional"
    assert t.key_provenance.algorithm == "HyUCC"   # no "+name-tiebreak"


@requires_discovery
def test_the_name_breaks_a_tie_between_two_equally_unique_columns():
    df = pd.DataFrame({
        "paid": [1.5, 2.5, 3.5, 4.5],     # unique, and ends in the letters "id"
        "record_id": [10, 11, 12, 13],    # unique, and is an identifier
        "region": ["EU", "US", "EU", "US"],
    })
    t = _table(df)
    assert t.primary_key == "record_id"
    assert t.key_provenance.algorithm == "HyUCC+name-tiebreak"


@requires_discovery
def test_a_tie_with_no_identifier_name_falls_through_to_dtype():
    df = pd.DataFrame({"amount": [1.5, 2.5, 3.5],
                       "label": ["a", "b", "c"]})
    t = _table(df)
    assert t.primary_key == "label"
    assert t.key_provenance.algorithm == "HyUCC+dtype-tiebreak"


# --------------------------------------------------------------------------
# composite candidate keys -- previously not discovered at all
# --------------------------------------------------------------------------

@requires_discovery
def test_a_composite_key_is_found_where_no_single_column_is_unique():
    df = pd.DataFrame({
        "day": ["mon", "mon", "tue", "tue"],
        "slot": ["am", "pm", "am", "pm"],
        "load": [5, 5, 5, 5],   # constant: in no unique combination
    })
    t = _table(df)
    assert t.primary_key == ["day", "slot"]
    assert t.key_provenance.algorithm == "HyUCC"


@requires_discovery
def test_the_source_data_still_conforms_to_a_spec_with_a_composite_key():
    df = pd.DataFrame({
        "day": ["mon", "mon", "tue", "tue"],
        "slot": ["am", "pm", "am", "pm"],
        "load": [5, 5, 5, 5],
    })
    spec = DatasetProfiler(df, name="t", sample=None).profile()
    assert validate_against_spec({"t": df}, spec).overall_ok


@requires_discovery
def test_a_single_column_key_wins_over_a_composite_one():
    df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "day": ["mon", "mon", "tue", "tue"],
        "slot": ["am", "pm", "am", "pm"],
        "load": [5, 5, 5, 5],
    })
    t = _table(df)
    assert t.primary_key == "id"
    # the composite key is still real, and is recorded as a unique constraint
    assert ["day", "slot"] in t.unique_constraints


@requires_discovery
def test_single_column_uniqueness_does_not_become_a_unique_constraint():
    """It already travels as constraints.unique, and populating
    unique_constraints costs the engine's vectorized path."""
    df = pd.DataFrame({"id": range(30), "g": ["a", "b", "c"] * 10})
    t = _table(df)
    assert t.unique_constraints == []
    assert t.columns[0].constraints.get("unique") is True


@requires_discovery
def test_no_key_at_all_is_still_reported_as_no_key():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    t = _table(df)
    assert t.primary_key is None
    assert t.key_provenance is None


@requires_discovery
def test_a_nullable_unique_column_is_not_promoted_to_a_key():
    df = pd.DataFrame({"record_id": [1.0, 2.0, None, 4.0]})
    t = _table(df)
    assert t.columns[0].constraints.get("unique") is True
    assert t.primary_key is None
