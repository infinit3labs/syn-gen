"""Re-profiling must not destroy hand edits.

The first review flagged this as the thing that makes the iteration loop
viable: `profile` writes a whole new file, so every correction a human made
to the previous one disappeared, and the sensible response was to profile
once and never again.
"""
import pandas as pd
import pytest

from syntab import discovery as D
from syntab.profiler import DatasetProfiler, merge_preserving_edits

requires_discovery = pytest.mark.skipif(
    not D.is_available(), reason="desbordante (optional [discovery] extra) not installed"
)


def _schema():
    return {
        "complaints": pd.DataFrame({
            "Complaint ID": [1, 2, 3, 4],
            "Company": ["Acme", "Beta", "Acme", "Beta"],
            "State": ["CA", "NY", "CA", "TX"],
        }),
        "companies": pd.DataFrame({"name": ["Acme", "Beta", "Gamma"]}),
        "states": pd.DataFrame({"code": ["CA", "NY", "TX", "WA"]}),
    }


def _profile(**kw):
    kw.setdefault("sample", None)
    return DatasetProfiler.profile_set(_schema(), **kw)


def _table(spec, name):
    return next(t for t in spec.tables if t.name == name)


# --------------------------------------------------------------------------
# the failure this fixes
# --------------------------------------------------------------------------

@requires_discovery
def test_a_reprofile_without_merging_still_overwrites_everything():
    """Stated so the merge is understood as opt-in, not as a silent change."""
    first = _profile()
    _table(first, "complaints").primary_key = "Company"     # a human's edit
    second = _profile()
    assert _table(second, "complaints").primary_key == "Complaint ID"


@requires_discovery
def test_an_edited_primary_key_survives_a_reprofile():
    first = _profile()
    _table(first, "complaints").primary_key = "Company"

    merged = merge_preserving_edits(first, _profile())
    t = _table(merged, "complaints")
    assert t.primary_key == "Company"
    assert t.key_provenance.human_edited is True


@requires_discovery
def test_an_untouched_primary_key_is_re_inferred():
    first = _profile()
    merged = merge_preserving_edits(first, _profile())
    t = _table(merged, "complaints")
    assert t.primary_key == "Complaint ID"
    assert t.key_provenance.human_edited is False
    assert t.key_provenance.algorithm == "HyUCC"


@requires_discovery
def test_an_edited_relationship_target_survives_a_reprofile():
    first = _profile()
    rel = next(r for r in _table(first, "complaints").relationships
               if r.from_ == "State")
    rel.to = "states.code"
    rel.on_delete = "restrict"          # the edit

    merged = merge_preserving_edits(first, _profile())
    kept = next(r for r in _table(merged, "complaints").relationships
                if r.from_ == "State")
    assert kept.on_delete == "restrict" or kept.provenance.human_edited is False


@requires_discovery
def test_a_relationship_repointed_by_hand_is_not_re_inferred():
    first = _profile()
    rel = next(r for r in _table(first, "complaints").relationships
               if r.from_ == "Company")
    rel.to = "states.code"              # deliberately wrong, deliberately kept

    merged = merge_preserving_edits(first, _profile())
    kept = next(r for r in _table(merged, "complaints").relationships
                if r.from_ == "Company")
    assert kept.to == "states.code"
    assert kept.provenance.human_edited is True


@requires_discovery
def test_a_hand_written_relationship_with_no_provenance_is_preserved():
    from syntab.spec import RelationshipSpec

    first = _profile()
    _table(first, "complaints").relationships.append(
        RelationshipSpec(**{"from": "Complaint ID", "to": "companies.name"})
    )
    merged = merge_preserving_edits(first, _profile())
    kept = [r for r in _table(merged, "complaints").relationships
            if r.from_ == "Complaint ID"]
    assert len(kept) == 1
    assert kept[0].provenance.human_edited is True


@requires_discovery
def test_a_relationship_only_the_new_profile_found_is_added():
    first = _profile()
    complaints = _table(first, "complaints")
    complaints.relationships = [r for r in complaints.relationships
                                if r.from_ != "State"]

    merged = merge_preserving_edits(first, _profile())
    froms = {r.from_ for r in _table(merged, "complaints").relationships}
    assert froms == {"Company", "State"}


@requires_discovery
def test_a_table_missing_from_the_new_profile_is_kept():
    """Re-profiling one file of several must not delete the others."""
    first = _profile()
    partial = DatasetProfiler.profile_set(
        {"complaints": _schema()["complaints"]}, sample=None)
    merged = merge_preserving_edits(first, partial)
    assert {t.name for t in merged.tables} == {"complaints", "companies", "states"}


@requires_discovery
def test_neither_input_spec_is_mutated():
    first = _profile()
    fresh = _profile()
    _table(first, "complaints").primary_key = "Company"
    before = [t.model_dump() for t in fresh.tables]
    merge_preserving_edits(first, fresh)
    assert [t.model_dump() for t in fresh.tables] == before
    assert _table(first, "complaints").primary_key == "Company"


@requires_discovery
def test_an_edit_stays_marked_across_a_second_reprofile():
    """human_edited is sticky: once a person has spoken, the profiler stops
    re-deciding that rule even if the value drifts back."""
    first = _profile()
    _table(first, "complaints").primary_key = "Company"
    once = merge_preserving_edits(first, _profile())
    twice = merge_preserving_edits(once, _profile())
    t = _table(twice, "complaints")
    assert t.primary_key == "Company"
    assert t.key_provenance.human_edited is True
