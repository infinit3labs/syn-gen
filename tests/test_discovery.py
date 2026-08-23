"""Tests for `syntab.discovery` -- the published-algorithm dependency search.

These cover the three things an integration of an external profiler gets
wrong: the frame handed to the C++ layer, null semantics, and the fact that a
raw approximate-FD result is dominated by near-key artefacts.
"""
import numpy as np
import pandas as pd
import pytest

from syntab import discovery as D

pytestmark = pytest.mark.skipif(
    not D.is_available(), reason="desbordante (optional [discovery] extra) not installed"
)


# --------------------------------------------------------------------------
# frame preparation
# --------------------------------------------------------------------------

def test_object_coercion_accepts_every_dtype_the_profiler_produces():
    """`astype(str)` is NOT usable; `astype(object)` is. See _as_desbordante_frame."""
    df = pd.DataFrame({
        "s": pd.Series(["a", "b", "a"], dtype="string"),
        "o": ["x", "y", "x"],
        "i": pd.array([1, 2, None], dtype="Int64"),
        "f": [1.0, 2.0, np.nan],
        "b": [True, False, True],
        "c": pd.Categorical(["p", "q", "p"]),
        "d": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-01"]),
    })
    # Must not raise: this is the exact call discovery makes.
    D.discover_unique_column_combinations(df)


def test_astype_str_is_rejected_by_the_binding_where_astype_object_is_not():
    """Pins the failure mode the coercion exists to avoid.

    A string column with nulls alongside a converted numeric column -- i.e.
    any real extract -- breaks under `astype(str)` on pandas 3.
    """
    import desbordante
    frame = pd.DataFrame({"a": ["x", None, "y"], "b": [1, 2, 3]})

    with pytest.raises(RuntimeError):
        desbordante.fd.algorithms.HyFD().load_data(table=frame.astype(str))

    # The coercion discovery actually uses must load the same frame.
    desbordante.fd.algorithms.HyFD().load_data(
        table=D._as_desbordante_frame(frame)
    )


def test_free_text_and_all_null_columns_are_pruned_from_the_fd_search():
    n = 200
    df = pd.DataFrame({
        "cat": ["a", "b"] * (n // 2),
        "narrative": [f"free text {i}" for i in range(n)],   # near-key
        "empty": [None] * n,
        "other": ["p", "q"] * (n // 2),
    })
    cols = D.discovery_columns(df)
    assert "narrative" not in cols
    assert "empty" not in cols
    assert set(cols) == {"cat", "other"}


# --------------------------------------------------------------------------
# functional dependencies
# --------------------------------------------------------------------------

def _hierarchy(dirty_rows: int = 0) -> pd.DataFrame:
    """sub -> parent, optionally with `dirty_rows` violations."""
    sub, parent = [], []
    for i in range(300):
        s = f"s{i % 10}"
        sub.append(s)
        parent.append(f"p{(i % 10) // 5}")
    for j in range(dirty_rows):
        parent[j] = "pX"
    return pd.DataFrame({"sub": sub, "parent": parent,
                         "noise": [f"n{i % 3}" for i in range(300)]})


def test_hyfd_finds_an_exact_functional_dependency():
    fds = D.discover_functional_dependencies(_hierarchy(), error=0.0)
    found = {(f.determinant, f.dependent): f for f in fds}
    fd = found[(("sub",), "parent")]
    assert fd.algorithm == "HyFD"
    assert fd.error == 0.0
    assert fd.exact
    assert fd.validated_on == "full"
    assert fd.support == 300


def test_exact_discovery_alone_misses_the_dependency_that_real_dirty_data_has():
    """The reason approximate FDs are needed at all, stated as a test."""
    dirty = _hierarchy(dirty_rows=3)
    exact_only = D.discover_functional_dependencies(dirty, error=0.0)
    assert (("sub",), "parent") not in {(f.determinant, f.dependent) for f in exact_only}

    with_pyro = D.discover_functional_dependencies(dirty, error=0.05)
    fd = {(f.determinant, f.dependent): f for f in with_pyro}[(("sub",), "parent")]
    assert fd.algorithm == "Pyro"
    assert 0.0 < fd.error <= 0.05
    assert fd.provenance().measure == "g1"


def test_the_error_threshold_is_the_knob_that_admits_dirt():
    dirty = _hierarchy(dirty_rows=30)
    key = (("sub",), "parent")
    tight = {(f.determinant, f.dependent) for f in
             D.discover_functional_dependencies(dirty, error=0.001)}
    loose = {(f.determinant, f.dependent) for f in
             D.discover_functional_dependencies(dirty, error=0.2, min_mu=0.1)}
    assert key not in tight
    assert key in loose


def test_near_key_determinants_are_rejected_by_mu_prime():
    """A high-cardinality column "determines" everything under g1 alone.

    This is the artefact that makes a naive AFD integration useless: with 400
    distinct dates over 400 rows, `day` satisfies any loose g1 bound against
    every column. mu' (Piatetsky-Shapiro & Matheus 1993) discounts it.
    """
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({
        "day": [f"d{i}" for i in range(n)],          # unique-ish, explains nothing
        "grp": [f"g{i % 8}" for i in range(n)],
        "label": [f"l{(i % 8) // 4}" for i in range(n)],
        "rand": rng.integers(0, 5, n).astype(str),
    })
    fds = D.discover_functional_dependencies(df, error=0.05, min_mu=D.DEFAULT_MIN_MU)
    determinants = {f.determinant for f in fds}
    assert ("day",) not in determinants, "a near-key determinant must not survive mu'"
    assert (("grp",), "label") in {(f.determinant, f.dependent) for f in fds}


def test_mu_prime_scores_a_real_dependency_high_and_a_random_one_low():
    rng = np.random.default_rng(1)
    n = 1000
    df = pd.DataFrame({
        "zip": [f"z{i % 50}" for i in range(n)],
        "state": [f"s{(i % 50) // 10}" for i in range(n)],
        "junk": rng.integers(0, 50, n).astype(str),
    })
    assert D.mu_prime(df, ["zip"], "state") > 0.95
    assert D.mu_prime(df, ["junk"], "state") < 0.2


def test_mu_prime_is_zero_for_a_key_determinant_and_a_constant_dependent():
    df = pd.DataFrame({"k": [f"k{i}" for i in range(50)],
                       "v": ["x"] * 50,
                       "w": [f"w{i % 5}" for i in range(50)]})
    assert D.mu_prime(df, ["k"], "w") == 0.0     # key determines everything
    assert D.mu_prime(df, ["w"], "v") == 0.0     # constant is determined by everything


def test_g1_error_matches_the_definition():
    # 4 rows, x -> y violated by one pair inside group "a".
    df = pd.DataFrame({"x": ["a", "a", "b", "b"], "y": ["1", "2", "3", "3"]})
    # sum cx^2 = 4 + 4 = 8 ; sum cxy^2 = 1 + 1 + 4 = 6 ; n^2 = 16
    assert D.g1_error(df, ["x"], "y") == pytest.approx((8 - 6) / 16)


def test_max_lhs_bounds_the_determinant_width():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({c: rng.integers(0, 3, 200).astype(str)
                       for c in "abcdefg"})
    fds = D.discover_functional_dependencies(df, error=0.05, min_mu=0.0, max_lhs=2)
    assert fds == [] or max(len(f.determinant) for f in fds) <= 2


# --------------------------------------------------------------------------
# candidate generation on a sample, validation on the full frame
# --------------------------------------------------------------------------

def test_a_dependency_true_only_in_the_sample_is_rejected_on_the_full_data():
    """Sampling proposes; the full frame decides. The HyFD pattern.

    The first 100 rows satisfy sub -> parent exactly. The remaining 900 break
    it well past any threshold. A sample-only answer reports the FD; candidate
    generation plus full validation must not.
    """
    n = 1000
    sub = [f"s{i % 5}" for i in range(n)]
    parent = [f"p{i % 5}" for i in range(100)] + [f"q{i}" for i in range(n - 100)]
    df = pd.DataFrame({"sub": sub, "parent": parent})

    head = df.head(100)
    assert D.holds_exactly(head, ["sub"], "parent")
    assert not D.holds_exactly(df, ["sub"], "parent")

    fds = D.discover_functional_dependencies(
        df, error=0.01, sample_rows=100,
    )
    assert (("sub",), "parent") not in {(f.determinant, f.dependent) for f in fds}
    assert all(f.validated_on == "full" and f.support == n for f in fds)


# --------------------------------------------------------------------------
# unique column combinations
# --------------------------------------------------------------------------

def test_hyucc_finds_the_key_and_not_the_non_keys():
    df = pd.DataFrame({
        "id": range(100),
        "grp": [i % 4 for i in range(100)],
        "text": [f"t{i % 20}" for i in range(100)],
    })
    uccs = D.discover_unique_column_combinations(df)
    assert ("id",) in {u.columns for u in uccs}
    assert ("grp",) not in {u.columns for u in uccs}
    assert all(u.algorithm == "HyUCC" and u.support == 100 for u in uccs)


def test_a_column_unique_only_within_the_sample_is_not_reported_as_a_key():
    """The failure a sample-only uniqueness verdict always has.

    Any column with more distinct values than the sample size is unique in the
    sample by construction. `dup` is unique in the first 50 rows and duplicated
    in the full 1000.
    """
    n = 1000
    df = pd.DataFrame({
        "dup": [f"v{i}" for i in range(50)] + [f"v{i % 50}" for i in range(n - 50)],
        "real": [f"r{i}" for i in range(n)],
    })
    uccs = D.discover_unique_column_combinations(df, sample_rows=50, seed=None)
    cols = {u.columns for u in uccs}
    assert ("dup",) not in cols


# --------------------------------------------------------------------------
# inclusion dependencies
# --------------------------------------------------------------------------

def test_spider_finds_an_inclusion_dependency_with_no_naming_convention():
    """The whole point: no column here is named `<table>_id`."""
    complaints = pd.DataFrame({"Company": ["Acme", "Beta", "Acme"],
                               "Amount": [1, 2, 3]})
    companies = pd.DataFrame({"name": ["Acme", "Beta", "Gamma"]})
    inds = D.discover_inclusion_dependencies(
        {"complaints": complaints, "companies": companies}
    )
    pairs = {(i.child_table, i.child_column, i.parent_table, i.parent_column)
             for i in inds}
    assert ("complaints", "Company", "companies", "name") in pairs
    one = next(i for i in inds
               if (i.child_table, i.child_column) == ("complaints", "Company"))
    assert one.algorithm == "SPIDER"
    assert one.error == 0.0
    assert one.provenance().citation.startswith("Bauckmann")


def test_a_nullable_foreign_key_is_still_detected():
    """SPIDER treats NULL as a value; a nullable FK is the ordinary case.

    Without the null-compacting view in `_ind_view` this returns nothing,
    which would lose exactly the relationships the P0 work was fixing.
    """
    child = pd.DataFrame({"fk": ["a", "b", None, "c"], "x": [1, 2, 3, 4]})
    parent = pd.DataFrame({"pk": ["a", "b", "c"]})
    inds = D.discover_inclusion_dependencies({"child": child, "parent": parent})
    pairs = {(i.child_table, i.child_column, i.parent_table, i.parent_column)
             for i in inds}
    assert ("child", "fk", "parent", "pk") in pairs


def test_an_all_null_column_yields_no_inclusion_dependency():
    child = pd.DataFrame({"fk": [None, None], "x": [1, 2]})
    parent = pd.DataFrame({"pk": ["a", "b"]})
    inds = D.discover_inclusion_dependencies({"child": child, "parent": parent})
    assert all(i.child_column != "fk" for i in inds)


def test_a_genuine_violation_still_blocks_an_exact_inclusion_dependency():
    child = pd.DataFrame({"fk": ["a", "b", "ZZZ"]})
    parent = pd.DataFrame({"pk": ["a", "b", "c"]})
    inds = D.discover_inclusion_dependencies({"child": child, "parent": parent})
    assert ("child", "fk", "parent", "pk") not in {
        (i.child_table, i.child_column, i.parent_table, i.parent_column) for i in inds
    }


def test_the_column_prefilter_restricts_the_search():
    child = pd.DataFrame({"fk": ["a", "b"], "other": ["a", "b"]})
    parent = pd.DataFrame({"pk": ["a", "b"]})
    inds = D.discover_inclusion_dependencies(
        {"child": child, "parent": parent}, columns={"child": ["fk"]},
    )
    assert all(i.child_column != "other" and i.parent_column != "other" for i in inds)


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------

def test_every_result_carries_an_algorithm_and_a_citation():
    df = _hierarchy()
    items = list(D.discover_functional_dependencies(df, error=0.01))
    items += list(D.discover_unique_column_combinations(df))
    assert items
    for item in items:
        p = item.provenance()
        assert p.algorithm in D.CITATIONS
        assert p.citation
        assert p.support == 300
        assert p.human_edited is False


def test_the_shared_pass_agrees_with_the_two_separate_measures():
    """dependency_measures shares the group-bys g1_error and mu_prime each do
    on their own. Equivalence is asserted rather than assumed, because the
    only reason it exists is speed."""
    rng = np.random.default_rng(11)
    n = 600
    df = pd.DataFrame({
        "a": [f"a{i % 7}" for i in range(n)],
        "b": [f"b{(i % 7) // 3}" for i in range(n)],
        "c": rng.integers(0, 4, n).astype(str),
        "const": ["k"] * n,
        "key": [f"k{i}" for i in range(n)],
    })
    cases = [(["a"], "b"), (["b"], "a"), (["c"], "b"), (["a", "c"], "b"),
             (["a"], "const"), (["key"], "b"), ([], "const"), ([], "a")]
    for lhs, rhs in cases:
        g1, mu = D.dependency_measures(df, lhs, rhs)
        assert g1 == pytest.approx(D.g1_error(df, lhs, rhs)), (lhs, rhs)
        if lhs:
            assert mu == pytest.approx(D.mu_prime(df, lhs, rhs)), (lhs, rhs)
        else:
            assert mu is None


def test_the_group_size_cache_does_not_change_the_answer():
    df = pd.DataFrame({"x": [f"x{i % 5}" for i in range(100)],
                       "y": [f"y{i % 5}" for i in range(100)],
                       "z": [f"z{i % 3}" for i in range(100)]})
    cache = {}
    first = [D.dependency_measures(df, ["x"], c, cache) for c in ("y", "z")]
    second = [D.dependency_measures(df, ["x"], c) for c in ("y", "z")]
    assert first == second
