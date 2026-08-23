"""Choosing which discovered dependencies generation can act on.

A conditional generator draws Y from P(Y | X), so a column can have exactly
one determinant. Discovery hands back many more edges than that, so something
has to choose -- and choosing greedily per column is demonstrably wrong: on the
CFPB source Product is determined by Issue (mu' 0.967) and by Sub-product
(0.728), so per-column selection gives Product the Issue edge, leaves
Sub-product with nothing, and the Product/Sub-product pair stays destroyed.

The choice is made over the graph instead: a maximum-weight spanning forest
(Chow & Liu 1968) with mu' as the edge weight.
"""
import pytest

from syntab.discovery import DiscoveredFD, choose_dependency_forest


def fd(determinant, dependent, mu, error=0.0, algorithm="Pyro"):
    return DiscoveredFD(
        determinant=tuple(determinant), dependent=dependent,
        algorithm=algorithm, error=error, mu_prime=mu,
        support=1000, validated_on="full",
    )


def as_pairs(edges):
    return [(e.determinant, e.dependent) for e in edges]


def undirected(edges):
    return {frozenset((e.determinant, e.dependent)) for e in edges}


# --------------------------------------------------------------------------
# the failure that motivates the whole construction
# --------------------------------------------------------------------------

CFPB = [
    fd(["Issue"], "Product", 0.967),
    fd(["Sub-product"], "Product", 0.728),
    fd(["Sub-issue"], "Issue", 0.614),
    fd(["Issue"], "Sub-product", 0.539),
    fd(["Product"], "Sub-product", 0.496),
]


def test_every_hierarchy_pair_gets_an_edge():
    """Per-column selection would leave Sub-product without a determinant."""
    edges = choose_dependency_forest(CFPB)
    assert undirected(edges) == {
        frozenset(("Issue", "Product")),
        frozenset(("Sub-product", "Product")),
        frozenset(("Sub-issue", "Issue")),
    }


def test_each_column_has_at_most_one_determinant():
    edges = choose_dependency_forest(CFPB)
    dependents = [e.dependent for e in edges]
    assert len(dependents) == len(set(dependents))


def test_the_result_is_acyclic():
    edges = choose_dependency_forest(CFPB)
    parent = {e.dependent: e.determinant for e in edges}
    for start in parent:
        seen, node = set(), start
        while node in parent:
            assert node not in seen, f"cycle through {node}"
            seen.add(node)
            node = parent[node]


def test_the_heaviest_edges_win():
    """The 0.539 and 0.496 edges lose to the 0.728 one; only one can stand."""
    edges = choose_dependency_forest(CFPB)
    assert len(edges) == 3
    assert max(e.strength for e in edges) == pytest.approx(0.967)


# --------------------------------------------------------------------------
# orientation
# --------------------------------------------------------------------------

def test_a_single_edge_keeps_its_discovered_direction():
    edges = choose_dependency_forest([fd(["zip"], "state", 0.98)])
    assert as_pairs(edges) == [("zip", "state")]
    assert edges[0].fd is not None
    assert edges[0].mu_prime == pytest.approx(0.98)


def test_a_chain_keeps_every_discovered_direction_when_it_can():
    edges = choose_dependency_forest([
        fd(["a"], "b", 0.9), fd(["b"], "c", 0.8),
    ])
    assert as_pairs(edges) == [("a", "b"), ("b", "c")]
    assert all(e.fd is not None for e in edges)


def test_a_reversed_edge_is_marked_as_reversed():
    """Two arrows into one column cannot both survive in a tree."""
    edges = choose_dependency_forest([
        fd(["a"], "b", 0.9), fd(["c"], "b", 0.8),
    ])
    assert len(edges) == 2
    reversed_edges = [e for e in edges if e.fd is None]
    assert len(reversed_edges) == 1
    assert reversed_edges[0].as_dict()["reversed"] is True
    kept = [e for e in edges if e.fd is not None][0]
    assert kept.as_dict()["reversed"] is False


def test_orientation_maximizes_agreement_with_discovered_directions():
    """Rooting at ``a`` respects both FDs; rooting anywhere else respects one."""
    edges = choose_dependency_forest([
        fd(["a"], "b", 0.9), fd(["b"], "c", 0.8), fd(["c"], "d", 0.7),
    ])
    assert as_pairs(edges) == [("a", "b"), ("b", "c"), ("c", "d")]


def test_a_reversed_edge_still_carries_the_pairs_strength():
    edges = choose_dependency_forest([fd(["a"], "b", 0.9), fd(["c"], "b", 0.8)])
    for e in edges:
        assert e.strength in (pytest.approx(0.9), pytest.approx(0.8))
    rev = [e for e in edges if e.fd is None][0]
    prov = rev.provenance()
    assert prov.mu_prime == pytest.approx(rev.strength)
    assert prov.citation


# --------------------------------------------------------------------------
# filtering
# --------------------------------------------------------------------------

def test_composite_determinants_are_ignored():
    edges = choose_dependency_forest([
        fd(["a", "b"], "c", 0.99), fd(["a"], "c", 0.6),
    ])
    assert as_pairs(edges) == [("a", "c")]


def test_columns_outside_the_allowed_set_are_ignored():
    edges = choose_dependency_forest(CFPB, columns=["Issue", "Product"])
    assert undirected(edges) == {frozenset(("Issue", "Product"))}


def test_self_dependencies_are_ignored():
    assert choose_dependency_forest([fd(["a"], "a", 0.99)]) == []


def test_no_dependencies_gives_no_edges():
    assert choose_dependency_forest([]) == []


def test_a_missing_mu_is_treated_as_no_strength():
    edges = choose_dependency_forest([
        DiscoveredFD(determinant=("a",), dependent="b", algorithm="HyFD",
                     error=0.0, mu_prime=None, support=10,
                     validated_on="full"),
        fd(["c"], "d", 0.6),
    ])
    assert ("c", "d") in as_pairs(edges)


def test_disconnected_groups_each_get_their_own_tree():
    edges = choose_dependency_forest([
        fd(["a"], "b", 0.9), fd(["c"], "d", 0.8),
    ])
    assert set(as_pairs(edges)) == {("a", "b"), ("c", "d")}


def test_selection_is_deterministic():
    first = as_pairs(choose_dependency_forest(CFPB))
    for _ in range(5):
        assert as_pairs(choose_dependency_forest(CFPB)) == first


def test_ties_do_not_produce_a_cycle():
    edges = choose_dependency_forest([
        fd(["a"], "b", 0.5), fd(["b"], "c", 0.5), fd(["c"], "a", 0.5),
    ])
    assert len(edges) == 2
    dependents = [e.dependent for e in edges]
    assert len(dependents) == len(set(dependents))
