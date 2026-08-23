"""Functional dependencies replace the correlation hints.

`metadata.suggested_depends_on` held pairs of numeric columns whose Pearson
coefficient exceeded 0.95, was consumed by nothing, and could not represent a
dependency: correlation is symmetric where a dependency has a direction, and
it is defined only between numeric columns. It is gone. This is its
replacement, and these tests pin both halves of that statement.
"""
import numpy as np
import pandas as pd
import pytest

from syntab import discovery as D
from syntab.profiler import DatasetProfiler

requires_discovery = pytest.mark.skipif(
    not D.is_available(), reason="desbordante (optional [discovery] extra) not installed"
)


def _table(df, **kw):
    kw.setdefault("sample", None)
    return DatasetProfiler(df, name="t", **kw).profile().tables[0]


def _fds(df, **kw):
    md = _table(df, **kw).metadata
    return md.get("discovery", {}).get("functional_dependencies", [])


def _hierarchy(n=300):
    return pd.DataFrame({
        "zip": [f"z{i % 20}" for i in range(n)],
        "state": [f"s{(i % 20) // 5}" for i in range(n)],
        "amount": np.arange(n, dtype="float64"),
        "double": np.arange(n, dtype="float64") * 2.0,   # r == 1.0
    })


# --------------------------------------------------------------------------
# the field that is gone
# --------------------------------------------------------------------------

def test_the_correlation_hint_field_no_longer_exists():
    """A perfectly correlated numeric pair used to populate it."""
    t = _table(_hierarchy())
    assert "suggested_depends_on" not in t.metadata


def test_the_correlation_method_is_gone_from_the_profiler():
    assert not hasattr(DatasetProfiler, "_infer_depends_on")


# --------------------------------------------------------------------------
# what replaced it
# --------------------------------------------------------------------------

@requires_discovery
def test_a_directional_dependency_between_string_columns_is_found():
    """Exactly what a correlation coefficient could never express: a
    dependency with a direction, between two non-numeric columns."""
    fds = _fds(_hierarchy(), discover_fds=True)
    pairs = {(tuple(f["determinant"]), f["dependent"]) for f in fds}
    assert (("zip",), "state") in pairs
    assert (("state",), "zip") not in pairs


@requires_discovery
def test_each_recorded_dependency_carries_its_algorithm_and_strength():
    fds = _fds(_hierarchy(), discover_fds=True)
    fd = next(f for f in fds if f["determinant"] == ["zip"])
    p = fd["provenance"]
    assert p["algorithm"] in ("HyFD", "Pyro")
    assert p["citation"]
    assert p["support"] == 300
    assert p["validated_on"] == "full"
    assert p["confidence"] == 1.0
    assert p["mu_prime"] > 0.9


@requires_discovery
def test_discovery_of_dependencies_is_off_unless_asked_for():
    assert _fds(_hierarchy()) == []


def test_asking_for_dependencies_without_the_extra_is_not_an_error():
    """discover_fds is a request, not a requirement; use discover=True to
    make a missing install fail loudly instead."""
    t = _table(_hierarchy(), discover=False, discover_fds=True)
    assert t.metadata.get("discovery") is None


@requires_discovery
def test_an_approximate_dependency_is_recorded_with_its_error():
    n = 300
    df = pd.DataFrame({
        "sub": [f"s{i % 10}" for i in range(n)],
        "parent": [f"p{(i % 10) // 5}" for i in range(n)],
    })
    df.loc[0, "parent"] = "pX"     # one violation: no longer exact
    fds = _fds(df, discover_fds=True, fd_error=0.05)
    fd = next(f for f in fds if f["determinant"] == ["sub"]
              and f["dependent"] == "parent")
    assert fd["provenance"]["algorithm"] == "Pyro"
    assert 0.0 < fd["provenance"]["error"] <= 0.05
    assert fd["provenance"]["confidence"] < 1.0


@requires_discovery
def test_a_near_key_determinant_does_not_reach_the_spec():
    """The artefact mu' exists to remove, checked end to end."""
    n = 400
    df = pd.DataFrame({
        "day": [f"d{i}" for i in range(n)],
        "grp": [f"g{i % 8}" for i in range(n)],
        "label": [f"l{(i % 8) // 4}" for i in range(n)],
    })
    fds = _fds(df, discover_fds=True, fd_error=0.05)
    assert all(f["determinant"] != ["day"] for f in fds)


@requires_discovery
def test_the_recorded_list_is_capped_and_strongest_first():
    from syntab.profiler import MAX_RECORDED_FDS

    rng = np.random.default_rng(3)
    n = 500
    data = {f"c{i}": rng.integers(0, 3, n).astype(str) for i in range(9)}
    data["zip"] = [f"z{i % 20}" for i in range(n)]
    data["state"] = [f"s{(i % 20) // 5}" for i in range(n)]
    fds = _fds(pd.DataFrame(data), discover_fds=True, fd_error=0.05)
    assert len(fds) <= MAX_RECORDED_FDS
    mus = [f["provenance"].get("mu_prime", 1.0) for f in fds]
    assert mus == sorted(mus, reverse=True)


@requires_discovery
def test_a_spec_with_dependencies_still_round_trips_through_yaml(tmp_path):
    from syntab.loaders import from_file, to_file

    spec = DatasetProfiler(_hierarchy(), name="t", sample=None,
                           discover_fds=True).profile()
    path = tmp_path / "spec.yaml"
    to_file(spec, path)
    again = from_file(path)
    before = spec.tables[0].metadata["discovery"]["functional_dependencies"]
    after = again.tables[0].metadata["discovery"]["functional_dependencies"]
    assert before == after
