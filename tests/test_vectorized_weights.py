"""Regression tests: `--vectorized` must accept the weights a profile produces.

`_gen_vectorized` passed `params["weights"]` straight to
`numpy.random.Generator.choice(p=...)`, which requires the probabilities to
sum to 1. The profiler rounds every category frequency to 4 decimal places, so
the weights it writes routinely sum to 0.9999 -- and `--vectorized` raised
`ValueError: Probabilities do not sum to 1` on 46% of profiled specs while the
row-by-row path, which uses `random.choices`, accepted all of them.
"""
import numpy as np
import pandas as pd
import pytest

from syntab.engine import GenerationEngine, _normalize_weights
from syntab.profiler import DatasetProfiler
from syntab.spec import ColumnSpec, Settings, Spec, SpecError, SpecMetadata, TableSpec


def _choice_spec(values, weights, rows=500, seed=1):
    return Spec(
        metadata=SpecMetadata(name="w"),
        settings=Settings(seed=seed),
        tables=[TableSpec(name="t", row_count=rows, columns=[
            ColumnSpec(name="c", dtype="str", generator="choice",
                       params={"values": values, "weights": weights}),
        ])],
    )


# --------------------------------------------------------------------------
# the reported failure
# --------------------------------------------------------------------------

def test_weights_rounded_to_four_places_are_accepted():
    """0.9999 is what the profiler writes; numpy rejects it unnormalized."""
    weights = [0.3333, 0.3333, 0.3333]
    assert sum(weights) != 1.0
    frames = GenerationEngine(
        _choice_spec(["a", "b", "c"], weights)
    ).run(vectorized=True).to_frames()
    assert len(frames["t"]) == 500
    assert set(frames["t"]["c"]) <= {"a", "b", "c"}


def test_weights_summing_above_one_are_accepted():
    weights = [0.3334, 0.3334, 0.3334]
    assert sum(weights) > 1.0
    GenerationEngine(_choice_spec(["a", "b", "c"], weights)).run(vectorized=True)


def test_integer_relative_weights_are_accepted_like_random_choices():
    """The row-by-row path has always taken these; the fast path must too."""
    spec = _choice_spec(["a", "b", "c"], [3, 1, 1])
    for vectorized in (False, True):
        GenerationEngine(spec).run(vectorized=vectorized)


@pytest.mark.parametrize("seed", range(12))
def test_profiled_categorical_specs_survive_the_vectorized_path(seed):
    """End to end: profile arbitrary data, then generate with --vectorized."""
    rng = np.random.default_rng(seed)
    k = int(rng.integers(3, 12))
    p = rng.dirichlet(np.ones(k))
    n = int(rng.integers(700, 5000))
    df = pd.DataFrame({"c": rng.choice([f"v{i}" for i in range(k)], size=n, p=p)})
    spec = DatasetProfiler(df, name="t", sample=None, seed=seed).profile()
    spec.tables[0].row_count = 200
    GenerationEngine(spec).run(vectorized=True)


def test_both_paths_agree_on_the_resulting_distribution():
    values, weights = ["a", "b", "c"], [0.5999, 0.3, 0.1]
    rows = 20_000
    out = {}
    for vectorized in (False, True):
        frames = GenerationEngine(
            _choice_spec(values, weights, rows=rows, seed=11)
        ).run(vectorized=vectorized).to_frames()
        out[vectorized] = frames["t"]["c"].value_counts(normalize=True)
    for v in values:
        assert abs(out[True][v] - out[False][v]) < 0.02


def test_bool_weights_are_normalized_too():
    spec = Spec(
        metadata=SpecMetadata(name="b"), settings=Settings(seed=2),
        tables=[TableSpec(name="t", row_count=4000, columns=[
            ColumnSpec(name="flag", dtype="bool", generator="bool",
                       params={"weights": [7, 3]}),
        ])],
    )
    frames = GenerationEngine(spec).run(vectorized=True).to_frames()
    assert 0.65 < frames["t"]["flag"].mean() < 0.75


# --------------------------------------------------------------------------
# the helper itself
# --------------------------------------------------------------------------

def test_normalize_weights_sums_to_one_within_numpy_tolerance():
    for weights in ([0.3333, 0.3333, 0.3333], [3, 1, 1], [1e-9, 1.0],
                    [0.238, 0.1754, 0.1568, 0.1128, 0.0662]):
        p = _normalize_weights(weights, len(weights))
        # numpy's own check is abs(sum(p) - 1) < sqrt(eps)
        assert abs(p.sum() - 1.0) < np.sqrt(np.finfo(np.float64).eps)


def test_normalize_weights_passes_none_through():
    assert _normalize_weights(None, 3) is None
    assert _normalize_weights([], 3) is None


@pytest.mark.parametrize("weights,k", [
    ([0.5, 0.5], 3),        # wrong count
    ([1.0, -0.5], 2),       # negative
    ([0.0, 0.0], 2),        # all zero
    ([float("nan"), 1.0], 2),
    ([float("inf"), 1.0], 2),
])
def test_normalize_weights_rejects_nonsense(weights, k):
    with pytest.raises(SpecError):
        _normalize_weights(weights, k)
