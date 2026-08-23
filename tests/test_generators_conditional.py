"""The ``conditional`` generator: sample Y from P(Y | X), not from P(Y).

Independent per-column sampling reproduces every marginal and destroys every
joint. On the CFPB source the marginals of Product and Sub-product score 0.98
and 0.96, and the pair scores 0.16 -- the hierarchy is gone. This generator is
the mechanism that fixes that: for a dependency X -> Y it draws Y from the
observed conditional distribution given the X value already in the row.
"""
import math
import random

import pytest
from faker import Faker

from syntab.generators import GenContext, build_generator, conditional_params
from syntab.spec import ColumnSpec, Settings, Spec, SpecMetadata, TableSpec
from syntab.engine import GenerationEngine
from syntab.spec import SpecError


def _ctx(row, params, seed=0):
    return GenContext(row=dict(row), parents={}, rng=random.Random(seed),
                      faker=Faker(), params=params)


HIERARCHY = {
    "on": ["sub_product"],
    "keys": [["checking"], ["savings"], ["fixed rate"], [None]],
    "values": [["bank account"], ["bank account"], ["mortgage"],
               ["bank account", "mortgage"]],
    "weights": [[1.0], [1.0], [1.0], [0.5, 0.5]],
    "default": {"values": ["bank account", "mortgage"], "weights": [0.5, 0.5]},
}


# --------------------------------------------------------------------------
# the core behaviour
# --------------------------------------------------------------------------

def test_a_deterministic_dependency_is_reproduced_exactly():
    gen = build_generator("conditional")
    params = dict(HIERARCHY)
    for sub, expected in (("checking", "bank account"),
                          ("savings", "bank account"),
                          ("fixed rate", "mortgage")):
        for seed in range(5):
            assert gen(_ctx({"sub_product": sub}, params, seed)) == expected


def test_a_null_determinant_is_a_key_like_any_other():
    gen = build_generator("conditional")
    seen = {gen(_ctx({"sub_product": None}, dict(HIERARCHY), s)) for s in range(40)}
    assert seen == {"bank account", "mortgage"}


def test_nan_matches_the_null_key():
    """A float NaN out of pandas must find the JSON ``null`` key."""
    gen = build_generator("conditional")
    assert gen(_ctx({"sub_product": float("nan")}, dict(HIERARCHY), 0)) in (
        "bank account", "mortgage")


def test_an_unseen_key_falls_back_to_the_default():
    gen = build_generator("conditional")
    assert gen(_ctx({"sub_product": "never seen"}, dict(HIERARCHY), 0)) in (
        "bank account", "mortgage")


def test_an_unseen_key_without_a_default_is_an_error():
    params = {k: v for k, v in HIERARCHY.items() if k != "default"}
    gen = build_generator("conditional")
    with pytest.raises(ValueError, match="no conditional distribution"):
        gen(_ctx({"sub_product": "never seen"}, params, 0))


def test_weights_within_a_key_are_honoured():
    params = {
        "on": ["x"],
        "keys": [["a"]],
        "values": [["p", "q"]],
        "weights": [[0.9, 0.1]],
    }
    gen = build_generator("conditional")
    draws = [gen(_ctx({"x": "a"}, params, None)) for _ in range(4000)]
    assert 0.85 < draws.count("p") / len(draws) < 0.95


def test_unnormalized_weights_are_normalized():
    params = {"on": ["x"], "keys": [["a"]], "values": [["p", "q"]],
              "weights": [[9, 1]]}
    gen = build_generator("conditional")
    draws = [gen(_ctx({"x": "a"}, params, None)) for _ in range(4000)]
    assert 0.85 < draws.count("p") / len(draws) < 0.95


def test_a_null_dependent_value_is_generated_as_null():
    params = {"on": ["x"], "keys": [["a"]], "values": [[None]], "weights": [[1.0]]}
    gen = build_generator("conditional")
    assert gen(_ctx({"x": "a"}, params, 0)) is None


def test_a_composite_determinant_matches_on_the_tuple():
    params = {
        "on": ["a", "b"],
        "keys": [["1", "x"], ["1", "y"], ["2", "x"]],
        "values": [["A"], ["B"], ["C"]],
        "weights": [[1.0], [1.0], [1.0]],
    }
    gen = build_generator("conditional")
    assert gen(_ctx({"a": "1", "b": "x"}, params, 0)) == "A"
    assert gen(_ctx({"a": "1", "b": "y"}, params, 0)) == "B"
    assert gen(_ctx({"a": "2", "b": "x"}, params, 0)) == "C"


def test_a_key_type_that_drifted_through_json_still_matches():
    """A spec round-tripped through JSON can hand back 2139 where the row has '2139'."""
    params = {"on": ["zip"], "keys": [[2139]], "values": [["MA"]],
              "weights": [[1.0]]}
    gen = build_generator("conditional")
    assert gen(_ctx({"zip": "2139"}, params, 0)) == "MA"


# --------------------------------------------------------------------------
# malformed params are rejected, not silently mis-sampled
# --------------------------------------------------------------------------

@pytest.mark.parametrize("params, match", [
    ({"keys": [["a"]], "values": [["A"]], "weights": [[1.0]]}, "params.on"),
    ({"on": ["x"], "values": [["A"]], "weights": [[1.0]]}, "params.keys"),
    ({"on": ["x"], "keys": [["a"]], "weights": [[1.0]]}, "params.values"),
    ({"on": ["x"], "keys": [["a"], ["b"]], "values": [["A"]],
      "weights": [[1.0]]}, "same length"),
    ({"on": ["x"], "keys": [["a"]], "values": [["A", "B"]],
      "weights": [[1.0]]}, "weights"),
    ({"on": ["x", "y"], "keys": [["a"]], "values": [["A"]],
      "weights": [[1.0]]}, "determinant column"),
    ({"on": ["x"], "keys": [["a"]], "values": [[]], "weights": [[]]},
     "at least one value"),
    ({"on": ["x"], "keys": [["a"]], "values": [["A"]], "weights": [[0.0]]},
     "must not sum to zero"),
    ({"on": ["x"], "keys": [["a"]], "values": [["A"]], "weights": [[-1.0]]},
     "non-negative"),
])
def test_malformed_conditional_params_are_rejected(params, match):
    gen = build_generator("conditional")
    with pytest.raises(ValueError, match=match):
        gen(_ctx({"x": "a", "y": "b"}, params, 0))


def test_the_index_is_built_once_and_reused():
    """208k rows must not rebuild the lookup table 208k times."""
    params = dict(HIERARCHY)
    gen = build_generator("conditional")
    gen(_ctx({"sub_product": "checking"}, params, 0))
    built = params.get("__conditional_index__")
    assert built is not None
    gen(_ctx({"sub_product": "savings"}, params, 0))
    assert params["__conditional_index__"] is built


# --------------------------------------------------------------------------
# conditional_params: the builder the profiler uses
# --------------------------------------------------------------------------

def test_conditional_params_reproduces_the_observed_joint():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({
        "sub": ["checking"] * 30 + ["savings"] * 20 + ["fixed rate"] * 50,
        "prod": ["bank"] * 50 + ["mortgage"] * 50,
    })
    params = conditional_params(frame, ["sub"], "prod")
    assert params["on"] == ["sub"]
    index = dict(zip((tuple(k) for k in params["keys"]),
                     zip(params["values"], params["weights"])))
    assert index[("checking",)][0] == ["bank"]
    assert index[("fixed rate",)][0] == ["mortgage"]


def test_conditional_params_carries_nulls_on_both_sides():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"sub": ["a", None, None], "prod": ["A", None, "B"]})
    params = conditional_params(frame, ["sub"], "prod")
    keys = [tuple(k) for k in params["keys"]]
    assert (None,) in keys
    values = params["values"][keys.index((None,))]
    assert None in values and "B" in values


def test_conditional_params_default_is_the_marginal():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"sub": ["a"] * 3 + ["b"], "prod": ["A", "A", "A", "B"]})
    params = conditional_params(frame, ["sub"], "prod")
    default = params["default"]
    total = dict(zip(default["values"], default["weights"]))
    assert math.isclose(total["A"], 0.75, abs_tol=1e-6)
    assert math.isclose(total["B"], 0.25, abs_tol=1e-6)


def test_conditional_params_weights_sum_to_one_per_key():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"sub": list("aaabbb"), "prod": list("AABBBC")})
    params = conditional_params(frame, ["sub"], "prod")
    for w in params["weights"]:
        assert math.isclose(sum(w), 1.0, abs_tol=1e-9)


# --------------------------------------------------------------------------
# end to end through the engine
# --------------------------------------------------------------------------

def _hierarchy_spec(seed=7, row_count=600, depends_on=("sub_product",)):
    return Spec(
        metadata=SpecMetadata(name="complaints"),
        settings=Settings(seed=seed),
        tables=[TableSpec(
            name="complaints",
            row_count=row_count,
            columns=[
                # Declared AFTER the column it depends on is not required:
                # the engine orders on the dependency, not on declaration.
                ColumnSpec(
                    name="product", dtype="str", generator="conditional",
                    depends_on=list(depends_on),
                    params=dict(HIERARCHY),
                ),
                ColumnSpec(
                    name="sub_product", dtype="str", generator="choice",
                    params={"values": ["checking", "savings", "fixed rate"],
                            "weights": [0.3, 0.2, 0.5]},
                ),
            ],
        )],
    )


def test_engine_regenerates_the_hierarchy():
    frame = GenerationEngine(_hierarchy_spec()).run().to_frames()["complaints"]
    pairs = set(zip(frame["sub_product"], frame["product"]))
    assert pairs <= {("checking", "bank account"), ("savings", "bank account"),
                     ("fixed rate", "mortgage")}
    assert len(pairs) == 3


def test_engine_orders_the_determinant_first_without_depends_on():
    """A hand-written spec that forgets depends_on must still work.

    The dependency is in ``params.on``; the engine reads it rather than
    requiring the author to restate it.
    """
    spec = _hierarchy_spec(depends_on=())
    frame = GenerationEngine(spec).run().to_frames()["complaints"]
    pairs = set(zip(frame["sub_product"], frame["product"]))
    assert pairs <= {("checking", "bank account"), ("savings", "bank account"),
                     ("fixed rate", "mortgage")}


def test_engine_rejects_an_unknown_determinant_column():
    spec = _hierarchy_spec()
    spec.tables[0].columns[0].params = dict(HIERARCHY, on=["nope"])
    spec.tables[0].columns[0].depends_on = []
    with pytest.raises(SpecError, match="nope"):
        GenerationEngine(spec)


def test_engine_rejects_a_conditional_cycle():
    spec = _hierarchy_spec()
    spec.tables[0].columns[1].generator = "conditional"
    spec.tables[0].columns[1].params = {
        "on": ["product"], "keys": [["bank account"]],
        "values": [["checking"]], "weights": [[1.0]],
    }
    with pytest.raises(SpecError, match="[Cc]yclic"):
        GenerationEngine(spec).run()


def test_conditional_column_falls_off_the_vectorized_path():
    """Correctness before speed: the fast path has no notion of row context."""
    frame = GenerationEngine(_hierarchy_spec()).run(
        vectorized=True).to_frames()["complaints"]
    pairs = set(zip(frame["sub_product"], frame["product"]))
    assert pairs <= {("checking", "bank account"), ("savings", "bank account"),
                     ("fixed rate", "mortgage")}
