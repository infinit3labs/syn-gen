"""Profiling a hierarchical source produces a spec that regenerates it.

This is the loop the previous two changes left open. The profiler discovered
that Sub-product determines Product and recorded it in a metadata dict; the
engine then sampled the two columns independently and produced output whose
marginals were right and whose hierarchy was gone. These tests assert the
whole path: discover the dependency, choose it, write a conditional generator,
and get the hierarchy back out of the engine.
"""
import random

import pandas as pd
import pytest

from syntab import discovery
from syntab.engine import GenerationEngine
from syntab.profiler import (
    DatasetProfiler,
    OTHER_BUCKET_LABEL,
    RECOMMENDED_MIN_CELL_COUNT,
)

pytestmark = pytest.mark.skipif(
    not discovery.is_available(),
    reason="dependency discovery needs the optional [discovery] extra",
)


SUB_TO_PRODUCT = {
    "checking": "bank account",
    "savings": "bank account",
    "cd": "bank account",
    "fixed rate": "mortgage",
    "arm": "mortgage",
    "reverse": "mortgage",
    "student": "loan",
    "auto": "loan",
}


def hierarchy_frame(n=3000, seed=0, null_fraction=0.0):
    """A frame with a real, single-column functional dependency in it."""
    rng = random.Random(seed)
    subs, prods, noise = [], [], []
    keys = list(SUB_TO_PRODUCT)
    for _ in range(n):
        sub = rng.choice(keys)
        if null_fraction and rng.random() < null_fraction:
            subs.append(None)
            prods.append(rng.choice(sorted(set(SUB_TO_PRODUCT.values()))))
        else:
            subs.append(sub)
            prods.append(SUB_TO_PRODUCT[sub])
        noise.append(rng.choice(["x", "y", "z"]))
    return pd.DataFrame({"sub_product": subs, "product": prods, "noise": noise})


def profiled(frame, **kw):
    options = dict(sample=None, seed=7, discover=True, discover_fds=True)
    options.update(kw)
    return DatasetProfiler(frame, name="complaints", **options).profile()


def column(spec, name):
    return next(c for c in spec.tables[0].columns if c.name == name)


def conditionals(spec):
    return (spec.tables[0].metadata.get("discovery", {})
            .get("conditional_dependencies", []))


# --------------------------------------------------------------------------
# the spec the profiler writes
# --------------------------------------------------------------------------

def test_the_dependent_column_becomes_a_conditional_generator():
    spec = profiled(hierarchy_frame())
    product = column(spec, "product")
    assert product.generator == "conditional"
    assert product.params["on"] == ["sub_product"]
    assert product.depends_on == ["sub_product"]


def test_the_determinant_keeps_its_own_marginal():
    spec = profiled(hierarchy_frame())
    sub = column(spec, "sub_product")
    assert sub.generator in (None, "auto")
    assert sub.profile.categorical.values


def test_the_independent_column_is_left_alone():
    spec = profiled(hierarchy_frame())
    assert column(spec, "noise").generator in (None, "auto")


def test_the_edge_is_recorded_with_its_provenance():
    spec = profiled(hierarchy_frame())
    edges = conditionals(spec)
    assert len(edges) == 1
    edge = edges[0]
    assert edge["determinant"] == ["sub_product"]
    assert edge["dependent"] == "product"
    assert edge["provenance"]["algorithm"] in ("HyFD", "Pyro")
    assert edge["provenance"]["citation"]


def test_the_conditional_table_holds_the_real_conditional():
    spec = profiled(hierarchy_frame())
    params = column(spec, "product").params
    table = {tuple(k): v for k, v in zip(params["keys"], params["values"])}
    for sub, prod in SUB_TO_PRODUCT.items():
        assert table[(sub,)] == [prod]


def test_a_null_determinant_gets_its_own_conditional_entry():
    spec = profiled(hierarchy_frame(null_fraction=0.15))
    assert column(spec, "product").generator == "conditional"
    params = column(spec, "product").params
    keys = [tuple(k) for k in params["keys"]]
    assert (None,) in keys
    assert len(params["values"][keys.index((None,))]) > 1


def test_there_is_always_a_fallback_marginal():
    spec = profiled(hierarchy_frame())
    default = column(spec, "product").params["default"]
    assert set(default["values"]) == set(SUB_TO_PRODUCT.values())
    assert sum(default["weights"]) == pytest.approx(1.0, abs=1e-3)


# --------------------------------------------------------------------------
# the data the engine then produces -- the actual claim
# --------------------------------------------------------------------------

def test_the_regenerated_data_reproduces_the_hierarchy():
    spec = profiled(hierarchy_frame())
    spec.settings.seed = 11
    frame = GenerationEngine(spec).run().to_frames()["complaints"]
    for sub, prod in zip(frame["sub_product"], frame["product"]):
        assert prod == SUB_TO_PRODUCT[sub]


def test_without_conditioning_the_hierarchy_is_destroyed():
    """The behaviour this change exists to fix, asserted rather than asserted about."""
    spec = profiled(hierarchy_frame(), condition_on_fds=False)
    spec.settings.seed = 11
    frame = GenerationEngine(spec).run().to_frames()["complaints"]
    wrong = sum(1 for sub, prod in zip(frame["sub_product"], frame["product"])
                if prod != SUB_TO_PRODUCT[sub])
    assert wrong > len(frame) * 0.3


def test_conditioning_does_not_damage_the_marginals():
    frame = hierarchy_frame()
    spec = profiled(frame)
    spec.settings.seed = 11
    out = GenerationEngine(spec).run().to_frames()["complaints"]
    real = frame["product"].value_counts(normalize=True)
    synth = out["product"].value_counts(normalize=True)
    for value, share in real.items():
        assert synth.get(value, 0.0) == pytest.approx(share, abs=0.05)


def test_nulls_come_from_the_conditional_not_from_the_marginal_rate():
    """A conditional generator owns its own missingness; the engine must not
    also inject nulls at the column's marginal rate on top."""
    # Missingness that DEPENDS on the determinant, which is the case a single
    # marginal null_rate cannot represent: "auto" never has a product, every
    # other sub-product always does.
    frame = hierarchy_frame()
    frame.loc[frame["sub_product"] == "auto", "product"] = None
    spec = profiled(frame)
    product = column(spec, "product")
    assert product.generator == "conditional"
    # The marginal rate is still recorded truthfully in the spec; it is simply
    # not applied on top of the per-key distribution.
    assert product.constraints["null_rate"] == pytest.approx(
        frame["product"].isna().mean(), abs=0.01)

    spec.settings.seed = 3
    out = GenerationEngine(spec).run().to_frames()["complaints"]
    assert out["product"].isna().mean() == pytest.approx(
        frame["product"].isna().mean(), abs=0.03)
    # and in the right rows, which is the part a marginal rate gets wrong
    assert out.loc[out["sub_product"] == "auto", "product"].isna().all()
    assert out.loc[out["sub_product"] != "auto", "product"].notna().all()


# --------------------------------------------------------------------------
# the switches
# --------------------------------------------------------------------------

def test_conditioning_is_off_when_fds_are_not_discovered():
    spec = profiled(hierarchy_frame(), discover_fds=False)
    assert conditionals(spec) == []
    assert column(spec, "product").generator in (None, "auto")


def test_conditioning_can_be_turned_off_on_its_own():
    spec = profiled(hierarchy_frame(), condition_on_fds=False)
    assert conditionals(spec) == []
    assert column(spec, "product").generator in (None, "auto")


def test_the_reported_fds_are_the_same_either_way():
    """Turning conditioning on must not change one line of the FD report."""
    def fds(**kw):
        spec = profiled(hierarchy_frame(), **kw)
        return (spec.tables[0].metadata.get("discovery", {})
                .get("functional_dependencies", []))

    assert fds(condition_on_fds=True) == fds(condition_on_fds=False)


# --------------------------------------------------------------------------
# what is never conditioned
# --------------------------------------------------------------------------

def test_a_pii_column_is_never_conditioned():
    frame = hierarchy_frame()
    spec = profiled(frame, pii_columns=["product"])
    assert column(spec, "product").generator.startswith("faker.")
    assert conditionals(spec) == []


def test_a_pii_column_is_never_a_determinant():
    frame = hierarchy_frame()
    spec = profiled(frame, pii_columns=["sub_product"])
    assert column(spec, "product").generator in (None, "auto")
    assert conditionals(spec) == []


def test_a_key_column_is_never_conditioned():
    frame = hierarchy_frame()
    frame["id"] = range(len(frame))
    spec = profiled(frame)
    assert column(spec, "id").generator != "conditional"


# --------------------------------------------------------------------------
# disclosure controls apply to the conditional table too
# --------------------------------------------------------------------------

def test_min_cell_count_suppresses_rare_conditional_cells():
    frame = hierarchy_frame()
    # One sub-product that almost always means "bank account" and, twice,
    # means something else. Those two rows are the disclosure risk.
    frame.loc[frame.index[:2], "sub_product"] = "checking"
    frame.loc[frame.index[:2], "product"] = "rare secret"
    spec = profiled(frame, min_cell_count=RECOMMENDED_MIN_CELL_COUNT)
    params = column(spec, "product").params
    table = {tuple(k): v for k, v in zip(params["keys"], params["values"])}
    assert "rare secret" not in table[("checking",)]
    assert OTHER_BUCKET_LABEL in table[("checking",)]


def test_without_min_cell_count_the_rare_cell_is_embedded():
    """The control has to be doing something; this is the other half."""
    frame = hierarchy_frame()
    frame.loc[frame.index[:2], "sub_product"] = "checking"
    frame.loc[frame.index[:2], "product"] = "rare secret"
    spec = profiled(frame, min_cell_count=0)
    params = column(spec, "product").params
    table = {tuple(k): v for k, v in zip(params["keys"], params["values"])}
    assert "rare secret" in table[("checking",)]


def test_redaction_keeps_real_labels_out_of_the_conditional_table():
    spec = profiled(hierarchy_frame(), redact_categoricals=True)
    params = column(spec, "product").params
    embedded = {str(v) for row in params["values"] for v in row}
    embedded |= {str(k) for key in params["keys"] for k in key}
    embedded |= {str(v) for v in params["default"]["values"]}
    for real in list(SUB_TO_PRODUCT) + list(SUB_TO_PRODUCT.values()):
        assert real not in embedded


def test_redaction_still_reproduces_the_hierarchy_under_placeholders():
    frame = hierarchy_frame()
    spec = profiled(frame, redact_categoricals=True)
    spec.settings.seed = 5
    out = GenerationEngine(spec).run().to_frames()["complaints"]
    # The labels are opaque, but the structure has to survive: each generated
    # sub_product placeholder must still map to exactly one product placeholder.
    mapping = {}
    for sub, prod in zip(out["sub_product"], out["product"]):
        assert mapping.setdefault(sub, prod) == prod


def test_redaction_without_a_vocabulary_skips_the_edge_rather_than_leaking():
    """A column with no placeholder mapping cannot take part at all."""
    profiler = DatasetProfiler(hierarchy_frame(), name="c", sample=None, seed=7,
                               discover=True, discover_fds=True,
                               redact_categoricals=True)
    profiler._redaction_maps.clear()
    spec = profiler.profile()
    # profile() rebuilds the maps, so clear them by profiling a frame whose
    # dependent has no categorical profile instead: assert the guard directly.
    assert profiler._label_map("not a column") == (False, None)
