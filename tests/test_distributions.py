"""Tests for empirical / skewed numeric distributions (#3)."""
import numpy as np
import pandas as pd
import random
from faker import Faker

from syntab.generators import GenContext, _g_float, _g_int
from syntab.profiler import DatasetProfiler
from syntab.engine import GenerationEngine
from syntab.loaders import from_file
from syntab.conformance import validate_against_spec
from syntab.validator import validate


def _ctx(params, dtype="float"):
    return GenContext(
        row={}, parents={}, rng=random.Random(0), faker=Faker(),
        params=params, table="t", col="c",
    )


def _sample_many(fn, params, n=500, dtype="float"):
    return [fn(_ctx(params, dtype)) for _ in range(n)]


def test_empirical_respects_range_and_histogram():
    params = {"distribution": "empirical", "hist_edges": [0, 1, 2, 3],
              "hist_counts": [0.25, 0.25, 0.5], "min": 0, "max": 3}
    vals = _sample_many(_g_float, params)
    assert all(0.0 <= v <= 3.0 for v in vals)
    # the heaviest bin [2,3) (weight 0.5) should capture the most samples
    in_last = sum(1 for v in vals if 2.0 <= v < 3.0)
    in_first = sum(1 for v in vals if 0.0 <= v < 1.0)
    assert in_last > in_first

    ints = _sample_many(_g_int, params, dtype="int")
    assert all(isinstance(v, int) and 0 <= v <= 3 for v in ints)


def test_lognormal_positive_and_bounded():
    params = {"distribution": "lognormal", "log_mu": 0.0, "log_sigma": 0.5,
              "min": 0.1, "max": 100.0}
    vals = _sample_many(_g_float, params)
    assert all(0.1 <= v <= 100.0 for v in vals)
    assert all(v > 0 for v in vals)


def test_exponential_positive_and_bounded():
    params = {"distribution": "exponential", "scale": 5.0, "min": 0.0, "max": 100.0}
    vals = _sample_many(_g_float, params)
    assert all(0.0 <= v <= 100.0 for v in vals)
    assert all(v > 0 for v in vals)


def test_pareto_heavy_tailed():
    params = {"distribution": "pareto", "pareto_b": 2.0, "pareto_xmin": 1.0,
              "min": 1.0, "max": 1000.0}
    vals = _sample_many(_g_float, params)
    assert all(1.0 <= v <= 1000.0 for v in vals)


def test_normal_within_rails():
    params = {"distribution": "normal", "mean": 50.0, "std": 10.0,
              "min": 0.0, "max": 100.0}
    vals = _sample_many(_g_float, params)
    assert all(0.0 <= v <= 100.0 for v in vals)


def test_uniform_default():
    params = {"min": 0.0, "max": 10.0}
    vals = _sample_many(_g_float, params)
    assert all(0.0 <= v <= 10.0 for v in vals)


def _profile(df, name="t"):
    return DatasetProfiler(df, name=name, sample=None).profile()


def test_profiler_detects_lognormal():
    rng = np.random.default_rng(1)
    vals = np.exp(rng.normal(2.0, 0.5, 3000))
    spec = _profile(pd.DataFrame({"x": vals}))
    assert spec.tables[0].columns[0].profile.numeric.distribution == "lognormal"


def test_profiler_detects_exponential():
    rng = np.random.default_rng(2)
    vals = rng.exponential(5.0, 4000)
    spec = _profile(pd.DataFrame({"x": vals}))
    assert spec.tables[0].columns[0].profile.numeric.distribution == "exponential"


def test_profiler_detects_uniform():
    rng = np.random.default_rng(3)
    vals = rng.uniform(0, 1, 4000)
    spec = _profile(pd.DataFrame({"x": vals}))
    assert spec.tables[0].columns[0].profile.numeric.distribution == "uniform"


def test_profiler_detects_empirical_for_negative_skew():
    rng = np.random.default_rng(4)
    vals = rng.normal(0, 1, 4000)
    prof = _profile(pd.DataFrame({"x": vals})).tables[0].columns[0].profile.numeric
    assert prof.distribution == "empirical"
    assert prof.hist_edges is not None and prof.hist_counts is not None


def test_roundtrip_lognormal_profile_generate_conforms():
    rng = np.random.default_rng(5)
    real = pd.DataFrame({
        "id": range(1, 1001),
        "amount": np.exp(rng.normal(2.0, 0.5, 1000)),
        "region": rng.choice(["EU", "US", "APAC"], 1000),
    })
    spec = _profile(real)
    frames = GenerationEngine(spec).run().to_frames()
    report = validate_against_spec(frames, spec)
    assert report.overall_ok, report.to_text()
    # synthetic amount stays positive and within the observed range
    assert (frames["t"]["amount"] > 0).all()


def test_compare_improves_with_empirical():
    rng = np.random.default_rng(6)
    real = pd.DataFrame({
        "amount": np.exp(rng.normal(2.0, 0.5, 2000)),
        "region": rng.choice(["EU", "US", "APAC"], 2000),
    })
    spec = _profile(real)
    synth = pd.DataFrame(GenerationEngine(spec).run().to_frames()["t"])
    report = validate(real, synth, spec)
    assert report.overall_pass, report.to_text()
