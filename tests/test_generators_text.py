"""Text generation samples from observed distributions, not from the maximum.

The defect these pin: a profiled free-text column resolved to the ``string``
generator with ``length`` set to the MAXIMUM observed length, and ``string``
filled that width from a uniform ``[a-zA-Z0-9]`` alphabet. Every synthetic
value came out at exactly the longest length in the source, made of uniform
alphanumeric noise -- no spaces, no punctuation, digits as common as vowels.

Both halves are fixed here: lengths are drawn from the observed length
distribution, and characters from an observed character distribution when one
is supplied.
"""
import random

import numpy as np
import pandas as pd
import pytest
from faker import Faker

from syntab.engine import GenerationEngine
from syntab.generators import GenContext, empirical_text_params
from syntab.generators import BUILTINS
from syntab.infer import resolve_generator
from syntab.profiler import DatasetProfiler
from syntab.validator import validate

SEED = 31
WORDS = ["the", "account", "was", "closed", "without", "notice", "and",
         "balance", "remains", "disputed", "since", "january", "letter"]


def _ctx(params, seed=SEED):
    return GenContext(row={}, parents={}, rng=random.Random(seed),
                      faker=Faker(), params=params)


def _gen(params, n=400, seed=SEED):
    ctx = _ctx(params, seed)
    return [BUILTINS["string"](ctx) for _ in range(n)]


def _prose(n=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    return [" ".join(rng.choice(WORDS, int(rng.integers(3, 20)))) for _ in range(n)]


# ---------------------------------------------------------------------------
# The string generator
# ---------------------------------------------------------------------------

def test_scalar_length_still_produces_a_fixed_width():
    """Back-compat: a hand-authored spec asking for length=12 gets length 12."""
    assert {len(v) for v in _gen({"length": 12})} == {12}


def test_min_and_max_length_produce_a_spread():
    lengths = {len(v) for v in _gen({"min_length": 5, "max_length": 20})}
    assert min(lengths) >= 5 and max(lengths) <= 20
    assert len(lengths) > 5, "lengths should vary across the range"


def test_length_values_sample_the_discrete_distribution():
    values = _gen({"length_values": [3, 30], "length_weights": [0.9, 0.1]}, n=2000)
    lengths = [len(v) for v in values]
    assert set(lengths) == {3, 30}
    short = sum(1 for n in lengths if n == 3) / len(lengths)
    assert short == pytest.approx(0.9, abs=0.05)


def test_length_histogram_samples_the_observed_shape():
    values = _gen({"length_edges": [0.0, 10.0, 20.0],
                   "length_counts": [1.0, 9.0]}, n=2000)
    lengths = [len(v) for v in values]
    assert max(lengths) <= 20
    upper = sum(1 for n in lengths if n > 10) / len(lengths)
    assert upper == pytest.approx(0.9, abs=0.06)


def test_char_values_replace_the_uniform_alphanumeric_alphabet():
    values = _gen({"length": 50, "char_values": ["a", " "],
                   "char_weights": [0.5, 0.5]}, n=200)
    assert set("".join(values)) == {"a", " "}


def test_char_weights_are_respected():
    values = _gen({"length": 200, "char_values": ["x", "y"],
                   "char_weights": [0.8, 0.2]}, n=200)
    joined = "".join(values)
    assert joined.count("x") / len(joined) == pytest.approx(0.8, abs=0.03)


def test_default_alphabet_is_unchanged_when_nothing_is_supplied():
    values = _gen({"length": 10})
    assert set("".join(values)) <= set(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")


# ---------------------------------------------------------------------------
# empirical_text_params
# ---------------------------------------------------------------------------

def test_empirical_text_params_captures_lengths_and_characters():
    source = _prose()
    params = empirical_text_params(source)
    assert params["min_length"] == min(len(s) for s in source)
    assert params["max_length"] == max(len(s) for s in source)
    assert " " in params["char_values"], "spaces must survive; prose has spaces"
    assert sum(params["char_weights"]) == pytest.approx(1.0)


def test_empirical_text_params_round_trips_through_the_generator():
    """The whole point: params derived from a source reproduce that source."""
    source = _prose()
    synth = _gen(empirical_text_params(source), n=len(source))

    real_s = pd.Series(source)
    synth_s = pd.Series(synth)
    # Mean length within 10%.
    assert abs(real_s.str.len().mean() - synth_s.str.len().mean()) \
        / real_s.str.len().mean() < 0.1
    # And the lengths actually vary.
    assert synth_s.str.len().nunique() > 20
    # The validator's text check passes on it.
    report = validate(pd.DataFrame({"t": real_s}), pd.DataFrame({"t": synth_s}))
    assert report.overall_pass, report.to_text()


def test_empirical_text_params_handles_nulls_and_empties():
    params = empirical_text_params(["abc", None, "", float("nan"), "de"])
    assert params["min_length"] == 0
    assert params["max_length"] == 3


def test_empirical_text_params_caps_the_character_set():
    source = ["".join(chr(0x4E00 + i) for i in range(50)) for _ in range(10)]
    params = empirical_text_params(source, max_characters=10)
    assert len(params["char_values"]) == 10
    assert sum(params["char_weights"]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# infer: the observed RANGE, not the maximum
# ---------------------------------------------------------------------------

def test_infer_resolves_a_length_profile_to_a_range_not_the_maximum():
    from syntab.spec import ColumnProfile, ColumnSpec

    col = ColumnSpec(name="t", dtype="str", generator="auto",
                     profile=ColumnProfile(length=(11, 151)))
    gen, params = resolve_generator(col)
    assert gen == "string"
    assert params["min_length"] == 11
    assert params["max_length"] == 151
    assert "length" not in params, (
        "a scalar length pins every value to one width -- that is the defect")


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

def test_profiled_free_text_no_longer_generates_at_a_single_length():
    real = pd.DataFrame({"narrative": _prose()})
    spec = DatasetProfiler(real, name="t", sample=None, seed=1).profile()
    synth = GenerationEngine(spec).run().to_frames()["t"]

    real_len = real["narrative"].str.len()
    synth_len = synth["narrative"].astype(str).str.len()

    assert synth_len.nunique() > 20, (
        f"all values at {synth_len.unique()[:5]} -- generated at a fixed length")
    assert synth_len.max() <= real_len.max()
    assert synth_len.min() >= real_len.min()
    # Mean length is in the right neighbourhood rather than pinned to the max.
    assert abs(synth_len.mean() - real_len.mean()) / real_len.mean() < 0.15
