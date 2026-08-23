"""Text columns can fail the validator.

``_compare_text`` has always computed the two metrics that catch a broken
text generator -- a length-mean difference and a character-distribution TVD --
and has always thrown the result away. The status was hard-coded down from
``fail`` to ``warn``, and the overall verdict excluded text columns outright,
on the rationale that free text "cannot be matched semantically".

That rationale does not survive contact with the metrics actually being
computed. Neither one is semantic. Length distribution and character
frequency are structural properties, they are exactly what a broken text
generator gets wrong, and a check that computes the right number and then
refuses to act on it is worse than no check: it reports a clean bill of
health it has the evidence to contradict.

These tests pin the un-suppressed behaviour, so it cannot quietly come back.
"""
import numpy as np
import pandas as pd

from syntab.validator import validate

SEED = 21
ALNUM = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def _prose(n=1500, seed=SEED):
    """Free text with a spread of lengths and an English-ish alphabet."""
    rng = np.random.default_rng(seed)
    words = ["the", "account", "was", "closed", "without", "notice", "and",
             "the", "balance", "remains", "disputed", "since", "january"]
    out = []
    for _ in range(n):
        k = int(rng.integers(3, 20))
        out.append(" ".join(rng.choice(words, k)))
    return pd.Series(out)


def _at_max_length_uniform_alphabet(real, n=1500, seed=SEED):
    """The defect: every value at the maximum observed length, uniform chars."""
    rng = np.random.default_rng(seed + 1)
    width = int(real.str.len().max())
    return pd.Series(["".join(rng.choice(list(ALNUM), width)) for _ in range(n)])


def test_broken_text_column_is_reported_as_fail_not_warn():
    real = _prose()
    df_real = pd.DataFrame({"narrative": real})
    df_synth = pd.DataFrame({"narrative": _at_max_length_uniform_alphabet(real)})

    report = validate(df_real, df_synth)
    col = next(c for c in report.columns if c.name == "narrative")

    assert col.kind == "text"
    assert col.status == "fail", (
        f"text column downgraded to {col.status!r} at distance {col.distance}"
    )
    assert report.n_fail >= 1


def test_broken_text_column_fails_the_overall_verdict():
    """A dataset whose only defect is its text must not report OVERALL: PASS."""
    real = _prose()
    df_real = pd.DataFrame({"narrative": real})
    df_synth = pd.DataFrame({"narrative": _at_max_length_uniform_alphabet(real)})

    report = validate(df_real, df_synth)
    assert report.overall_pass is False, report.to_text()


def test_the_two_text_metrics_are_both_surfaced():
    real = _prose()
    df_real = pd.DataFrame({"narrative": real})
    df_synth = pd.DataFrame({"narrative": _at_max_length_uniform_alphabet(real)})

    col = next(c for c in validate(df_real, df_synth).columns
               if c.name == "narrative")
    # Both metrics independently detect this defect.
    assert col.details["char_tvd"] > 0.5
    length_ratio = abs(col.details["real_len_mean"] - col.details["synth_len_mean"])
    assert length_ratio / col.details["real_len_mean"] > 0.3


def test_faithful_text_still_passes():
    """Un-suppressing must not turn every text column red."""
    df_real = pd.DataFrame({"narrative": _prose(seed=1)})
    df_synth = pd.DataFrame({"narrative": _prose(seed=2)})
    report = validate(df_real, df_synth)
    col = next(c for c in report.columns if c.name == "narrative")
    assert col.status == "pass", report.to_text()
    assert report.overall_pass is True


def test_text_length_defect_alone_is_enough_to_fail():
    """Right characters, wrong lengths: still a defect, still a failure."""
    real = _prose(seed=3)
    # Same vocabulary and characters, but every value padded to one length.
    width = int(real.str.len().max())
    fixed = real.str.slice(0, 4).str.pad(width, side="right", fillchar="e")
    report = validate(pd.DataFrame({"narrative": real}),
                      pd.DataFrame({"narrative": fixed}))
    col = next(c for c in report.columns if c.name == "narrative")
    assert col.status == "fail", report.to_text()
