"""Regression tests: facts about the whole dataset are not read off a sample.

Two bugs with the same shape:

  * ``unique`` was ``nunique == len`` on the sample, so any column with more
    distinct values than the sample size was declared unique -- exactly the
    high-cardinality columns most likely to be tested for keyhood;
  * ``nullable`` was ``null_rate > 0`` measured on the sample, so a column
    null once in a million rows was written ``nullable: false``, and the
    conformance checker then hard-failed the source data against its own spec.
"""
import numpy as np
import pandas as pd

from syntab.conformance import validate_against_spec
from syntab.profiler import DatasetProfiler

N = 50_000
SAMPLE = 1_000
SEED = 0


# --------------------------------------------------------------------------
# uniqueness: candidates from the sample, validated on the full column
# --------------------------------------------------------------------------

def _almost_unique_frame():
    """``code`` is unique except for a single duplicated pair."""
    codes = [f"c{i:06d}" for i in range(N)]
    codes[-1] = codes[0]
    return pd.DataFrame({"code": codes})


def test_uniqueness_is_validated_against_the_full_column():
    df = _almost_unique_frame()
    # Precondition: the draw the profiler will take does look unique, so this
    # exercises the validation step rather than the candidate step.
    assert df.sample(n=SAMPLE, random_state=SEED)["code"].is_unique

    spec = DatasetProfiler(df, name="t", sample=SAMPLE, seed=SEED).profile()
    col = spec.tables[0].columns[0]
    assert col.constraints.get("unique") is not True
    assert spec.tables[0].primary_key is None


def test_a_genuinely_unique_column_is_still_marked_unique():
    df = pd.DataFrame({"code": [f"c{i:06d}" for i in range(N)]})
    spec = DatasetProfiler(df, name="t", sample=SAMPLE, seed=SEED).profile()
    assert spec.tables[0].columns[0].constraints.get("unique") is True


def test_high_cardinality_free_text_is_not_declared_a_key():
    """The motivating case: a wide text column, unique inside any small draw."""
    df = pd.DataFrame({
        "narrative": [f"complaint text number {i} " + "x" * (i % 40)
                      for i in range(N)] * 1,
    })
    # make it genuinely non-unique across the whole column
    df.loc[N - 1, "narrative"] = df.loc[0, "narrative"]
    assert df.sample(n=SAMPLE, random_state=SEED)["narrative"].is_unique
    spec = DatasetProfiler(df, name="t", sample=SAMPLE, seed=SEED).profile()
    assert spec.tables[0].columns[0].constraints.get("unique") is not True


def test_false_unique_would_have_broken_conformance_on_the_real_data():
    """End to end: a profiled spec must certify its own source dataset."""
    df = _almost_unique_frame()
    spec = DatasetProfiler(df, name="t", sample=SAMPLE, seed=SEED).profile()
    report = validate_against_spec({"t": df}, spec)
    assert report.overall_ok, report.to_text()


def test_uniqueness_ignores_nulls_like_the_conformance_checker():
    """SQL UNIQUE permits repeated NULLs; the two sides must agree."""
    vals = np.arange(N, dtype="float64")
    vals[10] = np.nan
    vals[20] = np.nan
    df = pd.DataFrame({"k": vals})
    spec = DatasetProfiler(df, name="t", sample=SAMPLE, seed=SEED).profile()
    col = spec.tables[0].columns[0]
    assert col.constraints.get("unique") is True
    # the column-level unique check agrees (primary-key selection for a
    # nullable column is a separate concern, covered in test_profiler_keys.py)
    report = validate_against_spec({"t": df}, spec)
    unique_check = next(c for t in report.tables for c in t.checks
                        if c.name == "k unique")
    assert unique_check.status == "pass"


# --------------------------------------------------------------------------
# nullable: from the full column, not the sample
# --------------------------------------------------------------------------

def _rarely_null_frame():
    vals = np.arange(N, dtype="float64")
    vals[-1] = np.nan  # exactly one null in 50,000 rows
    return pd.DataFrame({"measure": vals})


def test_nullable_is_derived_from_the_full_column():
    df = _rarely_null_frame()
    # Precondition: the sample sees no nulls at all.
    assert df.sample(n=SAMPLE, random_state=SEED)["measure"].notna().all()

    spec = DatasetProfiler(df, name="t", sample=SAMPLE, seed=SEED).profile()
    col = spec.tables[0].columns[0]
    assert col.constraints["nullable"] is True
    assert col.constraints["null_rate"] == round(1 / N, 4)


def test_nullable_false_no_longer_hard_fails_conformance_on_a_real_null():
    df = _rarely_null_frame()
    spec = DatasetProfiler(df, name="t", sample=SAMPLE, seed=SEED).profile()
    report = validate_against_spec({"t": df}, spec)
    assert report.overall_ok, report.to_text()
    assert [c for t in report.tables for c in t.checks
            if c.name.endswith("not null")] == []


def test_a_column_with_no_nulls_anywhere_is_still_not_nullable():
    df = pd.DataFrame({"measure": np.arange(N, dtype="float64")})
    spec = DatasetProfiler(df, name="t", sample=SAMPLE, seed=SEED).profile()
    col = spec.tables[0].columns[0]
    assert col.constraints["nullable"] is False
    assert col.constraints["null_rate"] == 0.0


def test_null_rate_is_exact_rather_than_a_sample_estimate():
    rng = np.random.default_rng(SEED)
    vals = rng.normal(0, 1, N)
    mask = rng.random(N) < 0.13
    vals[mask] = np.nan
    df = pd.DataFrame({"measure": vals})
    spec = DatasetProfiler(df, name="t", sample=SAMPLE, seed=SEED).profile()
    expected = round(float(mask.sum()) / N, 4)
    assert spec.tables[0].columns[0].constraints["null_rate"] == expected
