"""Regression tests: a profiled ``row_count`` is the SOURCE size, not the sample.

``_build_table_spec`` sampled the frame and then recorded ``len(df)`` of the
*sampled* frame as ``row_count``. Every profiled dataset therefore silently
shrank to ``--sample`` rows, and ``syntab check`` -- comparing the generated
output against that same wrong number -- certified the truncated result.
"""
import numpy as np
import pandas as pd

from syntab.conformance import validate_against_spec
from syntab.profiler import DatasetProfiler

N = 50_000
SAMPLE = 1_000
SEED = 0


def _big_frame():
    rng = np.random.default_rng(SEED)
    return pd.DataFrame({
        "row_id": np.arange(N),
        "amount": rng.normal(100, 15, N),
        "status": rng.choice(["open", "closed"], N),
    })


def test_row_count_is_the_source_size_not_the_sample_size():
    spec = DatasetProfiler(_big_frame(), name="t", sample=SAMPLE, seed=SEED).profile()
    assert spec.tables[0].row_count == N


def test_sample_size_is_recorded_separately_as_provenance():
    spec = DatasetProfiler(_big_frame(), name="t", sample=SAMPLE, seed=SEED).profile()
    profiling = spec.tables[0].metadata["profiling"]
    assert profiling["source_row_count"] == N
    assert profiling["sampled_rows"] == SAMPLE
    assert profiling["sampled"] is True


def test_unsampled_profile_reports_no_sampling():
    df = _big_frame().head(100)
    spec = DatasetProfiler(df, name="t", sample=SAMPLE, seed=SEED).profile()
    table = spec.tables[0]
    assert table.row_count == 100
    assert table.metadata["profiling"]["sampled_rows"] == 100
    assert table.metadata["profiling"]["sampled"] is False


def test_multi_table_profile_records_each_source_size():
    a = _big_frame()
    b = _big_frame().head(7_500)
    spec = DatasetProfiler.profile_set({"a": a, "b": b}, sample=SAMPLE, seed=SEED)
    by_name = {t.name: t for t in spec.tables}
    assert by_name["a"].row_count == N
    assert by_name["b"].row_count == 7_500
    assert by_name["a"].metadata["profiling"]["sampled_rows"] == SAMPLE
    assert by_name["b"].metadata["profiling"]["sampled_rows"] == SAMPLE


def test_conformance_compares_against_source_size_not_sampled_rows():
    df = _big_frame()
    spec = DatasetProfiler(df, name="t", sample=SAMPLE, seed=SEED).profile()

    # A dataset the size of the real source matches the spec.
    ok = validate_against_spec({"t": df}, spec)
    row_check = next(c for c in ok.tables[0].checks
                     if c.name == "row count matches spec")
    assert row_check.status == "pass", row_check.detail

    # A dataset the size of the *sample* does not -- which is exactly what the
    # old behaviour quietly certified as correct.
    warned = validate_against_spec({"t": df.head(SAMPLE)}, spec)
    row_check = next(c for c in warned.tables[0].checks
                     if c.name == "row count matches spec")
    assert row_check.status == "warn"
    assert f"actual={SAMPLE}" in row_check.detail
    # the message must explain the provenance rather than just look wrong
    assert "sampled to build it" in row_check.detail
