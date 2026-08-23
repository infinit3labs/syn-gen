"""Scale coverage for conditional profiling with many small determinant groups.

This is deliberately a measurement test rather than a wall-clock benchmark:
the timings are useful when comparing runs, but the regression assertion is
that the runtime lookup index is built once and reused for every generated row.
"""
import json
import random
import time

import pandas as pd

from syntab import generators
from syntab.generators import GenContext, build_generator, conditional_params
from syntab.profiler import DatasetProfiler, OTHER_BUCKET_LABEL


GROUP_COUNT = 1_000
ROWS_PER_GROUP = 6
MIN_CELL_COUNT = 5


def _sparse_small_cell_frame() -> pd.DataFrame:
    determinants = []
    dependents = []
    for group in range(GROUP_COUNT):
        determinant = f"group-{group:04d}"
        determinants.extend([determinant] * ROWS_PER_GROUP)
        dependents.extend(["stable"] * (ROWS_PER_GROUP - 1))
        dependents.append(f"rare-{group:04d}")
    return pd.DataFrame({"determinant": determinants, "dependent": dependents})


def _column(spec, name):
    return next(column for column in spec.tables[0].columns
                if column.name == name)


def test_sparse_conditional_profile_and_generation_scale_without_rebuilding_index(
    monkeypatch,
):
    frame = _sparse_small_cell_frame()

    profile_started = time.perf_counter()
    profiler = DatasetProfiler(
        frame,
        name="sparse_cells",
        sample=None,
        discover=False,
        max_categorical=GROUP_COUNT + 10,
        max_categorical_ratio=0.5,
        max_categorical_ratio_cap=GROUP_COUNT + 10,
        min_cell_count=MIN_CELL_COUNT,
    )
    spec = profiler.profile()

    determinant = _column(spec, "determinant")
    dependent = _column(spec, "dependent")
    params = profiler._conditional_params(frame, determinant, dependent)
    profile_seconds = time.perf_counter() - profile_started
    assert params is not None
    assert len(params["keys"]) == GROUP_COUNT
    assert all(set(values) == {"stable", OTHER_BUCKET_LABEL}
               for values in params["values"])

    dependent.generator = "conditional"
    dependent.depends_on = ["determinant"]
    dependent.params = params
    spec_bytes = len(spec.model_dump_json().encode("utf-8"))

    unsuppressed = conditional_params(frame, ["determinant"], "dependent")
    # The conditional table must not retain one sparse source value per group.
    assert len(json.dumps(params, separators=(",", ":"))) \
        < len(json.dumps(unsuppressed, separators=(",", ":")))

    build_calls = []
    real_build = generators._build_conditional_index

    def counting_build(current_params):
        build_calls.append(current_params)
        return real_build(current_params)

    monkeypatch.setattr(generators, "_build_conditional_index", counting_build)
    generator = build_generator("conditional")
    context = GenContext(
        row={"determinant": "group-0000"},
        parents={},
        rng=random.Random(7),
        faker=None,
        params=params,
    )

    first_started = time.perf_counter()
    assert generator(context) in ("stable", OTHER_BUCKET_LABEL)
    first_index_seconds = time.perf_counter() - first_started
    index = params["__conditional_index__"]

    steady_started = time.perf_counter()
    for row_number in range(GROUP_COUNT * 2):
        context.row = {"determinant": f"group-{row_number % GROUP_COUNT:04d}"}
        assert generator(context) in ("stable", OTHER_BUCKET_LABEL)
    steady_state_seconds = time.perf_counter() - steady_started

    assert len(build_calls) == 1
    assert build_calls[0] is params
    assert params["__conditional_index__"] is index

    # Keep the measurements available under ``pytest -s`` without making test
    # success depend on the host's CPU, scheduler, or filesystem.
    print(
        "INF-247 sparse conditional: "
        f"profile={profile_seconds:.4f}s "
        f"spec={spec_bytes}B "
        f"first_index={first_index_seconds:.6f}s "
        f"steady_state={steady_state_seconds:.4f}s"
    )
