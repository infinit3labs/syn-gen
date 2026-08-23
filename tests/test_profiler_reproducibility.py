import pandas as pd
import pytest

from syntab.loaders import spec_to_dict
from syntab.profiler import DatasetProfiler, merge_preserving_edits


def _frame():
    return pd.DataFrame({
        "id": [1, 2, 3, 4],
        "status": ["open", "closed", "open", "closed"],
    })


def test_same_input_and_controls_produce_the_same_profile():
    first = DatasetProfiler(_frame(), name="events", sample=2, seed=17).profile()
    second = DatasetProfiler(_frame(), name="events", sample=2, seed=17).profile()

    assert spec_to_dict(first) == spec_to_dict(second)


def test_profile_records_reproducibility_and_sampling_controls():
    spec = DatasetProfiler(_frame(), name="events", sample=2, seed=17).profile()

    profiling = spec.tables[0].metadata["profiling"]
    assert profiling["source_row_count"] == 4
    assert profiling["sampled_rows"] == 2
    assert profiling["sampled"] is True
    assert profiling["requested_sample"] == 2
    assert profiling["seed"] == 17


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample": -1},
        {"max_categorical": -1},
        {"max_categorical_ratio": -0.1},
        {"max_categorical_ratio": 1.1},
        {"max_categorical_ratio_cap": -1},
        {"min_cell_count": -1},
    ],
)
def test_invalid_profiling_controls_fail_before_profiling(kwargs):
    with pytest.raises(ValueError, match="profiling control"):
        DatasetProfiler(_frame(), **kwargs)


def test_reprofile_merge_preserves_hand_authored_name_relationship_without_discovery():
    users = pd.DataFrame({"id": [1, 2], "name": ["a", "b"]})
    orders = pd.DataFrame({"id": [10, 11], "user_id": [1, 2]})
    first = DatasetProfiler.profile_set(
        {"users": users, "orders": orders}, sample=None, discover=False
    )
    orders_table = next(t for t in first.tables if t.name == "orders")
    relationship = orders_table.relationships[0]
    relationship.to = "users.name"

    second = DatasetProfiler.profile_set(
        {"users": users, "orders": orders}, sample=None, discover=False
    )
    merged = merge_preserving_edits(first, second)
    kept = next(t for t in merged.tables if t.name == "orders").relationships[0]

    assert kept.to == "users.name"
    assert kept.provenance.human_edited is True
