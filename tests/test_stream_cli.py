"""CLI-level guarantees for `syntab generate --stream` (INF-236).

The engine already streams safely in-process (see test_streaming.py); these
tests cover the part only the CLI is responsible for: publishing the result
to the user-facing --out path only on success, and reporting progress.
"""
from click.testing import CliRunner

from syntab.cli import cli
from syntab.loaders import to_file
from syntab.spec import (
    ColumnSpec, RelationshipSpec, Settings, Spec, SpecMetadata, TableSpec,
)


def _spec(*, failing: bool = False) -> Spec:
    customers = TableSpec(
        name="customers", row_count=5, primary_key="id",
        columns=[
            ColumnSpec(name="id", dtype="int", generator="sequence",
                       constraints={"unique": True}),
            ColumnSpec(name="region", dtype="str", generator="choice",
                       params={"values": ["EU", "US"], "weights": [0.5, 0.5]}),
        ],
    )
    if failing:
        customers.rules = ["id <= 1"]
    return Spec(
        metadata=SpecMetadata(name="shop"),
        settings=Settings(seed=7, fallback="raise", max_rule_attempts=200),
        tables=[
            customers,
            TableSpec(
                name="orders", row_count=8, primary_key="id",
                columns=[
                    ColumnSpec(name="id", dtype="int", generator="sequence",
                               constraints={"unique": True}),
                    ColumnSpec(name="customer_id", dtype="int", generator="fk",
                               params={"ref": "customers.id"}),
                ],
                relationships=[RelationshipSpec(
                    **{"from": "customer_id", "to": "customers.id", "alias": "customer"}
                )],
            ),
        ],
    )


def _write_spec(path, *, failing: bool = False) -> str:
    to_file(_spec(failing=failing), str(path))
    return str(path)


def test_stream_publishes_out_only_on_success(tmp_path):
    spec_path = _write_spec(tmp_path / "spec.yaml")
    out = tmp_path / "out"

    result = CliRunner().invoke(
        cli, ["generate", "--spec", spec_path, "--out", str(out),
              "--format", "csv", "--stream", "--chunk-size", "1"],
    )

    assert result.exit_code == 0, result.output
    assert {p.name for p in out.iterdir()} == {"customers.csv", "orders.csv"}
    # no staging/backup leftovers beside the published output
    assert {p.name for p in tmp_path.iterdir()} == {"spec.yaml", "out"}


def test_stream_failure_leaves_no_output_on_fresh_out(tmp_path):
    spec_path = _write_spec(tmp_path / "spec.yaml", failing=True)
    out = tmp_path / "out"

    result = CliRunner().invoke(
        cli, ["generate", "--spec", spec_path, "--out", str(out),
              "--format", "csv", "--stream", "--chunk-size", "1"],
    )

    assert result.exit_code == 1
    assert "Could not satisfy rules" in result.output
    assert not out.exists()
    # the staging directory must not survive a failed run either
    assert list(tmp_path.iterdir()) == [tmp_path / "spec.yaml"]


def test_stream_failure_preserves_prior_successful_output(tmp_path):
    spec_path = _write_spec(tmp_path / "spec.yaml")
    out = tmp_path / "out"
    runner = CliRunner()

    good = runner.invoke(
        cli, ["generate", "--spec", spec_path, "--out", str(out),
              "--format", "csv", "--stream"],
    )
    assert good.exit_code == 0, good.output
    before = {p.name: p.read_text() for p in out.iterdir()}

    failing_spec_path = _write_spec(tmp_path / "spec_failing.yaml", failing=True)
    failed = runner.invoke(
        cli, ["generate", "--spec", failing_spec_path, "--out", str(out),
              "--format", "csv", "--stream", "--chunk-size", "1"],
    )

    assert failed.exit_code == 1
    after = {p.name: p.read_text() for p in out.iterdir()}
    assert after == before
    # no leftover backup/staging directories next to `out`
    assert {p.name for p in tmp_path.iterdir()} == {
        "spec.yaml", "spec_failing.yaml", "out",
    }


def test_stream_progress_flag_reports_to_stderr(tmp_path):
    spec_path = _write_spec(tmp_path / "spec.yaml")
    out = tmp_path / "out"

    result = CliRunner().invoke(
        cli, ["generate", "--spec", spec_path, "--out", str(out),
              "--format", "csv", "--stream", "--chunk-size", "2", "--progress"],
    )

    assert result.exit_code == 0, result.output
    assert "customers: 2/5" in result.stderr
    assert "orders: 8/8" in result.stderr


def test_stream_without_progress_flag_is_quiet(tmp_path):
    spec_path = _write_spec(tmp_path / "spec.yaml")
    out = tmp_path / "out"

    result = CliRunner().invoke(
        cli, ["generate", "--spec", spec_path, "--out", str(out),
              "--format", "csv", "--stream", "--chunk-size", "2"],
    )

    assert result.exit_code == 0, result.output
    assert result.stderr == ""
