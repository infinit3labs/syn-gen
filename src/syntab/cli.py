"""Command-line interface for syntab."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

import click
import pandas as pd

from . import formats
from . import generators
from .engine import GenerationEngine, SpecError
from .loaders import from_file, spec_to_dict, to_file
from .profiler import (
    DEFAULT_MAX_CATEGORICAL,
    DEFAULT_MAX_CATEGORICAL_RATIO,
    DEFAULT_MAX_CATEGORICAL_RATIO_CAP,
)
from .spec import Spec, TableSpec, ColumnSpec, RelationshipSpec, SpecMetadata, Settings


@click.group()
def cli() -> None:
    """syntab — Spec-driven synthetic tabular data generator."""


@cli.command()
@click.option("--spec", "spec_path", required=True, help="Path to Spec YAML/JSON.")
@click.option("--out", "out", required=True, help="Output path (file or directory).")
@click.option("--format", "fmt", default=None, help="csv|json|jsonl|parquet|sql.")
@click.option("--seed", default=None, type=int, help="Override spec seed.")
@click.option("--stream", is_flag=True,
              help="Stream tables to disk as they are generated (csv/jsonl "
                   "directory output; lower memory for large datasets).")
@click.option("--vectorized", is_flag=True,
              help="Use the column-wise fast path where possible (tables with "
                   "rules/dependencies/Faker/self-refs/M2M/uniqueness fall back "
                   "to row-by-row generation).")
def generate(spec_path: str, out: str, fmt: str | None, seed: int | None,
             stream: bool, vectorized: bool) -> None:
    """Generate data from a Spec."""
    spec = from_file(spec_path)
    if seed is not None:
        spec.settings.seed = seed
    try:
        engine = GenerationEngine(spec)
        if stream:
            eff_fmt = fmt or Path(out).suffix.lstrip(".").lower() or "csv"
            if eff_fmt not in ("csv", "jsonl"):
                click.echo("Error: --stream only supports csv or jsonl.", err=True)
                sys.exit(2)
            sink = formats.CSVSink(out) if eff_fmt == "csv" else formats.JSONLSink(out)
            engine.run(stream_sink=sink, vectorized=vectorized)
            sink.close()
            total = sum(t.row_count for t in spec.tables)
            click.echo(f"Streamed {total} rows across {len(spec.tables)} table(s) to {out}")
            return
        result = engine.run(vectorized=vectorized)
    except (SpecError, Exception) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    result.write(out, fmt, spec=spec)
    total = sum(len(v) for v in result.tables.values())
    click.echo(f"Wrote {total} rows across {len(result.tables)} table(s) to {out}")


@cli.command()
def list_generators() -> None:
    """List built-in and registered custom generators."""
    click.echo("Built-in generators:")
    for name in sorted(generators.BUILTINS):
        click.echo(f"  - {name}")
    click.echo("Faker providers: faker.<provider>  (e.g. faker.name)")
    click.echo("Custom (registered):")
    for name in sorted(generators.get_registry()):
        click.echo(f"  - custom:{name}")
    click.echo("Import path: module.path:Attr")


@cli.command()
def init() -> None:
    """Print a starter Spec YAML to stdout."""
    import yaml as _yaml

    spec = Spec(
        metadata=SpecMetadata(name="example", description="Starter dataset"),
        settings=Settings(seed=42),
        tables=[
            TableSpec(
                name="users",
                row_count=100,
                primary_key="id",
                columns=[
                    ColumnSpec(name="id", dtype="int", generator="sequence", constraints={"unique": True}),
                    ColumnSpec(name="name", dtype="str", generator="faker.name", pii=True),
                    ColumnSpec(name="email", dtype="str", generator="faker.email", constraints={"unique": True}),
                    ColumnSpec(name="age", dtype="int", generator="int", params={"min": 18, "max": 90}),
                    ColumnSpec(name="region", dtype="str", generator="choice",
                               params={"values": ["EU", "US", "APAC"], "weights": [0.4, 0.4, 0.2]}),
                ],
            ),
            TableSpec(
                name="orders",
                row_count=500,
                primary_key="id",
                columns=[
                    ColumnSpec(name="id", dtype="int", generator="sequence", constraints={"unique": True}),
                    ColumnSpec(name="user_id", dtype="int", generator="fk", params={"ref": "users.id"}),
                    ColumnSpec(name="total", dtype="float", generator="float",
                               params={"min": 1, "max": 500, "decimals": 2}),
                    ColumnSpec(name="discount", dtype="float", generator="float",
                               params={"min": 0, "max": 50, "decimals": 2}, depends_on=["total"]),
                ],
                relationships=[RelationshipSpec(**{"from": "user_id", "to": "users.id", "alias": "user"})],
                rules=["discount <= total * 0.5"],
            ),
        ],
    )
    click.echo(_yaml.safe_dump(spec_to_dict(spec), sort_keys=False))


@cli.command()
@click.option("--spec", "spec_path", required=True, help="Path to Spec YAML/JSON.")
def validate(spec_path: str) -> None:
    """Validate a Spec (structure + references + rule syntax)."""
    try:
        spec = from_file(spec_path)
        GenerationEngine(spec)
    except Exception as e:
        click.echo(f"Invalid: {e}", err=True)
        sys.exit(1)
    click.echo(f"OK: {spec.metadata.name} ({len(spec.tables)} tables)")


@cli.command()
@click.argument("datasets", nargs=-1, required=True)
@click.option("--out", "out", required=True, help="Output Spec path (YAML or JSON).")
@click.option("--name", default=None, help="Dataset/set name.")
@click.option("--sample", default=5000, type=int, show_default=True,
              help="Rows to sample for profiling. Row counts, nullability and "
                   "uniqueness are always measured on the full dataset.")
@click.option("--seed", default=42, type=int, help="Seed for sampling + generation.")
@click.option("--max-categorical", "max_categorical",
              default=DEFAULT_MAX_CATEGORICAL, type=int, show_default=True,
              help="A column with at most this many distinct values is "
                   "profiled as categorical -- its real values and their "
                   "frequencies are written into the spec -- instead of being "
                   "treated as free text. Raising it captures more real "
                   "categoricals and embeds more real values in the spec.")
@click.option("--max-categorical-ratio", "max_categorical_ratio",
              default=DEFAULT_MAX_CATEGORICAL_RATIO, type=float, show_default=True,
              help="Also treat a column as categorical when its distinct count "
                   f"is at most this fraction of the rows (hard cap: "
                   f"{DEFAULT_MAX_CATEGORICAL_RATIO_CAP} distinct values). "
                   "Catches genuine categoricals that exceed "
                   "--max-categorical in absolute terms.")
@click.option("--redact-categoricals", "redact_categoricals", is_flag=True,
              help="Replace categorical values in the written spec with "
                   "opaque placeholder tokens (value_001, value_002, ...). "
                   "Cardinality and the frequency vector are preserved, so "
                   "generation still works; the value labels stop being "
                   "derived from the source data. Use when the spec will be "
                   "committed or shared and the category labels themselves "
                   "are sensitive.")
@click.option("--pii", "pii", default=None,
              help="Comma-separated column names to treat as PII.")
@click.option("--pii-strategy", "pii_strategy", default="faker",
              type=click.Choice(["faker", "mask", "redact", "hash"]),
              help="Anonymization strategy for --pii columns.")
def profile(datasets: tuple, out: str, name: str, sample: int, seed: int,
            max_categorical: int, max_categorical_ratio: float,
            redact_categoricals: bool, pii: str, pii_strategy: str) -> None:
    """Profile one or more datasets into a Spec.

    With a single dataset, produces a single-table Spec. With multiple
    datasets, foreign-key relationships between them are inferred. Columns
    named via --pii are flagged PII and anonymized (default: faker substitution,
    so no real value is ever baked into the spec).

    Note that a categorical column's real values and their exact frequencies
    ARE written into the output spec. Treat a profiled spec as derived from the
    source data, not as source code, and review it before sharing it.
    """
    from .profiler import DatasetProfiler

    pii_cols = [c.strip() for c in (pii or "").split(",") if c.strip()] or None
    # profiler options threaded through every construction path below
    opts = dict(max_categorical=max_categorical,
                max_categorical_ratio=max_categorical_ratio,
                redact_categoricals=redact_categoricals)
    if len(datasets) == 1:
        profiler = DatasetProfiler.from_file(
            datasets[0], name=name, sample=sample, seed=seed,
            pii_columns=pii_cols, pii_strategy=pii_strategy, **opts,
        )
        spec = profiler.profile()
    else:
        dfs = {}
        for d in datasets:
            profiler = DatasetProfiler.from_file(
                d, sample=sample, seed=seed, pii_columns=pii_cols,
                pii_strategy=pii_strategy, **opts,
            )
            dfs[profiler.name] = profiler.full
        spec = DatasetProfiler.profile_set(
            dfs, name=name, sample=sample, seed=seed,
            pii_columns=pii_cols, pii_strategy=pii_strategy, **opts,
        )
    to_file(spec, out)
    n_rel = sum(len(t.relationships) for t in spec.tables)
    n_pii = sum(1 for t in spec.tables for c in t.columns if c.pii)
    click.echo(
        f"Profiled {len(datasets)} dataset(s) -> {out}: "
        f"{len(spec.tables)} table(s), {n_rel} relationship(s), {n_pii} PII column(s)"
    )


def _read_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    if p.suffix.lower() in (".json",):
        return pd.read_json(p)
    return pd.read_csv(p)


def _read_tables(path: str) -> Dict[str, pd.DataFrame]:
    """Load one or many tables from a file or directory.

    * a directory of ``<table>.<ext>`` files (csv/parquet/jsonl) or a combined
      ``<name>.json`` mapping table -> rows;
    * a combined ``.json`` file mapping table -> rows;
    * a single data file (csv/parquet/json/jsonl) treated as one table.
    """
    p = Path(path)
    if p.is_dir():
        frames: Dict[str, pd.DataFrame] = {}
        for f in sorted(p.iterdir()):
            suf = f.suffix.lower()
            if suf == ".csv":
                frames[f.stem] = pd.read_csv(f)
            elif suf == ".parquet":
                frames[f.stem] = pd.read_parquet(f)
            elif suf == ".jsonl":
                frames[f.stem] = pd.read_json(f, lines=True)
            elif suf == ".json":
                obj = json.loads(f.read_text(encoding="utf-8"))
                if isinstance(obj, dict) and obj and all(isinstance(v, list) for v in obj.values()):
                    for k, v in obj.items():
                        frames[k] = pd.DataFrame(v)
        if not frames:
            raise click.ClickException(f"No loadable tables found in directory: {path}")
        return frames
    if p.suffix.lower() == ".json":
        obj = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(obj, dict) and obj and all(isinstance(v, list) for v in obj.values()):
            return {k: pd.DataFrame(v) for k, v in obj.items()}
    df = _read_any(path)
    return {p.stem: df}


@cli.command()
@click.option("--real", "real_path", required=True, help="Path to the real dataset.")
@click.option("--synthetic", "synth_path", required=True, help="Path to the synthetic dataset.")
@click.option("--spec", "spec_path", default=None, help="Optional Spec for column types.")
@click.option("--out", "out", default=None, help="Write the JSON report to this path.")
def compare(real_path: str, synth_path: str, spec_path: str, out: str) -> None:
    """Compare real vs synthetic data column distributions."""
    from .validator import validate

    real = _read_any(real_path)
    synth = _read_any(synth_path)
    spec = from_file(spec_path) if spec_path else None
    report = validate(real, synth, spec)
    click.echo(report.to_text())
    if out:
        Path(out).write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
        click.echo(f"Report written to {out}")
    sys.exit(0 if report.overall_pass else 1)


@cli.command()
@click.option("--spec", "spec_path", required=True, help="Path to the Spec YAML/JSON.")
@click.option("--data", "data", required=True,
              help="Generated dataset: a directory or file of table(s) (csv/parquet/json/jsonl).")
def check(spec_path: str, data: str) -> None:
    """Certify that generated data conforms to its Spec (PK, FK, rules, ...)."""
    from .conformance import validate_against_spec

    spec = from_file(spec_path)
    try:
        frames = _read_tables(data)
    except Exception as e:
        click.echo(f"Error reading data: {e}", err=True)
        sys.exit(2)
    report = validate_against_spec(frames, spec)
    click.echo(report.to_text())
    sys.exit(0 if report.overall_ok else 1)


if __name__ == "__main__":
    cli()
