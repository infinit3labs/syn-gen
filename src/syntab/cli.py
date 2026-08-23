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
from . import quality
from .disclosure import DisclosureBudget, DisclosureReport
from .engine import GenerationEngine, SpecError
from .loaders import from_file, spec_to_dict, to_file
from .profiler import (
    DEFAULT_CONDITION_ON_FDS,
    DEFAULT_MAX_CATEGORICAL,
    DEFAULT_MAX_CATEGORICAL_RATIO,
    DEFAULT_MAX_CATEGORICAL_RATIO_CAP,
    DEFAULT_MIN_CELL_COUNT,
    RECOMMENDED_MIN_CELL_COUNT,
)
from .discovery import (
    DEFAULT_CONDITIONAL_FD_ERROR,
    DEFAULT_FD_ERROR,
    DEFAULT_IND_ERROR,
    DEFAULT_MIN_MU,
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
@click.option("--chunk-size", default=None, type=int,
              help="Rows per streamed sink write. Defaults to one write per table.")
@click.option("--vectorized", is_flag=True,
              help="Use the column-wise fast path where possible (tables with "
                   "rules/dependencies/Faker/self-refs/M2M/uniqueness fall back "
                   "to row-by-row generation).")
def generate(spec_path: str, out: str, fmt: str | None, seed: int | None,
             stream: bool, chunk_size: int | None, vectorized: bool) -> None:
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
            engine.run(stream_sink=sink, vectorized=vectorized, chunk_size=chunk_size)
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
@click.option("--min-cell-count", "min_cell_count",
              default=DEFAULT_MIN_CELL_COUNT, type=int, show_default=True,
              help="Minimum cell-size suppression. Categorical values "
                   "occurring fewer than K times IN THE SOURCE DATA are "
                   "generalized into a single '__other__' bucket, preserving "
                   "total frequency mass. Rare values are the ones that "
                   "identify people (k-anonymity), so this is the control "
                   f"that matters most. 0 disables it; {RECOMMENDED_MIN_CELL_COUNT} "
                   "is the usual floor in published SDC practice. Left off by "
                   "default because it changes the statistical content of the "
                   "profile and that is the data holder's call -- but the "
                   "disclosure summary reports what it would have caught "
                   "either way.")
@click.option("--redact-categoricals", "redact_categoricals", is_flag=True,
              help="Replace categorical values in the written spec with "
                   "opaque placeholder tokens (value_001, value_002, ...). "
                   "Cardinality and the frequency vector are preserved, so "
                   "generation still works; the value labels stop being "
                   "derived from the source data. Use when the spec will be "
                   "committed or shared and the category labels themselves "
                   "are sensitive.")
@click.option("--max-unredacted-values", type=click.IntRange(min=0),
              default=None, help="Fail before writing when the disclosure "
              "report contains more unredacted categorical values than this.")
@click.option("--max-rare-values", type=click.IntRange(min=0), default=None,
              help="Fail before writing when more rare categorical values "
              "remain than this budget allows.")
@click.option("--max-conditional-cells", type=click.IntRange(min=0),
              default=None, help="Fail before writing when the disclosure "
              "report contains more conditional cells than this.")
@click.option("--pii", "pii", default=None,
              help="Comma-separated column names to treat as PII.")
@click.option("--pii-strategy", "pii_strategy", default="faker",
              type=click.Choice(["faker", "mask", "redact", "hash"]),
              help="Anonymization strategy for --pii columns.")
@click.option("--discover/--no-discover", "discover", default=None,
              help="Discover keys and foreign keys with published algorithms "
                   "(HyUCC for unique column combinations, SPIDER for "
                   "inclusion dependencies) instead of column-name rules. "
                   "Default: on when the optional 'discovery' extra is "
                   "installed, off otherwise. --discover makes a missing "
                   "install an error rather than a silent downgrade.")
@click.option("--discover-fds", "discover_fds", is_flag=True,
              help="Also discover functional dependencies (HyFD for exact "
                   "ones, Pyro for approximate ones) and record them in the "
                   "spec with their algorithm, error and strength. OFF by "
                   "default: it is the expensive part of profiling and its "
                   "output is a report for a human rather than an input to "
                   "generation.")
@click.option("--condition-on-fds/--no-condition-on-fds", "condition_on_fds",
              default=DEFAULT_CONDITION_ON_FDS, show_default=True,
              help="Generate a column that a discovered dependency determines "
                   "by sampling from the observed conditional distribution "
                   "given its determinant, instead of independently from its "
                   "own marginal. This is what makes --discover-fds change "
                   "the DATA rather than only the report: without it, a "
                   "hierarchy like Product/Sub-product is measured, recorded "
                   "and then destroyed at generation time. Requires "
                   "--discover-fds. NOTE that a conditional distribution "
                   "embeds more of the source than the two marginals it "
                   "replaces -- the disclosure summary says how much.")
@click.option("--conditional-fd-error", "conditional_fd_error",
              default=DEFAULT_CONDITIONAL_FD_ERROR, type=float,
              show_default=True,
              help="Error bound for dependencies used to CONDITION "
                   "generation, as opposed to --fd-error, which bounds the "
                   "ones REPORTED. Looser on purpose: reporting a dependency "
                   "claims it holds, while conditioning on one claims "
                   "nothing -- an 80%-clean dependency yields an 80%-"
                   "concentrated conditional and generation reproduces that "
                   "too. On the CFPB source the hierarchy edges sit at g1 "
                   "0.018-0.019, above --fd-error, because a fifth of "
                   "Sub-product and a third of Sub-issue are NULL.")
@click.option("--fd-error", "fd_error", default=DEFAULT_FD_ERROR, type=float,
              show_default=True,
              help="Error bound for approximate functional dependencies: the "
                   "fraction of tuple pairs allowed to violate one (the g1 "
                   "measure). 0 restricts the search to exact dependencies, "
                   "which on real dirty data finds almost nothing.")
@click.option("--fd-min-mu", "fd_min_mu", default=DEFAULT_MIN_MU, type=float,
              show_default=True,
              help="Minimum cardinality-corrected strength (mu-prime) for a "
                   "reported dependency. Without it a near-key column appears "
                   "to determine every other column in the table.")
@click.option("--ind-error", "ind_error", default=DEFAULT_IND_ERROR, type=float,
              show_default=True,
              help="Error bound for inclusion dependencies. 0 requires a "
                   "foreign key to hold exactly; raise it for an extract with "
                   "a few orphaned rows.")
@click.option("--merge-into", "merge_into", default=None,
              help="Path to an existing spec to fold this profile into. "
                   "Rules a human has edited since the spec was written are "
                   "kept, detected by comparing each rule against a "
                   "fingerprint of what the profiler last wrote. Without this "
                   "a re-profile overwrites the file and every hand "
                   "correction with it.")
def profile(datasets: tuple, out: str, name: str, sample: int, seed: int,
            max_categorical: int, max_categorical_ratio: float,
            min_cell_count: int, redact_categoricals: bool,
            max_unredacted_values: int, max_rare_values: int,
            max_conditional_cells: int,
            pii: str, pii_strategy: str, discover, discover_fds: bool,
            condition_on_fds: bool, conditional_fd_error: float,
            fd_error: float, fd_min_mu: float, ind_error: float,
            merge_into: str) -> None:
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
                redact_categoricals=redact_categoricals,
                min_cell_count=min_cell_count,
                discover=discover, discover_fds=discover_fds,
                condition_on_fds=condition_on_fds,
                conditional_fd_error=conditional_fd_error,
                fd_error=fd_error, fd_min_mu=fd_min_mu, ind_error=ind_error)
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
    if merge_into:
        from .profiler import merge_preserving_edits
        spec = merge_preserving_edits(from_file(merge_into), spec)
    report = DisclosureReport.from_spec(spec)
    violations = report.check_budget(DisclosureBudget(
        max_unredacted_values=max_unredacted_values,
        max_rare_values=max_rare_values,
        max_conditional_cells=max_conditional_cells,
    ))
    if violations:
        details = "; ".join(
            f"{v.metric}={v.actual} exceeds budget {v.limit}"
            for v in violations
        )
        raise click.ClickException(
            f"disclosure budget exceeded; no spec written: {details}"
        )
    to_file(spec, out)
    n_rel = sum(len(t.relationships) for t in spec.tables)
    n_pii = sum(1 for t in spec.tables for c in t.columns if c.pii)
    n_fd = sum(len(t.metadata.get("discovery", {})
                   .get("functional_dependencies", []))
               for t in spec.tables)
    n_kept = sum(
        1 for t in spec.tables
        for prov in [t.key_provenance] + [r.provenance for r in t.relationships]
        if prov is not None and prov.human_edited
    )
    click.echo(
        f"Profiled {len(datasets)} dataset(s) -> {out}: "
        f"{len(spec.tables)} table(s), {n_rel} relationship(s), {n_pii} PII column(s)"
    )
    # Say which of the two paths produced the structure. A spec that says
    # "SPIDER over 208,398 rows" and one that says "the column name ended in
    # _id" warrant different amounts of review, and the difference should not
    # require opening the file to see.
    algorithms = sorted({
        prov.algorithm.split("+")[0]
        for t in spec.tables
        for prov in [t.key_provenance] + [r.provenance for r in t.relationships]
        if prov is not None
    })
    if algorithms:
        click.echo(f"  structure inferred by: {', '.join(algorithms)}")
    if n_fd:
        click.echo(f"  {n_fd} functional dependenc"
                   f"{'y' if n_fd == 1 else 'ies'} recorded")
    conditionals = [c for t in spec.tables
                    for c in t.metadata.get("discovery", {})
                    .get("conditional_dependencies", [])]
    if conditionals:
        click.echo(f"  {len(conditionals)} column(s) generated conditionally:")
        for c in conditionals:
            note = " (arrow reversed to keep the graph a forest)" if c.get(
                "reversed") else ""
            mu = c.get("provenance", {}).get("mu_prime")
            strength = f"  mu'={mu}" if mu is not None else ""
            click.echo(f"      {c['determinant'][0]} -> {c['dependent']}"
                       f"{strength}{note}")
    elif discover_fds and condition_on_fds:
        click.echo("  no dependency was strong enough to condition generation "
                   "on; every column is sampled independently")
    if merge_into:
        click.echo(f"  merged into {merge_into}: {n_kept} hand-edited rule(s) preserved")
    # The disclosure summary goes to stderr, deliberately. It is a notice about
    # the artefact rather than part of it, and stderr is the stream that
    # survives `syntab profile ... | tee`, redirection and CI log capture -- the
    # situations in which someone is least likely to be reading closely.
    click.echo(report.to_text(spec_path=out), err=True)


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
    """Compare real vs synthetic per-column distributions (pass/warn/fail).

    MARGINALS ONLY. This compares each column against its counterpart in
    isolation and says nothing about the relationships between columns, so a
    dataset can pass here with every correlation in it destroyed. Use
    `syntab quality` for a graded score that includes column pair trends.
    """
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


@cli.command(name="quality")
@click.option("--real", "real_path", required=True, help="Path to the real dataset.")
@click.option("--synthetic", "synth_path", required=True, help="Path to the synthetic dataset.")
@click.option("--spec", "spec_path", default=None,
              help="Optional Spec. Used for column types and key columns, "
                   "which are more reliable than inferring them from the data.")
@click.option("--out", "out", default=None, help="Write the JSON report to this path.")
@click.option("--verbose", is_flag=True, help="Show the per-column breakdown.")
@click.option("--min-score", "min_score", default=None, type=float,
              help="Exit non-zero if the overall score falls below this. For "
                   "CI. Left unset, this command always exits 0 -- a quality "
                   "score is a measurement, not a verdict.")
@click.option("--sample", default=None, type=int,
              help="Rows to sample for the quadratic pair-trend pass and the "
                   f"character-distribution metrics (default {quality.DEFAULT_SUBSAMPLE}). "
                   "0 disables sampling.")
@click.option("--no-pair-trends", "no_pair_trends", is_flag=True,
              help="Skip Column Pair Trends. Note that this removes the only "
                   "property that detects destroyed relationships between "
                   "columns, which is the failure mode a rule-based generator "
                   "is most exposed to.")
def quality_cmd(real_path: str, synth_path: str, spec_path: str, out: str,
                verbose: bool, min_score: float, sample: int,
                no_pair_trends: bool) -> None:
    """Score synthetic fidelity against real data (graded, 0..1).

    One of THREE reports, which answer different questions:

      syntab quality   -- how closely does the synthetic data RESEMBLE the
                          real data? Graded. Column shapes, column pair
                          trends, coverage, boundary adherence, missingness.
      syntab diagnose  -- is the synthetic data structurally VALID? Pass/fail.
      syntab check     -- does the data honour its SPEC? Pass/fail, and needs
                          no real data to run.

    Fidelity is a matter of degree, so this one reports a score rather than a
    verdict. Use --min-score to turn it into a gate.
    """
    real = _read_any(real_path)
    synth = _read_any(synth_path)
    spec = from_file(spec_path) if spec_path else None
    eff_sample = quality.DEFAULT_SUBSAMPLE if sample is None else (sample or None)
    report = quality.quality_report(
        real, synth, spec=spec, sample=eff_sample,
        pair_trends=not no_pair_trends,
    )
    click.echo(report.to_text(verbose=verbose))
    if out:
        Path(out).write_text(json.dumps(report.to_dict(), indent=2, default=str),
                             encoding="utf-8")
        click.echo(f"Report written to {out}")
    if min_score is not None and report.overall_score < min_score:
        click.echo(f"Overall score {report.overall_score:.4f} is below "
                   f"--min-score {min_score}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--real", "real_path", required=True, help="Path to the real dataset.")
@click.option("--synthetic", "synth_path", required=True, help="Path to the synthetic dataset.")
@click.option("--spec", "spec_path", default=None,
              help="Optional Spec. Supplies column types and the key columns "
                   "checked for uniqueness.")
@click.option("--out", "out", default=None, help="Write the JSON report to this path.")
@click.option("--verbose", is_flag=True, help="Show every check, not just failures.")
@click.option("--tolerance", default=0.0, type=float, show_default=True,
              help="How far below 1.0 a diagnostic metric may fall and still "
                   "pass. The default is strict on purpose: these checks are "
                   "for things that are broken, not things that are imprecise.")
def diagnose(real_path: str, synth_path: str, spec_path: str, out: str,
             verbose: bool, tolerance: float) -> None:
    """Check that synthetic data is structurally valid (pass/fail).

    Data Structure (do the columns match?) and Data Validity (is every value
    one the source could have produced -- in range, a real category, a unique
    key?). These catch output that is broken rather than merely low-fidelity;
    `syntab quality` scores the fidelity.
    """
    real = _read_any(real_path)
    synth = _read_any(synth_path)
    spec = from_file(spec_path) if spec_path else None
    report = quality.diagnostic_report(real, synth, spec=spec, tolerance=tolerance)
    click.echo(report.to_text(verbose=verbose))
    if out:
        Path(out).write_text(json.dumps(report.to_dict(), indent=2, default=str),
                             encoding="utf-8")
        click.echo(f"Report written to {out}")
    sys.exit(0 if report.overall_ok else 1)


@cli.command()
@click.option("--spec", "spec_path", required=True, help="Path to the Spec YAML/JSON.")
@click.option("--data", "data", required=True,
              help="Generated dataset: a directory or file of table(s) (csv/parquet/json/jsonl).")
def check(spec_path: str, data: str) -> None:
    """Certify that generated data conforms to its Spec (PK, FK, rules, ...).

    The third of the three reports, and the only one that needs no real data:
    it checks the output against the contract it was generated from rather
    than against a source dataset. `syntab quality` and `syntab diagnose`
    answer the other two questions.
    """
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
