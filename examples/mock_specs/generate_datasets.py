"""Generate example datasets for every mock spec under examples/mock_specs/.

For each spec, runs the engine and writes the tables to
``examples/mock_specs/datasets/<spec>/`` in several formats:
  * csv/      one <table>.csv per table
  * parquet/  one <table>.parquet per table
  * jsonl/    one <table>.jsonl per table
  * <spec>.json  all tables combined

Run from the repo root:  python examples/mock_specs/generate_datasets.py
"""
from __future__ import annotations

import os
import sys

import pandas as pd
from syntab.engine import GenerationEngine
from syntab.formats import write as write_tables
from syntab.loaders import from_file

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

MOCK = os.path.join(REPO_ROOT, "examples", "mock_specs")
OUT = os.path.join(MOCK, "datasets")


def main() -> None:
    specs = sorted(
        f for f in os.listdir(MOCK)
        if f.endswith((".yaml", ".yml")) and not f.startswith(".")
    )
    for fname in specs:
        name = os.path.splitext(fname)[0]
        spec = from_file(os.path.join(MOCK, fname))
        frames = GenerationEngine(spec).run().to_frames()
        n_rows = sum(len(df) for df in frames.values())

        spec_dir = os.path.join(OUT, name)
        os.makedirs(spec_dir, exist_ok=True)
        tables = {t: df.to_dict(orient="records") for t, df in frames.items()}

        write_tables(tables, os.path.join(spec_dir, "csv"), "csv")
        write_tables(tables, os.path.join(spec_dir, "parquet"), "parquet")
        write_tables(tables, os.path.join(spec_dir, "jsonl"), "jsonl")
        write_tables(tables, os.path.join(spec_dir, f"{name}.json"), "json")

        print(f"[OK] {name}: {len(frames)} tables, {n_rows} rows -> {spec_dir}")


if __name__ == "__main__":
    main()
