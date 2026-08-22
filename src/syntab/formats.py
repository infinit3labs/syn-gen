"""Output writers: csv, json, jsonl, parquet, sql."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def _to_frames(tables: Dict[str, List[Dict[str, Any]]]) -> Dict[str, pd.DataFrame]:
    return {name: pd.DataFrame(rows) for name, rows in tables.items()}


def _serialize_cell(v: Any) -> Any:
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return v


def _frames_serializable(frames: Dict[str, pd.DataFrame]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for name, df in frames.items():
        out[name] = [_serialize_row(r) for r in df.to_dict(orient="records")]
    return out


def _serialize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {k: _serialize_cell(v) for k, v in row.items()}


def write(tables: Dict[str, List[Dict[str, Any]]], out: str, fmt: str | None = None,
          spec: Any = None) -> None:
    fmt = fmt or Path(out).suffix.lstrip(".").lower() or "csv"
    if fmt == "sql":
        if spec is not None:
            from .spec import expand_many_to_many
            spec = expand_many_to_many(spec)
        _write_sql(tables, out, spec)
        return

    frames = _frames_with_columns(tables, spec)
    if fmt in ("csv", "json", "jsonl", "parquet"):
        _write_tablewise(frames, out, fmt)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def _write_tablewise(frames: Dict[str, pd.DataFrame], out: str, fmt: str) -> None:
    out_path = Path(out)
    if out_path.suffix == "":
        # extensionless path -> directory with one file per table
        out_path.mkdir(parents=True, exist_ok=True)
        ext = "jsonl" if fmt == "jsonl" else fmt
        for name, df in frames.items():
            _write_one(df, out_path / f"{name}.{ext}", fmt)
    else:
        if fmt in ("json", "jsonl"):
            serial = _frames_serializable(frames)
            if fmt == "json":
                out_path.write_text(json.dumps(serial, indent=2), encoding="utf-8")
            else:
                with out_path.open("w", encoding="utf-8") as f:
                    for name, rows in serial.items():
                        for r in rows:
                            f.write(json.dumps({name: r}) + "\n")
        else:
            if len(frames) > 1 and fmt in ("csv", "parquet"):
                raise ValueError("Multiple tables require a directory output for csv/parquet")
            name, df = next(iter(frames.items()))
            _write_one(df, out_path, fmt)


def _write_one(df: pd.DataFrame, path: Path, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        df.to_csv(path, index=False)
    elif fmt == "json":
        path.write_text(df.to_json(orient="records", indent=2, date_format="iso"), encoding="utf-8")
    elif fmt == "jsonl":
        df.to_json(path, orient="records", lines=True, date_format="iso")
    elif fmt == "parquet":
        try:
            df.to_parquet(path, index=False)
        except Exception as e:
            raise RuntimeError("Parquet support requires pyarrow: pip install pyarrow") from e


def _sql_type(dtype: Any) -> str:
    return {
        "int": "INTEGER", "float": "REAL", "bool": "BOOLEAN",
        "str": "TEXT", "uuid": "TEXT", "datetime": "TIMESTAMP", "date": "DATE",
    }.get(dtype, "TEXT")


def _frames_with_columns(tables, spec):
    """Build DataFrames, preserving column names for empty tables via the spec."""
    frames = _to_frames(tables)
    if spec is not None:
        spec_cols = {t.name: [c.name for c in t.columns] for t in spec.tables}
        for name, df in frames.items():
            if len(df) == 0 and name in spec_cols and spec_cols[name]:
                frames[name] = pd.DataFrame(tables[name], columns=spec_cols[name])
    return frames


def _write_sql(tables: Dict[str, List[Dict[str, Any]]], out: str, spec: Any = None) -> None:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    spec_tables = {t.name: t for t in spec.tables} if spec is not None else {}

    # Emit parent tables before children for cleaner DDL ordering.
    names = list(tables.keys())

    def _order_key(n: str) -> int:
        st = spec_tables.get(n)
        return sum(1 for _ in st.relationships) if st else 0

    names.sort(key=_order_key)

    for name in names:
        rows = tables[name]
        if not rows:
            if st is None:
                continue
            cols = [c.name for c in st.columns]
            if not cols:
                continue
        else:
            cols = list(rows[0].keys())
        st = spec_tables.get(name)
        type_for: Dict[str, str] = {}
        pk_scalar = None
        pk_list: List[str] = []
        uniques: set = set()
        not_nulls: set = set()
        fks: List[tuple] = []
        if st is not None:
            if isinstance(st.primary_key, str):
                pk_scalar = st.primary_key
            elif isinstance(st.primary_key, list):
                pk_list = list(st.primary_key)
            for c in st.columns:
                type_for[c.name] = _sql_type(c.dtype)
                if c.constraints.get("unique"):
                    uniques.add(c.name)
                if c.constraints.get("nullable") is False:
                    not_nulls.add(c.name)
            for rel in st.relationships:
                fks.append((rel.child_columns, rel.parent_table, rel.parent_columns))

        col_defs: List[str] = []
        for c in cols:
            bits = [f'"{c}" {type_for.get(c, "TEXT")}']
            if c in not_nulls:
                bits.append("NOT NULL")
            if c in uniques and c != pk_scalar:
                bits.append("UNIQUE")
            if c == pk_scalar:
                bits.append("PRIMARY KEY")
            col_defs.append(" ".join(bits))
        if pk_list:
            col_defs.append(
                "PRIMARY KEY (" + ", ".join(f'"{c}"' for c in pk_list) + ")"
            )
        if st is not None:
            for unique_columns in st.unique_constraints:
                col_defs.append(
                    "UNIQUE (" + ", ".join(f'"{c}"' for c in unique_columns) + ")"
                )
        for fcols, ptbl, pcols in fks:
            col_defs.append(
                f'FOREIGN KEY ({", ".join(f"\"{c}\"" for c in fcols)}) '
                f'REFERENCES "{ptbl}" ({", ".join(f"\"{c}\"" for c in pcols)})'
            )

        lines.append(
            f'CREATE TABLE IF NOT EXISTS "{name}" (\n  '
            + ",\n  ".join(col_defs)
            + "\n);"
        )
        for r in rows:
            keys = list(r.keys())
            vals = ", ".join(_sql_val(r[k]) for k in keys)
            lines.append(
                f'INSERT INTO "{name}" ({", ".join(f"\"{k}\"" for k in keys)}) '
                f"VALUES ({vals});"
            )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _sql_val(v: Any) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, (int, float)):
        return str(v)
    s = v.isoformat() if hasattr(v, "isoformat") else str(v)
    return "'" + s.replace("'", "''") + "'"


# ---------------------------------------------------------------------------
# Streaming sinks (large-scale generation): write each table to disk as soon
# as it is generated instead of holding every table in memory.
# ---------------------------------------------------------------------------

class RowSink:
    """Base class for streaming table sinks."""

    def write_table(self, name: str, rows: List[Dict[str, Any]]) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class CSVSink(RowSink):
    """Append each table to ``<out_dir>/<table>.csv`` (header written once)."""

    def __init__(self, out_dir: str):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._written: set = set()

    def write_table(self, name: str, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        path = self.dir / f"{name}.csv"
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False, header=name not in self._written, mode="a")
        self._written.add(name)


class JSONLSink(RowSink):
    """Append each table to ``<out_dir>/<table>.jsonl``."""

    def __init__(self, out_dir: str):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def write_table(self, name: str, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        path = self.dir / f"{name}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(_serialize_row(r)) + "\n")
