"""Certify that a generated (or real) dataset conforms to a Spec.

This closes the loop started by ``profile`` and ``generate``: it re-checks the
*output* of the engine against the contract it was built from. Checks, per
table:

  * all spec columns are present;
  * row count matches (warning only);
  * primary key uniqueness (and per-column ``unique`` constraints);
  * not-null constraints and declared ``null_rate`` tolerance;
  * foreign-key integrity (every child value exists in the parent key);
  * business rules, including cross-join rules that reference a parent alias.

A :class:`SpecConformance` report aggregates the results and is also exposed as
a ``syntab check`` CLI command.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from .rules import compile_rules
from .spec import Spec, expand_many_to_many

REPORT_SCHEMA_VERSION = "1"


@dataclass
class Check:
    scope: str
    name: str
    status: str  # pass | warn | fail
    detail: str = ""


@dataclass
class TableConformance:
    table: str
    checks: List[Check] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.status != "fail" for c in self.checks)


@dataclass
class SpecConformance:
    tables: List[TableConformance] = field(default_factory=list)

    @property
    def overall_ok(self) -> bool:
        return all(t.ok for t in self.tables)

    def to_text(self) -> str:
        lines = []
        n_pass = n_warn = n_fail = 0
        for t in self.tables:
            for c in t.checks:
                if c.status == "pass":
                    n_pass += 1
                elif c.status == "warn":
                    n_warn += 1
                else:
                    n_fail += 1
        verdict = "CONFORMS" if self.overall_ok else "NON-CONFORMING"
        lines.append(f"Conformance: {verdict}  (pass={n_pass} warn={n_warn} fail={n_fail})")
        lines.append("-" * 64)
        for t in self.tables:
            lines.append(f"table {t.table}: {'OK' if t.ok else 'FAIL'}")
            for c in t.checks:
                mark = {"pass": "ok", "warn": "warn", "fail": "FAIL"}[c.status]
                line = f"  [{mark}] {c.name}"
                if c.detail:
                    line += f" -- {c.detail}"
                lines.append(line)
        return "\n".join(lines)

    def to_dict(self, compact: bool = False) -> dict:
        summary = {
            "report_type": "conformance",
            "report_schema_version": REPORT_SCHEMA_VERSION,
            "status": "pass" if self.overall_ok else "fail",
            "overall_ok": self.overall_ok,
            "table_count": len(self.tables),
        }
        if compact:
            checks = [c for t in self.tables for c in t.checks]
            summary.update({
                "pass_count": sum(c.status == "pass" for c in checks),
                "warn_count": sum(c.status == "warn" for c in checks),
                "fail_count": sum(c.status == "fail" for c in checks),
            })
            return summary
        return {
            **summary,
            "tables": [
                {
                    "table": t.table,
                    "ok": t.ok,
                    "checks": [
                        {"scope": c.scope, "name": c.name,
                         "status": c.status, "detail": c.detail}
                        for c in t.checks
                    ],
                }
                for t in self.tables
            ],
        }


def _coerce(v: Any) -> Any:
    if isinstance(v, pd.Timestamp):
        return v
    return v


def validate_against_spec(
    frames: Dict[str, pd.DataFrame], spec: Spec
) -> SpecConformance:
    """Validate ``frames`` (table_name -> DataFrame) against ``spec``."""
    spec = expand_many_to_many(spec)
    result = SpecConformance()
    for table in spec.tables:
        checks: List[Check] = []
        df = frames.get(table.name)

        if df is None:
            checks.append(Check(table.name, f"table '{table.name}' present", "fail",
                                 "no data provided for this table"))
            result.tables.append(TableConformance(table.name, checks))
            continue

        # column presence
        missing = [c.name for c in table.columns if c.name not in df.columns]
        if missing:
            checks.append(Check(table.name, "all spec columns present", "fail",
                                 f"missing: {missing}"))
            result.tables.append(TableConformance(table.name, checks))
            continue
        else:
            checks.append(Check(table.name, "all spec columns present", "pass"))

        # Row count is compared against ``table.row_count``, which for a
        # profiled spec is the size of the SOURCE dataset. How many rows the
        # profiler actually read lives under ``metadata.profiling.sampled_rows``
        # and is provenance only -- it is never the target, so it must not be
        # compared against. It is surfaced in the message so an operator can
        # tell a genuine mismatch from a profiling artefact.
        if table.row_count is not None and len(df) != table.row_count:
            detail = f"spec={table.row_count} actual={len(df)}"
            profiling = (table.metadata or {}).get("profiling") or {}
            if profiling.get("sampled"):
                detail += (
                    f" (spec row_count is the profiled source size; "
                    f"{profiling.get('sampled_rows')} rows were sampled to build it)"
                )
            checks.append(Check(table.name, "row count matches spec", "warn", detail))
        else:
            checks.append(Check(table.name, "row count matches spec", "pass"))

        # primary key uniqueness
        if table.primary_key:
            dup = int(df.duplicated(subset=table.primary_key).sum())
            checks.append(Check(
                table.name, "primary key unique",
                "pass" if dup == 0 else "fail",
                "" if dup == 0 else f"{dup} duplicate key row(s)",
            ))
        for unique_columns in table.unique_constraints:
            if any(c not in df.columns for c in unique_columns):
                checks.append(Check(
                    table.name, f"unique constraint {unique_columns}", "fail",
                    "column unavailable",
                ))
                continue
            dup = int(df.duplicated(subset=unique_columns).sum())
            checks.append(Check(
                table.name, f"unique constraint {unique_columns}",
                "pass" if dup == 0 else "fail",
                "" if dup == 0 else f"{dup} duplicate row(s)",
            ))

        # per-column unique + nullability + null_rate
        for col in table.columns:
            series = df[col.name]
            if col.constraints.get("unique"):
                d = int(series.dropna().duplicated().sum())
                checks.append(Check(
                    table.name, f"{col.name} unique",
                    "pass" if d == 0 else "fail",
                    "" if d == 0 else f"{d} duplicate value(s)",
                ))
            nullable = col.constraints.get("nullable", True)
            nulls = int(series.isna().sum())
            if nullable is False and nulls > 0:
                checks.append(Check(
                    table.name, f"{col.name} not null", "fail",
                    f"{nulls} null value(s)",
                ))
            else:
                declared = float(col.constraints.get("null_rate", 0.0))
                actual = round(nulls / len(df), 4) if len(df) else 0.0
                if declared and actual > declared + 0.05:
                    checks.append(Check(
                        table.name, f"{col.name} null_rate", "warn",
                        f"declared={declared} actual={actual}",
                    ))
                else:
                    checks.append(Check(table.name, f"{col.name} null_rate", "pass"))

        # foreign-key integrity
        for rel in table.relationships:
            parent_tbl = rel.parent_table
            child_cols = rel.child_columns
            parent_cols = rel.parent_columns
            parent_df = frames.get(parent_tbl)
            if (parent_df is None
                    or any(c not in parent_df.columns for c in parent_cols)
                    or any(c not in df.columns for c in child_cols)):
                checks.append(Check(table.name, f"FK {child_cols} -> {rel.to}", "fail",
                                     "parent table/column unavailable for check"))
                continue
            child_vals = {
                tuple(_coerce(row[c]) for c in child_cols)
                for _, row in df[list(child_cols)].dropna().iterrows()
            }
            parent_vals = {
                tuple(_coerce(row[c]) for c in parent_cols)
                for _, row in parent_df[list(parent_cols)].dropna().iterrows()
            }
            violations = len(child_vals - parent_vals)
            checks.append(Check(
                table.name, f"FK {child_cols} -> {rel.to}",
                "pass" if violations == 0 else "fail",
                "" if violations == 0 else f"{violations} orphan value(s)",
            ))

            if rel.ordered and rel.order_column:
                order_col = rel.order_column
                if order_col not in df.columns or any(c not in df.columns for c in child_cols):
                    checks.append(Check(
                        table.name, f"ordered {order_col} within {child_cols}",
                        "fail", "order column or FK column unavailable",
                    ))
                    continue
                bad = 0
                grp = df.dropna(subset=list(child_cols))
                for _, sub in grp.groupby(list(child_cols)):
                    series = sub[order_col].dropna()
                    if len(series) < 2:
                        continue
                    if not (series.diff().dropna() > 0).all():
                        bad += 1
                checks.append(Check(
                    table.name, f"ordered {order_col} within {child_cols}",
                    "pass" if bad == 0 else "fail",
                    "" if bad == 0 else f"{bad} group(s) not strictly increasing",
                ))

        # business rules (in-table + cross-join via parent aliases)
        rules = compile_rules(table.rules)
        if rules:
            parent_index: Dict[str, tuple] = {}
            for rel in table.relationships:
                p_tbl = rel.parent_table
                p_cols = rel.parent_columns
                p_df = frames.get(p_tbl)
                idx: Dict[Any, dict] = {}
                if p_df is not None and all(c in p_df.columns for c in p_cols):
                    for rec in p_df.to_dict(orient="records"):
                        key = tuple(_coerce(rec[c]) for c in p_cols)
                        idx[key] = rec
                parent_index[rel.alias or p_tbl] = (rel, idx)

            rows = df.to_dict(orient="records")
            total_viol = 0
            for row in rows:
                parents: Dict[str, dict] = {}
                for alias, (rel, idx) in parent_index.items():
                    key = tuple(_coerce(row.get(c)) for c in rel.child_columns)
                    parents[alias] = idx.get(key, {})
                for rule in rules:
                    if not rule.check(row, parents):
                        total_viol += 1
                        break
            checks.append(Check(
                table.name, f"{len(rules)} business rule(s)",
                "pass" if total_viol == 0 else "fail",
                "" if total_viol == 0 else f"{total_viol} row(s) violate a rule",
            ))

        result.tables.append(TableConformance(table.name, checks))
    return result
