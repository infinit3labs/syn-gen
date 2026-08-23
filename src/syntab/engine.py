"""Generation engine.

Consumes a validated :class:`Spec`, topologically orders tables (by foreign
keys) and columns (by ``depends_on``), generates rows, and enforces business
rules with the hybrid strategy:

  * regenerate the row's free columns (rejection sampling) up to
    ``max_rule_attempts``;
  * on each attempt, re-pick FK parents when a failing rule references a
    parent alias (so the value can be made consistent by construction);
  * if still failing, apply ``settings.fallback`` (raise | drop | null).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from faker import Faker

from .generators import GenContext, build_generator, _parse_dt
from .infer import conditional_determinants, resolve_generator
from . import formats
from .rules import Rule, compile_rules
from .spec import ColumnSpec, Spec, SpecError, TableSpec, expand_many_to_many


class RuleViolation(Exception):
    pass


# ---------------------------------------------------------------------------
# PII anonymization helpers
# ---------------------------------------------------------------------------
_PII_PROVIDER_HINTS = [
    ("email", "email"), ("phone", "phone_number"), ("mobile", "phone_number"),
    ("address", "address"), ("city", "city"), ("country", "country"),
    ("company", "company"), ("first_name", "first_name"),
    ("last_name", "last_name"), ("name", "name"), ("ssn", "ssn"),
    ("zip", "zipcode"), ("postal", "zipcode"),
]


def _pii_faker_value(col, ctx: "GenContext") -> Any:
    """Generate a faker value appropriate to the column's name / dtype."""
    name_l = (col.name or "").lower()
    provider = None
    for key, prov in _PII_PROVIDER_HINTS:
        if key in name_l:
            provider = prov
            break
    if provider is None:
        if col.dtype in ("int", "float"):
            n = ctx.faker.random_int(min=0, max=10 ** 6)
            return float(n) / 100.0 if col.dtype == "float" else n
        provider = "word"
    method = getattr(ctx.faker, provider, None)
    if method is None:
        method = ctx.faker.word
    return method()


def _pii_transform(strategy: str, val: Any, col, ctx: "GenContext") -> Any:
    """Apply a mask / redact / hash transformation to an already-generated value."""
    if strategy == "mask":
        if isinstance(val, str):
            if "@" in val:
                local, domain = val.split("@", 1)
                mlocal = (local[0] + "*" * (len(local) - 1)) if len(local) > 1 else "*"
                return f"{mlocal}@{domain}"
            if len(val) <= 1:
                return "*"
            return val[0] + "*" * (len(val) - 2) + val[-1]
        if isinstance(val, bool):
            return val
        if isinstance(val, float):
            return round(val, -1)
        if isinstance(val, int):
            return (val // 10) * 10
        if isinstance(val, (_dt.datetime, _dt.date)):
            if isinstance(val, _dt.datetime):
                return val.replace(hour=0, minute=0, second=0, microsecond=0)
            return val
        return val
    if strategy == "redact":
        if isinstance(val, str):
            return "REDACTED"
        if isinstance(val, bool):
            return False
        if isinstance(val, (int, float)):
            return 0
        return None
    if strategy == "hash":
        return hashlib.sha256(f"{col.name}|{val}".encode()).hexdigest()[:12]
    return val


@dataclass
class TableResult:
    name: str
    rows: List[Dict[str, Any]] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


class GenerationResult:
    def __init__(self, tables: Dict[str, List[Dict[str, Any]]],
                 columns: Optional[Dict[str, List[str]]] = None):
        self.tables = tables
        self.columns = columns or {}

    def to_frames(self) -> Dict[str, pd.DataFrame]:
        return {
            name: pd.DataFrame(rows, columns=self.columns.get(name))
            for name, rows in self.tables.items()
        }

    def write(self, out: str, fmt: Optional[str] = None, spec: Any = None) -> None:
        formats.write(self.tables, out, fmt, spec)


def _topo_tables(tables: List[TableSpec]) -> List[TableSpec]:
    by_name = {t.name: t for t in tables}
    deps: Dict[str, List[str]] = {t.name: [] for t in tables}
    for t in tables:
        for rel in t.relationships:
            parent = rel.parent_table
            if parent in by_name and parent != t.name:
                deps[t.name].append(parent)
    # Kahn
    indeg = {n: len(set(d)) for n, d in deps.items()}
    queue = [n for n, d in indeg.items() if indeg[n] == 0]
    order: List[str] = []
    while queue:
        n = queue.pop(0)
        order.append(n)
        for m in by_name:
            if n in deps[m]:
                indeg[m] -= 1
                if indeg[m] == 0:
                    queue.append(m)
    if len(order) != len(tables):
        raise SpecError("Cyclic foreign-key dependencies detected among tables")
    return [by_name[n] for n in order], deps


def _topo_columns(table: TableSpec, fk_cols: set) -> List[ColumnSpec]:
    cols = [c for c in table.columns if c.name not in fk_cols]
    by_name = {c.name: c for c in cols}
    deps: Dict[str, set] = {c.name: set() for c in cols}
    for c in cols:
        # params.on of a conditional generator is a generation dependency in
        # the same sense depends_on is: the determinant must exist in the row
        # before the dependent can be drawn conditioned on it.
        for d in list(c.depends_on) + conditional_determinants(c):
            if d in by_name and d != c.name:
                deps[c.name].add(d)
    indeg = {n: len(v) for n, v in deps.items()}
    queue = [n for n, d in indeg.items() if d == 0]
    order: List[str] = []
    while queue:
        n = queue.pop(0)
        order.append(n)
        for m in by_name:
            if n in deps[m]:
                indeg[m] -= 1
                if indeg[m] == 0:
                    queue.append(m)
    if len(order) != len(cols):
        raise SpecError(f"Cyclic depends_on in table '{table.name}'")
    return [by_name[n] for n in order]


class _ParentSelector:
    """Plan parent assignments for one child relationship."""

    def __init__(self, rel, parent_rows, child_count: int, rng):
        if not parent_rows:
            raise SpecError(f"No parent rows available for {rel.to}")
        self.rel = rel
        self.rng = rng
        self.fixed = False
        self.assignments: List[Dict[str, Any]] = []
        self.index = 0

        if rel.cardinality == "one_to_one":
            if child_count > len(parent_rows):
                raise SpecError(
                    f"one_to_one relationship {rel.to} needs at least "
                    f"{child_count} parent rows, found {len(parent_rows)}"
                )
            self.assignments = list(rng.sample(parent_rows, child_count))
            self.fixed = True
        elif rel.allocation == "weighted":
            weights = [max(0.0, float(row.get(rel.weight_column, 0) or 0))
                       for row in parent_rows]
            if sum(weights) <= 0:
                raise SpecError(
                    f"Relationship {rel.to} has no positive weights in "
                    f"{rel.weight_column}"
                )
            self.assignments = rng.choices(parent_rows, weights=weights, k=child_count)
            self.fixed = True
        elif (rel.allocation == "balanced"
              or rel.min_children > 0
              or rel.max_children is not None):
            n_parents = len(parent_rows)
            minimum = rel.min_children
            maximum = rel.max_children
            if minimum * n_parents > child_count:
                raise SpecError(
                    f"Relationship {rel.to} requires at least "
                    f"{minimum * n_parents} children, found {child_count}"
                )
            if maximum is not None and maximum * n_parents < child_count:
                raise SpecError(
                    f"Relationship {rel.to} allows at most "
                    f"{maximum * n_parents} children, found {child_count}"
                )
            counts = [minimum] * n_parents
            remaining = child_count - sum(counts)
            cursor = 0
            while remaining:
                if maximum is not None and counts[cursor] >= maximum:
                    cursor = (cursor + 1) % n_parents
                    continue
                counts[cursor] += 1
                remaining -= 1
                cursor = (cursor + 1) % n_parents
            for parent, count in zip(parent_rows, counts):
                self.assignments.extend([parent] * count)
            rng.shuffle(self.assignments)
            self.fixed = True

    def next(self):
        if not self.fixed:
            return None
        parent = self.assignments[self.index]
        self.index += 1
        return parent


def _validate_m2m_feasible(meta: dict, n_a: int, n_b: int) -> None:
    """Raise SpecError when a many-to-many link budget is impossible."""
    k = meta["row_count"]
    min_a, max_a = meta["min_a"], meta["max_a"]
    min_b, max_b = meta["min_b"], meta["max_b"]
    if k <= 0:
        raise SpecError(f"many_to_many '{meta['name']}' needs row_count > 0")
    if min_a < 0 or min_b < 0:
        raise SpecError(f"many_to_many '{meta['name']}' min bounds must be >= 0")
    if max_a is not None and max_a < min_a:
        raise SpecError(f"many_to_many '{meta['name']}' max_per_a < min_per_a")
    if max_b is not None and max_b < min_b:
        raise SpecError(f"many_to_many '{meta['name']}' max_per_b < min_per_b")
    if max_a is not None and max_a > n_b:
        raise SpecError(
            f"many_to_many '{meta['name']}' max_per_a ({max_a}) exceeds "
            f"{meta['table_b']} size ({n_b})"
        )
    if max_b is not None and max_b > n_a:
        raise SpecError(
            f"many_to_many '{meta['name']}' max_per_b ({max_b}) exceeds "
            f"{meta['table_a']} size ({n_a})"
        )
    if k < max(n_a * min_a, n_b * min_b):
        raise SpecError(
            f"many_to_many '{meta['name']}' row_count ({k}) too small for "
            f"minimum link bounds"
        )
    upper = min(n_a * (max_a if max_a is not None else n_b),
                n_b * (max_b if max_b is not None else n_a))
    if k > upper:
        raise SpecError(
            f"many_to_many '{meta['name']}' row_count ({k}) exceeds feasible "
            f"maximum ({upper})"
        )
    if meta["unique"] and k > n_a * n_b:
        raise SpecError(
            f"many_to_many '{meta['name']}' cannot keep {k} unique pairs with "
            f"only {n_a * n_b} possible combinations"
        )


def _generate_m2m_pairs(a_rows, b_rows, meta: dict, rng) -> List[tuple]:
    """Greedily build ``row_count`` (A, B) pairs honouring degree bounds."""
    pk_a, pk_b = meta["pk_a"], meta["pk_b"]
    a_vals = [r[pk_a] for r in a_rows]
    b_vals = [r[pk_b] for r in b_rows]
    n_a, n_b = len(a_vals), len(b_vals)
    min_a, max_a = meta["min_a"], meta["max_a"]
    min_b, max_b = meta["min_b"], meta["max_b"]
    k = meta["row_count"]
    unique = meta["unique"]

    counts_a = {a: 0 for a in a_vals}
    counts_b = {b: 0 for b in b_vals}
    used = set()
    pairs: List[tuple] = []

    def add(a, b):
        pairs.append((a, b))
        used.add((a, b))
        counts_a[a] += 1
        counts_b[b] += 1

    # Phase 1: satisfy the minimum degree on each side.
    progress = True
    while progress:
        progress = False
        for a in a_vals:
            while counts_a[a] < min_a:
                cands = [b for b in b_vals
                         if counts_b[b] < (max_b if max_b is not None else n_b)
                         and (a, b) not in used]
                if not cands:
                    break
                add(a, rng.choice(cands))
                progress = True
        for b in b_vals:
            while counts_b[b] < min_b:
                cands = [a for a in a_vals
                         if counts_a[a] < (max_a if max_a is not None else n_a)
                         and (a, b) not in used]
                if not cands:
                    break
                add(rng.choice(cands), b)
                progress = True

    # Phase 2: fill up to row_count.
    while len(pairs) < k:
        cands_a = [a for a in a_vals
                   if counts_a[a] < (max_a if max_a is not None else n_b)]
        cands_b = [b for b in b_vals
                   if counts_b[b] < (max_b if max_b is not None else n_a)]
        if not cands_a or not cands_b:
            break
        a = rng.choice(cands_a)
        b = rng.choice(cands_b)
        tries = 0
        while unique and (a, b) in used and tries < 200:
            a = rng.choice(cands_a)
            b = rng.choice(cands_b)
            tries += 1
        if unique and (a, b) in used:
            found = False
            for aa in cands_a:
                for bb in cands_b:
                    if (aa, bb) not in used:
                        a, b = aa, bb
                        found = True
                        break
                if found:
                    break
            if not found:
                break
        add(a, b)

    if len(pairs) != k:
        raise SpecError(
            f"many_to_many '{meta['name']}' could not place {k} links; "
            f"constraints are too tight for the parent table sizes"
        )
    return pairs


def _apply_group_order(rows: List[Dict[str, Any]], child_cols: List[str],
                       order_col: str) -> None:
    """Number ``order_col`` 1..n within each parent group defined by the FK columns."""
    groups: Dict[tuple, List[int]] = {}
    for i, row in enumerate(rows):
        key = tuple(row.get(c) for c in child_cols)
        groups.setdefault(key, []).append(i)
    for idxs in groups.values():
        for position, i in enumerate(idxs, start=1):
            rows[i][order_col] = position


# Generators that can be produced in a single vectorized (column-wise) pass.
_VECTORIZABLE = {
    "const", "int", "float", "bool", "uuid", "datetime", "date", "choice",
    "sequence", "fk",
}


def _can_vectorize(table: TableSpec) -> bool:
    """Whether a table can use the fast column-wise generation path.

    Anything needing row-by-row semantics — business rules, column dependencies,
    conditional generators, Faker/text columns, self-references, many-to-many
    junctions, or uniqueness constraints — falls back to the row engine.
    """
    if table.rules:
        return False
    if table.metadata.get("m2m"):
        return False
    if table.unique_constraints:
        return False
    if isinstance(table.primary_key, list):
        return False
    if any(rel.parent_table == table.name for rel in table.relationships):
        return False  # self-reference needs a growing parent pool
    for c in table.columns:
        if c.depends_on or c.when:
            return False
        if c.constraints.get("unique"):
            return False
        if c.pii_strategy == "faker":
            return False
        g, _ = resolve_generator(c)
        if g is None or g.startswith("faker.") or g.startswith("custom:") or ":" in g:
            return False
        if g not in _VECTORIZABLE:
            return False
    return True


def _sample_numeric_vectorized(params: dict, n: int, rng_np) -> np.ndarray:
    """Vectorized analogue of ``generators._sample_numeric``."""
    dist = params.get("distribution")
    lo = float(params["min"]) if params.get("min") is not None else None
    hi = float(params["max"]) if params.get("max") is not None else None

    if dist == "empirical" and params.get("hist_edges") and params.get("hist_counts"):
        edges = np.asarray(params["hist_edges"], dtype=float)
        counts = np.asarray(params["hist_counts"], dtype=float)
        if counts.sum() <= 0 or len(edges) < 2:
            return np.full(n, float(edges[0]) if len(edges) else 0.0)
        cdf = np.cumsum(counts)
        idx = np.clip(np.searchsorted(cdf, rng_np.random(n) * cdf[-1], side="left"),
                      0, len(counts) - 1)
        lo_e, hi_e = edges[idx], edges[idx + 1]
        frac = rng_np.random(n)
        return np.where(hi_e > lo_e, lo_e + frac * (hi_e - lo_e), lo_e)

    if dist == "lognormal" and params.get("log_mu") is not None and params.get("log_sigma") not in (None, "", 0):
        vals = rng_np.lognormal(float(params["log_mu"]), float(params["log_sigma"]), n)
    elif dist == "exponential":
        vals = rng_np.exponential(float(params.get("scale", 1)), n)
    elif dist == "pareto":
        b = float(params.get("pareto_b", 1.0))
        xmin = float(params.get("pareto_xmin", lo if (lo is not None and lo > 0) else 1.0))
        vals = xmin * rng_np.pareto(b, n)
    elif dist == "normal" and params.get("mean") is not None and params.get("std") not in (None, "", 0):
        vals = rng_np.normal(float(params["mean"]), float(params["std"]), n)
    else:
        a = lo if lo is not None else 0.0
        b = hi if hi is not None else a + 1.0
        return rng_np.uniform(a, b, n)

    if lo is not None:
        vals = np.maximum(vals, lo)
    if hi is not None:
        vals = np.minimum(vals, hi)
    return vals


def _normalize_weights(weights: Any, k: int) -> Optional[np.ndarray]:
    """Turn spec weights into probabilities that sum to exactly 1.

    ``numpy.random.Generator.choice`` requires ``p`` to sum to 1 to within
    ``sqrt(eps)`` and raises "Probabilities do not sum to 1" otherwise. The
    row-by-row path uses ``random.choices``, which accepts arbitrary positive
    weights and normalizes internally -- so the two generation paths disagreed
    about which specs were even valid.

    They disagreed on most profiled specs. The profiler rounds every category
    frequency to 4 decimal places, so the weights it writes routinely sum to
    0.9999 or 1.0001; over 200 profiled single-column specs, ``--vectorized``
    raised on 46% of them and the row-by-row path on none. Hand-authored
    integer weights such as ``[3, 1, 1]`` failed the same way.

    Normalizing here rather than at load time keeps the spec file saying what
    the profiler actually measured, and matches what ``random.choices`` does
    on the other path.
    """
    if not weights:
        return None
    w = np.asarray(weights, dtype=float)
    if w.size != k:
        raise SpecError(f"choice: {w.size} weight(s) for {k} value(s)")
    if not np.isfinite(w).all() or bool((w < 0).any()):
        raise SpecError("choice: weights must be finite and non-negative")
    total = float(w.sum())
    if total <= 0:
        raise SpecError("choice: weights must not sum to zero")
    return w / total


def _gen_vectorized(col: ColumnSpec, g: str, params: dict, n: int, rng_np,
                    fk_index: Dict[str, List[Any]]) -> List[Any]:
    """Return a python list of ``n`` values for one vectorizable column."""
    if g == "const":
        return [params.get("value")] * n
    if g == "sequence":
        return list(range(1, n + 1))
    if g == "fk":
        ref = params.get("ref")
        if not ref:
            raise SpecError("fk generator requires params.ref (e.g. 'users.id')")
        choices = fk_index.get(ref)
        if choices is None or not choices:
            raise SpecError(f"No parent key values found for fk ref '{ref}'")
        return list(rng_np.choice(choices, size=n))
    if g == "uuid":
        return [str(uuid.uuid4()) for _ in range(n)]
    if g == "choice":
        values = params.get("values")
        if not values:
            raise SpecError("choice generator requires params.values")
        p = _normalize_weights(params.get("weights"), len(values))
        return list(rng_np.choice(values, size=n, p=p))
    if g == "bool":
        p = _normalize_weights(params.get("weights"), 2)
        if p is not None:
            arr = rng_np.random(n) < float(p[0])
        else:
            arr = rng_np.random(n) < float(params.get("p", 0.5))
        return arr.tolist()
    if g == "int":
        lo = int(params.get("min", 0))
        hi = int(params.get("max", 100)) if params.get("max") is not None else lo + 100
        step = int(params.get("step", 1))
        if params.get("distribution") in ("empirical", "lognormal", "exponential", "pareto", "normal"):
            vals = np.round(_sample_numeric_vectorized(params, n, rng_np)).astype(float)
            vals = np.clip(vals, lo, hi)
            if step != 1:
                vals = lo + ((vals - lo) // step) * step
                vals = np.minimum(vals, hi)
            return vals.astype(int).tolist()
        if step != 1:
            span = max(0, (hi - lo) // step)
            return (lo + rng_np.integers(0, span + 1, n) * step).astype(int).tolist()
        return rng_np.integers(lo, hi + 1, n).astype(int).tolist()
    if g == "float":
        vals = _sample_numeric_vectorized(params, n, rng_np)
        lo = float(params.get("min", 0.0)) if params.get("min") is not None else None
        hi = float(params.get("max", 1.0)) if params.get("max") is not None else None
        if lo is not None:
            vals = np.maximum(vals, lo)
        if hi is not None:
            vals = np.minimum(vals, hi)
        decimals = params.get("decimals")
        if decimals is not None:
            vals = np.round(vals, int(decimals))
        return vals.tolist()
    if g == "datetime" or g == "date":
        p = params
        if p.get("start") is None or p.get("end") is None:
            start_d = _dt.datetime(2000, 1, 1)
            end_d = _dt.datetime(2030, 1, 1)
        else:
            start_d = _parse_dt(p["start"])
            end_d = _parse_dt(p["end"])
        delta = (end_d - start_d).total_seconds()
        offs = rng_np.uniform(0, max(0.0, delta), n)
        out = [start_d + _dt.timedelta(seconds=float(o)) for o in offs]
        if g == "date":
            out = [d.date() if isinstance(d, _dt.datetime) else d for d in out]
        return out
    # Should be unreachable: _can_vectorize filters unknown generators.
    raise SpecError(f"Cannot vectorize generator '{g}'")


def _build_table_vectorized(self, table, col_order, fk_cols, fk_rows, fk_index,
                            parent_selectors, rng, rng_np, n) -> List[Dict[str, Any]]:
    """Generate every column of ``table`` in one column-wise pass."""
    col_values: Dict[str, List[Any]] = {}

    rel_assign: Dict[str, List[Dict[str, Any]]] = {}
    for rel in table.relationships:
        if rel.parent_table == table.name:
            continue
        alias = rel.alias or rel.parent_table
        sel = parent_selectors.get(alias)
        prows = fk_rows.get(rel.parent_table, [])
        if not prows:
            raise SpecError(f"No parent rows available for {rel.to}")
        if sel is not None and sel.fixed:
            rel_assign[alias] = sel.assignments
        else:
            idx = rng_np.integers(0, len(prows), n)
            rel_assign[alias] = [prows[int(i)] for i in idx]

    for col in col_order:
        g, params = resolve_generator(col)
        vals = _gen_vectorized(col, g, params, n, rng_np, fk_index)
        if col.pii_strategy in ("mask", "redact", "hash"):
            vals = [_pii_transform(col.pii_strategy, v, col, None) for v in vals]
        nullable = col.constraints.get("nullable", True)
        null_rate = float(col.constraints.get("null_rate", 0.0))
        if nullable and null_rate > 0:
            mask = rng_np.random(n) < null_rate
            vals = [None if m else v for m, v in zip(mask, vals)]
        col_values[col.name] = vals

    for rel in table.relationships:
        if rel.parent_table == table.name:
            continue
        alias = rel.alias or rel.parent_table
        assignments = rel_assign[alias]
        for child_col, parent_col in zip(rel.child_columns, rel.parent_columns):
            col_values[child_col] = [assignments[i][parent_col] for i in range(n)]

    return [
        {c.name: col_values[c.name][i] for c in table.columns}
        for i in range(n)
    ]


class GenerationEngine:
    def __init__(self, spec: Spec):
        self.spec = expand_many_to_many(spec)
        self._validate()

    def _validate(self) -> None:
        table_names = {t.name for t in self.spec.tables}
        col_index: Dict[str, set] = {}
        for t in self.spec.tables:
            col_index[t.name] = {c.name for c in t.columns}
            for unique_columns in t.unique_constraints:
                if not unique_columns:
                    raise SpecError(f"Empty unique constraint in table '{t.name}'")
                for column in unique_columns:
                    if column not in col_index[t.name]:
                        raise SpecError(
                            f"Unique constraint column unknown: {t.name}.{column}"
                        )
        # relationships + rules
        for t in self.spec.tables:
            for rel in t.relationships:
                parent_tbl = rel.parent_table
                if parent_tbl not in table_names:
                    raise SpecError(f"Relationship target unknown: {rel.to}")
                parent_columns = rel.parent_columns
                child_columns = rel.child_columns
                if len(parent_columns) != len(child_columns):
                    raise SpecError(
                        f"Relationship column count mismatch in {t.name}: "
                        f"{child_columns} -> {parent_tbl}.{parent_columns}"
                    )
                for parent_col in parent_columns:
                    if parent_col not in col_index.get(parent_tbl, set()):
                        raise SpecError(
                            f"Relationship target column unknown: {parent_tbl}.{parent_col}"
                        )
                for child_col in child_columns:
                    if child_col not in col_index[t.name]:
                        raise SpecError(
                            f"Relationship 'from' column unknown: {child_col} in {t.name}"
                        )
                if rel.allocation not in ("uniform", "balanced", "weighted"):
                    raise SpecError(f"Unknown relationship allocation: {rel.allocation}")
                if rel.cardinality not in ("many_to_one", "one_to_one"):
                    raise SpecError(f"Unknown relationship cardinality: {rel.cardinality}")
                if rel.min_children < 0:
                    raise SpecError("Relationship min_children cannot be negative")
                if rel.max_children is not None and rel.max_children < rel.min_children:
                    raise SpecError("Relationship max_children must be >= min_children")
                if rel.allocation == "weighted" and not rel.weight_column:
                    raise SpecError("Weighted relationships require weight_column")
                if rel.ordered and not rel.order_column:
                    raise SpecError(
                        f"Ordered relationship {rel.to} requires order_column"
                    )
                if rel.ordered and rel.order_column not in col_index[t.name]:
                    raise SpecError(
                        f"order_column '{rel.order_column}' unknown in {t.name}"
                    )
                if rel.max_depth is not None and rel.max_depth < 1:
                    raise SpecError(
                        f"Relationship max_depth must be >= 1: {rel.max_depth}"
                    )
                if (rel.min_children > 0 or rel.max_children is not None) \
                        and rel.allocation == "uniform" \
                        and rel.cardinality == "many_to_one":
                    raise SpecError(
                        f"Relationship {rel.to} sets min/max_children but uses "
                        f"uniform allocation; use balanced/weighted (or "
                        f"cardinality one_to_one) so the bounds are honoured"
                    )
                if not (0.0 <= rel.root_fraction <= 1.0):
                    raise SpecError(
                        f"Relationship root_fraction must be in [0, 1]: {rel.root_fraction}"
                    )
                if parent_tbl == t.name:
                    # Self-reference: the FK must be nullable because the first
                    # row (and any additional root) has no parent to point at.
                    for child_col in child_columns:
                        col = next(
                            (c for c in t.columns if c.name == child_col), None
                        )
                        if col and not col.constraints.get("nullable", True):
                            raise SpecError(
                                f"Self-referencing FK {t.name}.{child_col} must be "
                                f"nullable (root rows require NULL)"
                            )
            try:
                compile_rules(t.rules)
            except Exception as e:
                raise SpecError(f"Invalid rule in table '{t.name}': {e}")
            for c in t.columns:
                for d in c.depends_on:
                    if d not in col_index[t.name]:
                        raise SpecError(f"depends_on '{d}' unknown in {t.name}.{c.name}")
                for d in conditional_determinants(c):
                    if d not in col_index[t.name]:
                        raise SpecError(
                            f"conditional generator determinant '{d}' unknown "
                            f"in {t.name}.{c.name} (params.on)"
                        )
                    if d == c.name:
                        raise SpecError(
                            f"conditional generator in {t.name}.{c.name} is "
                            f"conditioned on itself (params.on)"
                        )
                for branch in c.when:
                    try:
                        condition = Rule(branch.condition)
                    except Exception as e:
                        raise SpecError(
                            f"Invalid conditional generator in '{t.name}.{c.name}': {e}"
                        )
                    for ref in condition.column_refs():
                        if ref not in col_index[t.name]:
                            raise SpecError(
                                f"Conditional generator reference '{ref}' unknown "
                                f"in {t.name}.{c.name}"
                            )

    def run(self, stream_sink: Any = None, vectorized: bool = False,
            chunk_size: Optional[int] = None,
            progress: Optional[Callable[[str, int, int], None]] = None) -> GenerationResult:
        """Generate tables, optionally writing streamed output in chunks.

        ``progress`` receives ``(table_name, emitted, table_total)`` after
        each successful sink write. Exceptions propagate so prior chunks stay
        available to the caller.
        """
        if chunk_size is not None and chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        spec = self.spec
        seed = spec.settings.seed
        rng = __import__("random").Random(seed)
        rng_np = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        faker = Faker()
        if seed is not None:
            faker.seed_instance(seed)

        fk_rows: Dict[str, List[Dict[str, Any]]] = {}
        fk_index: Dict[str, List[Any]] = {}
        out: Dict[str, List[Dict[str, Any]]] = {}
        columns: Dict[str, List[str]] = {}

        topo_order, deps = _topo_tables(spec.tables)
        # transitive dependents: tables reachable from each table via relationships
        all_dependents: Dict[str, set] = {t.name: set() for t in spec.tables}
        for t in reversed(topo_order):
            for p in deps.get(t.name, []):
                all_dependents[p].add(t.name)
                all_dependents[p] |= all_dependents[t.name]
        done: set = set()

        for table in topo_order:
            rules = compile_rules(table.rules)
            fk_cols = {
                child for rel in table.relationships for child in rel.child_columns
            }
            col_order = _topo_columns(table, fk_cols)

            # resolved generators + params per column
            gen_fns: Dict[str, Callable[[GenContext], Any]] = {}
            gen_params: Dict[str, dict] = {}
            gen_names: Dict[str, str] = {}
            when_fns: Dict[str, list] = {}
            for c in table.columns:
                gstr, params = resolve_generator(c)
                if gstr is None:
                    raise SpecError(f"No generator resolved for {table.name}.{c.name}")
                gen_fns[c.name] = build_generator(gstr)
                gen_params[c.name] = params
                gen_names[c.name] = gstr
                branches = []
                for branch in c.when:
                    branch_spec = ColumnSpec(
                        name=c.name,
                        dtype=c.dtype,
                        generator=branch.generator,
                        params=dict(branch.params),
                    )
                    branch_name, branch_params = resolve_generator(branch_spec)
                    if branch_name is None:
                        raise SpecError(
                            f"No generator resolved for conditional branch "
                            f"{table.name}.{c.name}"
                        )
                    branches.append((
                        Rule(branch.condition),
                        branch_name,
                        build_generator(branch_name),
                        branch_params,
                    ))
                when_fns[c.name] = branches

            # Reverse dependency graph used to regenerate derived columns when
            # a referenced column changes during targeted rule retries.
            column_dependents: Dict[str, set] = {c.name: set() for c in col_order}
            for c in col_order:
                refs = set(c.depends_on)
                for branch in c.when:
                    refs.update(Rule(branch.condition).column_refs())
                for ref in refs:
                    if ref in column_dependents and ref != c.name:
                        column_dependents[ref].add(c.name)
            changed = True
            while changed:
                changed = False
                for source in list(column_dependents):
                    expanded = set(column_dependents[source])
                    for child in list(expanded):
                        expanded |= column_dependents.get(child, set())
                    if expanded != column_dependents[source]:
                        column_dependents[source] = expanded
                        changed = True

            # per-column uniqueness sets + sequence counters
            unique_sets: Dict[str, Optional[set]] = {}
            seq_counters: Dict[str, int] = {}
            for c in table.columns:
                unique_sets[c.name] = set() if c.constraints.get("unique") else None
                seq_counters[c.name] = 0
            key_constraints: List[List[str]] = []
            if table.primary_key:
                key_constraints.append(
                    [table.primary_key]
                    if isinstance(table.primary_key, str)
                    else list(table.primary_key)
                )
            key_constraints.extend([list(cols) for cols in table.unique_constraints])
            key_sets = {tuple(cols): set() for cols in key_constraints}

            parent_selectors: Dict[str, _ParentSelector] = {}
            self_refs = [rel for rel in table.relationships
                         if rel.parent_table == table.name]
            self_ref_pool: Dict[str, List[Dict[str, Any]]] = {
                rel.alias or rel.parent_table: [] for rel in self_refs
            }
            self_ref_depths: Dict[str, List[int]] = {
                rel.alias or rel.parent_table: [] for rel in self_refs
            }
            m2m = table.metadata.get("m2m")
            m2m_parent_plans: List[Dict[str, Dict[str, Any]]] = []
            if m2m is not None:
                a_rows = fk_rows[m2m["table_a"]]
                b_rows = fk_rows[m2m["table_b"]]
                _validate_m2m_feasible(m2m, len(a_rows), len(b_rows))
                pairs = _generate_m2m_pairs(a_rows, b_rows, m2m, rng)
                a_by = {r[m2m["pk_a"]]: r for r in a_rows}
                b_by = {r[m2m["pk_b"]]: r for r in b_rows}
                m2m_parent_plans = [
                    {m2m["alias_a"]: a_by[av], m2m["alias_b"]: b_by[bv]}
                    for av, bv in pairs
                ]
            for rel in table.relationships:
                if m2m is not None:
                    continue  # junction links are injected per row
                if rel.parent_table == table.name:
                    continue  # self-reference is resolved dynamically
                parent_selectors[rel.alias or rel.parent_table] = _ParentSelector(
                    rel,
                    fk_rows.get(rel.parent_table, []),
                    table.row_count,
                    rng,
                )

            rows: List[Dict[str, Any]] = []
            stream_buffer: List[Dict[str, Any]] = []
            emitted = 0
            ordered_table = any(rel.ordered and rel.order_column for rel in table.relationships)
            attempts_total = spec.settings.max_rule_attempts
            fallback = spec.settings.fallback

            def emit_chunk(chunk: List[Dict[str, Any]]) -> None:
                nonlocal emitted
                if stream_sink is None or not chunk:
                    return
                stream_sink.write_table(table.name, chunk)
                emitted += len(chunk)
                if progress is not None:
                    progress(table.name, emitted, table.row_count)

            used_vectorized = vectorized and _can_vectorize(table)
            if used_vectorized:
                rows = _build_table_vectorized(
                    self, table, col_order, fk_cols, fk_rows, fk_index,
                    parent_selectors,
                    rng, rng_np, table.row_count,
                )
                if stream_sink is not None and chunk_size is not None and not ordered_table:
                    stream_buffer.extend(rows)
            else:
                for i in range(table.row_count):
                    if m2m is not None:
                        parent_plan = m2m_parent_plans[i]
                    else:
                        parent_plan = {
                            alias: selector.next()
                            for alias, selector in parent_selectors.items()
                            if selector.fixed
                        }
                    # Self-referencing relationships: the parent pool grows as rows
                    # are emitted, so every FK points at an already-generated row.
                    # The first row (empty pool) and any row below root_fraction are
                    # roots with a NULL foreign key.
                    root_aliases: set = set()
                    for rel in self_refs:
                        alias = rel.alias or rel.parent_table
                        if not self_ref_pool[alias]:
                            root_aliases.add(alias)
                        elif rng.random() < rel.root_fraction:
                            root_aliases.add(alias)
                    row = self._generate_row(
                        table, col_order, rules, gen_fns, gen_params, gen_names, when_fns,
                        column_dependents,
                        unique_sets, key_sets, seq_counters, fk_rows, fk_index,
                    rng, faker, attempts_total, parent_plan,
                    self_ref_pool, root_aliases, self_ref_depths,
                )
                    if row is None:
                        if fallback == "drop":
                            continue
                        if fallback == "null":
                            row = self._nulled_row(table, col_order, fk_rows, rng, faker)
                        else:
                            raise RuleViolation(
                                f"Could not satisfy rules for a row in '{table.name}' "
                                f"after {attempts_total} attempts (fallback={fallback})."
                            )
                    # commit sequence increments + unique values already added per accepted row
                    rows.append(row)
                    if stream_sink is not None and chunk_size is not None and not ordered_table:
                        stream_buffer.append(row)
                        if len(stream_buffer) >= chunk_size:
                            emit_chunk(stream_buffer)
                            stream_buffer = []
                    for rel in self_refs:
                        alias = rel.alias or rel.parent_table
                        pool = self_ref_pool[alias]
                        depths = self_ref_depths[alias]
                        child_cols = rel.child_columns
                        parent_cols = rel.parent_columns
                        if row.get(child_cols[0]) is None:
                            new_depth = 0
                        else:
                            key = tuple(row.get(c) for c in child_cols)
                            new_depth = 0
                            for r, d in zip(pool, depths):
                                if tuple(r.get(pc) for pc in parent_cols) == key:
                                    new_depth = d + 1
                                    break
                        depths.append(new_depth)
                        pool.append(row)

            # Ordered relationships: number each child row 1..n within its
            # parent group so events come out in order per parent.
            for rel in table.relationships:
                if rel.ordered and rel.order_column:
                    _apply_group_order(
                        rows, rel.child_columns, rel.order_column
                    )

            if stream_sink is not None:
                if chunk_size is None:
                    emit_chunk(rows)
                elif ordered_table:
                    # Ordering is a post-processing step, so defer all writes
                    # for this table until the final order is known.
                    for start in range(0, len(rows), chunk_size):
                        emit_chunk(rows[start:start + chunk_size])
                elif stream_buffer:
                    emit_chunk(stream_buffer)

            fk_rows[table.name] = rows
            for c in table.columns:
                fk_index[f"{table.name}.{c.name}"] = [r.get(c.name) for r in rows]

            if stream_sink is None:
                out[table.name] = rows
            columns[table.name] = [c.name for c in table.columns]

            done.add(table.name)
            # free parent tables whose every dependent table is already done
            for a in list(fk_rows.keys()):
                if a != table.name and all_dependents.get(a, set()) <= done:
                    fk_rows.pop(a, None)

        return GenerationResult(out, columns)

    def _generate_row(self, table, col_order, rules, gen_fns, gen_params,
                      gen_names, when_fns, column_dependents,
                      unique_sets, key_sets, seq_counters, fk_rows, fk_index,
                      rng, faker, attempts_total, parent_plan,
                      self_ref_pool=None, root_aliases=None, self_ref_depths=None):
        all_columns = {c.name for c in col_order}
        fk_columns = {
            child for rel in table.relationships for child in rel.child_columns
        }
        parent_dependent_columns = {
            c.name for c in col_order if set(c.depends_on).intersection(fk_columns)
        }
        parents: Dict[str, Dict[str, Any]] = {}
        row: Dict[str, Any] = {}
        selected_names: Dict[str, str] = {}
        regenerate_all = True
        regenerate: set = set(all_columns)

        for _attempt in range(attempts_total):
            if regenerate_all:
                parents = {}
                row = {}
                for rel in table.relationships:
                    to_tbl = rel.parent_table
                    parent_columns = rel.parent_columns
                    child_columns = rel.child_columns
                    alias = rel.alias or to_tbl
                    if to_tbl == table.name:
                        # self-reference: parent pool contains rows generated
                        # before the current one.
                        pool = (self_ref_pool or {}).get(alias, [])
                        depths = (self_ref_depths or {}).get(alias, [])
                        if (root_aliases and alias in root_aliases) or not pool:
                            parents[alias] = None
                            for child_col in child_columns:
                                row[child_col] = None
                            continue
                        parent = parent_plan.get(alias)
                        if parent is None:
                            if rel.max_depth is not None:
                                eligible = [p for p, d in zip(pool, depths)
                                            if d < rel.max_depth]
                                if not eligible:
                                    # No parent can extend the tree further;
                                    # this row becomes an additional root.
                                    parents[alias] = None
                                    for child_col in child_columns:
                                        row[child_col] = None
                                    continue
                                parent = rng.choice(eligible)
                            else:
                                parent = rng.choice(pool)
                        parents[alias] = parent
                        for child_col, parent_col in zip(child_columns, parent_columns):
                            row[child_col] = parent[parent_col]
                        continue
                    prows = fk_rows.get(to_tbl, [])
                    if not prows:
                        raise SpecError(f"No parent rows available for {rel.to}")
                    parent = parent_plan.get(alias) or rng.choice(prows)
                    parents[alias] = parent
                    for child_col, parent_col in zip(child_columns, parent_columns):
                        row[child_col] = parent[parent_col]
                row["__parents__"] = parents
                regenerate = set(all_columns)

            # Generate only the columns implicated by the previous failed
            # rules. On the first/full attempt this is every free column.
            for col in col_order:
                if col.name not in regenerate:
                    continue

                gen = gen_fns[col.name]
                params = gen_params[col.name]
                gen_name = gen_names[col.name]
                for condition, branch_name, branch_gen, branch_params in when_fns.get(col.name, []):
                    try:
                        matched = condition.check(row, parents)
                    except Exception:
                        matched = False
                    if matched:
                        gen = branch_gen
                        params = branch_params
                        gen_name = branch_name
                        break
                selected_names[col.name] = gen_name

                nullable = col.constraints.get("nullable", True)
                null_rate = float(col.constraints.get("null_rate", 0.0))
                # A conditional generator carries NULL as a value inside each
                # per-key distribution, so it owns its own missingness.
                # Injecting nulls on top at the column's marginal rate would
                # double-count them, and would put them in the wrong rows:
                # P(NULL | X) varies enormously by X -- some CFPB Issues have
                # no sub-issue at all -- and one marginal rate applied
                # uniformly is exactly the independence assumption this
                # generator exists to remove.
                if (col.dtype != "fk" and nullable and null_rate > 0
                        and gen_name != "conditional"
                        and rng.random() < null_rate):
                    row[col.name] = None
                    continue

                cctx = GenContext(
                    row=row, parents=parents, rng=rng, faker=faker, params=params,
                    table=table.name, col=col.name,
                    seq=seq_counters[col.name] + 1,
                    unique_set=unique_sets[col.name], fk_index=fk_index,
                )

                def make_value():
                    pii = col.pii_strategy
                    if pii == "faker" and not gen_name.startswith("faker."):
                        return _pii_faker_value(col, cctx)
                    value = gen(cctx)
                    if pii in ("mask", "redact", "hash"):
                        value = _pii_transform(pii, value, col, cctx)
                    return value

                val = make_value()
                uset = unique_sets[col.name]
                if uset is not None:
                    inner = 0
                    while val in uset and inner < 10000:
                        val = make_value()
                        inner += 1
                    if val in uset:
                        raise SpecError(
                            f"Cannot satisfy uniqueness for {table.name}.{col.name}"
                        )
                row[col.name] = val

            failed = [rule for rule in rules if not rule.check(row, parents)]
            duplicate_key_columns: set = set()
            for columns, used in key_sets.items():
                key = tuple(row.get(column) for column in columns)
                if key in used:
                    duplicate_key_columns.update(columns)

            if not failed and not duplicate_key_columns:
                # Commit unique values + sequence counters only after every
                # rule passes, preserving the original retry semantics.
                for col in col_order:
                    if unique_sets[col.name] is not None:
                        unique_sets[col.name].add(row.get(col.name))
                    if selected_names.get(col.name) == "sequence":
                        seq_counters[col.name] += 1
                for columns, used in key_sets.items():
                    used.add(tuple(row.get(column) for column in columns))
                row.pop("__parents__", None)
                return row

            refs = set(duplicate_key_columns)
            requires_parent_retry = False
            for rule in failed:
                refs.update(rule.column_refs())
                if (rule.parent_aliases()
                        or refs.intersection(fk_columns)
                        or refs.intersection(parent_dependent_columns)):
                    requires_parent_retry = True

            if requires_parent_retry:
                regenerate_all = True
                regenerate = set(all_columns)
                continue

            regenerate = {ref for ref in refs if ref in all_columns}
            expanded = set(regenerate)
            for ref in list(regenerate):
                expanded |= column_dependents.get(ref, set())
            regenerate = expanded
            if not regenerate:
                # A rule with no regenerable local columns cannot benefit from
                # targeted retry; fall back to a full attempt.
                regenerate_all = True
                regenerate = set(all_columns)
            else:
                regenerate_all = False

        return None

    def _nulled_row(self, table, col_order, fk_rows, rng, faker):
        # best-effort: build fk values then null free columns
        row: Dict[str, Any] = {}
        for rel in table.relationships:
            to_tbl = rel.parent_table
            prows = fk_rows.get(to_tbl, [])
            if prows:
                parent = rng.choice(prows)
                for child_col, parent_col in zip(rel.child_columns, rel.parent_columns):
                    row[child_col] = parent[parent_col]
        for col in col_order:
            row[col.name] = None
        return row
