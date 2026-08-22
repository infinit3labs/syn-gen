"""Dataset profiler: analyse an existing dataset and emit a compatible Spec.

The profiler infers, per column:
  * a logical ``dtype`` (int | float | bool | datetime | date | uuid | str)
  * a ``profile`` (numeric / categorical / datetime / string stats)
  * structural hints (unique -> ``constraints.unique``, candidate primary key)

Every column is emitted with ``generator: auto`` so the generation engine can
re-synthesize a compatible dataset from the captured statistics (the profiling
round-trip). Built-in logic only; no external ML required.
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import re

from .spec import (
    CategoricalProfile,
    ColumnProfile,
    ColumnSpec,
    NumericProfile,
    RelationshipSpec,
    Settings,
    Spec,
    SpecMetadata,
    TableSpec,
)

_UUID_RE = __import__("re").compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def _py(v: Any) -> Any:
    if hasattr(v, "item"):
        try:
            return v.item()
        except Exception:
            return v
    if isinstance(v, (pd.Timestamp, _dt.datetime, _dt.date)):
        return v
    return v


def _is_uuid(s: pd.Series) -> bool:
    sample = s.dropna().astype(str).head(200)
    if len(sample) == 0:
        return False
    return bool(sample.map(lambda x: bool(_UUID_RE.match(x))).mean() > 0.9)


def _string_pattern(values: List[str]) -> str:
    """Map a representative value to a bothify-style pattern."""
    rep = max(values, key=len) if values else ""
    out = []
    for ch in rep:
        if ch.isalpha():
            out.append("?")
        elif ch.isdigit():
            out.append("#")
        else:
            out.append(ch)
    return "".join(out)


class DatasetProfiler:
    def __init__(
        self,
        df: pd.DataFrame,
        name: Optional[str] = None,
        source: Optional[str] = None,
        sample: Optional[int] = 5000,
        seed: Optional[int] = None,
        max_categorical: int = 50,
        pii_columns: Optional[List[str]] = None,
        pii_strategy: str = "faker",
    ):
        self.full = df
        self.name = name or "profiled"
        self.source = source or "unknown"
        self.sample = sample
        self.seed = seed
        self.max_categorical = max_categorical
        self.pii_columns = set(pii_columns or [])
        self.pii_strategy = pii_strategy

    # ----- PII helpers -----
    _EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    _SSN_RE = re.compile(r"^\d{3}-\d{2}-\d{4}$")
    _PHONE_RE = re.compile(r"^\+?[\d\s().-]{7,}$")

    @classmethod
    def _looks_like_pii(cls, s: pd.Series) -> Optional[str]:
        """Heuristically detect a PII column from its string values.

        Returns a suggested faker provider, or None.
        """
        sample = s.dropna().astype(str).head(50)
        if len(sample) == 0:
            return None
        if sample.map(lambda x: bool(cls._EMAIL_RE.match(x))).mean() > 0.8:
            return "email"
        if sample.map(lambda x: bool(cls._SSN_RE.match(x))).mean() > 0.8:
            return "ssn"
        if sample.map(lambda x: bool(cls._PHONE_RE.match(x))).mean() > 0.8:
            return "phone_number"
        return None

    @staticmethod
    def _faker_provider_for(name: str, dtype: str) -> str:
        name_l = (name or "").lower()
        mapping = [
            ("email", "email"), ("phone", "phone_number"), ("mobile", "phone_number"),
            ("address", "address"), ("city", "city"), ("country", "country"),
            ("company", "company"), ("first_name", "first_name"),
            ("last_name", "last_name"), ("name", "name"), ("ssn", "ssn"),
            ("zip", "zipcode"), ("postal", "zipcode"),
        ]
        for key, prov in mapping:
            if key in name_l:
                return prov
        if dtype in ("int", "float"):
            return "random_number"
        return "word"

    # ----- factory -----
    @classmethod
    def from_file(cls, path: str, **kw) -> "DatasetProfiler":
        p = Path(path)
        if p.suffix.lower() == ".parquet":
            df = pd.read_parquet(p)
        elif p.suffix.lower() in (".csv", ".txt"):
            df = pd.read_csv(p)
        elif p.suffix.lower() == ".json":
            df = pd.read_json(p)
        elif p.suffix.lower() in (".xlsx", ".xls"):
            df = pd.read_excel(p)
        else:
            raise ValueError(f"Unsupported data file: {p.suffix}")
        name = kw.pop("name", None) or p.stem
        return cls(df, name=name, source=str(p), **kw)

    # ----- public -----
    def profile(self) -> Spec:
        table = self._build_table_spec(self.full, self.name, self.source)
        meta = SpecMetadata(
            name=self.name,
            description=f"Profiled from {self.source}",
            tags=["profiled"],
            source=f"profiled:{self.source}",
            created_at=_dt.datetime.now(),
        )
        return Spec(
            metadata=meta,
            settings=Settings(seed=self.seed, fallback="raise", max_rule_attempts=100),
            tables=[table],
        )

    def _build_table_spec(self, df: pd.DataFrame, name: str, source: str) -> TableSpec:
        if self.sample and len(df) > self.sample:
            df = df.sample(n=self.sample, random_state=self.seed)
        n = len(df)
        cols: List[ColumnSpec] = []
        for c in df.columns:
            spec = self._profile_column(str(c), df[c], n)
            if spec is not None:
                cols.append(spec)
        pk = self._detect_pk(cols)
        return TableSpec(name=name, row_count=n, primary_key=pk, columns=cols)

    @classmethod
    def profile_set(
        cls,
        tables: Dict[str, pd.DataFrame],
        name: Optional[str] = None,
        sample: Optional[int] = 5000,
        seed: Optional[int] = None,
        max_categorical: int = 50,
        pii_columns: Optional[List[str]] = None,
        pii_strategy: str = "faker",
    ) -> Spec:
        """Profile several related tables and infer foreign-key relationships.

        Relationship inference matches a column ``<parent>_id`` to a table whose
        name matches ``<parent>`` (or its plural) and whose primary-key values
        are a superset of the column's values. Strong numeric correlations are
        recorded as ``metadata.suggested_depends_on`` hints (non-breaking).
        """
        profilers = {
            t: cls(df, name=t, sample=sample, seed=seed, max_categorical=max_categorical,
                   pii_columns=pii_columns, pii_strategy=pii_strategy)
            for t, df in tables.items()
        }
        table_specs = {
            t: profilers[t]._build_table_spec(df, t, f"profiled:{t}")
            for t, df in tables.items()
        }
        rels = cls._infer_relationships(table_specs, tables)
        for t, rel_list in rels.items():
            table_specs[t].relationships = rel_list
        cls._infer_depends_on(table_specs, tables)
        meta = SpecMetadata(
            name=name or "profiled_set",
            description="Profiled from multiple related tables",
            tags=["profiled", "multi-table"],
            source=f"profiled:{list(tables)}",
            created_at=_dt.datetime.now(),
        )
        return Spec(
            metadata=meta,
            settings=Settings(seed=seed, fallback="raise", max_rule_attempts=100),
            tables=list(table_specs.values()),
        )

    @staticmethod
    def _infer_relationships(
        table_specs: Dict[str, "TableSpec"], dfs: Dict[str, pd.DataFrame]
    ) -> Dict[str, List[RelationshipSpec]]:
        rels: Dict[str, List[RelationshipSpec]] = {t: [] for t in table_specs}
        names = set(table_specs)
        for cname, cts in table_specs.items():
            cdf = dfs[cname]
            for col in cts.columns:
                colname = col.name
                if colname == "id" or not colname.endswith("_id"):
                    continue
                base = colname[:-3]
                candidates = [b for b in (base, base + "s", base.rstrip("s"))
                              if b in names and b != cname]
                if not candidates:
                    continue
                child_vals = {str(v) for v in cdf[colname].dropna().unique()}
                if not child_vals:
                    continue
                for pname in candidates:
                    pts = table_specs[pname]
                    pkp = pts.primary_key
                    if not pkp:
                        continue
                    parent_vals = {str(v) for v in dfs[pname][pkp].dropna().unique()}
                    if child_vals.issubset(parent_vals):
                        rels[cname].append(RelationshipSpec(
                            **{"from": colname, "to": f"{pname}.{pkp}", "alias": pname}
                        ))
                        break
        return rels

    @staticmethod
    def _infer_depends_on(
        table_specs: Dict[str, "TableSpec"], dfs: Dict[str, pd.DataFrame]
    ) -> None:
        for t, cts in table_specs.items():
            df = dfs[t]
            num_cols = [c.name for c in cts.columns if c.dtype in ("int", "float")]
            sugg: Dict[str, List[str]] = {}
            for i, a in enumerate(num_cols):
                for b in num_cols[i + 1:]:
                    if a not in df.columns or b not in df.columns:
                        continue
                    try:
                        r = df[[a, b]].dropna().corr().iloc[0, 1]
                    except Exception:
                        r = 0
                    if r is not None and abs(float(r)) > 0.95:
                        sugg.setdefault(a, []).append(b)
                        sugg.setdefault(b, []).append(a)
            if sugg:
                md = dict(cts.metadata or {})
                md["suggested_depends_on"] = sugg
                cts.metadata = md

    # ----- internals -----
    def _profile_column(self, name: str, series: pd.Series, n: int) -> Optional[ColumnSpec]:
        s = series.dropna()
        null_rate = round(1 - len(s) / n, 4) if n else 1.0
        if len(s) == 0:
            return None  # fully-null column: skip

        dtype, profile = self._infer_type_and_profile(s)
        params: dict = {}
        constraints: dict = {"nullable": null_rate > 0, "null_rate": null_rate}

        # uniqueness
        n_unique = int(s.nunique())
        unique = n_unique == len(s) and len(s) > 0
        if unique:
            constraints["unique"] = True

        generator = "auto"
        # integer id-like sequential column -> sequence
        if dtype == "int" and unique and self._looks_sequential(s):
            generator = "sequence"

        spec = ColumnSpec(
            name=name,
            dtype=dtype,
            generator=generator,
            params=params,
            constraints=constraints,
            profile=profile,
        )

        # PII handling: flag the column and make sure no real value is baked
        # into the generated spec (otherwise a profiled dataset would leak PII).
        is_pii = name in self.pii_columns
        detected_provider = None
        if dtype == "str" and not is_pii:
            detected_provider = self._looks_like_pii(s)
            if detected_provider:
                is_pii = True
        if is_pii:
            strategy = self.pii_strategy if name in self.pii_columns else "faker"
            spec.pii = True
            spec.pii_strategy = strategy
            # Never bake real values into the spec: substitute a faker generator
            # and drop captured values / patterns / stats. The chosen strategy
            # is then applied on top of the synthetic value at generation time.
            provider = detected_provider or self._faker_provider_for(name, dtype)
            spec.generator = f"faker.{provider}"
            spec.profile = None
            spec.params = {}
            spec.constraints = {
                k: v for k, v in spec.constraints.items()
                if k in ("nullable", "null_rate")
            }

        return spec

    def _infer_type_and_profile(self, s: pd.Series) -> Tuple[str, Optional[ColumnProfile]]:
        if pd.api.types.is_bool_dtype(s.dtype):
            return "bool", None

        if pd.api.types.is_integer_dtype(s.dtype):
            return "int", ColumnProfile(numeric=self._numeric_profile(s))

        if pd.api.types.is_float_dtype(s.dtype):
            # integral floats (e.g. ids stored as float w/ NaNs) -> int
            if (s.dropna() % 1 == 0).all():
                return "int", ColumnProfile(numeric=self._numeric_profile(s))
            return "float", ColumnProfile(numeric=self._numeric_profile(s))

        # object / string
        vals = s.astype(str)
        # datetime?
        try:
            parsed = pd.to_datetime(s, errors="coerce", format="mixed")
        except (ValueError, TypeError):
            parsed = pd.to_datetime(s, errors="coerce")
        if parsed.notna().mean() > 0.8:
            mn, mx = parsed.min(), parsed.max()
            return "datetime", ColumnProfile(
                datetime_range=(_py(mn), _py(mx))
            )
        if _is_uuid(vals):
            return "uuid", None

        # categorical vs free text
        n_unique = int(vals.nunique())
        ratio = n_unique / len(vals)
        if ratio < 0.5 and n_unique <= self.max_categorical:
            counts = vals.value_counts()
            chosen = counts.head(self.max_categorical)
            total = float(chosen.sum())
            values = {str(k): round(float(v) / total, 4) for k, v in chosen.items()}
            return "str", ColumnProfile(
                categorical=CategoricalProfile(values=values, null_rate=0.0)
            )

        # free text: length stats (+ pattern if short)
        lengths = vals.str.len()
        prof = ColumnProfile(length=(int(lengths.min()), int(lengths.max())))
        if lengths.max() <= 30:
            prof.string_pattern = _string_pattern(vals.head(50).tolist())
        return "str", prof

    def _numeric_profile(self, s: pd.Series) -> NumericProfile:
        smin = float(s.min())
        smax = float(s.max())
        smean = float(s.mean())
        sstd = float(s.std()) if len(s) > 1 else 0.0
        dist = None
        log_mu = log_sigma = scale = pareto_b = pareto_xmin = None
        edges = counts = None

        if smax > smin:
            unif_std = (smax - smin) / (12 ** 0.5)
            is_uniform = abs(sstd - unif_std) < 0.1 * unif_std
            pos = (s > 0)

            if is_uniform:
                dist = "uniform"
            elif pos.all() and sstd > 0 and smean > 0:
                cv = sstd / smean
                # exponential: mean ~= std and a near-zero minimum
                if smin < (smax - smin) * 0.05 and abs(smean - sstd) < 0.1 * smean:
                    scale = smean
                    dist = "exponential"
                elif cv > 0.5:
                    logv = np.log(s)
                    log_mu, log_sigma = float(logv.mean()), float(logv.std())
                    if cv < 2.0:
                        dist = "lognormal"
                    elif smean > smin:
                        pareto_xmin = smin
                        pareto_b = smean / (smean - smin)
                        dist = "pareto"
                    else:
                        dist = "lognormal"
                else:
                    # near-symmetric positive data
                    dist = "normal"
            else:
                # not confidently uniform / positive-skewed -> match the real
                # shape directly via an empirical histogram
                dist = "empirical"

            if dist == "empirical":
                nb = min(50, max(2, int(s.nunique())))
                hist, edge = np.histogram(s, bins=nb)
                total = float(hist.sum())
                edges = [float(x) for x in edge]
                counts = [float(h) / total for h in hist]

        return NumericProfile(
            min=smin, max=smax, mean=smean, std=sstd, distribution=dist,
            log_mu=log_mu, log_sigma=log_sigma, scale=scale,
            pareto_b=pareto_b, pareto_xmin=pareto_xmin,
            hist_edges=edges, hist_counts=counts,
        )

    def _looks_sequential(self, s: pd.Series) -> bool:
        try:
            arr = s.dropna().sort_values().reset_index(drop=True)
            if len(arr) < 2:
                return False
            step = arr.iloc[1] - arr.iloc[0]
            if step not in (1, -1):
                return False
            expected = range(int(arr.iloc[0]), int(arr.iloc[0]) + len(arr) * int(step), int(step))
            return list(arr.astype(int).values) == list(expected)
        except Exception:
            return False

    def _detect_pk(self, cols: List[ColumnSpec]) -> Optional[str]:
        candidates = [c for c in cols if c.constraints.get("unique")]
        if not candidates:
            return None
        for c in candidates:
            low = c.name.lower()
            if low in ("id", "complaint id") or low.endswith("id") or low.endswith("_id"):
                return c.name
        # otherwise first unique integer column
        for c in candidates:
            if c.dtype in ("int", "str"):
                return c.name
        return candidates[0].name
