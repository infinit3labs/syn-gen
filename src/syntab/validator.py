"""Validation: compare real vs synthetic data at the column level.

For each column we compute a distributional distance and a pass/warn/fail
status against configurable thresholds:

  * numeric / datetime -> Kolmogorov-Smirnov statistic D (supremum difference
    of empirical CDFs), computed without external deps.
  * categorical / boolean -> Total Variation Distance (0.5 * sum |p_i - q_i|).
  * text / string -> length-statistic distance + character-distribution TVD
    (structural similarity only; free text cannot be matched semantically).

The report aggregates per-column results and an overall verdict.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .spec import Spec


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ColumnReport:
    name: str
    kind: str
    metric: str
    distance: float
    threshold: float
    status: str  # pass | warn | fail
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    columns: List[ColumnReport] = field(default_factory=list)
    real_rows: int = 0
    synthetic_rows: int = 0
    n_pass: int = 0
    n_warn: int = 0
    n_fail: int = 0
    mean_distance: float = 0.0
    overall_pass: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["overall_pass"] = self.overall_pass
        return d

    def to_text(self) -> str:
        lines = [
            f"Validation: real={self.real_rows} rows, synthetic={self.synthetic_rows} rows",
            f"  pass={self.n_pass}  warn={self.n_warn}  fail={self.n_fail}  "
            f"mean_distance={self.mean_distance:.4f}",
            f"  OVERALL: {'PASS' if self.overall_pass else 'FAIL'}",
            "-" * 72,
            f"{'column':<28}{'kind':<12}{'metric':<8}{'dist':<8}{'thr':<8}status",
            "-" * 72,
        ]
        for c in self.columns:
            lines.append(
                f"{c.name[:27]:<28}{c.kind:<12}{c.metric:<8}{c.distance:<8.4f}"
                f"{c.threshold:<8.2f}{c.status}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------

def _ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    if len(a) == 0 or len(b) == 0:
        return 1.0
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / len(a)
    cdf_b = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _tvd(p: Dict[Any, float], q: Dict[Any, float]) -> float:
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def _freqs(series: pd.Series) -> Dict[Any, float]:
    vc = series.value_counts(dropna=True)
    total = float(vc.sum())
    if total == 0:
        return {}
    return {str(k): float(v) / total for k, v in vc.items()}


def _char_freqs(series: pd.Series) -> Dict[str, float]:
    text = "".join(series.dropna().astype(str).tolist())
    if not text:
        return {}
    counts: Dict[str, float] = {}
    for ch in text:
        counts[ch] = counts.get(ch, 0.0) + 1.0
    total = float(sum(counts.values()))
    return {k: v / total for k, v in counts.items()}


# ---------------------------------------------------------------------------
# Per-kind comparison
# ---------------------------------------------------------------------------

def _compare_numeric(real: pd.Series, synth: pd.Series, thr: Dict[str, float]):
    r = real.dropna().astype(float).to_numpy()
    s = synth.dropna().astype(float).to_numpy()
    d = _ks_statistic(r, s)
    status = "pass" if d <= thr["pass"] else ("warn" if d <= thr["warn"] else "fail")
    details = {
        "real_mean": float(np.mean(r)) if len(r) else None,
        "synth_mean": float(np.mean(s)) if len(s) else None,
        "real_std": float(np.std(r)) if len(r) else None,
        "synth_std": float(np.std(s)) if len(s) else None,
    }
    return d, status, details


def _compare_categorical(real: pd.Series, synth: pd.Series, thr: Dict[str, float]):
    d = _tvd(_freqs(real), _freqs(synth))
    status = "pass" if d <= thr["pass"] else ("warn" if d <= thr["warn"] else "fail")
    return d, status, {}


def _compare_boolean(real: pd.Series, synth: pd.Series, thr: Dict[str, float]):
    rp = float(real.dropna().astype(bool).mean()) if real.notna().any() else 0.0
    sp = float(synth.dropna().astype(bool).mean()) if synth.notna().any() else 0.0
    d = abs(rp - sp)
    status = "pass" if d <= thr["pass"] else ("warn" if d <= thr["warn"] else "fail")
    details = {"real_p": rp, "synth_p": sp}
    return d, status, details


def _compare_datetime(real: pd.Series, synth: pd.Series, thr: Dict[str, float]):
    try:
        rp = pd.to_datetime(real, errors="coerce", format="mixed")
        sp = pd.to_datetime(synth, errors="coerce", format="mixed")
    except (ValueError, TypeError):
        rp = pd.to_datetime(real, errors="coerce")
        sp = pd.to_datetime(synth, errors="coerce")
    r = rp.dropna().astype("int64").to_numpy()
    s = sp.dropna().astype("int64").to_numpy()
    d = _ks_statistic(r, s)
    status = "pass" if d <= thr["pass"] else ("warn" if d <= thr["warn"] else "fail")
    details = {
        "real_range": [
            str(pd.to_datetime(real, errors="coerce").min()),
            str(pd.to_datetime(real, errors="coerce").max()),
        ],
        "synth_range": [
            str(pd.to_datetime(synth, errors="coerce").min()),
            str(pd.to_datetime(synth, errors="coerce").max()),
        ],
    }
    return d, status, details


def _compare_text(real: pd.Series, synth: pd.Series, thr: Dict[str, float]):
    rl = real.dropna().astype(str).str.len()
    sl = synth.dropna().astype(str).str.len()
    len_mean_diff = abs(float(rl.mean()) - float(sl.mean())) / max(1.0, float(rl.mean()))
    char_tvd = _tvd(_char_freqs(real), _char_freqs(synth))
    d = max(len_mean_diff, char_tvd)
    status = "pass" if d <= thr["pass"] else ("warn" if d <= thr["warn"] else "fail")
    # Free text cannot be matched semantically -> never a hard fail.
    if status == "fail":
        status = "warn"
    details = {
        "real_len_mean": float(rl.mean()),
        "synth_len_mean": float(sl.mean()),
        "char_tvd": char_tvd,
        "note": "structural only; free text is not matched semantically",
    }
    return d, status, details


# ---------------------------------------------------------------------------
# Kind inference + orchestration
# ---------------------------------------------------------------------------

def _infer_kind(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series.dtype):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series.dtype):
        return "numeric"
    s = series.dropna().astype(str)
    try:
        parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    except (ValueError, TypeError):
        parsed = pd.to_datetime(series, errors="coerce")
    if parsed.notna().mean() > 0.8:
        return "datetime"
    n_unique = s.nunique()
    if n_unique / max(1, len(s)) < 0.5 and n_unique <= 50:
        return "categorical"
    return "text"


_DEFAULT_THRESHOLDS = {
    "numeric": {"pass": 0.1, "warn": 0.2},
    "datetime": {"pass": 0.1, "warn": 0.2},
    "categorical": {"pass": 0.1, "warn": 0.25},
    "boolean": {"pass": 0.1, "warn": 0.2},
    "text": {"pass": 0.35, "warn": 0.6},
}


def validate(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    spec: Optional[Spec] = None,
    thresholds: Optional[Dict[str, Dict[str, float]]] = None,
) -> ValidationReport:
    thr = _DEFAULT_THRESHOLDS.copy()
    if thresholds:
        for k, v in thresholds.items():
            thr[k] = {**thr.get(k, {}), **v}

    # align on shared columns
    cols = [c for c in real.columns if c in synthetic.columns]
    reports: List[ColumnReport] = []
    dists: List[float] = []

    for name in cols:
        rcol = real[name]
        scol = synthetic[name]
        kind = _infer_kind(rcol)
        t = thr.get(kind, {"pass": 0.1, "warn": 0.2})
        if kind == "numeric":
            d, status, det = _compare_numeric(rcol, scol, t)
            metric = "ks"
        elif kind == "datetime":
            d, status, det = _compare_datetime(rcol, scol, t)
            metric = "ks"
        elif kind == "boolean":
            d, status, det = _compare_boolean(rcol, scol, t)
            metric = "tvd"
        elif kind == "categorical":
            d, status, det = _compare_categorical(rcol, scol, t)
            metric = "tvd"
        else:
            d, status, det = _compare_text(rcol, scol, t)
            metric = "max(tvd,len)"
        reports.append(ColumnReport(name=name, kind=kind, metric=metric,
                                    distance=round(d, 4), threshold=t["pass"],
                                    status=status, details=det))
        dists.append(d)

    n_pass = sum(1 for c in reports if c.status == "pass")
    n_warn = sum(1 for c in reports if c.status == "warn")
    n_fail = sum(1 for c in reports if c.status == "fail")
    # Free-text columns are structural-only and never hard-fail the verdict.
    overall = all(c.status != "fail" for c in reports if c.kind != "text")
    return ValidationReport(
        columns=reports,
        real_rows=len(real),
        synthetic_rows=len(synthetic),
        n_pass=n_pass, n_warn=n_warn, n_fail=n_fail,
        mean_distance=round(float(np.mean(dists)), 4) if dists else 0.0,
        overall_pass=overall,
    )
