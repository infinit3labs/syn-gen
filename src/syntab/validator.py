"""Validation: compare real vs synthetic data at the column level.

For each column we compute a distributional distance and a pass/warn/fail
status against configurable thresholds:

  * numeric / datetime -> Kolmogorov-Smirnov statistic D (supremum difference
    of empirical CDFs).
  * categorical / boolean -> Total Variation Distance (0.5 * sum |p_i - q_i|).
  * text / string -> length-statistic distance + character-distribution TVD.

The report aggregates per-column results and an overall verdict.

Scope
-----
This module is MARGINALS-ONLY by construction: it compares each column
against its counterpart in isolation and says nothing about the
relationships between columns. That is a real limit, not an oversight, and
it is the reason :mod:`syntab.quality` exists -- a dataset can pass every
check here while every correlation in it has been destroyed. Use
:func:`syntab.quality.quality_report` for a graded fidelity score including
column pair trends, and :func:`syntab.quality.diagnostic_report` for
structural validity. This remains as the quick per-column pass/warn/fail
gate, and shares its metric primitives with that module so the two cannot
drift apart.

A note on text columns
----------------------
Text results used to be suppressed: a ``fail`` was rewritten to ``warn``,
and text columns were excluded from the overall verdict, on the rationale
that free text "cannot be matched semantically". Both suppressions are gone.

The rationale did not describe the metrics actually being computed. Neither
the length-mean difference nor the character-distribution TVD is semantic;
both are structural, and both are precisely what a broken text generator
gets wrong. A check that computes the right number and then declines to act
on it is worse than no check at all, because it issues a clean bill of
health it holds the evidence to contradict. Text columns now fail like any
other column.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .quality import (
    _char_frequencies as _char_freqs,
    _ks_statistic,
    _tvd,
    _value_frequencies as _freqs,
    infer_kind as _infer_kind,
)
from .spec import Spec

REPORT_SCHEMA_VERSION = "1"


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

    def to_dict(self, compact: bool = False) -> Dict[str, Any]:
        d = asdict(self)
        d["report_type"] = "comparison"
        d["report_schema_version"] = REPORT_SCHEMA_VERSION
        d["status"] = "pass" if self.overall_pass else "fail"
        if compact:
            return {
                key: d[key]
                for key in (
                    "report_type", "report_schema_version", "status",
                    "overall_pass", "real_rows", "synthetic_rows",
                    "n_pass", "n_warn", "n_fail", "mean_distance",
                )
            }
        d["overall_pass"] = self.overall_pass
        return d

    def to_text(self) -> str:
        lines = [
            f"Validation: real={self.real_rows} rows, synthetic={self.synthetic_rows} rows",
            f"  pass={self.n_pass}  warn={self.n_warn}  fail={self.n_fail}  "
            f"mean_distance={self.mean_distance:.4f}",
            f"  OVERALL: {'PASS' if self.overall_pass else 'FAIL'}",
            "  (marginals only -- see `syntab quality` for pair trends)",
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
    """Compare two text columns structurally.

    Two metrics, both structural, neither semantic:

      * ``len_mean_diff`` -- relative difference in mean value length. Catches
        a generator emitting every value at one length.
      * ``char_tvd`` -- total variation distance between the character-unigram
        distributions. Catches a generator drawing from the wrong alphabet.

    The worse of the two is the reported distance, and it is reported
    honestly: a text column fails like any other column.
    """
    rl = real.dropna().astype(str).str.len()
    sl = synth.dropna().astype(str).str.len()
    len_mean_diff = abs(float(rl.mean()) - float(sl.mean())) / max(1.0, float(rl.mean()))
    char_tvd = _tvd(_char_freqs(real), _char_freqs(synth))
    d = max(len_mean_diff, char_tvd)
    status = "pass" if d <= thr["pass"] else ("warn" if d <= thr["warn"] else "fail")
    details = {
        "real_len_mean": float(rl.mean()),
        "synth_len_mean": float(sl.mean()),
        "len_mean_diff": len_mean_diff,
        "char_tvd": char_tvd,
        "note": "structural comparison of length and character distributions",
    }
    return d, status, details


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

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
    overall = all(c.status != "fail" for c in reports)
    return ValidationReport(
        columns=reports,
        real_rows=len(real),
        synthetic_rows=len(synthetic),
        n_pass=n_pass, n_warn=n_warn, n_fail=n_fail,
        mean_distance=round(float(np.mean(dists)), 4) if dists else 0.0,
        overall_pass=overall,
    )
