"""Graded quality scoring and structural diagnostics for synthetic data.

syntab reports on a generated dataset with THREE separate artefacts. They
answer three different questions and are deliberately not merged:

  1. QUALITY -- :func:`quality_report`, graded 0..1, never pass/fail.
     "How closely does the synthetic data resemble the real data?"
     Fidelity is a matter of degree, so it is scored, not judged.

  2. DIAGNOSTIC -- :func:`diagnostic_report`, pass/fail.
     "Is the synthetic data structurally valid at all?"
     These catch things that are simply broken -- a category that does not
     exist in the source, a value outside the observed range, a duplicated
     key -- rather than merely low-fidelity.

  3. CONFORMANCE -- :mod:`syntab.conformance`, pass/fail against the Spec.
     "Does the output honour the contract it was generated from?"
     This is a different question again: it checks the data against the
     declared Spec (primary keys, foreign keys, null rates, business rules),
     not against a real dataset. A dataset can conform perfectly to its Spec
     and still score badly here, and vice versa.

Why a graded report at all
--------------------------
The previous evaluation path (:mod:`syntab.validator`) compared per-column
marginal distributions and nothing else. A synthetic dataset that reproduces
every marginal exactly while destroying every relationship between columns
scored a clean pass. That is the canonical failure mode of a rule-based
generator, which samples each column independently by construction, so it is
precisely the failure this module exists to detect. Column Pair Trends is the
property that catches it.

Metric definitions
------------------
Every metric here is a named, published metric taken from SDMetrics
(DataCebo, MIT licensed) -- the field-standard synthetic-data metrics suite --
and each function is named after its SDMetrics counterpart. The definitions
were taken from the SDMetrics source and documentation:

  * https://docs.sdv.dev/sdmetrics/
  * https://github.com/sdv-dev/SDMetrics

They are implemented natively rather than imported. See ``docs/quality.md``
for the dependency decision and its reasoning; the short version is that
SDMetrics pins ``pandas<3.0.0`` and syntab runs pandas 3.x, so depending on
it would force a downgrade of syntab's core data dependency.

``tests/test_quality_sdmetrics_parity.py`` cross-checks these implementations
against SDMetrics itself whenever SDMetrics happens to be importable, so the
claim that these are the same metrics is tested rather than asserted.

Metric               | SDMetrics name        | Applies to
---------------------|-----------------------|---------------------------
ks_complement        | KSComplement          | numeric, datetime
tv_complement        | TVComplement          | categorical, boolean
correlation_similarity | CorrelationSimilarity | numeric/datetime pairs
contingency_similarity | ContingencySimilarity | all other pairs
range_coverage       | RangeCoverage         | numeric, datetime
category_coverage    | CategoryCoverage      | categorical, boolean
boundary_adherence   | BoundaryAdherence     | numeric, datetime
category_adherence   | CategoryAdherence     | categorical, boolean
missing_value_similarity | MissingValueSimilarity | every column
key_uniqueness       | KeyUniqueness         | declared key columns
table_structure      | TableStructure        | whole table

Free-text columns have no SDMetrics sdtype -- SDV excludes them from scoring.
syntab scores them anyway, because a rule-based generator emits text and a
defect there is invisible otherwise. The two text metrics are both standard
metrics applied to a derived distribution, not new metrics: KSComplement over
the value-length distribution, and TVComplement over the character-unigram
distribution. Both are marked ``(derived)`` in the report so a reader can tell
them apart from the SDMetrics-native scores.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .spec import Spec

# Rows sampled before computing the O(n_cols^2) pair-trend metrics and the
# character-distribution metrics. SDMetrics exposes the same control as
# ``num_rows_subsample``. Pair trends on a 200k-row source with 18 columns is
# 153 groupby passes; the metric is a distributional summary, so a large
# sample gives the same answer for a fraction of the work.
DEFAULT_SUBSAMPLE = 50_000

# Number of bins used to discretize a continuous column before computing
# ContingencySimilarity. Matches the SDMetrics default.
NUM_DISCRETE_BINS = 10


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def _as_numeric(series: pd.Series) -> pd.Series:
    """Coerce datetimes to int64 nanoseconds; leave other numerics alone."""
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return series.astype("int64")
    if pd.api.types.is_numeric_dtype(series.dtype):
        return series
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.notna().any():
        return parsed.astype("int64")
    return pd.to_numeric(series, errors="coerce")


def _ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov D: sup |F_a(x) - F_b(x)|."""
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    if len(a) == 0 or len(b) == 0:
        return 1.0
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / len(a)
    cdf_b = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _tvd(p: Dict[Any, float], q: Dict[Any, float]) -> float:
    """Total variation distance between two discrete distributions."""
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def _value_frequencies(series: pd.Series) -> Dict[Any, float]:
    vc = series.value_counts(dropna=True)
    total = float(vc.sum())
    if total == 0:
        return {}
    return {str(k): float(v) / total for k, v in vc.items()}


def _subsample(series_or_frame, sample: Optional[int], seed: int = 0):
    """Deterministically take at most ``sample`` rows."""
    if not sample or len(series_or_frame) <= sample:
        return series_or_frame
    return series_or_frame.sample(sample, random_state=seed)


def _char_frequencies(series: pd.Series, sample: Optional[int] = DEFAULT_SUBSAMPLE,
                      seed: int = 0) -> Dict[str, float]:
    """Character-unigram frequencies over the non-null values of ``series``.

    Vectorized via numpy rather than a Python loop over characters: a free-text
    column of a few hundred thousand rows is hundreds of megabytes of text, and
    a per-character loop over that is minutes, not seconds.
    """
    s = _subsample(series.dropna().astype(str), sample, seed)
    joined = "".join(s.tolist())
    if not joined:
        return {}
    codes = np.frombuffer(joined.encode("utf-32-le"), dtype=np.uint32)
    values, counts = np.unique(codes, return_counts=True)
    total = float(counts.sum())
    return {chr(int(v)): int(c) / total for v, c in zip(values, counts)}


def infer_kind(series: pd.Series) -> str:
    """Classify a column as numeric | datetime | categorical | boolean | text.

    This is syntab's stand-in for an SDV ``sdtype``. It is inferred from the
    REAL column so that real and synthetic are always scored with the same
    metric even when the synthetic column has drifted to another dtype.
    """
    if pd.api.types.is_bool_dtype(series.dtype):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series.dtype):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return "datetime"
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


_CONTINUOUS = ("numeric", "datetime")
_DISCRETE = ("categorical", "boolean")


# ---------------------------------------------------------------------------
# Single-column metrics (SDMetrics definitions)
# ---------------------------------------------------------------------------

def ks_complement(real: pd.Series, synthetic: pd.Series) -> float:
    """SDMetrics ``KSComplement``: 1 - the two-sample KS D statistic."""
    r = _as_numeric(real).dropna().to_numpy()
    s = _as_numeric(synthetic).dropna().to_numpy()
    if len(r) == 0 or len(s) == 0:
        return float("nan")
    return 1.0 - _ks_statistic(r, s)


def tv_complement(real: pd.Series, synthetic: pd.Series) -> float:
    """SDMetrics ``TVComplement``: 1 - total variation distance."""
    r = real.dropna()
    s = synthetic.dropna()
    if len(r) == 0 or len(s) == 0:
        return float("nan")
    return 1.0 - _tvd(_value_frequencies(r), _value_frequencies(s))


def range_coverage(real: pd.Series, synthetic: pd.Series) -> float:
    """SDMetrics ``RangeCoverage``: how much of the real range is spanned."""
    r = _as_numeric(real).dropna()
    s = _as_numeric(synthetic).dropna()
    if len(r) == 0 or len(s) == 0:
        return float("nan")
    min_r, max_r = float(r.min()), float(r.max())
    if min_r == max_r:
        return float("nan")
    normalized_min = max((float(s.min()) - min_r) / (max_r - min_r), 0.0)
    normalized_max = max((max_r - float(s.max())) / (max_r - min_r), 0.0)
    return max(1.0 - (normalized_min + normalized_max), 0.0)


def category_coverage(real: pd.Series, synthetic: pd.Series) -> float:
    """SDMetrics ``CategoryCoverage``: fraction of real categories present."""
    r = set(real.dropna().unique())
    s = set(synthetic.dropna().unique())
    if not r:
        return float("nan")
    return len(s & r) / len(r)


def boundary_adherence(real: pd.Series, synthetic: pd.Series) -> float:
    """SDMetrics ``BoundaryAdherence``: fraction inside the real [min, max]."""
    r = _as_numeric(real).dropna()
    s = _as_numeric(synthetic).dropna()
    if len(r) == 0 or len(s) == 0:
        return float("nan")
    valid = s.between(r.min(), r.max())
    return float(valid.sum()) / len(s)


def category_adherence(real: pd.Series, synthetic: pd.Series) -> float:
    """SDMetrics ``CategoryAdherence``: fraction of synthetic values seen in real.

    Unlike :func:`category_coverage` this is a validity check, not a fidelity
    one: it asks whether the generator invented a category that never occurs
    in the source. An invented category is broken output, not merely
    imprecise output, which is why it belongs in the diagnostic report.

    Null handling matters here and is easy to get wrong. A null is adherent
    when the real column also has nulls, and a violation when it does not --
    a generator emitting NULL into a column that is never null in the source
    HAS invented a value. ``Series.isin(Series)`` gives exactly that
    behaviour, because pandas matches NaN against NaN, which is why the real
    column is passed through whole rather than as a set of its non-null
    values. Dropping the nulls first scores a column that is 83% null at
    0.17 while it is faithfully reproducing that column.
    """
    if len(synthetic) == 0:
        return float("nan")
    return float(synthetic.isin(real).mean())


def missing_value_similarity(real: pd.Series, synthetic: pd.Series) -> float:
    """SDMetrics ``MissingValueSimilarity``: 1 - |null_rate difference|."""
    if len(real) == 0 or len(synthetic) == 0:
        return float("nan")
    r = float(real.isna().sum()) / len(real)
    s = float(synthetic.isna().sum()) / len(synthetic)
    return 1.0 - abs(r - s)


def key_uniqueness(synthetic: pd.DataFrame) -> float:
    """SDMetrics ``KeyUniqueness``: fraction of key rows neither dup nor null."""
    if isinstance(synthetic, pd.Series):
        synthetic = synthetic.to_frame()
    if len(synthetic) == 0:
        return float("nan")
    is_nan = synthetic.isna().all(axis=1)
    bad = synthetic.duplicated() | is_nan
    return 1.0 - float(bad.sum()) / len(synthetic)


def table_structure(real: pd.DataFrame, synthetic: pd.DataFrame) -> float:
    """SDMetrics ``TableStructure``: Jaccard over (column name, dtype) pairs."""
    r = set(zip(real.columns, map(str, real.dtypes)))
    s = set(zip(synthetic.columns, map(str, synthetic.dtypes)))
    union = r | s
    if not union:
        return float("nan")
    return len(r & s) / len(union)


# ---------------------------------------------------------------------------
# Text metrics (standard metrics over a derived distribution)
# ---------------------------------------------------------------------------

def text_length_complement(real: pd.Series, synthetic: pd.Series,
                           sample: Optional[int] = DEFAULT_SUBSAMPLE) -> float:
    """KSComplement over the distribution of string LENGTHS.

    Catches a generator that emits every value at a single length -- the
    classic symptom of filling free text at the maximum observed length.
    """
    r = _subsample(real.dropna().astype(str), sample).str.len()
    s = _subsample(synthetic.dropna().astype(str), sample).str.len()
    if len(r) == 0 or len(s) == 0:
        return float("nan")
    return 1.0 - _ks_statistic(r.to_numpy(), s.to_numpy())


def text_character_complement(real: pd.Series, synthetic: pd.Series,
                              sample: Optional[int] = DEFAULT_SUBSAMPLE) -> float:
    """TVComplement over the character-unigram distribution.

    Catches a generator drawing from a uniform ``[a-zA-Z0-9]`` alphabet when
    the source is English prose: no spaces, no punctuation, digits as common
    as vowels.
    """
    rf = _char_frequencies(real, sample)
    sf = _char_frequencies(synthetic, sample)
    if not rf or not sf:
        return float("nan")
    return 1.0 - _tvd(rf, sf)


# ---------------------------------------------------------------------------
# Column-pair metrics (SDMetrics definitions)
# ---------------------------------------------------------------------------

def _discretize(real: pd.Series, synthetic: pd.Series,
                bins: int = NUM_DISCRETE_BINS) -> Tuple[np.ndarray, np.ndarray]:
    """Bin a continuous pair of columns on the REAL column's histogram edges."""
    r = _as_numeric(real)
    s = _as_numeric(synthetic)
    clean = r.dropna()
    if len(clean) == 0:
        return np.zeros(len(r), dtype=int), np.zeros(len(s), dtype=int)
    edges = np.histogram_bin_edges(clean, bins=bins)
    edges = np.asarray(edges, dtype=float)
    edges[0], edges[-1] = -np.inf, np.inf
    return np.digitize(r, bins=edges), np.digitize(s, bins=edges)


def correlation_similarity(real: pd.DataFrame, synthetic: pd.DataFrame,
                           coefficient: str = "Pearson") -> float:
    """SDMetrics ``CorrelationSimilarity``: 1 - |corr_real - corr_synth| / 2."""
    c1, c2 = list(real.columns)[:2]
    r = real[[c1, c2]].apply(_as_numeric).dropna()
    s = synthetic[[c1, c2]].apply(_as_numeric).dropna()
    if len(r) < 2 or len(s) < 2:
        return float("nan")
    method = "pearson" if coefficient.lower() == "pearson" else "spearman"
    corr_r = r[c1].corr(r[c2], method=method)
    corr_s = s[c1].corr(s[c2], method=method)
    if corr_r is None or corr_s is None or math.isnan(corr_r) or math.isnan(corr_s):
        return float("nan")
    return 1.0 - abs(float(corr_r) - float(corr_s)) / 2.0


def contingency_similarity(real: pd.DataFrame, synthetic: pd.DataFrame,
                           continuous_columns: Sequence[str] = (),
                           bins: int = NUM_DISCRETE_BINS) -> float:
    """SDMetrics ``ContingencySimilarity``: 1 - TVD over the 2-D joint table."""
    c1, c2 = list(real.columns)[:2]
    r = real[[c1, c2]].copy()
    s = synthetic[[c1, c2]].copy()
    for col in continuous_columns:
        if col in (c1, c2):
            r[col], s[col] = _discretize(real[col], synthetic[col], bins)
    # Cast to string so that mixed dtypes between real and synthetic still
    # line up on a common index -- the metric is over category labels.
    for col in (c1, c2):
        r[col] = r[col].astype(str)
        s[col] = s[col].astype(str)
    if len(r) == 0 or len(s) == 0:
        return float("nan")
    joint_r = r.groupby([c1, c2], dropna=False).size() / len(r)
    joint_s = s.groupby([c1, c2], dropna=False).size() / len(s)
    index = joint_r.index.union(joint_s.index, sort=False).drop_duplicates()
    joint_r = joint_r.reindex(index, fill_value=0.0)
    joint_s = joint_s.reindex(index, fill_value=0.0)
    return 1.0 - float((joint_r - joint_s).abs().fillna(0.0).sum()) / 2.0


# ---------------------------------------------------------------------------
# Report types
# ---------------------------------------------------------------------------

def _mean(scores: Iterable[float]) -> float:
    vals = [v for v in scores if v is not None and not math.isnan(v)]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def _fmt(score: float) -> str:
    return "  n/a " if score is None or math.isnan(score) else f"{score:6.4f}"


@dataclass
class MetricResult:
    """One metric applied to one column (or one column pair)."""

    column: str
    metric: str
    score: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Property:
    """A named group of metrics, scored as the mean of its members.

    ``contributes`` marks whether the property feeds the headline Quality
    Score. See :class:`QualityReport` for why three of the five do not.
    """

    name: str
    description: str
    results: List[MetricResult] = field(default_factory=list)
    contributes: bool = True

    @property
    def score(self) -> float:
        return _mean(r.score for r in self.results)


@dataclass
class QualityReport:
    """Graded fidelity report. Scores, never a verdict.

    ``overall_score`` is the unweighted mean of **Column Shapes** and
    **Column Pair Trends**, which is exactly what SDMetrics' Quality Score
    aggregates -- those are the only two properties in its QualityReport.

    Coverage, Boundary Adherence and Missing Value Similarity are reported
    alongside them, graded, but do not feed the headline number. That is a
    deliberate choice and not an oversight. Averaging all five dilutes the
    one property that detects the failure this report exists to catch: on a
    dataset whose columns were independently shuffled -- identical marginals,
    every relationship destroyed -- the other four properties are all exactly
    1.0 by construction, so a five-way mean scores it 0.90 and buries the
    defect. Over the two SDMetrics properties the same dataset scores ~0.75,
    which is a number that reads like the problem it is.

    ``overall_score_all_properties`` exposes the five-way mean for anyone who
    wants it. Properties with no applicable columns are skipped rather than
    counted as zero.
    """

    properties: List[Property] = field(default_factory=list)
    real_rows: int = 0
    synthetic_rows: int = 0
    #: Key columns held out of the graded properties. Reported rather than
    #: dropped silently, so the exclusion is visible in the output.
    excluded_key_columns: List[str] = field(default_factory=list)

    @property
    def overall_score(self) -> float:
        """SDMetrics' Quality Score: mean of Column Shapes and Pair Trends."""
        return _mean(p.score for p in self.properties if p.contributes)

    @property
    def overall_score_all_properties(self) -> float:
        """Unweighted mean across every property, dilution included."""
        return _mean(p.score for p in self.properties)

    def get_property(self, name: str) -> Optional[Property]:
        return next((p for p in self.properties if p.name == name), None)

    def to_dict(self) -> Dict[str, Any]:
        both = self.overall_score_all_properties
        return {
            "overall_score": None if math.isnan(self.overall_score) else round(self.overall_score, 4),
            "overall_score_all_properties": None if math.isnan(both) else round(both, 4),
            "scored_properties": [p.name for p in self.properties if p.contributes],
            "excluded_key_columns": list(self.excluded_key_columns),
            "real_rows": self.real_rows,
            "synthetic_rows": self.synthetic_rows,
            "properties": [
                {
                    "name": p.name,
                    "description": p.description,
                    "contributes_to_score": p.contributes,
                    "score": None if math.isnan(p.score) else round(p.score, 4),
                    "results": [
                        {"column": r.column, "metric": r.metric,
                         "score": None if math.isnan(r.score) else round(r.score, 4),
                         "details": r.details}
                        for r in p.results
                    ],
                }
                for p in self.properties
            ],
        }

    def to_text(self, verbose: bool = False) -> str:
        scored = [p.name for p in self.properties if p.contributes]
        lines = [
            "QUALITY REPORT (graded fidelity -- scores, not a verdict)",
            f"  real={self.real_rows} rows  synthetic={self.synthetic_rows} rows",
            "",
            f"  Overall Score: {_fmt(self.overall_score)}",
            f"    = mean of {' + '.join(scored)}",
            "",
            f"  {'Property':<26}{'Score':>8}     Metrics",
            "  " + "-" * 68,
        ]
        for p in self.properties:
            metrics = sorted({r.metric for r in p.results})
            mark = "  * " if p.contributes else "    "
            lines.append(
                f"  {p.name:<26}{_fmt(p.score):>8}{mark}{', '.join(metrics) or '--'}")
        lines += [
            "  " + "-" * 68,
            "  * counts toward the Overall Score (SDMetrics' two Quality "
            "Score properties).",
            "    The rest are graded and reported, but averaging all five "
            "dilutes the one",
            "    property that detects destroyed relationships. "
            f"Five-way mean: {_fmt(self.overall_score_all_properties).strip()}",
        ]
        if self.excluded_key_columns:
            lines.append(
                "    Key columns excluded (checked by KeyUniqueness in "
                f"`syntab diagnose`): {', '.join(self.excluded_key_columns)}")
        if verbose:
            for p in self.properties:
                lines += ["", f"  {p.name} -- {p.description}",
                          "  " + "-" * 68,
                          f"  {'column':<40}{'metric':<22}{'score':>8}"]
                for r in sorted(p.results, key=lambda x: (math.isnan(x.score), x.score)):
                    lines.append(f"  {r.column[:39]:<40}{r.metric:<22}{_fmt(r.score):>8}")
        return "\n".join(lines)


@dataclass
class DiagnosticResult:
    """One structural check. Pass/fail, with the underlying score kept."""

    column: str
    metric: str
    score: float
    status: str  # pass | fail | n/a

    @property
    def ok(self) -> bool:
        return self.status != "fail"


@dataclass
class DiagnosticProperty:
    name: str
    description: str
    results: List[DiagnosticResult] = field(default_factory=list)

    @property
    def score(self) -> float:
        return _mean(r.score for r in self.results)

    @property
    def ok(self) -> bool:
        return all(r.ok for r in self.results)


@dataclass
class DiagnosticReport:
    """Structural validity report. Pass/fail, not graded.

    A diagnostic metric is expected to be exactly 1.0. Anything less means the
    generator produced output that is broken rather than merely imprecise: a
    value outside the observed range, a category that does not exist in the
    source, a duplicated key, a missing column. ``tolerance`` loosens the bar
    for callers who need to, but the default is strict on purpose.
    """

    properties: List[DiagnosticProperty] = field(default_factory=list)
    real_rows: int = 0
    synthetic_rows: int = 0
    tolerance: float = 0.0

    @property
    def overall_ok(self) -> bool:
        return all(p.ok for p in self.properties)

    def failures(self) -> List[DiagnosticResult]:
        return [r for p in self.properties for r in p.results if not r.ok]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_ok": self.overall_ok,
            "real_rows": self.real_rows,
            "synthetic_rows": self.synthetic_rows,
            "tolerance": self.tolerance,
            "properties": [
                {
                    "name": p.name,
                    "description": p.description,
                    "score": None if math.isnan(p.score) else round(p.score, 4),
                    "ok": p.ok,
                    "results": [
                        {"column": r.column, "metric": r.metric,
                         "score": None if math.isnan(r.score) else round(r.score, 4),
                         "status": r.status}
                        for r in p.results
                    ],
                }
                for p in self.properties
            ],
        }

    def to_text(self, verbose: bool = False) -> str:
        verdict = "VALID" if self.overall_ok else "INVALID"
        lines = [
            "DIAGNOSTIC REPORT (structural validity -- pass/fail)",
            f"  real={self.real_rows} rows  synthetic={self.synthetic_rows} rows",
            "",
            f"  OVERALL: {verdict}",
            "",
            f"  {'Property':<26}{'Score':>8}  Result",
            "  " + "-" * 68,
        ]
        for p in self.properties:
            lines.append(f"  {p.name:<26}{_fmt(p.score):>8}  {'ok' if p.ok else 'FAIL'}")
        failures = self.failures()
        if failures:
            lines += ["", "  Failures:", "  " + "-" * 68]
            for r in failures:
                lines.append(f"  [FAIL] {r.column[:39]:<40}{r.metric:<22}{_fmt(r.score):>8}")
        if verbose:
            for p in self.properties:
                lines += ["", f"  {p.name} -- {p.description}", "  " + "-" * 68]
                for r in p.results:
                    lines.append(
                        f"  [{r.status:<4}] {r.column[:39]:<40}{r.metric:<22}{_fmt(r.score):>8}"
                    )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _shared_columns(real: pd.DataFrame, synthetic: pd.DataFrame) -> List[str]:
    return [c for c in real.columns if c in synthetic.columns]


_SPEC_DTYPE_KINDS = {
    "int": "numeric", "float": "numeric", "bool": "boolean",
    "datetime": "datetime", "date": "datetime",
}


def _spec_kinds(spec: Optional[Spec], table: Optional[str],
                columns: Sequence[str]) -> Dict[str, str]:
    """Column kinds declared by the Spec.

    A declared kind beats an inferred one. :func:`infer_kind` has to guess
    from the data -- its categorical rule is a cardinality heuristic, which is
    unreliable on small frames and on high-cardinality categoricals. When a
    Spec is available that guess is unnecessary: a profiled column that
    carries a ``categorical`` profile IS categorical, whatever its cardinality
    happens to look like in a given sample.
    """
    if spec is None:
        return {}
    kinds: Dict[str, str] = {}
    for t in spec.tables:
        if table is not None and t.name != table:
            continue
        for col in t.columns:
            if col.name not in columns:
                continue
            prof = col.profile
            if prof is not None:
                if prof.categorical is not None:
                    kinds[col.name] = "categorical"
                    continue
                if prof.numeric is not None:
                    kinds[col.name] = "numeric"
                    continue
                if prof.datetime_range is not None:
                    kinds[col.name] = "datetime"
                    continue
            declared = _SPEC_DTYPE_KINDS.get(col.dtype)
            if declared is not None:
                kinds[col.name] = declared
    return kinds


def _resolve_kinds(real: pd.DataFrame, columns: Sequence[str],
                   spec: Optional[Spec], table: Optional[str]) -> Dict[str, str]:
    declared = _spec_kinds(spec, table, columns)
    return {c: declared.get(c) or infer_kind(real[c]) for c in columns}


def _key_columns(spec: Optional[Spec], table: Optional[str],
                 columns: Sequence[str]) -> List[List[str]]:
    """Key column groups declared in the Spec, for KeyUniqueness."""
    if spec is None:
        return []
    keys: List[List[str]] = []
    for t in spec.tables:
        if table is not None and t.name != table:
            continue
        pk = t.primary_key
        if isinstance(pk, str):
            pk = [pk]
        if pk and all(c in columns for c in pk):
            keys.append(list(pk))
        for uc in t.unique_constraints:
            if all(c in columns for c in uc) and list(uc) not in keys:
                keys.append(list(uc))
        for col in t.columns:
            if col.constraints.get("unique") and col.name in columns:
                if [col.name] not in keys:
                    keys.append([col.name])
    return keys


def quality_report(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    spec: Optional[Spec] = None,
    table: Optional[str] = None,
    sample: Optional[int] = DEFAULT_SUBSAMPLE,
    seed: int = 0,
    pair_trends: bool = True,
) -> QualityReport:
    """Score how closely ``synthetic`` resembles ``real``. Graded, 0..1.

    Properties, in SDMetrics' vocabulary:

      * **Column Shapes** -- per-column marginals. KSComplement for
        numeric/datetime, TVComplement for categorical/boolean, and the two
        derived text metrics for free text.
      * **Column Pair Trends** -- the relationships BETWEEN columns.
        CorrelationSimilarity for continuous pairs, ContingencySimilarity
        otherwise. This is the property a marginals-only validator misses,
        and the one a rule-based generator is most exposed to.
      * **Coverage** -- does the synthetic data span the real data?
        RangeCoverage and CategoryCoverage.
      * **Boundary Adherence** -- graded here (the diagnostic report asks the
        same question as pass/fail).
      * **Missing Value Similarity** -- is missingness reproduced?

    ``sample`` bounds the row count used for the quadratic pair-trend pass
    and the character-distribution metrics; ``None`` disables subsampling.
    """
    cols = _shared_columns(real, synthetic)
    # Key columns are excluded from every graded property, matching SDMetrics'
    # treatment of the `id` sdtype -- it appears in none of its property
    # metric maps. A surrogate key has no distribution worth reproducing and
    # no meaningful relationship with any other column, so scoring it adds
    # noise in both directions: a synthetic key range that legitimately does
    # not overlap the real one drags Column Shapes down, while ~50 pairings
    # of a key against everything else pull Column Pair Trends toward the
    # middle. Key integrity is checked by KeyUniqueness in the diagnostic
    # report, which is the right question to ask of a key.
    key_members = {c for key in _key_columns(spec, table, cols) for c in key}
    excluded_keys = sorted(key_members)
    cols = [c for c in cols if c not in key_members]
    kinds = _resolve_kinds(real, cols, spec, table)

    shapes = Property(
        "Column Shapes",
        "Per-column marginal distributions (KSComplement / TVComplement).")
    # These three are graded and reported but do not feed the headline score
    # -- see QualityReport for why.
    coverage = Property(
        "Coverage",
        "Does the synthetic data span the real data's range and categories?",
        contributes=False)
    boundary = Property(
        "Boundary Adherence",
        "Fraction of synthetic values inside the real [min, max].",
        contributes=False)
    missing = Property(
        "Missing Value Similarity",
        "Is the real null rate reproduced?",
        contributes=False)

    for c in cols:
        kind = kinds[c]
        r, s = real[c], synthetic[c]
        if kind in _CONTINUOUS:
            shapes.results.append(MetricResult(c, "KSComplement", ks_complement(r, s)))
            coverage.results.append(MetricResult(c, "RangeCoverage", range_coverage(r, s)))
            boundary.results.append(
                MetricResult(c, "BoundaryAdherence", boundary_adherence(r, s)))
        elif kind in _DISCRETE:
            shapes.results.append(MetricResult(c, "TVComplement", tv_complement(r, s)))
            coverage.results.append(
                MetricResult(c, "CategoryCoverage", category_coverage(r, s)))
        else:  # text
            length = text_length_complement(r, s, sample)
            chars = text_character_complement(r, s, sample)
            # Worst case of the two, mirroring the existing validator's
            # max(len_diff, char_tvd). A text column that reproduces character
            # frequencies at a single fixed length is not a good text column,
            # so averaging the two would hide exactly the defect we care about.
            finite = [v for v in (length, chars) if not math.isnan(v)]
            score = min(finite) if finite else float("nan")
            shapes.results.append(MetricResult(
                c, "TextSimilarity (derived)", score,
                {"KSComplement_over_lengths": None if math.isnan(length) else round(length, 4),
                 "TVComplement_over_characters": None if math.isnan(chars) else round(chars, 4),
                 "note": "worst of the two; standard metrics over derived distributions"}))
        missing.results.append(
            MetricResult(c, "MissingValueSimilarity", missing_value_similarity(r, s)))

    trends = Property(
        "Column Pair Trends",
        "Relationships BETWEEN columns (CorrelationSimilarity / "
        "ContingencySimilarity). The property a marginals-only check misses.")
    if pair_trends:
        scorable = [c for c in cols if kinds[c] != "text"]
        r_s = _subsample(real, sample, seed)
        s_s = _subsample(synthetic, sample, seed)
        for i, c1 in enumerate(scorable):
            for c2 in scorable[i + 1:]:
                k1, k2 = kinds[c1], kinds[c2]
                pair = f"{c1} | {c2}"
                if k1 in _CONTINUOUS and k2 in _CONTINUOUS:
                    trends.results.append(MetricResult(
                        pair, "CorrelationSimilarity",
                        correlation_similarity(r_s[[c1, c2]], s_s[[c1, c2]])))
                else:
                    continuous = [c for c in (c1, c2) if kinds[c] in _CONTINUOUS]
                    trends.results.append(MetricResult(
                        pair, "ContingencySimilarity",
                        contingency_similarity(
                            r_s[[c1, c2]], s_s[[c1, c2]], continuous)))

    properties = [shapes, trends, coverage, boundary, missing]
    return QualityReport(
        properties=[p for p in properties if p.results],
        real_rows=len(real),
        synthetic_rows=len(synthetic),
        excluded_key_columns=excluded_keys,
    )


def diagnostic_report(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    spec: Optional[Spec] = None,
    table: Optional[str] = None,
    tolerance: float = 0.0,
) -> DiagnosticReport:
    """Check that ``synthetic`` is structurally valid. Pass/fail, not graded.

    Properties, in SDMetrics' vocabulary:

      * **Data Structure** -- TableStructure. Do the columns (and their
        dtypes) match at all?
      * **Data Validity** -- BoundaryAdherence for numeric/datetime,
        CategoryAdherence for categorical/boolean, KeyUniqueness for key
        columns declared in the Spec.

    Each metric is expected to be exactly 1.0; ``tolerance`` widens that.
    """
    threshold = 1.0 - tolerance

    def _status(score: float) -> str:
        if score is None or math.isnan(score):
            return "n/a"
        return "pass" if score >= threshold - 1e-9 else "fail"

    structure = DiagnosticProperty(
        "Data Structure",
        "Do the synthetic columns and dtypes match the real ones? (TableStructure)")
    struct_score = table_structure(real, synthetic)
    structure.results.append(DiagnosticResult(
        "<table>", "TableStructure", struct_score, _status(struct_score)))

    validity = DiagnosticProperty(
        "Data Validity",
        "Is every synthetic value a value the source could have produced? "
        "(BoundaryAdherence / CategoryAdherence / KeyUniqueness)")

    cols = _shared_columns(real, synthetic)
    kinds = _resolve_kinds(real, cols, spec, table)
    keys = _key_columns(spec, table, cols)
    # A key column gets KeyUniqueness INSTEAD of BoundaryAdherence, matching
    # SDMetrics' DataValidity property. Synthetic keys are not supposed to
    # fall inside the real key range -- reusing real identifiers would be a
    # disclosure problem, not a success -- so range is the wrong question to
    # ask of them. Uniqueness is the right one.
    key_members = {c for key in keys for c in key}

    for c in cols:
        if c in key_members:
            continue
        kind = kinds[c]
        r, s = real[c], synthetic[c]
        if kind in _CONTINUOUS:
            score = boundary_adherence(r, s)
            validity.results.append(
                DiagnosticResult(c, "BoundaryAdherence", score, _status(score)))
        elif kind in _DISCRETE:
            score = category_adherence(r, s)
            validity.results.append(
                DiagnosticResult(c, "CategoryAdherence", score, _status(score)))
        # Free text has no closed value domain, so there is nothing structural
        # to check: any string is a structurally valid string. Text quality is
        # scored in the quality report instead.

    for key in keys:
        score = key_uniqueness(synthetic[key])
        validity.results.append(DiagnosticResult(
            ", ".join(key), "KeyUniqueness", score, _status(score)))

    return DiagnosticReport(
        properties=[p for p in (structure, validity) if p.results],
        real_rows=len(real),
        synthetic_rows=len(synthetic),
        tolerance=tolerance,
    )
