"""Dependency discovery using published data-profiling algorithms.

This module exists to replace hand-rolled heuristics with the algorithms the
data-profiling literature actually specifies, as surveyed in

    Abedjan, Golab & Naumann, "Profiling relational data: a survey",
    The VLDB Journal 24(4), 2015.

Every algorithm below is a named, published one, executed through the
Desbordante C++ implementations (Chernishev et al., "Solving Data Quality
Problems with Desbordante: a Demo", 2023; https://desbordante.unidata-platform.ru/).
Nothing here re-implements a discovery algorithm. What this module does own is
the glue the literature does not specify: how a pandas frame is handed to the
C++ layer, which candidates are worth generating, and how a discovered
dependency is turned into something a Spec can hold.

Algorithms used
---------------
HyFD    exact functional dependencies.
        Papenbrock & Naumann, "A Hybrid Approach to Functional Dependency
        Discovery", SIGMOD 2016.

Pyro    approximate (relaxed) functional dependencies with an error bound.
        Kruse & Naumann, "Efficient Discovery of Approximate Dependencies",
        PVLDB 11(7), 2018. The bound is the g1 error of Kivinen & Mannila,
        "Approximate Inference of Functional Dependencies from Relations",
        Theoretical Computer Science 149(1), 1995: the fraction of ordered
        tuple pairs that agree on the determinant and disagree on the
        dependent.

HyUCC   unique column combinations (candidate keys).
        Papenbrock & Naumann, "A Hybrid Approach for Efficient Unique Column
        Combination Discovery", BTW 2017.

SPIDER  unary inclusion dependencies (the foreign-key candidates).
        Bauckmann, Leser, Naumann & Tietz, "Efficiently Detecting Inclusion
        Dependencies", ICDE 2007.

One construction here is this module's own rather than Desbordante's:
``choose_dependency_forest`` turns the discovered dependencies into the
in-degree-one dependency graph that generation can act on, following

    Chow & Liu, "Approximating discrete probability distributions with
    dependence trees", IEEE Transactions on Information Theory 14(3), 1968.

It is a selection over discovered dependencies, not a discovery algorithm.

BINDER (Papenbrock et al., PVLDB 2015) is the other standard IND algorithm and
is named in Desbordante's own materials, but it is NOT exposed by the Python
bindings of desbordante 2.4.1: ``desbordante.ind.algorithms`` offers exactly
``Spider``, ``Mind`` and ``Faida``. SPIDER is used here because it is the
exact, sort-merge algorithm of the three and its cost on the real workload is
seconds, so there is no reason to trade exactness for Faida's approximation.

Interestingness
---------------
Approximate FDs need a second filter, and this is the one place where a naive
integration goes wrong. Under the g1 bound a near-key determines everything:
on the 208k-row CFPB source, ``Date received`` (1133 distinct values) comes
back "determining" thirteen unrelated columns at error 0.001, purely because
few tuple pairs share a date. The published correction is mu' (mu-prime), the
cardinality-corrected dependency measure of

    Piatetsky-Shapiro & Matheus, "Measuring Data Dependencies in Large
    Databases", KDD-93 Workshop on Knowledge Discovery in Databases, 1993,

which discounts the predictive power of X for Y by the number of distinct X
values, so a high-cardinality determinant has to earn its score. It is applied
as a post-filter over the algorithm's own output; it is not a substitute for
discovery. See ``mu_prime``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .spec import InferenceProvenance

# ---------------------------------------------------------------------------
# Optional dependency
# ---------------------------------------------------------------------------
# Desbordante is AGPL-3.0-only and is a compiled extension with no Windows
# wheel, so it is an optional extra (see pyproject.toml). Import failure is a
# supported state: callers ask ``is_available()`` and fall back, or call a
# discovery function and get a DiscoveryUnavailable with an actionable message.

try:  # pragma: no cover - exercised by whichever branch the host supports
    import desbordante as _desbordante
except Exception as _exc:  # pragma: no cover
    _desbordante = None
    _IMPORT_ERROR: Optional[BaseException] = _exc
else:
    _IMPORT_ERROR = None


class DiscoveryUnavailable(RuntimeError):
    """Raised when a discovery routine is called without Desbordante installed."""


def is_available() -> bool:
    """Whether the Desbordante bindings imported successfully."""
    return _desbordante is not None


def require_desbordante():
    if _desbordante is None:
        raise DiscoveryUnavailable(
            "Dependency discovery needs the 'desbordante' package, which is an "
            "optional extra: pip install 'syntab[discovery]'. Note it is "
            "AGPL-3.0-only and ships no Windows wheel. "
            f"Import failed with: {_IMPORT_ERROR!r}"
        )
    return _desbordante


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# g1 error bound handed to Pyro. Real data is dirty, so exact FDs alone miss
# the dependencies that actually describe it -- on the CFPB source the whole
# Product/Sub-product/Issue hierarchy is approximate, not exact. 0.01 says "at
# most 1% of tuple pairs may violate this". It is a threshold on tolerated
# dirtiness, not a confidence level, and there is no universally right value:
# raise it for messier sources, lower it toward 0 to approach HyFD.
DEFAULT_FD_ERROR = 0.01

# mu' floor for reporting an approximate FD. See module docstring. 0.5 means
# "the determinant must remove at least half of the uncertainty that its own
# cardinality does not already explain". Measured on the CFPB source this
# separates the real dependencies (ZIP code -> State 0.985, Issue -> Product
# 0.967, Sub-product -> Product 0.728, Sub-issue -> Issue 0.614) from the
# near-key artefacts (Date received -> Product 0.102, State -> Product 0.006).
DEFAULT_MIN_MU = 0.5

# Approximate-IND error bound. 0.0 means an inclusion dependency must hold
# exactly. Foreign keys in real extracts are often *almost* clean, so this is
# exposed; it stays exact by default because a spurious FK changes generated
# output structurally, where a spurious FD only adds a line to a report.
DEFAULT_IND_ERROR = 0.0

# Row count above which candidate generation runs on a sample. See
# ``_sampled_frame`` and ``discover_functional_dependencies`` for the
# candidate-generate / full-validate split.
DEFAULT_SAMPLE_ROWS = 50_000

# Lattice depth cap. The FD search space is exponential in the column count,
# and a determinant of more than three columns is not a dependency anyone acts
# on -- on the CFPB source every FD with a 4+ column determinant is a
# combinatorial artefact of one near-key column. This is Desbordante's own
# ``max_lhs`` option, which is the documented way to bound the search.
DEFAULT_MAX_LHS = 3

# Columns whose distinct-value ratio exceeds this are excluded from discovery.
# A free-text column (the CFPB narrative is 202,516 distinct values in 208,398
# rows) is a near-key: it participates in no interesting dependency and it
# dominates the runtime. Excluding it is a pre-filter on the candidate set, not
# a change to any algorithm.
DEFAULT_MAX_UNIQUE_RATIO = 0.9

# Above this column count a table is treated as wide and candidate generation
# is forced onto a sample regardless of row count.
DEFAULT_WIDE_TABLE_COLUMNS = 25


CITATIONS = {
    "HyFD": "Papenbrock & Naumann, SIGMOD 2016",
    "Pyro": "Kruse & Naumann, PVLDB 11(7), 2018",
    "HyUCC": "Papenbrock & Naumann, BTW 2017",
    "SPIDER": "Bauckmann, Leser, Naumann & Tietz, ICDE 2007",
}


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiscoveredFD:
    """A functional dependency ``determinant -> dependent``."""

    determinant: Tuple[str, ...]
    dependent: str
    algorithm: str
    error: float          # g1 error; 0.0 for an exact FD
    mu_prime: Optional[float]
    support: int          # rows the dependency was measured against
    validated_on: str     # "full" | "sample"

    @property
    def exact(self) -> bool:
        return self.error == 0.0

    def provenance(self) -> InferenceProvenance:
        return InferenceProvenance(
            algorithm=self.algorithm,
            citation=CITATIONS.get(self.algorithm),
            measure="exact" if self.exact else "g1",
            confidence=round(1.0 - self.error, 6),
            error=round(self.error, 6),
            mu_prime=None if self.mu_prime is None else round(self.mu_prime, 6),
            support=self.support,
            validated_on=self.validated_on,
        )

    def as_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "determinant": list(self.determinant),
            "dependent": self.dependent,
            "provenance": self.provenance().model_dump(
                mode="json", exclude_none=True, exclude_defaults=True
            ),
        }
        return d


@dataclass(frozen=True)
class DiscoveredUCC:
    """A unique column combination -- a candidate key."""

    columns: Tuple[str, ...]
    algorithm: str
    support: int
    validated_on: str

    def provenance(self) -> InferenceProvenance:
        return InferenceProvenance(
            algorithm=self.algorithm,
            citation=CITATIONS.get(self.algorithm),
            measure="exact",
            confidence=1.0,
            support=self.support,
            validated_on=self.validated_on,
        )


@dataclass(frozen=True)
class DiscoveredIND:
    """A unary inclusion dependency ``child.column SUBSET-OF parent.column``."""

    child_table: str
    child_column: str
    parent_table: str
    parent_column: str
    algorithm: str
    error: float
    support: int
    validated_on: str

    def provenance(self) -> InferenceProvenance:
        return InferenceProvenance(
            algorithm=self.algorithm,
            citation=CITATIONS.get(self.algorithm),
            measure="exact" if self.error == 0.0 else "ind_error",
            confidence=round(1.0 - self.error, 6),
            error=round(self.error, 6),
            support=self.support,
            validated_on=self.validated_on,
        )


# ---------------------------------------------------------------------------
# Frame preparation
# ---------------------------------------------------------------------------


def value_kind(s: pd.Series) -> str:
    """Coarse comparability class of a column, for key matching.

    Two columns are only candidates for a key relationship if they hold the
    same kind of value. ``is_numeric_dtype`` is true of a boolean column, so
    bool has to be tested first.

    This is the canonical implementation; ``profiler.DatasetProfiler._value_kind``
    delegates here so the type-compatibility rule that gates foreign-key
    matching exists exactly once.
    """
    if pd.api.types.is_bool_dtype(s):
        return "bool"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    return "string"


def _as_desbordante_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a frame into something the pybind11 Table caster accepts.

    ``object`` dtype, and specifically NOT ``df.astype(str)``, which is the
    obvious way to normalize a frame and is wrong here. Under pandas 3
    ``astype(str)`` produces the new ``str`` dtype, and the pybind11 Table
    caster rejects the result whenever a converted column carries nulls -- the
    real CFPB frame fails this way. The error is a bare "Unable to cast Python
    instance of type <class 'tuple'> to C++ type '?'" naming neither the
    column nor the dtype, so it is worth pinning in a test rather than
    rediscovering.

    ``astype(object)`` loads every dtype tested: pandas StringDtype (with and
    without nulls), object, Int64, float64, bool, category and datetime64.
    """
    return df.astype(object)


def discovery_columns(
    df: pd.DataFrame,
    max_unique_ratio: float = DEFAULT_MAX_UNIQUE_RATIO,
) -> List[str]:
    """Columns worth including in a dependency search.

    Drops all-null columns (they carry no dependency) and near-key columns
    whose distinct-value ratio exceeds ``max_unique_ratio``. The second is the
    free-text guard: such a column determines every other column trivially and
    is determined by none, so it contributes only runtime. Note that an actual
    key column is caught by this too -- that is intended for FD search, and
    key discovery goes through HyUCC on the unpruned frame instead.
    """
    keep: List[str] = []
    n = len(df)
    if n == 0:
        return []
    for c in df.columns:
        s = df[c]
        nn = int(s.notna().sum())
        if nn == 0:
            continue
        if float(s.nunique(dropna=True)) / n > max_unique_ratio:
            continue
        keep.append(str(c))
    return keep


def _sampled_frame(
    df: pd.DataFrame, sample_rows: int, seed: Optional[int]
) -> Tuple[pd.DataFrame, bool]:
    if sample_rows and len(df) > sample_rows:
        return df.sample(n=sample_rows, random_state=seed), True
    return df, False


# ---------------------------------------------------------------------------
# mu' -- cardinality-corrected dependency measure
# ---------------------------------------------------------------------------


def _pdep(df: pd.DataFrame, determinant: Sequence[str], dependent: str) -> Tuple[float, int]:
    """Probabilistic dependence pdep(X -> Y), and |X|.

    pdep(X -> Y) = (1/n) * sum_x sum_y count(x,y)^2 / count(x): the probability
    that two tuples drawn from the same X group agree on Y. Piatetsky-Shapiro &
    Matheus (1993), section 2.

    Computed with two group-by passes rather than a Python loop over groups --
    the loop form took 26 s on a 208k-row column with 208k distinct values,
    which is exactly the case a near-key determinant produces.
    """
    n = len(df)
    xs = list(determinant)
    if n == 0:
        return 0.0, 0
    if not xs:
        # Empty determinant: pdep reduces to the marginal pdep of Y.
        return _pdep_marginal(df[dependent]), 1
    cxy = df.groupby(xs + [dependent], dropna=False, observed=True).size()
    cx = df.groupby(xs, dropna=False, observed=True).size()
    levels = list(range(len(xs)))
    per_x = (cxy.astype("float64") ** 2).groupby(level=levels, observed=True).sum()
    aligned = cx.reindex(per_x.index).astype("float64")
    return float((per_x / aligned).sum() / n), int(len(cx))


def _pdep_marginal(s: pd.Series) -> float:
    n = len(s)
    if n == 0:
        return 0.0
    counts = s.value_counts(dropna=False).to_numpy(dtype="float64")
    return float((counts ** 2).sum() / (n * n))


def mu_prime(df: pd.DataFrame, determinant: Sequence[str], dependent: str) -> float:
    """Cardinality-corrected dependency measure mu' of X -> Y.

    mu' = 1 - (n - |X|)/(n - 1) * (1 - pdep(X -> Y)) / (1 - pdep(Y))

    Piatetsky-Shapiro & Matheus, "Measuring Data Dependencies in Large
    Databases", KDD-93. The correction term is the point: pdep(X -> Y) rises
    mechanically with the number of distinct X values, so a column with many
    distinct values scores well without predicting anything. mu' is the
    expected pdep of a *random* X with the same cardinality subtracted out, so
    a random determinant scores about 0 whatever its cardinality.

    Two degenerate cases return 0.0 rather than 1.0, because both are true and
    uninformative: a constant dependent (pdep(Y) == 1, everything determines
    it) and a determinant that is a key (|X| == n, it determines everything).
    Reporting either as a strong dependency is how an integration of an AFD
    algorithm ends up recommending nonsense; a key is discovered as a key, by
    HyUCC, not as a determinant of every other column.
    """
    n = len(df)
    if n < 2:
        return 0.0
    pdep_y = _pdep_marginal(df[dependent])
    if pdep_y >= 1.0:
        return 0.0
    pdep_xy, k = _pdep(df, determinant, dependent)
    if k >= n:
        return 0.0
    correction = (n - k) / (n - 1)
    value = 1.0 - correction * (1.0 - pdep_xy) / (1.0 - pdep_y)
    if math.isnan(value):
        return 0.0
    return float(max(0.0, min(1.0, value)))


def g1_error(df: pd.DataFrame, determinant: Sequence[str], dependent: str) -> float:
    """The g1 error of X -> Y: the fraction of ordered tuple pairs violating it.

    Kivinen & Mannila, "Approximate Inference of Functional Dependencies from
    Relations", TCS 149(1), 1995. This is the same measure Pyro bounds, and it
    is recomputed here on the FULL frame so a dependency proposed from a sample
    is reported with its true error rather than the sample's.

    g1 = (sum_x count(x)^2 - sum_x,y count(x,y)^2) / n^2
    """
    n = len(df)
    if n < 2:
        return 0.0
    xs = list(determinant)
    if not xs:
        counts = df[dependent].value_counts(dropna=False).to_numpy(dtype="float64")
        return float((n * n - (counts ** 2).sum()) / (n * n))
    cx = df.groupby(xs, dropna=False, observed=True).size().astype("float64")
    cxy = df.groupby(xs + [dependent], dropna=False, observed=True).size().astype("float64")
    return float(((cx ** 2).sum() - (cxy ** 2).sum()) / (n * n))


def dependency_measures(
    df: pd.DataFrame,
    determinant: Sequence[str],
    dependent: str,
    _group_sizes: Optional[Dict[Tuple[str, ...], "pd.Series"]] = None,
) -> Tuple[float, Optional[float]]:
    """Both measures of X -> Y in one pass: (g1 error, mu\').

    ``g1_error`` and ``mu_prime`` each need the same two group-by results --
    the sizes of the X groups and of the X,Y groups -- and validating a
    candidate needs both. Computing them separately did the work twice: on the
    208k-row CFPB source, 272 candidates cost 13.2 s of g1 plus 17.0 s of mu\'.
    This shares the group-bys and caches the X sizes across candidates with the
    same determinant, which most candidates have.

    Returns ``mu' = None`` for an empty determinant, where it is undefined (a
    constant column is not a dependency anyone acts on, and the empty-LHS FD
    already says the column is constant).
    """
    n = len(df)
    xs = list(determinant)
    if n < 2:
        return 0.0, None
    if not xs:
        counts = df[dependent].value_counts(dropna=False).to_numpy(dtype="float64")
        return float((n * n - (counts ** 2).sum()) / (n * n)), None

    key = tuple(xs)
    cx = None if _group_sizes is None else _group_sizes.get(key)
    if cx is None:
        cx = df.groupby(xs, dropna=False, observed=True).size().astype("float64")
        if _group_sizes is not None:
            _group_sizes[key] = cx
    cxy = (df.groupby(xs + [dependent], dropna=False, observed=True)
             .size().astype("float64"))

    sum_cx2 = float((cx ** 2).sum())
    sum_cxy2 = float((cxy ** 2).sum())
    g1 = (sum_cx2 - sum_cxy2) / (n * n)

    pdep_y = _pdep_marginal(df[dependent])
    k = int(len(cx))
    if pdep_y >= 1.0 or k >= n:
        return g1, 0.0
    levels = list(range(len(xs)))
    per_x = (cxy ** 2).groupby(level=levels, observed=True).sum()
    pdep_xy = float((per_x / cx.reindex(per_x.index)).sum() / n)
    mu = 1.0 - ((n - k) / (n - 1)) * (1.0 - pdep_xy) / (1.0 - pdep_y)
    if math.isnan(mu):
        mu = 0.0
    return g1, float(max(0.0, min(1.0, mu)))


def holds_exactly(df: pd.DataFrame, determinant: Sequence[str], dependent: str) -> bool:
    """Whether X -> Y holds on every row of ``df``."""
    xs = list(determinant)
    if len(df) < 2:
        return True
    if not xs:
        return int(df[dependent].nunique(dropna=False)) <= 1
    grouped = df.groupby(xs, dropna=False, observed=True)[dependent]
    return bool(grouped.nunique(dropna=False).max() <= 1)


# ---------------------------------------------------------------------------
# Functional dependencies
# ---------------------------------------------------------------------------


def discover_functional_dependencies(
    df: pd.DataFrame,
    *,
    error: float = DEFAULT_FD_ERROR,
    min_mu: float = DEFAULT_MIN_MU,
    max_lhs: int = DEFAULT_MAX_LHS,
    sample_rows: int = DEFAULT_SAMPLE_ROWS,
    max_unique_ratio: float = DEFAULT_MAX_UNIQUE_RATIO,
    wide_table_columns: int = DEFAULT_WIDE_TABLE_COLUMNS,
    threads: int = 0,
    seed: Optional[int] = 0,
) -> List[DiscoveredFD]:
    """Discover functional dependencies with HyFD (exact) and Pyro (approximate).

    ``error`` is Pyro's g1 bound. ``error == 0`` runs HyFD alone; any positive
    value runs both, because the exact FDs are the subset of the approximate
    ones that are worth reporting as exact and Pyro does not label them.

    CANDIDATE GENERATION / FULL VALIDATION. When the frame exceeds
    ``sample_rows`` rows, or is wider than ``wide_table_columns`` columns, the
    algorithms run on a sample and every dependency they propose is then
    re-measured against the whole frame; only those still within ``error`` and
    above ``min_mu`` on the full data survive. This is the structure HyFD
    itself uses internally -- sampling proposes, validation decides -- and it
    matters here for a measurable reason: on the 208k-row CFPB source, HyFD
    reports 191 exact FDs from a 5,000-row sample and 29 from the full data.
    Roughly 85% of a sample-only answer is an artefact of the sample.

    Because validation can only reject, sampling costs recall, not precision: a
    dependency the sample never proposed is never found. The default threshold
    is set high (50k rows) for that reason -- the full 208k x 18 CFPB frame
    runs HyFD in ~5 s, so sampling is a guard for pathological widths, not the
    normal path.
    """
    d = require_desbordante()
    cols = discovery_columns(df, max_unique_ratio=max_unique_ratio)
    if len(cols) < 2:
        return []
    full = df[cols]
    wide = len(cols) > wide_table_columns
    work, sampled = _sampled_frame(full, sample_rows if not wide else min(sample_rows, 10_000), seed)
    sampled = sampled or (wide and len(work) < len(full))
    table = _as_desbordante_frame(work)
    names = list(table.columns)

    candidates: Dict[Tuple[Tuple[str, ...], str], str] = {}

    exact = d.fd.algorithms.HyFD()
    exact.load_data(table=table)
    exact.execute(max_lhs=max_lhs, threads=threads)
    for fd in exact.get_fds():
        key = (tuple(names[i] for i in fd.lhs_indices), names[fd.rhs_index])
        candidates[key] = "HyFD"

    if error > 0:
        approx = d.afd.algorithms.Pyro()
        approx.load_data(table=table)
        approx.execute(error=error, max_lhs=max_lhs, threads=threads,
                       seed=0 if seed is None else int(seed))
        for fd in approx.get_fds():
            key = (tuple(names[i] for i in fd.lhs_indices), names[fd.rhs_index])
            candidates.setdefault(key, "Pyro")

    support = len(full)
    validated_on = "full"
    out: List[DiscoveredFD] = []
    # Shared across candidates: the X group sizes depend only on the
    # determinant, and most candidates share one with several others.
    group_sizes: Dict[Tuple[str, ...], Any] = {}
    for lhs, rhs in sorted(candidates):
        algorithm = candidates[(lhs, rhs)]
        err, mu = dependency_measures(full, lhs, rhs, group_sizes)
        if err > error + 1e-12:
            continue
        if lhs and (mu is None or mu < min_mu):
            continue
        out.append(DiscoveredFD(
            determinant=lhs,
            dependent=rhs,
            # HyFD's claim is only about the sample when the sample was what it
            # saw; if the full-data error is nonzero the dependency is
            # approximate and Pyro's bound is the honest attribution.
            algorithm=algorithm if (err == 0.0 or algorithm == "Pyro") else "Pyro",
            error=err,
            mu_prime=mu,
            support=support,
            validated_on=validated_on,
        ))
    out.sort(key=lambda f: (len(f.determinant), -(f.mu_prime or 1.0), f.determinant, f.dependent))
    return out


# ---------------------------------------------------------------------------
# Choosing which dependencies generation can actually act on
# ---------------------------------------------------------------------------

# Error bound used when discovering dependencies to CONDITION GENERATION on,
# as opposed to dependencies to REPORT. These are different questions and they
# deserve different thresholds.
#
# Reporting an FD is a claim: "this rule holds in your data". DEFAULT_FD_ERROR
# is deliberately tight (0.01) because a reported FD that does not hold is a
# false statement about the source.
#
# Conditioning on a dependency claims nothing. Drawing Y from the observed
# P(Y | X) reproduces the observed joint whether or not X determines Y
# functionally -- if the dependency is only 80% clean, the conditional is 80%
# concentrated and generation reproduces that too. So the question here is not
# "is this a rule?" but "is X informative about Y?", which is what mu' already
# measures.
#
# The bound still exists because the conditional table has to stay small, and
# a dependency that is mostly noise produces a wide, sparse table with a large
# disclosure surface and little fidelity to show for it.
#
# It is set at 0.05 for a measurable reason. On the 208,398-row CFPB source the
# hierarchy edges have g1 errors of 0.019 (Sub-product -> Product) and 0.018
# (Issue -> Sub-issue), both above DEFAULT_FD_ERROR, because roughly a fifth of
# Sub-product and a third of Sub-issue are NULL and every null group counts as
# a violation. At 0.01 those two edges are invisible and the hierarchy stays
# broken; at 0.05 they are found, with mu' of 0.73 and 0.51.
DEFAULT_CONDITIONAL_FD_ERROR = 0.05


@dataclass(frozen=True)
class ConditionalEdge:
    """One ``determinant -> dependent`` edge that generation will act on."""

    determinant: str
    dependent: str
    mu_prime: float           # strength in the CHOSEN direction (0.0 if unmeasured)
    strength: float           # undirected strength that won the edge its place
    fd: Optional[DiscoveredFD] = None   # the FD in the chosen direction, if any

    def provenance(self) -> InferenceProvenance:
        if self.fd is not None:
            return self.fd.provenance()
        # The edge was selected on the reverse direction's evidence: the pair
        # is dependent, the arrow was turned round to keep the graph a forest.
        return InferenceProvenance(
            algorithm="Pyro",
            citation=CITATIONS.get("Pyro"),
            measure="g1",
            mu_prime=round(self.strength, 6),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "determinant": [self.determinant],
            "dependent": self.dependent,
            "reversed": self.fd is None,
            "provenance": self.provenance().model_dump(
                mode="json", exclude_none=True, exclude_defaults=True
            ),
        }


def choose_dependency_forest(
    fds: Sequence[DiscoveredFD],
    columns: Optional[Sequence[str]] = None,
) -> List[ConditionalEdge]:
    """Pick the dependencies to condition generation on: a spanning forest.

    WHY A FOREST AND NOT JUST "USE EVERY DISCOVERED FD".

    A conditional generator draws Y from P(Y | X), so Y can have exactly one
    determinant -- in-degree one. Discovery hands back many more edges than
    that: on the CFPB source, Product is determined by Issue (mu' 0.967) and
    by Sub-product (mu' 0.728), and Sub-product is determined by Product
    (0.496) and by Issue (0.539). Taking the strongest determinant for each
    column independently gives Product a parent, leaves Sub-product without
    one, and the Product/Sub-product pair -- the thing that is broken -- stays
    broken. The choice has to be made over the graph, not per column.

    So: treat the columns as vertices and each discovered dependency as an
    undirected edge weighted by the stronger of its two directions, take a
    maximum-weight spanning forest, and orient each tree away from a root.
    That is Chow & Liu's dependency-tree construction ("Approximating discrete
    probability distributions with dependence trees", IEEE Trans. Information
    Theory 14(3), 1968), with mu' (Piatetsky-Shapiro & Matheus, KDD-93) in
    place of mutual information as the edge weight -- mu' because it is what
    ``syntab.discovery`` already measures and because it is cardinality-
    corrected, so a near-key column does not win every edge it touches. The
    resulting per-column conditionals are the in-degree-one case of the
    Bayesian-network factorization that ``generators._g_conditional`` samples.

    On the CFPB hierarchy this picks all three edges -- Issue-Product (0.967),
    Sub-product-Product (0.728), Sub-issue-Issue (0.614) -- where per-column
    selection picks one.

    ORIENTATION. Within a tree the arrows are determined by the root, and the
    root is chosen to maximize the total mu' of the edges read in their chosen
    direction, which prefers orientations that agree with a discovered FD.
    Where it cannot -- a path A -> B <- C has to become A -> B -> C or
    A <- B <- C -- the reversed edge is marked ``reversed`` in the spec. This
    costs nothing in fidelity: P(X)P(Y|X) and P(Y)P(X|Y) both reproduce the
    joint exactly. It costs something in readability, so it is recorded.

    Only single-column determinants are considered. A composite determinant
    would make the stored table approach the full joint, which is both a much
    larger file and a much larger disclosure; see
    ``profiler.MAX_CONDITIONAL_LHS``.
    """
    allowed = set(columns) if columns is not None else None

    # Best FD per ordered pair.
    directed: Dict[Tuple[str, str], DiscoveredFD] = {}
    for f in fds:
        if len(f.determinant) != 1:
            continue
        u, v = f.determinant[0], f.dependent
        if u == v:
            continue
        if allowed is not None and (u not in allowed or v not in allowed):
            continue
        mu = f.mu_prime if f.mu_prime is not None else 0.0
        best = directed.get((u, v))
        if best is None or (best.mu_prime or 0.0) < mu:
            directed[(u, v)] = f

    # Undirected candidate edges, weighted by the stronger direction.
    undirected: Dict[Tuple[str, str], float] = {}
    for (u, v), f in directed.items():
        pair = (u, v) if u <= v else (v, u)
        mu = f.mu_prime if f.mu_prime is not None else 0.0
        if mu > undirected.get(pair, -1.0):
            undirected[pair] = mu

    # Kruskal: heaviest edge first, skip anything that would close a cycle.
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        parent[ra] = rb
        return True

    chosen: List[Tuple[str, str, float]] = []
    for (a, b), w in sorted(undirected.items(), key=lambda kv: (-kv[1], kv[0])):
        if union(a, b):
            chosen.append((a, b, w))

    # Group the forest into trees and orient each away from its best root.
    adjacency: Dict[str, List[Tuple[str, float]]] = {}
    for a, b, w in chosen:
        adjacency.setdefault(a, []).append((b, w))
        adjacency.setdefault(b, []).append((a, w))

    def orient(root: str) -> List[Tuple[str, str, float]]:
        """BFS from ``root``, emitting (parent, child, undirected weight)."""
        out: List[Tuple[str, str, float]] = []
        seen = {root}
        queue = [root]
        while queue:
            node = queue.pop(0)
            for nxt, w in sorted(adjacency.get(node, [])):
                if nxt in seen:
                    continue
                seen.add(nxt)
                out.append((node, nxt, w))
                queue.append(nxt)
        return out

    def directed_mu(u: str, v: str) -> float:
        f = directed.get((u, v))
        return (f.mu_prime or 0.0) if f is not None else 0.0

    edges: List[ConditionalEdge] = []
    visited: set = set()
    for vertex in sorted(adjacency):
        if vertex in visited:
            continue
        component = sorted(_component(adjacency, vertex))
        visited.update(component)
        best_orientation, best_score = None, None
        for root in component:
            oriented = orient(root)
            score = sum(directed_mu(u, v) for u, v, _ in oriented)
            if best_score is None or score > best_score:
                best_orientation, best_score = oriented, score
        for u, v, w in (best_orientation or []):
            edges.append(ConditionalEdge(
                determinant=u, dependent=v,
                mu_prime=directed_mu(u, v), strength=w,
                fd=directed.get((u, v)),
            ))
    edges.sort(key=lambda e: (-e.strength, e.determinant, e.dependent))
    return edges


def _component(adjacency: Dict[str, List[Tuple[str, float]]], start: str) -> set:
    seen = {start}
    queue = [start]
    while queue:
        node = queue.pop(0)
        for nxt, _ in adjacency.get(node, []):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return seen


# ---------------------------------------------------------------------------
# Unique column combinations
# ---------------------------------------------------------------------------


def discover_unique_column_combinations(
    df: pd.DataFrame,
    *,
    max_columns: int = 4,
    sample_rows: int = DEFAULT_SAMPLE_ROWS,
    threads: int = 0,
    seed: Optional[int] = 0,
) -> List[DiscoveredUCC]:
    """Discover unique column combinations with HyUCC.

    Papenbrock & Naumann, BTW 2017. Unlike FD discovery this runs on the whole
    frame including high-cardinality columns, because a key is precisely a
    high-cardinality column -- pruning them would prune the answer.

    Candidates from a sample are re-checked against the full frame with
    ``duplicated``, because a sample makes any column with more distinct values
    than the sample size look unique by construction. Combinations wider than
    ``max_columns`` are dropped: they are real, but a 5-column candidate key is
    not one a generator should adopt.
    """
    d = require_desbordante()
    if df.shape[1] == 0 or len(df) == 0:
        return []
    work, sampled = _sampled_frame(df, sample_rows, seed)
    table = _as_desbordante_frame(work)
    names = list(table.columns)

    algo = d.ucc.algorithms.HyUCC()
    algo.load_data(table=table)
    algo.execute(threads=threads)

    out: List[DiscoveredUCC] = []
    for ucc in algo.get_uccs():
        cols = tuple(names[i] for i in ucc.indices)
        if not cols or len(cols) > max_columns:
            continue
        if sampled:
            subset = df[list(cols)].dropna()
            if subset.empty or bool(subset.duplicated().any()):
                continue
        else:
            subset = df[list(cols)].dropna()
            if subset.empty:
                continue
        out.append(DiscoveredUCC(
            columns=cols, algorithm="HyUCC",
            support=len(df), validated_on="full",
        ))
    out.sort(key=lambda u: (len(u.columns), u.columns))
    return out


# ---------------------------------------------------------------------------
# Inclusion dependencies
# ---------------------------------------------------------------------------


def _ind_view(df: pd.DataFrame, columns: Sequence[str]) -> Optional[pd.DataFrame]:
    """A frame whose per-column VALUE SETS match ``df``'s, with no nulls.

    SPIDER treats a null as an ordinary value, so a nullable foreign key -- the
    ordinary case, not an edge case -- is reported as not contained in its
    parent. ``is_null_equal_null`` does not change this; it governs equality
    between two nulls, not their participation. SQL's rule is the opposite: a
    NULL foreign key satisfies referential integrity vacuously.

    So each column's non-null values are compacted to the top and the tail is
    padded by repeating that column's own first value. A unary inclusion
    dependency depends only on the set of distinct values in each column, and
    padding with a value already present changes neither the set nor SPIDER's
    distinct-value-based error, while keeping the frame rectangular as the
    binding requires. Rows are meaningless in the result, which is fine: unary
    INDs do not read across columns.
    """
    data: Dict[str, List[Any]] = {}
    height = 0
    for c in columns:
        vals = df[c].dropna().tolist()
        if not vals:
            continue
        data[c] = vals
        height = max(height, len(vals))
    if not data or height == 0:
        return None
    for c, vals in data.items():
        if len(vals) < height:
            vals.extend([vals[0]] * (height - len(vals)))
    return pd.DataFrame(data, columns=list(data)).astype(object)


def discover_inclusion_dependencies(
    tables: Dict[str, pd.DataFrame],
    *,
    error: float = DEFAULT_IND_ERROR,
    columns: Optional[Dict[str, Sequence[str]]] = None,
    threads: int = 0,
) -> List[DiscoveredIND]:
    """Discover unary inclusion dependencies with SPIDER.

    Bauckmann, Leser, Naumann & Tietz, ICDE 2007. An inclusion dependency
    ``A SUBSET-OF B`` is the evidence a foreign key is made of; turning INDs
    into foreign keys is a separate decision and lives in the profiler.

    ``columns`` restricts the search per table. The profiler passes the
    type-compatibility pre-filter through it: a column that cannot be
    key-compatible with anything in another table is not worth sorting. That
    pruning is sound (an IND between incomparable value kinds is not a key
    relationship) and it is a pre-filter only -- recall now comes from SPIDER,
    not from a column-name rule.
    """
    d = require_desbordante()
    order = [t for t in tables if len(tables[t]) > 0]
    views: Dict[str, pd.DataFrame] = {}
    for t in order:
        cols = list(columns.get(t, tables[t].columns)) if columns else list(tables[t].columns)
        cols = [c for c in cols if c in tables[t].columns]
        view = _ind_view(tables[t], cols)
        if view is not None:
            views[t] = view
    order = [t for t in order if t in views]
    if len(order) == 0:
        return []

    frames = [views[t] for t in order]
    algo = d.ind.algorithms.Spider()
    algo.load_data(tables=frames)
    algo.execute(error=error, threads=threads)

    out: List[DiscoveredIND] = []
    for ind in algo.get_inds():
        lhs, rhs = ind.get_lhs(), ind.get_rhs()
        if len(lhs.column_indices) != 1 or len(rhs.column_indices) != 1:
            continue  # unary only; n-ary INDs are a separate discovery problem
        ct, pt = order[lhs.table_index], order[rhs.table_index]
        cc = list(views[ct].columns)[lhs.column_indices[0]]
        pc = list(views[pt].columns)[rhs.column_indices[0]]
        if ct == pt and cc == pc:
            continue
        out.append(DiscoveredIND(
            child_table=ct, child_column=cc,
            parent_table=pt, parent_column=pc,
            algorithm="SPIDER",
            error=float(ind.get_error()),
            support=len(tables[ct]),
            validated_on="full",
        ))
    out.sort(key=lambda i: (i.child_table, i.child_column, i.parent_table, i.parent_column))
    return out
