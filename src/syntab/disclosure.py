"""Disclosure summary for a profiled Spec.

`syntab profile` reads real data and writes a file users are expected to commit
to version control. That file embeds real source values: the distinct values
and exact frequencies of every categorical column, the real min/max of every
numeric column, the real first/last timestamp of every datetime column, and a
character pattern derived from real values for short free-text columns.

None of that is new, and none of it is necessarily wrong -- it is what makes
the profiling round trip work. What was missing is that nothing said so. The
failure was never the behaviour; it was that a user had no way to know an
assessment was needed, so the assessment did not happen.

This module turns a Spec into a report of what it discloses. It reads only the
Spec, so it describes the artefact that will actually be shared rather than the
profiling run that produced it -- which is the right object to describe.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .profiler import OTHER_BUCKET_LABEL, RECOMMENDED_MIN_CELL_COUNT
from .spec import Spec

_WIDTH = 78
_RULE = "=" * _WIDTH


@dataclass
class CategoricalDisclosure:
    table: str
    column: str
    n_values: int          # distinct labels in the spec (excl. the other bucket)
    rare_values: int       # of those, how many are rare in the SOURCE
    redacted: bool
    suppressed: int
    min_cell_count: Optional[int]

    @property
    def qualified(self) -> str:
        return f"{self.table}.{self.column}"


@dataclass
class ConditionalDisclosure:
    """A conditional distribution embedded in the spec.

    Materially different from the marginals it sits beside, and the reason
    this class exists rather than the conditional being folded into
    ``CategoricalDisclosure``. A marginal says which values occur and how
    often. A conditional says which COMBINATIONS occur and in what proportion
    -- P(Sub-product | Product) for every observed pair. That is a
    contingency table of the source, and it is a larger disclosure than the
    two marginals it replaces even though it makes the synthetic data better.

    A fidelity win that widens the privacy surface without saying so is
    exactly what this module exists to prevent, so it is counted separately
    and named in the notice.
    """

    table: str
    determinant: List[str]
    dependent: str
    n_keys: int            # distinct determinant values with their own row
    n_cells: int           # (determinant, dependent) pairs recorded in total
    redacted: bool         # labels on both sides are placeholders
    suppressed_cells: int  # cells generalized into the other bucket

    @property
    def qualified(self) -> str:
        return f"{'/'.join(self.determinant)} -> {self.dependent}"


@dataclass
class PiiDisclosure:
    table: str
    column: str
    signals: List[str]
    generator: Optional[str]

    @property
    def qualified(self) -> str:
        return f"{self.table}.{self.column}"


@dataclass(frozen=True)
class DisclosureBudget:
    """Optional release limits for disclosure metrics already in the report.

    These are governance checks, not a differential-privacy guarantee. They
    make a release decision explicit without pretending that counting fields
    is equivalent to adding calibrated privacy noise.
    """

    max_unredacted_values: Optional[int] = None
    max_rare_values: Optional[int] = None
    max_conditional_cells: Optional[int] = None

    def __post_init__(self) -> None:
        for name in ("max_unredacted_values", "max_rare_values",
                     "max_conditional_cells"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class DisclosureViolation:
    metric: str
    actual: int
    limit: int


@dataclass
class DisclosureReport:
    """What a profiled Spec reveals about the data it was profiled from."""

    n_columns: int = 0
    categoricals: List[CategoricalDisclosure] = field(default_factory=list)
    conditionals: List[ConditionalDisclosure] = field(default_factory=list)
    pii: List[PiiDisclosure] = field(default_factory=list)
    key_candidates: List[str] = field(default_factory=list)
    n_numeric_ranges: int = 0
    n_empirical_histograms: int = 0
    n_datetime_ranges: int = 0
    n_string_shapes: int = 0

    # ---- derived ---------------------------------------------------------
    @property
    def open_categoricals(self) -> List[CategoricalDisclosure]:
        """Categorical columns whose real labels are in the spec."""
        return [c for c in self.categoricals if not c.redacted]

    @property
    def redacted_categoricals(self) -> List[CategoricalDisclosure]:
        return [c for c in self.categoricals if c.redacted]

    @property
    def embedded_values(self) -> int:
        return sum(c.n_values for c in self.open_categoricals)

    @property
    def rare_values(self) -> int:
        return sum(c.rare_values for c in self.open_categoricals)

    @property
    def suppressed_values(self) -> int:
        return sum(c.suppressed for c in self.categoricals)

    @property
    def suppression_applied(self) -> bool:
        return any(c.min_cell_count for c in self.categoricals)

    @property
    def redaction_applied(self) -> bool:
        return bool(self.redacted_categoricals)

    @property
    def min_cell_count(self) -> Optional[int]:
        ks = {c.min_cell_count for c in self.categoricals if c.min_cell_count}
        return min(ks) if ks else None

    @property
    def conditional_keys(self) -> int:
        return sum(c.n_keys for c in self.conditionals)

    @property
    def conditional_cells(self) -> int:
        return sum(c.n_cells for c in self.conditionals)

    @property
    def suppressed_conditional_cells(self) -> int:
        return sum(c.suppressed_cells for c in self.conditionals)

    @property
    def needs_review(self) -> bool:
        """Whether anything here warrants a human decision before sharing.

        Conditionals count even when redacted. Redaction removes the labels;
        it does not remove the fact that the spec records which category
        co-occurs with which, and how often.
        """
        return bool(self.embedded_values or self.pii or self.key_candidates
                    or self.n_numeric_ranges or self.n_datetime_ranges
                    or self.n_string_shapes or self.conditionals)

    def check_budget(self, budget: DisclosureBudget) -> List[DisclosureViolation]:
        """Return the configured budget limits this report exceeds.

        A missing limit means that metric is not governed by this budget. The
        checks use report values after redaction/generalization, so a caller
        can verify that those controls actually brought a spec within policy.
        """
        checks = (
            ("unredacted categorical values", self.embedded_values,
             budget.max_unredacted_values),
            ("rare categorical values", self.rare_values,
             budget.max_rare_values),
            ("conditional cells", self.conditional_cells,
             budget.max_conditional_cells),
        )
        return [
            DisclosureViolation(metric, actual, limit)
            for metric, actual, limit in checks
            if limit is not None and actual > limit
        ]

    # ---- construction ----------------------------------------------------
    @classmethod
    def from_spec(cls, spec: Spec) -> "DisclosureReport":
        rep = cls()
        multi = len(spec.tables) > 1
        for table in spec.tables:
            rep.n_columns += len(table.columns)

            prof_meta = (table.metadata or {}).get("profiling") or {}
            for name in prof_meta.get("identifying_key_candidates", []):
                rep.key_candidates.append(
                    f"{table.name}.{name}" if multi else str(name))

            for col in table.columns:
                if col.generator == "conditional":
                    rep.conditionals.append(
                        _conditional_disclosure(table.name, col))
                if col.pii:
                    rep.pii.append(PiiDisclosure(
                        table=table.name, column=col.name,
                        signals=list(col.pii_detected_by or []),
                        generator=col.generator,
                    ))
                prof = col.profile
                if prof is None:
                    continue
                if prof.categorical is not None and prof.categorical.values:
                    cat = prof.categorical
                    n = sum(1 for k in cat.values if k != OTHER_BUCKET_LABEL)
                    rep.categoricals.append(CategoricalDisclosure(
                        table=table.name, column=col.name, n_values=n,
                        rare_values=cat.rare_value_count,
                        redacted=cat.redacted,
                        suppressed=cat.suppressed_values,
                        min_cell_count=cat.min_cell_count,
                    ))
                if prof.numeric is not None and (
                        prof.numeric.min is not None or prof.numeric.max is not None):
                    rep.n_numeric_ranges += 1
                    if prof.numeric.hist_edges:
                        rep.n_empirical_histograms += 1
                if prof.datetime_range is not None:
                    rep.n_datetime_ranges += 1
                if prof.string_pattern is not None or prof.length is not None:
                    rep.n_string_shapes += 1
        return rep

    # ---- rendering -------------------------------------------------------
    def to_text(self, spec_path: Optional[str] = None) -> str:
        L: List[str] = [_RULE, " DISCLOSURE NOTICE"]
        if spec_path:
            L.append(f" {spec_path}")
        L += [" This spec is derived from your source data. Review it before"
              " sharing.", _RULE]

        # --- what real values are in the file ---
        L.append("")
        L.append(" Real source values embedded in this spec")
        opens = self.open_categoricals
        if opens:
            L.append(f"   {self.embedded_values:,} distinct categorical value(s) "
                     f"from {len(opens)} of {self.n_columns} column(s):")
            for c in sorted(opens, key=lambda c: -c.n_values)[:10]:
                extra = (f"   ({c.rare_values:,} occur < "
                         f"{RECOMMENDED_MIN_CELL_COUNT} times in the source)"
                         if c.rare_values else "")
                L.append(f"     {c.qualified:<34} {c.n_values:>7,} value(s)"
                         f"{extra}")
            if len(opens) > 10:
                L.append(f"     ... and {len(opens) - 10} more column(s)")
        else:
            L.append("   no categorical value labels "
                     + ("(all redacted)" if self.redaction_applied else "(none found)"))

        for count, what in (
            (self.n_numeric_ranges,
             "numeric column(s) embed the real min/max of the source column"),
            (self.n_empirical_histograms,
             "  of those, empirical histograms embed real bin edges"),
            (self.n_datetime_ranges,
             "datetime column(s) embed the real first/last timestamp"),
            (self.n_string_shapes,
             "free-text column(s) embed a length or character pattern taken "
             "from real values"),
        ):
            if count:
                L.append(f"   {count:,} {what}")

        # --- conditional structure ---
        # Deliberately its own section rather than a line in the one above.
        # Everything above is about which VALUES are in the file. This is
        # about which COMBINATIONS are, which is a different and larger
        # statement about the source, and it is new: it only appears when
        # generation was conditioned on a discovered dependency.
        if self.conditionals:
            L.append("")
            L.append(f" Conditional distributions embedded in this spec:"
                     f" {len(self.conditionals)}")
            for c in sorted(self.conditionals,
                            key=lambda c: (-c.n_cells, c.dependent))[:10]:
                note = "  (labels redacted)" if c.redacted else ""
                L.append(f"     {c.qualified:<34} {c.n_keys:>5,} key(s),"
                         f" {c.n_cells:>6,} cell(s){note}")
            if len(self.conditionals) > 10:
                L.append(f"     ... and {len(self.conditionals) - 10} more")
            L.append(f"   {self.conditional_cells:,} cell(s) in total. A"
                     f" conditional records which values CO-OCCUR")
            L.append("   and in what proportion -- a contingency table of the"
                     " source. That is")
            L.append("   MORE than the marginal frequency vectors it sits"
                     " beside, which record")
            L.append("   only which values exist. It is what makes the"
                     " synthetic data preserve")
            L.append("   the source's structure, and it is a wider disclosure"
                     " than not doing so.")
            if self.suppressed_conditional_cells:
                L.append(f"   {self.suppressed_conditional_cells:,} cell(s)"
                         f" suppressed into '{OTHER_BUCKET_LABEL}' by"
                         f" --min-cell-count.")
            if any(c.redacted for c in self.conditionals):
                L.append("   Redacted conditionals carry no real labels, but"
                         " they still carry the")
                L.append("   co-occurrence structure between the categories"
                         " they stand for.")
            L.append("   Turn this off with --no-condition-on-fds; the cost is"
                     " that generated")
            L.append("   data loses the relationships between these columns.")

        # --- PII ---
        L.append("")
        if self.pii:
            L.append(f" PII columns flagged: {len(self.pii)} of {self.n_columns}")
            for p in self.pii:
                sig = ", ".join(p.signals) or "unknown"
                L.append(f"     {p.qualified:<28} {sig:<26} -> {p.generator}")
            L.append("   Flagged columns carry no real values: the profile is"
                     " dropped and a")
            L.append("   synthetic generator substituted.")
        else:
            L.append(f" PII columns flagged: none of {self.n_columns}")

        if self.key_candidates:
            L.append("")
            L.append(f" Id-like columns NOT auto-flagged -- your call:"
                     f" {len(self.key_candidates)}")
            L.append(f"     {', '.join(self.key_candidates)}")
            L.append("   These name an identifier but are keys. Anonymizing a key"
                     " breaks primary-")
            L.append("   and foreign-key integrity, so the profiler will not do it"
                     " for you.")

        # --- controls ---
        L.append("")
        L.append(" Disclosure controls")
        if self.redaction_applied:
            L.append(f"   --redact-categoricals   APPLIED to "
                     f"{len(self.redacted_categoricals)} column(s)")
        else:
            L.append("   --redact-categoricals   not applied")
        if self.suppression_applied:
            L.append(f"   --min-cell-count        APPLIED at K="
                     f"{self.min_cell_count} "
                     f"({self.suppressed_values:,} value(s) generalized into "
                     f"'{OTHER_BUCKET_LABEL}')")
        else:
            L.append(f"   --min-cell-count        not applied "
                     f"-- explicitly disabled with 0 "
                     f"(recommended: {RECOMMENDED_MIN_CELL_COUNT}, the "
                     f"default)")
        if self.rare_values:
            L.append(f"       {self.rare_values:,} embedded value(s) occur fewer "
                     f"than {RECOMMENDED_MIN_CELL_COUNT} times in the source.")
            L.append("       Low-frequency values are the ones that identify"
                     " individuals.")

        # --- the ask ---
        L.append("")
        if self.needs_review:
            L.append(" A profiled spec is DERIVED FROM SOURCE DATA, not source"
                     " code. Review it")
            L.append(" before committing it to version control or sharing it"
                     " outside the")
            L.append(" boundary the source data lives in. Synthetic data is not"
                     " automatically")
            L.append(" anonymous.")
        else:
            L.append(" No real source values are embedded in this spec.")
        # Always cited, in both branches: the reader of a clean notice is
        # exactly the person who should know what the controls are for before
        # the next profile run, which may not be clean.
        L.append(" What a spec contains, and what these controls do:"
                 " docs/disclosure.md")
        L.append(_RULE)
        return "\n".join(L)


def _conditional_disclosure(table: str, col) -> ConditionalDisclosure:
    """Measure the conditional table on a column, tolerating a broken one.

    The notice has to render for any spec it is handed, including one a human
    has edited into an inconsistent state -- refusing to print a disclosure
    notice because the spec is malformed would suppress the notice exactly
    when the file is least trustworthy.
    """
    params = col.params or {}
    keys = params.get("keys") or []
    values = params.get("values") or []
    cells = 0
    suppressed = 0
    for row in values:
        try:
            cells += len(row)
            suppressed += sum(1 for v in row if v == OTHER_BUCKET_LABEL)
        except TypeError:
            continue
    prof_cat = col.profile.categorical if col.profile else None
    return ConditionalDisclosure(
        table=table,
        determinant=[str(c) for c in (params.get("on") or [])],
        dependent=col.name,
        n_keys=len(keys),
        n_cells=cells,
        redacted=bool(prof_cat and prof_cat.redacted),
        suppressed_cells=suppressed,
    )


def summarize(spec: Spec) -> DisclosureReport:
    return DisclosureReport.from_spec(spec)
