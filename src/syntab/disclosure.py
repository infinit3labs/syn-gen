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
class PiiDisclosure:
    table: str
    column: str
    signals: List[str]
    generator: Optional[str]

    @property
    def qualified(self) -> str:
        return f"{self.table}.{self.column}"


@dataclass
class DisclosureReport:
    """What a profiled Spec reveals about the data it was profiled from."""

    n_columns: int = 0
    categoricals: List[CategoricalDisclosure] = field(default_factory=list)
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
    def needs_review(self) -> bool:
        """Whether anything here warrants a human decision before sharing."""
        return bool(self.embedded_values or self.pii or self.key_candidates
                    or self.n_numeric_ranges or self.n_datetime_ranges
                    or self.n_string_shapes)

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
                     f"(recommended: {RECOMMENDED_MIN_CELL_COUNT})")
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


def summarize(spec: Spec) -> DisclosureReport:
    return DisclosureReport.from_spec(spec)
