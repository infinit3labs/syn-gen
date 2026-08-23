"""Canonical Spec models.

A Spec is the single contract used both for authoring data generation and for
the output of a future profiling module. Everything is defined as pydantic
models so loading YAML/JSON validates structure automatically.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class SpecError(Exception):
    pass


class SpecMetadata(BaseModel):
    name: str
    version: str = "1.0.0"
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    source: Optional[str] = None  # e.g. "profiled:orders.csv" or "hand-authored"
    owner: Optional[str] = None
    created_at: Optional[datetime] = None
    license: Optional[str] = None
    # The syntab package version that emitted the spec. Recorded by the
    # profiler (``syntab.__version__``) so a reviewer can tell whether a spec
    # was written by a version that still produced the behaviour it claims
    # to. ``None`` on hand-authored specs.
    syntab_version: Optional[str] = None
    # The merge report from the most recent ``--merge-into`` profile, when
    # one ran. A dict (not a Pydantic model) so the per-call summary is free
    # to evolve without a spec-version bump. ``None`` on hand-authored specs
    # and on profiles that did not use ``--merge-into``.
    merge: Optional[Dict[str, Any]] = None


class InferenceProvenance(BaseModel):
    """Where an inferred rule came from, and how strong the evidence was.

    Attached to anything the profiler decides rather than reads: a primary
    key, a foreign key, a functional dependency. The point is that a profiled
    spec is a *draft* a human edits, and until now nothing in the file
    distinguished "SPIDER proved this inclusion dependency over 208,398 rows"
    from "a column name ended in _id". Those warrant different amounts of
    trust and different review effort.

    ``fingerprint`` is what makes re-profiling non-destructive. It records the
    value the profiler itself wrote. On a re-profile, a rule whose current
    value no longer matches its fingerprint was changed by a human, so the
    change is preserved and ``human_edited`` is set rather than the edit being
    silently overwritten. See ``profiler.merge_preserving_edits``.
    """

    # "HyFD" | "Pyro" | "HyUCC" | "SPIDER" | "name-heuristic" | "human"
    algorithm: str
    citation: Optional[str] = None
    # "exact" | "g1" | "ind_error" | "name-match"
    measure: Optional[str] = None
    confidence: Optional[float] = None
    error: Optional[float] = None
    # Cardinality-corrected dependency measure (Piatetsky-Shapiro & Matheus,
    # KDD-93). Recorded separately from ``confidence`` because they answer
    # different questions: confidence is how often the rule holds, mu_prime is
    # how much of that is explained by the determinant's cardinality alone.
    mu_prime: Optional[float] = None
    # Rows the rule was measured against.
    support: Optional[int] = None
    # "full" | "sample" -- whether the evidence is the whole dataset or a draw.
    validated_on: Optional[str] = None
    human_edited: bool = False
    fingerprint: Optional[str] = None


class NumericProfile(BaseModel):
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    # uniform | normal | empirical | lognormal | exponential | pareto
    distribution: Optional[str] = None
    # skewed-distribution parameters (populated when relevant)
    log_mu: Optional[float] = None
    log_sigma: Optional[float] = None
    scale: Optional[float] = None
    pareto_b: Optional[float] = None
    pareto_xmin: Optional[float] = None
    # empirical CDF (histogram), populated when distribution == "empirical"
    hist_edges: Optional[List[float]] = None
    hist_counts: Optional[List[float]] = None
    null_rate: float = 0.0


class CategoricalProfile(BaseModel):
    values: Dict[str, float] = Field(default_factory=dict)  # value -> frequency
    null_rate: float = 0.0
    # True when ``values`` holds placeholder tokens rather than real source
    # values (profiled with --redact-categoricals). The frequency vector and
    # the cardinality are still real -- that is what generation needs -- but
    # the value labels carry no information from the source data. Recorded so
    # a reader of the spec, and `syntab compare`, can tell the difference.
    redacted: bool = False
    # Minimum cell-size suppression (--min-cell-count). ``min_cell_count`` is
    # the K that was applied, None when none was; ``suppressed_values`` is how
    # many distinct source values were generalized into the "__other__" bucket.
    min_cell_count: Optional[int] = None
    suppressed_values: int = 0
    # How many of the values embedded above occur fewer than
    # profiler.RECOMMENDED_MIN_CELL_COUNT (5) times in the source data. These
    # are the values that carry the re-identification risk: a category with one
    # member identifies that member. Recorded even when no suppression was
    # applied, precisely so that a spec which needs review says so on its face.
    rare_value_count: int = 0


class ColumnProfile(BaseModel):
    numeric: Optional[NumericProfile] = None
    categorical: Optional[CategoricalProfile] = None
    string_pattern: Optional[str] = None
    length: Optional[tuple[int, int]] = None
    datetime_range: Optional[tuple[datetime, datetime]] = None


class WhenClause(BaseModel):
    """Conditional generator branch for a column."""

    condition: str = Field(alias="if")
    generator: str
    params: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}


class ColumnSpec(BaseModel):
    name: str
    dtype: str  # int | float | str | bool | datetime | date | uuid
    generator: Optional[str] = None  # faker.email | custom:name | mypkg:Gen | "auto" | None
    params: Dict[str, Any] = Field(default_factory=dict)
    # constraints: min/max/length/pattern/choices/weights/unique/nullable/null_rate/decimals
    constraints: Dict[str, Any] = Field(default_factory=dict)
    depends_on: List[str] = Field(default_factory=list)
    when: List[WhenClause] = Field(default_factory=list)
    pii: bool = False
    # anonymization strategy applied at generation time:
    # faker | mask | redact | hash  (None -> no special handling)
    pii_strategy: Optional[str] = None
    # Which detection signal(s) flagged this column, in the profiler's
    # vocabulary: "explicit" (named in --pii), "column-name", "value-pattern".
    # Recorded because they warrant different follow-up -- a value-pattern hit
    # is evidence about the data, a column-name hit is an inference from a
    # label that may be wrong in either direction. None when not profiled or
    # not flagged, so it is omitted from a serialized spec entirely.
    pii_detected_by: Optional[List[str]] = None
    description: Optional[str] = None
    profile: Optional[ColumnProfile] = None


class RelationshipSpec(BaseModel):
    from_: Union[str, List[str]] = Field(alias="from")
    # Scalar form: "users.id". Composite form: "users" + to_columns.
    to: str
    to_columns: Optional[List[str]] = None
    alias: Optional[str] = None
    on_delete: str = "cascade"
    # Parent assignment controls. ``uniform`` preserves the original behavior.
    allocation: str = "uniform"  # uniform | balanced | weighted
    cardinality: str = "many_to_one"  # many_to_one | one_to_one
    min_children: int = 0
    max_children: Optional[int] = None
    weight_column: Optional[str] = None
    # Self-referencing relationships only: fraction of rows that are roots
    # (foreign key left NULL). The first generated row is always a root so the
    # hierarchy has an anchor; additional roots appear with this probability.
    root_fraction: float = 0.0
    # Ordered generation: when set, ``order_column`` receives a 1..n sequence
    # within each parent group, producing time-/event-ordered children.
    ordered: bool = False
    order_column: Optional[str] = None
    # Self-referencing relationships only: cap the tree depth so hierarchies
    # stay realistic (e.g. an org chart with at most N reporting levels).
    max_depth: Optional[int] = None
    # How this relationship was inferred, when it was inferred rather than
    # authored. None on a hand-written spec.
    provenance: Optional[InferenceProvenance] = None

    model_config = {"populate_by_name": True}

    @property
    def parent_table(self) -> str:
        return self.to.split(".", 1)[0]

    @property
    def child_columns(self) -> List[str]:
        return [self.from_] if isinstance(self.from_, str) else list(self.from_)

    @property
    def parent_columns(self) -> List[str]:
        if self.to_columns:
            return list(self.to_columns)
        _, _, column = self.to.partition(".")
        return [column]


class ManyToManySpec(BaseModel):
    """Declarative many-to-many relationship.

    Lowered into a junction table with two foreign keys (one per parent) plus
    two relationships. The engine then fills the links while honouring the
    per-parent degree bounds and, by default, keeping each (A, B) pair unique.
    """

    name: str  # junction table name
    table_a: str
    table_b: str
    column_a: Optional[str] = None  # defaults to f"{table_a}_id"
    column_b: Optional[str] = None  # defaults to f"{table_b}_id"
    row_count: int
    min_per_a: int = 0
    max_per_a: Optional[int] = None
    min_per_b: int = 0
    max_per_b: Optional[int] = None
    unique_pairs: bool = True
    allocation: str = "balanced"  # informational; links are generated greedily


class TableSpec(BaseModel):
    name: str
    row_count: int
    primary_key: Optional[Union[str, List[str]]] = None
    # Evidence for ``primary_key`` when it was discovered rather than declared.
    key_provenance: Optional[InferenceProvenance] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    columns: List[ColumnSpec]
    unique_constraints: List[List[str]] = Field(default_factory=list)
    relationships: List[RelationshipSpec] = Field(default_factory=list)
    rules: List[str] = Field(default_factory=list)


class Settings(BaseModel):
    seed: Optional[int] = None
    fallback: str = "raise"  # raise | drop | null
    max_rule_attempts: int = 100
    default_format: str = "csv"


class Spec(BaseModel):
    spec_version: str = "1.0"
    metadata: SpecMetadata
    settings: Settings = Field(default_factory=Settings)
    tables: List[TableSpec]
    many_to_many: List[ManyToManySpec] = Field(default_factory=list)


def _col_dtype(table: TableSpec, col_name: str) -> str:
    for c in table.columns:
        if c.name == col_name:
            return c.dtype
    return "int"


def expand_many_to_many(spec: "Spec") -> "Spec":
    """Lower ``many_to_many`` declarations into concrete junction tables.

    Idempotent: a spec that has already been expanded (its ``many_to_many``
    list is empty) is returned unchanged.
    """
    if not spec.many_to_many:
        return spec

    tables = list(spec.tables)
    m2m_aliases = ("__m2m_a__", "__m2m_b__")
    for m in spec.many_to_many:
        a = next((t for t in spec.tables if t.name == m.table_a), None)
        b = next((t for t in spec.tables if t.name == m.table_b), None)
        if a is None:
            raise SpecError(f"many_to_many parent table unknown: {m.table_a}")
        if b is None:
            raise SpecError(f"many_to_many parent table unknown: {m.table_b}")
        if not isinstance(a.primary_key, str):
            raise SpecError(
                f"many_to_many requires a scalar primary key on '{m.table_a}'"
            )
        if not isinstance(b.primary_key, str):
            raise SpecError(
                f"many_to_many requires a scalar primary key on '{m.table_b}'"
            )
        if any(t.name == m.name for t in tables):
            raise SpecError(f"many_to_many junction name collides: {m.name}")

        col_a = m.column_a or f"{m.table_a}_id"
        col_b = m.column_b or f"{m.table_b}_id"
        pk_a, pk_b = a.primary_key, b.primary_key
        dtype_a = _col_dtype(a, pk_a)
        dtype_b = _col_dtype(b, pk_b)
        alias_a, alias_b = m2m_aliases

        rel_a = RelationshipSpec(
            **{"from": col_a, "to": f"{m.table_a}.{pk_a}", "alias": alias_a,
               "allocation": m.allocation}
        )
        rel_b = RelationshipSpec(
            **{"from": col_b, "to": f"{m.table_b}.{pk_b}", "alias": alias_b,
               "allocation": m.allocation}
        )
        meta = {
            "name": m.name, "table_a": m.table_a, "table_b": m.table_b,
            "col_a": col_a, "col_b": col_b, "pk_a": pk_a, "pk_b": pk_b,
            "alias_a": alias_a, "alias_b": alias_b,
            "min_a": m.min_per_a, "max_a": m.max_per_a,
            "min_b": m.min_per_b, "max_b": m.max_per_b,
            "unique": m.unique_pairs, "row_count": m.row_count,
        }

        if m.unique_pairs:
            junction = TableSpec(
                name=m.name, row_count=m.row_count,
                primary_key=[col_a, col_b],
                columns=[
                    ColumnSpec(name=col_a, dtype=dtype_a, generator="fk",
                               constraints={"nullable": False}),
                    ColumnSpec(name=col_b, dtype=dtype_b, generator="fk",
                               constraints={"nullable": False}),
                ],
                relationships=[rel_a, rel_b],
                metadata={"m2m": meta},
            )
        else:
            junction = TableSpec(
                name=m.name, row_count=m.row_count, primary_key="id",
                columns=[
                    ColumnSpec(name="id", dtype="int", generator="sequence"),
                    ColumnSpec(name=col_a, dtype=dtype_a, generator="fk",
                               constraints={"nullable": False}),
                    ColumnSpec(name=col_b, dtype=dtype_b, generator="fk",
                               constraints={"nullable": False}),
                ],
                relationships=[rel_a, rel_b],
                metadata={"m2m": meta},
            )
        tables.append(junction)

    new = spec.model_copy(deep=True)
    new.tables = tables
    new.many_to_many = []
    return new
