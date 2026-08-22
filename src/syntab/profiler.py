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
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import discovery
from .spec import (
    CategoricalProfile,
    ColumnProfile,
    ColumnSpec,
    InferenceProvenance,
    NumericProfile,
    RelationshipSpec,
    Settings,
    Spec,
    SpecMetadata,
    TableSpec,
)

_UUID_RE = re.compile(
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


# Identifier tokenization. Splits snake_case, kebab-case, space-separated and
# camelCase/PascalCase names into word tokens while keeping runs of capitals
# together, so "Complaint ID" -> ["Complaint", "ID"] and "userId" ->
# ["user", "Id"]. This is the conventional identifier-splitting alternation;
# the order matters, because the all-caps run has to be tried before the
# single-capital-then-lowercase form.
_NAME_TOKEN_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+")

# Trailing tokens that mark a column as an identifier.
_ID_TOKENS = {"id", "uuid", "guid"}

# ---------------------------------------------------------------------------
# Minimum cell-size suppression
# ---------------------------------------------------------------------------
# The standard statistical-disclosure-control response to rare categorical
# values. The framing is k-anonymity (Samarati & Sweeney 1998; Sweeney 2002,
# "k-Anonymity: A Model for Protecting Privacy"): a record is safe to release
# only if it is indistinguishable from at least k-1 others on the quasi-
# identifiers. A category with one member fails that immediately -- the
# category *is* the identifier -- and a profiled spec publishes exactly the
# distinct values and exact frequencies that make it visible.
#
# The two standard responses are SUPPRESSION (remove the cell) and
# GENERALIZATION (coarsen it until the cell is big enough). This implements
# generalization: values below the threshold are collapsed into a single
# residual "other" category, which is the conventional move because it keeps
# the total mass intact -- suppression alone would silently change the
# frequency vector the generator reproduces.
#
# Worth saying plainly, because the opposite is widely assumed: synthetic data
# is not automatically anonymous. A generator that faithfully reproduces a
# category with two members reproduces the fact that those two people exist and
# what distinguishes them; the ICO's 2025 anonymisation guidance and the
# NIST/UK-ONS work on synthetic data both treat synthetic output as requiring
# its own disclosure assessment rather than as de-identified by construction.
# Fidelity and disclosure risk trade against each other. This is a control on
# that trade, not a proof of anything.

# Suppression is OFF by default. See DatasetProfiler._categorical_profile for
# why, and note that the disclosure summary reports how many values fall below
# RECOMMENDED_MIN_CELL_COUNT whether or not suppression is enabled.
DEFAULT_MIN_CELL_COUNT = 0

# The threshold to reach for when you do enable it, and the one the disclosure
# summary measures against. Five is the most widely used minimum cell size in
# published SDC practice -- it is the common floor in national statistical
# office rules and in health-data release policy (CMS uses a stricter 11 for
# Medicare claims, some agencies use 3). There is no principled universal
# value: K is a policy decision about acceptable risk, and this constant is a
# default for that decision, not a substitute for it.
RECOMMENDED_MIN_CELL_COUNT = 5

# Label for the residual bucket. Deliberately not a plausible real value, so
# nobody mistakes it for one when reading a spec.
OTHER_BUCKET_LABEL = "__other__"


def _suppress_small_cells(
    freqs: Dict[str, float],
    source_counts: "pd.Series",
    k: int,
) -> Tuple[Dict[str, float], int]:
    """Generalize categories with a source count below ``k`` into one bucket.

    Returns ``(values, n_suppressed)``.

    Includes SECONDARY (complementary) suppression: if the residual bucket is
    itself smaller than k, it discloses its own members just as badly as the
    cells it absorbed -- a bucket of one is the value it hid. The standard fix
    is to keep absorbing, smallest surviving category first, until the bucket
    clears the threshold. Without this step a single rare value simply gets
    renamed to "__other__" and nothing is protected.
    """
    kept: Dict[str, float] = {}
    rolled: Dict[str, float] = {}
    for label, freq in freqs.items():
        if int(source_counts.get(label, 0)) < k:
            rolled[label] = freq
        else:
            kept[label] = freq

    if not rolled:
        return freqs, 0

    bucket = sum(int(source_counts.get(l, 0)) for l in rolled)
    for label in sorted(kept, key=lambda l: int(source_counts.get(l, 0))):
        if bucket >= k:
            break
        rolled[label] = kept.pop(label)
        bucket += int(source_counts.get(label, 0))

    out = dict(kept)
    out[OTHER_BUCKET_LABEL] = sum(rolled.values())
    return out, len(rolled)


# Placeholder token for a redacted categorical value. Zero-padded so the tokens
# sort in the same (descending-frequency) order the profiler emits them in.
_REDACTED_VALUE_PREFIX = "value_"


def _redact_categorical_values(values: Dict[str, float]) -> Dict[str, float]:
    """Replace real categorical labels with opaque tokens, keeping the shape.

    Cardinality and the frequency vector are preserved exactly -- those are
    what the generation engine consumes (``infer.resolve_generator`` turns them
    into ``choice`` values + weights) -- while the labels themselves stop being
    derived from the source data. Ordering is the profiler's descending-
    frequency order, so ``value_001`` is always the modal category.

    PLACEHOLDERS RATHER THAN HASHES, deliberately. A hash of a low-cardinality
    categorical is not a de-identification measure: the domain is small and
    usually guessable, so an unsalted digest is recovered by hashing the
    candidate list and matching -- the failure demonstrated at scale on the
    2014 NYC taxi release, where MD5-hashed medallion numbers were recovered
    by brute force over the known medallion format. A keyed hash would fix
    that, but it would also require key management this tool has no business
    inventing, and it buys nothing here: nothing downstream needs to join on
    these labels. An opaque counter is strictly safer and strictly simpler.
    """
    width = max(3, len(str(len(values))))
    return {
        f"{_REDACTED_VALUE_PREFIX}{i:0{width}d}": freq
        for i, freq in enumerate(values.values(), start=1)
    }


# ---------------------------------------------------------------------------
# Structure discovery
# ---------------------------------------------------------------------------
# Keys, foreign keys and column dependencies are discovered with the published
# algorithms in ``syntab.discovery`` (HyFD, Pyro, HyUCC, SPIDER) rather than
# inferred from column names. Discovery needs the optional ``[discovery]``
# extra; when it is absent the profiler falls back to the name-based rules
# that preceded it and says so in the provenance it records, so a spec always
# states which of the two produced each rule.
#
# ``None`` means "use discovery if it is installed". Pass True to make its
# absence an error, False to force the fallback.
DEFAULT_DISCOVER = None

# Widest candidate key the profiler will adopt as a primary key. A discovered
# 5-column unique combination is real and is almost never the table's key.
MAX_KEY_COLUMNS = 3

# Cap on composite unique constraints written into a spec. Populating
# ``unique_constraints`` turns off the engine's vectorized generation path
# (see engine._is_vectorizable), so the profiler emits only genuinely
# composite ones -- single-column uniqueness already travels as
# ``constraints.unique``, which does not cost the fast path -- and only a
# handful of them.
MAX_COMPOSITE_UNIQUE_CONSTRAINTS = 3

# Functional-dependency discovery is OFF by default, and that is a considered
# choice rather than caution. FD discovery is the most expensive thing in this
# module -- on the 208k-row CFPB source it is the difference between a 3.6 s
# and a ~40 s profile -- and unlike key and foreign-key discovery its output
# does not currently steer generation (see the commit that removed
# ``suggested_depends_on`` for exactly why it cannot yet). Paying forty seconds
# by default for a report nothing consumes would be the wrong trade; paying it
# on request, to understand a dataset before editing its spec, is a good one.
DEFAULT_DISCOVER_FDS = False

# Cap on functional dependencies recorded in a spec, strongest first. A wide
# table can yield hundreds; a spec is something a human reads and edits.
MAX_RECORDED_FDS = 50


# Categorical-vs-free-text thresholds. See DatasetProfiler._is_categorical.
DEFAULT_MAX_CATEGORICAL = 50
DEFAULT_MAX_CATEGORICAL_RATIO = 0.05
DEFAULT_MAX_CATEGORICAL_RATIO_CAP = 500


class _NamedKey:
    """Stand-in for a discovered UCC when discovery did not run.

    Lets the primary-key selection run one code path whether the evidence came
    from HyUCC or from the per-column uniqueness check that preceded it.
    """

    __slots__ = ("columns",)

    def __init__(self, columns: Tuple[str, ...]):
        self.columns = columns


def _fingerprint(value: Any) -> str:
    """Stable digest of a value the profiler wrote into a spec.

    Recorded on the provenance so a later re-profile can tell its own previous
    output from a human edit. See ``merge_preserving_edits``.
    """
    import hashlib
    import json

    payload = json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _table_name_forms(table: str) -> set:
    """The base names a foreign-key column could carry for ``table``.

    ``users`` is referenced by ``user_id`` as often as by ``users_id``, so the
    singular and plural are both accepted. Used only to break a tie between
    equally well-covered candidate parents.
    """
    low = table.lower()
    forms = {low, low + "s"}
    if low.endswith("s"):
        forms.add(low[:-1])
    return forms


def _would_cycle(edges: set, child: str, parent: str) -> bool:
    """Whether adding child -> parent closes a cycle in the table graph.

    The engine generates a parent table before its children and raises on a
    cyclic spec, so a pair of tables whose keys happen to contain each other
    must not both become foreign keys.
    """
    seen = {parent}
    stack = [parent]
    while stack:
        node = stack.pop()
        if node == child:
            return True
        for a, b in edges:
            if a == node and b not in seen:
                seen.add(b)
                stack.append(b)
    return False


def _name_tokens(name: str) -> List[str]:
    return _NAME_TOKEN_RE.findall(name or "")


def _is_id_like(name: str) -> bool:
    """Whether a column name ends in an identifier *token*.

    Substring matching -- ``name.lower().endswith("id")`` -- is wrong here. It
    accepts ``paid``, ``void``, ``valid``, ``bid``, ``grid``, ``rapid``,
    ``humid``, ``squid`` and anything else that happens to end in those two
    letters, and a float column called ``paid`` was being promoted to primary
    key ahead of the real one. Tokenizing the name and testing only the final
    token fixes the whole class of false positives at once, and removes the
    need for the per-dataset exception ("complaint id") that had been added to
    paper over one instance of it.
    """
    tokens = _name_tokens(name)
    return bool(tokens) and tokens[-1].lower() in _ID_TOKENS


def _id_base(name: str) -> Optional[str]:
    """For an id-like name, the normalized prefix before the id token.

    ``user_id`` / ``userId`` / ``User ID`` -> ``user``. Returns None when the
    name is a bare identifier (``id``) or is not id-like at all.
    """
    tokens = _name_tokens(name)
    if len(tokens) < 2 or tokens[-1].lower() not in _ID_TOKENS:
        return None
    return "_".join(t.lower() for t in tokens[:-1])


# ---------------------------------------------------------------------------
# PII detection: column-name signal
# ---------------------------------------------------------------------------
# Names of the detection signals. Recorded on ColumnSpec.pii_detected_by so a
# reviewer can tell a declared flag from an inferred one, and a name-inferred
# flag from a value-inferred one -- they warrant different follow-up.
PII_SIGNAL_EXPLICIT = "explicit"        # named in --pii
PII_SIGNAL_NAME = "column-name"         # the column's NAME looks identifying
PII_SIGNAL_VALUE = "value-pattern"      # the column's VALUES look identifying

# Column-name patterns for the identifiers enumerated by the HIPAA Safe Harbor
# de-identification standard (45 CFR 164.514(b)(2)) -- names, geography finer
# than a state, dates tied to an individual, telephone/fax, email, SSN, medical
# record and health-plan numbers, account numbers, certificate and licence
# numbers, vehicle and device identifiers, IP and MAC addresses -- cross-checked
# against the direct identifiers listed in NIST SP 800-122 (Guide to Protecting
# the Confidentiality of PII), section 2.2 and Appendix A.
#
# Using a published enumeration rather than an ad-hoc one is the whole point.
# What was here before was not a de-identification standard; it was three
# regexes for the three identifier formats that happen to be easy to match.
#
# Matching is on NAME TOKENS, not substrings, reusing _name_tokens(). This is
# the same correctness argument as _is_id_like: `low.endswith("id")` matched
# `paid`, `void` and `squid`, and a substring test for "name" matches `rename`
# and one for "sin" matches `single` by exactly the same mechanism. A pattern
# matches when its token sequence appears as a contiguous run in the column's
# token list, so `home_address`, `addressLine1` and `Address` all match
# ("address",) while `readdressed` does not tokenize to it at all.
#
# Ordered, first match wins, so a multi-token pattern precedes any single-token
# pattern it contains: ("first", "name") is tried before ("name",).
_PII_NAME_PATTERNS: List[Tuple[Tuple[str, ...], str]] = [
    # -- names (Safe Harbor A) --
    (("first", "name"), "first_name"),
    (("given", "name"), "first_name"),
    (("fore", "name"), "first_name"),
    (("forename",), "first_name"),
    (("firstname",), "first_name"),
    (("middle", "name"), "first_name"),
    (("last", "name"), "last_name"),
    (("lastname",), "last_name"),
    (("family", "name"), "last_name"),
    (("maiden", "name"), "last_name"),
    (("surname",), "last_name"),
    (("full", "name"), "name"),
    (("fullname",), "name"),
    (("user", "name"), "user_name"),
    (("username",), "user_name"),
    (("patient",), "name"),
    (("guardian",), "name"),
    (("beneficiary",), "name"),
    (("policyholder",), "name"),
    (("name",), "name"),

    # -- electronic contact (Safe Harbor F, N) --
    (("email",), "email"),
    (("e", "mail"), "email"),
    (("ip", "address"), "ipv4"),
    (("ipv4",), "ipv4"),
    (("ipv6",), "ipv6"),
    (("mac", "address"), "mac_address"),

    # -- telephone / fax (Safe Harbor D, E) --
    (("phone",), "phone_number"),
    (("telephone",), "phone_number"),
    (("mobile",), "phone_number"),
    (("msisdn",), "phone_number"),
    (("fax",), "phone_number"),

    # -- government identifiers (Safe Harbor G) --
    (("ssn",), "ssn"),
    (("social", "security"), "ssn"),
    (("national", "insurance"), "ssn"),
    (("tax", "identification"), "ssn"),
    (("passport",), "passport_number"),

    # -- account and financial identifiers (Safe Harbor J) --
    (("account", "number"), "bban"),
    (("account", "no"), "bban"),
    (("bank", "account"), "bban"),
    (("iban",), "iban"),
    (("bban",), "bban"),
    (("sort", "code"), "aba"),
    (("routing", "number"), "aba"),
    (("credit", "card"), "credit_card_number"),
    (("card", "number"), "credit_card_number"),
    (("cardholder",), "name"),

    # -- health identifiers (Safe Harbor H, I) --
    (("medical", "record"), "bothify"),
    (("mrn",), "bothify"),
    (("health", "plan"), "bothify"),
    (("nhs", "number"), "bothify"),

    # -- certificate / licence / vehicle / device (Safe Harbor K, L, M) --
    (("driver", "licence"), "license_plate"),
    (("driver", "license"), "license_plate"),
    (("driving", "licence"), "license_plate"),
    (("licence", "number"), "license_plate"),
    (("license", "number"), "license_plate"),
    (("licence", "plate"), "license_plate"),
    (("license", "plate"), "license_plate"),
    (("number", "plate"), "license_plate"),
    (("registration", "plate"), "license_plate"),
    (("vin",), "vin"),
    (("imei",), "bothify"),
    (("serial", "number"), "bothify"),
    (("device", "serial"), "bothify"),

    # -- dates tied to an individual (Safe Harbor C) --
    (("date", "of", "birth"), "date_of_birth"),
    (("birth", "date"), "date_of_birth"),
    (("birthdate",), "date_of_birth"),
    (("birthday",), "date_of_birth"),
    (("dob",), "date_of_birth"),

    # -- geography finer than a state (Safe Harbor B) --
    (("home", "address"), "address"),
    (("mailing", "address"), "address"),
    (("billing", "address"), "address"),
    (("shipping", "address"), "address"),
    (("street", "address"), "street_address"),
    (("address",), "address"),
    (("addr",), "address"),
    (("street",), "street_address"),
    (("postcode",), "postcode"),
    (("postal", "code"), "postcode"),
    (("post", "code"), "postcode"),
    (("zipcode",), "zipcode"),
    (("zip", "code"), "zipcode"),
    (("zip",), "zipcode"),
    (("latitude",), "latitude"),
    (("longitude",), "longitude"),
]

# Tokens that, immediately before "name", say the column names a *thing*, not a
# person: `file_name`, `table_name`, `product_name`. Without this the bare
# ("name",) pattern -- which is the one that catches `patient_name` and is
# therefore the one worth having -- also strips the profile off every
# schema-metadata column in the dataset. The bare pattern is deliberately kept
# and qualified rather than dropped: under-flagging a person's name is the
# failure this whole change exists to fix.
_NON_PERSON_NAME_QUALIFIERS = frozenset({
    "app", "application", "attribute", "brand", "bucket", "category", "class",
    "cluster", "col", "column", "container", "currency", "dag", "dataset",
    "database", "db", "dir", "directory", "domain", "entity", "environment",
    "event", "feature", "field", "file", "folder", "font", "function", "group",
    "host", "hostname", "icon", "image", "index", "job", "key", "label",
    "language", "layer", "locale", "method", "metric", "model", "module",
    "node", "object", "operation", "package", "page", "param", "parameter",
    "partition", "path", "pipeline", "plan", "platform", "port", "process",
    "product", "project", "queue", "region", "repo", "repository", "resource",
    "role", "route", "rule", "schema", "service", "sheet", "site", "source",
    "stage", "step", "stream", "style", "subject", "system", "tab", "table",
    "tag", "task", "template", "tenant", "theme", "timezone", "topic", "type",
    "unit", "variable", "version", "view", "workflow", "zone",
})


def _contains_token_run(tokens: List[str], pattern: Tuple[str, ...]) -> int:
    """Index at which ``pattern`` occurs as a contiguous run in ``tokens``, or -1."""
    n, m = len(tokens), len(pattern)
    if m == 0 or m > n:
        return -1
    for i in range(n - m + 1):
        if tuple(tokens[i:i + m]) == pattern:
            return i
    return -1


def pii_name_signal(name: str) -> Optional[str]:
    """The faker provider suggested by a column NAME alone, or None.

    Returns None for id-like names (``_is_id_like``). A trailing ``id`` /
    ``uuid`` / ``guid`` token marks a *structural key* in this profiler:
    ``_detect_pk`` and ``_infer_relationships`` both key off it, and flagging
    one as PII strips its ``unique`` constraint and swaps its generator, which
    silently destroys the primary key and every foreign key pointing at it.

    A surrogate key can of course still be a direct identifier -- ``patient_id``
    and ``mrn_id`` are the obvious cases -- so these are not simply ignored.
    They are collected and reported by the disclosure summary as columns a
    human has to make a call on. Quietly anonymizing a key and breaking
    referential integrity is the one outcome worse than not flagging it.
    """
    tokens = [t.lower() for t in _name_tokens(name)]
    if not tokens or tokens[-1] in _ID_TOKENS:
        return None
    for pattern, provider in _PII_NAME_PATTERNS:
        at = _contains_token_run(tokens, pattern)
        if at < 0:
            continue
        if pattern == ("name",) and at > 0 and \
                tokens[at - 1] in _NON_PERSON_NAME_QUALIFIERS:
            continue
        return provider
    return None


# ---------------------------------------------------------------------------
# Re-profiling without destroying hand edits
# ---------------------------------------------------------------------------
# Profiling is meant to be iterative: profile, read the spec, correct what the
# algorithms got wrong, profile again with a better sample or a looser
# threshold. Until now the second step threw away the third -- ``profile``
# writes a whole new file and every correction went with it, which makes the
# loop unusable and is why people stop after one pass.
#
# The mechanism is drift detection, not a diff. Each inferred rule is written
# with a fingerprint of the value the profiler itself produced. On a re-profile
# the fingerprint is compared against what the spec now says:
#
#   value == fingerprint   -> nobody touched it; the new inference wins
#   value != fingerprint   -> a human changed it; keep the human's value and
#                             mark it human_edited so it is never silently
#                             re-inferred again
#   no provenance at all   -> hand-authored, or written before this existed;
#                             keep it, on the principle that an unexplained
#                             rule is more likely a person's than a machine's
#
# Scope: this covers what discovery writes -- the primary key and the
# relationships. Column-level edits (a corrected dtype, a pii flag, an edited
# categorical distribution) are NOT preserved yet; see the PR description.


def _provenance_is_stale(prov, current_value) -> bool:
    """Whether ``current_value`` differs from what the profiler last wrote."""
    if prov is None or not prov.fingerprint:
        return False
    return prov.fingerprint != _fingerprint(current_value)


def _mark_edited(prov: Optional[InferenceProvenance]) -> InferenceProvenance:
    if prov is None:
        return InferenceProvenance(algorithm="human", human_edited=True)
    updated = prov.model_copy(deep=True)
    updated.human_edited = True
    return updated


def merge_preserving_edits(existing: Spec, profiled: Spec) -> Spec:
    """Fold a fresh profile into an existing spec, keeping human corrections.

    Returns a new Spec; neither argument is mutated. Tables and relationships
    the fresh profile found but the existing spec does not have are added --
    the point of re-profiling is to learn something new, so a merge that only
    ever preserved would be as useless as one that only ever overwrote.
    """
    merged = profiled.model_copy(deep=True)
    old_tables = {t.name: t for t in existing.tables}

    for table in merged.tables:
        old = old_tables.get(table.name)
        if old is None:
            continue

        # ----- primary key -----
        keep_key = (
            old.primary_key is not None
            and (old.key_provenance is None
                 or old.key_provenance.human_edited
                 or _provenance_is_stale(old.key_provenance, old.primary_key))
        )
        if keep_key:
            table.primary_key = old.primary_key
            table.key_provenance = _mark_edited(old.key_provenance)

        # ----- relationships, matched on the child column(s) -----
        old_rels = {tuple(r.child_columns): r for r in old.relationships}
        kept: Dict[tuple, RelationshipSpec] = {}
        for key, rel in old_rels.items():
            # The fingerprint a discovered relationship carries is over
            # [child column, "parent.column"], so that is what is re-derived.
            edited = (
                rel.provenance is None
                or rel.provenance.human_edited
                or _provenance_is_stale(rel.provenance, [rel.from_, rel.to])
            )
            if edited:
                preserved = rel.model_copy(deep=True)
                preserved.provenance = _mark_edited(rel.provenance)
                kept[key] = preserved

        fresh = [r for r in table.relationships
                 if tuple(r.child_columns) not in kept]
        table.relationships = list(kept.values()) + fresh

    # Tables the existing spec has and the fresh profile does not are kept:
    # a source that was not re-read is not a source that went away.
    profiled_names = {t.name for t in merged.tables}
    for name, old in old_tables.items():
        if name not in profiled_names:
            merged.tables.append(old.model_copy(deep=True))

    return merged


class DatasetProfiler:
    def __init__(
        self,
        df: pd.DataFrame,
        name: Optional[str] = None,
        source: Optional[str] = None,
        sample: Optional[int] = 5000,
        seed: Optional[int] = None,
        max_categorical: int = DEFAULT_MAX_CATEGORICAL,
        max_categorical_ratio: float = DEFAULT_MAX_CATEGORICAL_RATIO,
        max_categorical_ratio_cap: int = DEFAULT_MAX_CATEGORICAL_RATIO_CAP,
        pii_columns: Optional[List[str]] = None,
        pii_strategy: str = "faker",
        redact_categoricals: bool = False,
        min_cell_count: int = DEFAULT_MIN_CELL_COUNT,
        discover: Optional[bool] = DEFAULT_DISCOVER,
        discover_fds: bool = DEFAULT_DISCOVER_FDS,
        fd_error: float = discovery.DEFAULT_FD_ERROR,
        fd_min_mu: float = discovery.DEFAULT_MIN_MU,
        ind_error: float = discovery.DEFAULT_IND_ERROR,
        discovery_sample_rows: int = discovery.DEFAULT_SAMPLE_ROWS,
        discovery_max_lhs: int = discovery.DEFAULT_MAX_LHS,
    ):
        self.full = df
        self.name = name or "profiled"
        self.source = source or "unknown"
        self.sample = sample
        self.seed = seed
        self.max_categorical = max_categorical
        self.max_categorical_ratio = max_categorical_ratio
        self.max_categorical_ratio_cap = max_categorical_ratio_cap
        self.pii_columns = set(pii_columns or [])
        self.pii_strategy = pii_strategy
        self.redact_categoricals = redact_categoricals
        self.min_cell_count = max(0, int(min_cell_count))
        if discover is True and not discovery.is_available():
            discovery.require_desbordante()  # raises with an actionable message
        self.discover = (
            discovery.is_available() if discover is None else bool(discover)
        )
        self.discover_fds = bool(discover_fds) and self.discover
        self.fd_error = fd_error
        self.fd_min_mu = fd_min_mu
        self.ind_error = ind_error
        self.discovery_sample_rows = discovery_sample_rows
        self.discovery_max_lhs = discovery_max_lhs
        # id-like columns whose base name looks identifying (patient_id). Not
        # auto-flagged -- see pii_name_signal -- but surfaced for review.
        self.identifying_key_candidates: List[str] = []

    # ----- PII helpers -----
    #
    # Two INDEPENDENT signals, either sufficient on its own to flag a column:
    #
    #   value-pattern -- the column's VALUES match a recognisable identifier
    #                    format (email, SSN, phone);
    #   column-name   -- the column's NAME matches a known identifier name
    #                    (see _PII_NAME_PATTERNS, above).
    #
    # They used to be chained rather than independent. `_faker_provider_for`,
    # the name-based mapping, was only ever reached from inside `if is_pii:`,
    # and the only things that could make `is_pii` true were an explicit --pii
    # list and the three value regexes below, applied to `str` columns only. So
    # the name signal could not *flag* anything: it could only pick a provider
    # for a column something else had already flagged. The consequence is that
    # `patient_name`, `home_address` and `date_of_birth` were never detected --
    # no regex matches a human name or a street address, and the one rule that
    # recognises them by name sat downstream of the flag it needed to set.
    #
    # The value regexes are kept as-is; they were never the bug. They are now
    # simply one of two inputs rather than the gate on the other.
    _EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    _SSN_RE = re.compile(r"^\d{3}-\d{2}-\d{4}$")
    _PHONE_RE = re.compile(r"^\+?[\d\s().-]{7,}$")

    @classmethod
    def _looks_like_pii(cls, s: pd.Series) -> Optional[str]:
        """Heuristically detect a PII column from its string values.

        Returns a suggested faker provider, or None. This is the *value*
        signal; the *name* signal is `pii_name_signal` and runs independently.
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
        """Provider for a column flagged PII, when no signal named one.

        Reached for a column listed in --pii whose name matches no pattern.
        Delegates to the same published pattern set the name signal uses, so
        there is one table rather than two that can drift apart, and falls back
        on dtype.
        """
        provider = pii_name_signal(name)
        if provider:
            return provider
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
        source_rows = len(df)
        # per-table, so a profiler reused across tables does not accumulate
        self.identifying_key_candidates = []
        if self.sample and source_rows > self.sample:
            sdf = df.sample(n=self.sample, random_state=self.seed)
        else:
            sdf = df
        n = len(sdf)
        cols: List[ColumnSpec] = []
        for c in df.columns:
            # The sample drives the expensive shape work; the full column is
            # passed alongside it so facts that must be true of the whole
            # dataset (nullability, uniqueness) are checked against all of it.
            spec = self._profile_column(str(c), sdf[c], full=df[c])
            if spec is not None:
                cols.append(spec)
        uccs = self._discover_uccs(df)
        pk, key_prov = self._detect_pk(cols, uccs, source_rows)
        unique_constraints = self._composite_unique_constraints(uccs, pk)
        fds = self._discover_fds(df)
        # ``row_count`` is the size of the SOURCE dataset, not of whatever
        # sample we happened to read. It is the contract "a dataset of this
        # shape has this many rows", and ``syntab check`` compares generated
        # output against it. Recording the sample size here silently shrank
        # every profiled dataset to --sample rows, and the conformance check
        # then certified the truncated result as correct.
        #
        # How much was actually read is provenance, so it is recorded
        # separately under ``metadata.profiling``. Nothing compares against it.
        return TableSpec(
            name=name,
            row_count=source_rows,
            primary_key=pk,
            key_provenance=key_prov,
            columns=cols,
            unique_constraints=unique_constraints,
            metadata={
                "profiling": {
                    "source_row_count": source_rows,
                    "sampled_rows": n,
                    "sampled": n < source_rows,
                    # id-like columns whose base name looks identifying
                    # (patient_id). Not auto-flagged, because anonymizing a key
                    # breaks PK/FK integrity -- recorded so the disclosure
                    # summary can put them in front of a human. See
                    # pii_name_signal.__doc__.
                    **({"identifying_key_candidates":
                        list(self.identifying_key_candidates)}
                       if self.identifying_key_candidates else {}),
                },
                **({"discovery": {"functional_dependencies": fds}}
                   if fds else {}),
            },
        )

    @classmethod
    def profile_set(
        cls,
        tables: Dict[str, pd.DataFrame],
        name: Optional[str] = None,
        sample: Optional[int] = 5000,
        seed: Optional[int] = None,
        max_categorical: int = DEFAULT_MAX_CATEGORICAL,
        max_categorical_ratio: float = DEFAULT_MAX_CATEGORICAL_RATIO,
        max_categorical_ratio_cap: int = DEFAULT_MAX_CATEGORICAL_RATIO_CAP,
        pii_columns: Optional[List[str]] = None,
        pii_strategy: str = "faker",
        redact_categoricals: bool = False,
        min_cell_count: int = DEFAULT_MIN_CELL_COUNT,
        discover: Optional[bool] = DEFAULT_DISCOVER,
        discover_fds: bool = DEFAULT_DISCOVER_FDS,
        fd_error: float = discovery.DEFAULT_FD_ERROR,
        fd_min_mu: float = discovery.DEFAULT_MIN_MU,
        ind_error: float = discovery.DEFAULT_IND_ERROR,
        discovery_sample_rows: int = discovery.DEFAULT_SAMPLE_ROWS,
        discovery_max_lhs: int = discovery.DEFAULT_MAX_LHS,
    ) -> Spec:
        """Profile several related tables and discover foreign keys.

        Foreign keys come from inclusion dependencies found by SPIDER
        (Bauckmann, Leser, Naumann & Tietz, ICDE 2007), not from column names.
        See ``_relationships_from_inds`` for how an IND becomes a foreign key,
        and ``_relationships_from_names`` for the name-based rule that is now
        only the fallback for when the optional discovery extra is absent.
        """
        profilers = {
            t: cls(df, name=t, sample=sample, seed=seed,
                   max_categorical=max_categorical,
                   max_categorical_ratio=max_categorical_ratio,
                   max_categorical_ratio_cap=max_categorical_ratio_cap,
                   pii_columns=pii_columns, pii_strategy=pii_strategy,
                   redact_categoricals=redact_categoricals,
                   min_cell_count=min_cell_count,
                   discover=discover, discover_fds=discover_fds,
                   fd_error=fd_error, fd_min_mu=fd_min_mu,
                   ind_error=ind_error,
                   discovery_sample_rows=discovery_sample_rows,
                   discovery_max_lhs=discovery_max_lhs)
            for t, df in tables.items()
        }
        table_specs = {
            t: profilers[t]._build_table_spec(df, t, f"profiled:{t}")
            for t, df in tables.items()
        }
        any_profiler = next(iter(profilers.values()), None)
        use_discovery = bool(any_profiler and any_profiler.discover)
        rels = cls._infer_relationships(
            table_specs, tables,
            use_discovery=use_discovery, ind_error=ind_error,
        )
        for t, rel_list in rels.items():
            table_specs[t].relationships = rel_list
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
    def _value_kind(s: pd.Series) -> str:
        """Coarse comparability class of a column, for foreign-key matching.

        Two columns are only candidates for a key relationship if they hold
        the same kind of value. Note that ``is_numeric_dtype`` is true of a
        boolean column, so bool has to be tested first.

        Delegates to ``discovery.value_kind``: the same rule now also has to
        gate what is handed to SPIDER and what is believed of its output, and
        two copies of a type-compatibility rule drift.
        """
        return discovery.value_kind(s)

    @classmethod
    def _key_values(cls, s: pd.Series) -> set:
        """The distinct key values of a column, compared on their own type.

        This used to be ``{str(v) for v in col.dropna().unique()}`` on both
        sides of the containment test, which is wrong in both directions:

        * a nullable foreign key is held by pandas in a float64 column, so a
          child value of 1 stringifies to ``"1.0"`` while the parent's int64 1
          stringifies to ``"1"``. The subset test failed and the relationship
          was silently dropped -- and a nullable FK is the ordinary case, not
          an edge case;
        * in the other direction, columns of genuinely different types could
          be matched because their string forms happened to coincide.

        So: integral numerics are normalized to int, and columns of different
        kinds are never compared at all.
        """
        s = s.dropna()
        kind = cls._value_kind(s)
        if kind == "bool":
            return set(s.astype(bool).tolist())
        if kind == "numeric":
            vals = s.astype("float64")
            # 2**53 is the largest integer float64 represents exactly. Past
            # that the float is already lossy, so narrowing to int would be
            # inventing precision rather than recovering it.
            if len(vals) and bool((vals % 1 == 0).all()) \
                    and float(vals.abs().max()) < 2 ** 53:
                return set(vals.astype("int64").tolist())
            return set(vals.tolist())
        if kind == "datetime":
            return set(pd.to_datetime(s).tolist())
        return set(s.astype(str).tolist())

    # ----- foreign keys -----------------------------------------------
    #
    # A foreign key is an inclusion dependency whose referenced side is a key.
    # That is the definition, and it is also the primary discriminating
    # feature in the published work on picking foreign keys out of a set of
    # INDs (Rostin, Albrecht, Bauckmann, Naumann & Leser, "A Machine Learning
    # Approach to Foreign Key Discovery", WebDB 2009). SPIDER supplies the
    # INDs; the rules below decide which of them are foreign keys.

    @classmethod
    def _infer_relationships(
        cls,
        table_specs: Dict[str, "TableSpec"],
        dfs: Dict[str, pd.DataFrame],
        *,
        use_discovery: bool = True,
        ind_error: float = discovery.DEFAULT_IND_ERROR,
    ) -> Dict[str, List[RelationshipSpec]]:
        """Foreign keys, from inclusion dependencies where possible.

        Falls back to the name-based rule when the discovery extra is absent,
        recording ``algorithm="name-heuristic"`` on the provenance so a spec
        never overstates the evidence behind a relationship.
        """
        if use_discovery and discovery.is_available():
            try:
                return cls._relationships_from_inds(
                    table_specs, dfs, ind_error=ind_error
                )
            except discovery.DiscoveryUnavailable:
                pass
        return cls._relationships_from_names(table_specs, dfs)

    @staticmethod
    def _normalize_key_column(s: pd.Series) -> pd.Series:
        """Put a column into the form key comparison happens in.

        Integral numerics are narrowed to a nullable integer so a nullable
        foreign key -- which pandas holds as float64, making 1 into "1.0" --
        compares equal to an int64 parent key of "1". This is the same
        normalization ``_key_values`` performs for the name-based path, moved
        upstream of SPIDER because SPIDER compares the values itself.

        2**53 is the largest integer float64 represents exactly; past it the
        float is already lossy, so narrowing would invent precision.
        """
        kind = discovery.value_kind(s)
        if kind != "numeric":
            return s
        vals = pd.to_numeric(s, errors="coerce").astype("Float64")
        nn = vals.dropna()
        if len(nn) and bool((nn % 1 == 0).all()) and float(nn.abs().max()) < 2 ** 53:
            return vals.astype("Int64")
        return vals

    @classmethod
    def _ind_candidate_columns(
        cls, table_specs: Dict[str, "TableSpec"], dfs: Dict[str, pd.DataFrame]
    ) -> Dict[str, List[str]]:
        """Which columns are worth handing to SPIDER.

        The promotion rule below only ever accepts an IND whose PARENT side is
        a table's primary key, so the referenced side of the search space is
        exactly the set of primary-key columns. A column is therefore a useful
        candidate only if it is a primary key, or if some other table's
        primary key could contain it. Two necessary conditions decide that:

          * same value kind -- the type-compatibility pruning introduced by
            the P0 work, kept because it is sound and cheap;
          * no more distinct values than the parent key has. Containment
            cannot hold otherwise, so this discards nothing real.

        On the CFPB source the second condition is what keeps the 202,516
        distinct free-text narratives out of a sort-merge over every column.
        This is a pre-filter on the candidate set. Recall now comes from
        SPIDER, not from a rule about column names.
        """
        keys: Dict[str, Tuple[str, str, int]] = {}
        for name, ts in table_specs.items():
            pk = ts.primary_key
            if not isinstance(pk, str) or pk not in dfs[name].columns:
                continue
            col = dfs[name][pk]
            keys[name] = (pk, discovery.value_kind(col), int(col.nunique(dropna=True)))

        out: Dict[str, List[str]] = {}
        for name, df in dfs.items():
            own_pk = keys.get(name, (None,))[0]
            chosen: List[str] = []
            for c in df.columns:
                c = str(c)
                if c == own_pk:
                    chosen.append(c)
                    continue
                kind = discovery.value_kind(df[c])
                distinct = int(df[c].nunique(dropna=True))
                if distinct == 0:
                    continue
                if any(pk_kind == kind and distinct <= pk_distinct
                       for other, (_, pk_kind, pk_distinct) in keys.items()
                       if other != name):
                    chosen.append(c)
            if chosen:
                out[name] = chosen
        return out

    @classmethod
    def _relationships_from_inds(
        cls,
        table_specs: Dict[str, "TableSpec"],
        dfs: Dict[str, pd.DataFrame],
        *,
        ind_error: float = discovery.DEFAULT_IND_ERROR,
    ) -> Dict[str, List[RelationshipSpec]]:
        """Promote SPIDER's inclusion dependencies to foreign keys.

        An IND ``child.c SUBSET-OF parent.p`` becomes a foreign key when:

        1. ``p`` IS the parent table's primary key. This is what a foreign key
           means, and it is the feature that does nearly all the work: on the
           CFPB source it is what rejects the reverse INDs (a dimension's
           values are trivially contained in the fact column they came from)
           and the accidental ones between two unconstrained fact columns.
        2. ``c`` is NOT the child table's own primary key. A mutual inclusion
           between two primary keys is a one-to-one correspondence with no
           direction the data can settle, and asserting one at random is worse
           than asserting none. This deliberately declines some real 1:1
           foreign keys; see the report.
        3. the two columns have the same value kind. SPIDER compares values
           after its own conversion, so the string "1" is contained in an
           integer column of 1s. The type-compatibility rule from the P0 work
           rejects that, here as a post-filter as well as a pre-filter.
        4. the tables differ. An intra-table IND into the table's own key is a
           self-reference, which the engine supports but only with the extra
           settings (nullable key, root_fraction, max_depth) that make a
           hierarchy terminate. Inferring one without them produces a spec
           that fails validation, so it is left to a human. Called out in the
           report rather than inferred.

        When a child column is contained in several primary keys, the one with
        the highest COVERAGE -- distinct child values over distinct parent key
        values -- wins. Coverage is one of the features Rostin et al. (WebDB
        2009) rank foreign-key candidates on: a child column that uses almost
        all of a key is far more likely to reference it than one that touches
        a handful of a much larger key's values. Column NAME breaks a
        remaining tie and nothing more, exactly as it does for primary keys.

        Finally, edges that would make the inter-table graph cyclic are
        dropped lowest-coverage-first: the engine generates parents before
        children and cannot satisfy a cycle.
        """
        frames = {t: df for t, df in dfs.items() if len(df) > 0}
        columns = cls._ind_candidate_columns(table_specs, frames)
        normalized = {
            t: pd.DataFrame(
                {c: cls._normalize_key_column(frames[t][c]) for c in cols}
            )
            for t, cols in columns.items()
        }
        if not normalized:
            return {t: [] for t in table_specs}

        inds = discovery.discover_inclusion_dependencies(
            normalized, error=ind_error
        )

        pk_of = {
            name: ts.primary_key if isinstance(ts.primary_key, str) else None
            for name, ts in table_specs.items()
        }
        scored: List[Tuple[float, bool, Any]] = []
        for ind in inds:
            if ind.child_table == ind.parent_table:
                continue                                        # rule 4
            if pk_of.get(ind.parent_table) != ind.parent_column:  # rule 1
                continue
            if pk_of.get(ind.child_table) == ind.child_column:    # rule 2
                continue
            child_raw = dfs[ind.child_table][ind.child_column]
            parent_raw = dfs[ind.parent_table][ind.parent_column]
            if discovery.value_kind(child_raw) != discovery.value_kind(parent_raw):
                continue                                        # rule 3
            parent_distinct = int(parent_raw.nunique(dropna=True))
            if parent_distinct == 0:
                continue
            coverage = int(child_raw.nunique(dropna=True)) / parent_distinct
            name_match = _id_base(ind.child_column) in _table_name_forms(
                ind.parent_table
            )
            scored.append((coverage, name_match, ind))

        # Best parent per child column: coverage, then the name, then a stable
        # alphabetical order so the result does not depend on dict ordering.
        best: Dict[Tuple[str, str], Tuple[float, bool, Any]] = {}
        for coverage, name_match, ind in scored:
            key = (ind.child_table, ind.child_column)
            current = best.get(key)
            candidate = (coverage, name_match, ind)
            if current is None or (
                (coverage, name_match, ind.parent_table)
                > (current[0], current[1], current[2].parent_table)
            ):
                best[key] = candidate

        rels: Dict[str, List[RelationshipSpec]] = {t: [] for t in table_specs}
        edges: set = set()
        for coverage, name_match, ind in sorted(
            best.values(), key=lambda x: (-x[0], x[2].child_table, x[2].child_column)
        ):
            if _would_cycle(edges, ind.child_table, ind.parent_table):
                continue
            edges.add((ind.child_table, ind.parent_table))
            rels[ind.child_table].append(RelationshipSpec(**{
                "from": ind.child_column,
                "to": f"{ind.parent_table}.{ind.parent_column}",
                "alias": ind.parent_table,
                "provenance": InferenceProvenance(
                    algorithm="SPIDER" if not name_match else "SPIDER+name-tiebreak",
                    citation=discovery.CITATIONS["SPIDER"],
                    measure="exact" if ind.error == 0.0 else "ind_error",
                    confidence=round(1.0 - ind.error, 6),
                    error=round(ind.error, 6),
                    support=ind.support,
                    validated_on="full",
                    fingerprint=_fingerprint(
                        [ind.child_column,
                         f"{ind.parent_table}.{ind.parent_column}"]
                    ),
                ),
            }))
        return rels

    @classmethod
    def _relationships_from_names(
        cls, table_specs: Dict[str, "TableSpec"], dfs: Dict[str, pd.DataFrame]
    ) -> Dict[str, List[RelationshipSpec]]:
        """The pre-discovery rule: ``<parent>_id`` plus a table-stem match.

        Kept as the fallback for installations without the optional discovery
        extra. Its recall is bounded by a naming convention -- on the CFPB
        source no column is named ``<table>_id``, so it finds nothing at all --
        which is the whole reason SPIDER is now the primary path.
        """
        rels: Dict[str, List[RelationshipSpec]] = {t: [] for t in table_specs}
        names = set(table_specs)
        for cname, cts in table_specs.items():
            cdf = dfs[cname]
            for col in cts.columns:
                colname = col.name
                if colname not in cdf.columns:
                    continue
                # Token-aware, so camelCase and spaced names work too; returns
                # None for a bare "id" or a non-identifier name.
                base = _id_base(colname)
                if not base:
                    continue
                singular = base[:-1] if base.endswith("s") else base
                candidates = [b for b in (base, base + "s", singular)
                              if b in names and b != cname]
                if not candidates:
                    continue
                child = cdf[colname]
                child_vals = cls._key_values(child)
                if not child_vals:
                    continue
                child_kind = cls._value_kind(child)
                for pname in candidates:
                    pkp = table_specs[pname].primary_key
                    if not isinstance(pkp, str):
                        continue  # no key, or a composite one
                    pdf = dfs[pname]
                    if pkp not in pdf.columns:
                        continue
                    parent = pdf[pkp]
                    if cls._value_kind(parent) != child_kind:
                        continue
                    if child_vals.issubset(cls._key_values(parent)):
                        rels[cname].append(RelationshipSpec(**{
                            "from": colname,
                            "to": f"{pname}.{pkp}",
                            "alias": pname,
                            "provenance": InferenceProvenance(
                                algorithm="name-heuristic",
                                measure="name-match",
                                support=int(len(cdf)),
                                validated_on="full",
                                fingerprint=_fingerprint(
                                    [colname, f"{pname}.{pkp}"]
                                ),
                            ),
                        }))
                        break
        return rels

    # ----- internals -----
    @staticmethod
    def _validate_unique(sample: pd.Series, full: Optional[pd.Series]) -> bool:
        """Decide uniqueness by candidate generation + validation.

        The sample can only ever *nominate* a candidate. Any column with more
        distinct values than the sample size is unique within a sample draw by
        construction, which is precisely the set of high-cardinality columns
        someone would want to test for keyhood -- so a sample-only verdict is
        wrong exactly where it matters. This is the structure sampling-based
        dependency discovery uses (HyFD and its descendants): cheap sampling
        proposes candidates, then every candidate is confirmed against the
        full data before it is believed.

        Validation is ``pandas.Series.is_unique`` -- one hash-table pass, and
        far cheaper than the sort a hand-rolled check would reach for. Nulls
        are dropped first to match the conformance checker, which follows SQL
        in not treating repeated NULLs as duplicate key values.
        """
        if len(sample) == 0:
            return False
        if int(sample.nunique()) != len(sample):
            return False  # not even a candidate
        if full is None:
            return True
        return bool(full.dropna().is_unique)

    def _profile_column(self, name: str, series: pd.Series,
                        full: Optional[pd.Series] = None) -> Optional[ColumnSpec]:
        s = series.dropna()
        if len(s) == 0:
            return None  # fully-null column: skip

        # Nullability comes from the FULL column, never the sample. A column
        # that is null in one row per million shows zero nulls in a 5k draw;
        # the profiler then wrote ``nullable: false``, and ``syntab check``
        # hard-failed the source dataset against its own spec on the first
        # real null. ``isna().sum()`` is a single vectorized pass, so there is
        # no performance reason to have approximated this from a sample.
        stats_src = full if full is not None else series
        total = len(stats_src)
        n_nulls = int(stats_src.isna().sum())
        null_rate = round(n_nulls / total, 4) if total else 1.0

        dtype, profile = self._infer_type_and_profile(s, full=full)
        params: dict = {}
        constraints: dict = {"nullable": n_nulls > 0, "null_rate": null_rate}

        # uniqueness: nominated on the sample, confirmed on the full column
        unique = self._validate_unique(s, full)
        if unique:
            constraints["unique"] = True

        generator = "auto"
        # Integer id-like sequential column -> sequence. Also decided on the
        # full column: a random sample of a contiguous sequence has gaps, so
        # asking the sample can only ever produce a false negative here.
        seq_src = full.dropna() if full is not None else s
        if dtype == "int" and unique and self._looks_sequential(seq_src):
            generator = "sequence"

        spec = ColumnSpec(
            name=name,
            dtype=dtype,
            generator=generator,
            params=params,
            constraints=constraints,
            profile=profile,
        )

        # PII handling. Each signal is evaluated on its own and any one of
        # them flags the column; see the _PII_NAME_PATTERNS commentary for why
        # the name signal used to be unreachable.
        signals: List[str] = []
        name_provider = pii_name_signal(name)
        value_provider = self._looks_like_pii(s) if dtype == "str" else None

        if name in self.pii_columns:
            signals.append(PII_SIGNAL_EXPLICIT)
        if name_provider:
            signals.append(PII_SIGNAL_NAME)
        if value_provider:
            signals.append(PII_SIGNAL_VALUE)

        # An id-like column whose base name looks identifying (patient_id,
        # ssn_id) is deliberately NOT auto-flagged -- anonymizing it would
        # break the primary key or a foreign key -- but it is recorded so the
        # disclosure summary can put it in front of a human. See
        # pii_name_signal.__doc__.
        if not signals and _is_id_like(name):
            base = _id_base(name)
            if base and pii_name_signal(base):
                self.identifying_key_candidates.append(name)

        if signals:
            strategy = self.pii_strategy if name in self.pii_columns else "faker"
            spec.pii = True
            spec.pii_strategy = strategy
            spec.pii_detected_by = signals
            # Never bake real values into the spec: substitute a faker
            # generator and drop captured values / patterns / stats. The chosen
            # strategy is then applied on top of the synthetic value at
            # generation time.
            #
            # Provider precedence: the value signal is evidence about the data
            # itself and beats a guess from the name, so an `email_backup`
            # column holding phone numbers gets phone_number, not email.
            provider = (value_provider or name_provider
                        or self._faker_provider_for(name, dtype))
            spec.generator = f"faker.{provider}"
            spec.profile = None
            spec.params = {}
            spec.constraints = {
                k: v for k, v in spec.constraints.items()
                if k in ("nullable", "null_rate")
            }

        return spec

    def _infer_type_and_profile(
        self, s: pd.Series, full: Optional[pd.Series] = None,
    ) -> Tuple[str, Optional[ColumnProfile]]:
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
        if self._is_categorical(n_unique, len(vals)):
            # Cell counts come from the FULL column, never the sample -- see
            # _categorical_profile.
            full_vals = (full.dropna().astype(str)
                         if full is not None else None)
            return "str", ColumnProfile(
                categorical=self._categorical_profile(vals, full_vals)
            )

        # free text: length stats (+ pattern if short)
        lengths = vals.str.len()
        prof = ColumnProfile(length=(int(lengths.min()), int(lengths.max())))
        if lengths.max() <= 30:
            prof.string_pattern = _string_pattern(vals.head(50).tolist())
        return "str", prof

    def _categorical_profile(
        self, vals: pd.Series, full_vals: Optional[pd.Series] = None,
    ) -> CategoricalProfile:
        """Build the categorical profile for a column, applying disclosure controls.

        This is the single place where real source values are copied into a
        spec, so it is the single place the disclosure controls have to act.
        Order matters: suppression works on real labels, so it must run before
        redaction destroys them.

        CELL COUNTS COME FROM THE FULL COLUMN. "How many people share this
        value" is a fact about the source data, and a sample cannot answer it:
        a value seen 3 times in a 5,000-row sample of 4M rows is not rare, and
        a value seen once might be one of thousands. Deciding suppression on
        sample counts would be the same mistake as deciding uniqueness on them
        (fixed in d3f78eb). Frequencies themselves are still sample-derived --
        that is existing behaviour and out of scope here -- so the emitted
        vector is unchanged when suppression is off.

        SUPPRESSION IS OFF BY DEFAULT (DEFAULT_MIN_CELL_COUNT = 0). Turning it
        on by default would silently change the statistical content of every
        profile -- a 5-row test frame would collapse entirely at K=5 -- and
        suppression is a policy decision belonging to whoever holds the data,
        not a default a library should make on their behalf. The answer to
        "then nobody will use it" is not a coerced default; it is the
        disclosure summary, which reports how many embedded values fall below
        RECOMMENDED_MIN_CELL_COUNT whether or not the flag was passed.
        """
        counts = vals.value_counts()
        # _is_categorical already bounds n_unique; head() is a belt-and-braces
        # guard so a future caller cannot make the profiler write an unbounded
        # number of real values into a spec file.
        chosen = counts.head(self._categorical_value_cap)
        total = float(chosen.sum())
        values = {str(k): float(v) / total for k, v in chosen.items()}

        source_counts = (full_vals.value_counts()
                         if full_vals is not None else counts)

        rare = sum(1 for label in values
                   if int(source_counts.get(label, 0)) < RECOMMENDED_MIN_CELL_COUNT)

        suppressed = 0
        if self.min_cell_count > 0:
            values, suppressed = _suppress_small_cells(
                values, source_counts, self.min_cell_count)
            rare = sum(1 for label in values
                       if label != OTHER_BUCKET_LABEL
                       and int(source_counts.get(label, 0))
                       < RECOMMENDED_MIN_CELL_COUNT)

        values = {k: round(v, 4) for k, v in values.items()}

        redacted = False
        if self.redact_categoricals:
            values = _redact_categorical_values(values)
            redacted = True

        return CategoricalProfile(
            values=values, null_rate=0.0, redacted=redacted,
            min_cell_count=self.min_cell_count or None,
            suppressed_values=suppressed,
            rare_value_count=rare,
        )

    @property
    def _categorical_value_cap(self) -> int:
        """Hard ceiling on how many distinct values may be written to a spec."""
        return max(self.max_categorical, self.max_categorical_ratio_cap)

    def _is_categorical(self, n_unique: int, n: int) -> bool:
        """Decide categorical vs free text for a string column.

        Two independent tests, either of which is sufficient.

        Absolute: at most ``max_categorical`` distinct values, and fewer than
        half the rows. This is the original rule and it covers the small fixed
        domain -- a status, a country code, a response category. The 0.5 guard
        stops a tiny frame whose rows are nearly all distinct from being
        called categorical.

        Ratio: the distinct count is at most ``max_categorical_ratio`` of the
        rows, capped at ``max_categorical_ratio_cap`` distinct values. A fixed
        absolute cutoff is simply wrong at scale -- 300 distinct product codes
        across 5M rows is unambiguously categorical, but 300 > 50, so the
        column fell through to the free-text branch and was re-synthesized as
        random characters with the right string length and nothing else. Low
        cardinality *relative to row count* is the standard way to make that
        call; the cap is what keeps the emitted spec bounded in size, and
        bounds how many real values it embeds.

        Raising ``max_categorical`` increases the number of real source values
        written verbatim into the spec file. That is the intended trade-off,
        but it is a disclosure trade-off, not only a fidelity one.
        """
        if n <= 0:
            return False
        ratio = n_unique / n
        if n_unique <= self.max_categorical and ratio < 0.5:
            return True
        return (ratio <= self.max_categorical_ratio
                and n_unique <= self.max_categorical_ratio_cap)

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

    def _discover_uccs(self, df: pd.DataFrame) -> Optional[List[Any]]:
        """Unique column combinations of the source table, or None.

        HyUCC (Papenbrock & Naumann, BTW 2017), via ``syntab.discovery``.
        Returns None -- not an empty list -- when discovery is switched off or
        unavailable, so the caller can tell "no keys" from "did not look".
        """
        if not self.discover:
            return None
        try:
            return discovery.discover_unique_column_combinations(
                df,
                max_columns=MAX_KEY_COLUMNS,
                sample_rows=self.discovery_sample_rows,
                seed=self.seed,
            )
        except discovery.DiscoveryUnavailable:
            return None

    def _discover_fds(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Functional dependencies of the source table, with provenance.

        HyFD for the exact ones and Pyro for the approximate ones, filtered by
        mu\' -- see ``syntab.discovery``, which cites all three. Recorded under
        ``metadata.discovery`` because ``metadata`` is the free-form provenance
        dict a TableSpec already has; no new schema field is introduced for a
        result that does not yet steer generation.

        THIS REPLACES ``metadata.suggested_depends_on``, which held pairs of
        Pearson-correlated numeric columns. Correlation is not dependency: it
        is symmetric where a dependency has a direction, it is defined only
        between numeric columns so it could never see ZIP code -> State, and
        a coefficient above 0.95 is not evidence that one column determines
        another. That field is gone; this is what a real answer looks like.
        """
        if not self.discover_fds:
            return []
        try:
            found = discovery.discover_functional_dependencies(
                df,
                error=self.fd_error,
                min_mu=self.fd_min_mu,
                max_lhs=self.discovery_max_lhs,
                sample_rows=self.discovery_sample_rows,
                seed=self.seed,
            )
        except discovery.DiscoveryUnavailable:
            return []
        # Strongest first, so a truncated list is the useful part of the list.
        found.sort(key=lambda f: (-(f.mu_prime or 1.0), len(f.determinant),
                                  f.determinant, f.dependent))
        return [f.as_dict() for f in found[:MAX_RECORDED_FDS]]

    def _composite_unique_constraints(
        self, uccs: Optional[List[Any]], pk: Optional[Any]
    ) -> List[List[str]]:
        """Composite candidate keys, as spec-level unique constraints.

        Only genuinely composite ones (2+ columns) and only a few: see
        MAX_COMPOSITE_UNIQUE_CONSTRAINTS for why the cap exists. Single-column
        uniqueness is already carried by ``constraints.unique`` on the column,
        which is where the engine and the conformance checker read it from.
        """
        if not uccs:
            return []
        pk_cols = tuple(pk) if isinstance(pk, list) else (pk,) if pk else ()
        out: List[List[str]] = []
        for u in uccs:
            if len(u.columns) < 2 or u.columns == pk_cols:
                continue
            out.append(list(u.columns))
            if len(out) >= MAX_COMPOSITE_UNIQUE_CONSTRAINTS:
                break
        return out

    def _detect_pk(
        self,
        cols: List[ColumnSpec],
        uccs: Optional[List[Any]] = None,
        source_rows: Optional[int] = None,
    ) -> Tuple[Optional[Any], Optional[InferenceProvenance]]:
        """Choose a primary key from discovered unique column combinations.

        A primary key is UNIQUE *and* NOT NULL. Both halves are checked: a
        unique-but-nullable column nominated as a key fails the conformance
        checker's own PK check, because pandas' ``duplicated`` counts repeated
        NaNs as duplicates where SQL UNIQUE does not.

        The uniqueness evidence is HyUCC's, over the whole table, not a naming
        convention. THE COLUMN NAME IS A TIE-BREAKER AND NOTHING ELSE: it is
        consulted only when more than one discovered candidate key survives
        the NOT NULL filter, which is the one situation where the data cannot
        distinguish them -- ``paid`` and ``record_id`` are both unique in
        ``{paid, record_id, region}`` and only the name says which is the key.
        When a single candidate survives, the name is never read, and the
        recorded provenance says which of the two decided.

        Composite keys: adopted only when no single-column candidate exists.
        A one-column key is what downstream generation is built around, and a
        table that has both is a table whose single-column key is the key.
        Among several composites the narrowest wins, ties broken
        lexicographically -- an arbitrary rule, chosen for determinism, and
        the reason the losers are still recorded as unique constraints.

        Falls back to the pre-existing per-column uniqueness check when
        discovery is unavailable, recording ``algorithm="name-heuristic"`` so
        the spec does not overstate its evidence.
        """
        by_name = {c.name: c for c in cols}

        def usable(name: str) -> bool:
            c = by_name.get(name)
            return bool(c and not c.constraints.get("nullable", True))

        discovered = uccs is not None
        if discovered:
            singles = [u for u in uccs
                       if len(u.columns) == 1 and usable(u.columns[0])]
            composites = [u for u in uccs
                          if 2 <= len(u.columns) <= MAX_KEY_COLUMNS
                          and all(usable(c) for c in u.columns)]
        else:
            # Pre-discovery behaviour: each column judged on its own.
            singles = [
                _NamedKey((c.name,)) for c in cols
                if c.constraints.get("unique")
                and not c.constraints.get("nullable", True)
            ]
            composites = []

        algorithm = "HyUCC" if discovered else "name-heuristic"
        support = source_rows

        if not singles:
            if composites:
                chosen = min(composites, key=lambda u: (len(u.columns), u.columns))
                return list(chosen.columns), InferenceProvenance(
                    algorithm=algorithm,
                    citation=discovery.CITATIONS.get(algorithm),
                    measure="exact", confidence=1.0, support=support,
                    validated_on="full",
                    fingerprint=_fingerprint(list(chosen.columns)),
                )
            return None, None

        # Ordered by the columns declaration order, not by the order the
        # discovery algorithm happened to emit its results in, so that the
        # last-resort "first candidate wins" tie-break is deterministic and
        # means what it says.
        position = {c.name: i for i, c in enumerate(cols)}
        names = sorted((u.columns[0] for u in singles),
                       key=lambda n: position.get(n, len(position)))
        if len(names) == 1:
            chosen, tie_broken_by = names[0], None
        else:
            chosen, tie_broken_by = self._break_key_tie(names, by_name)

        return chosen, InferenceProvenance(
            algorithm=algorithm if tie_broken_by is None
            else f"{algorithm}+{tie_broken_by}-tiebreak",
            citation=discovery.CITATIONS.get(algorithm),
            measure="exact", confidence=1.0, support=support,
            validated_on="full",
            fingerprint=_fingerprint(chosen),
        )

    @staticmethod
    def _break_key_tie(
        names: List[str], by_name: Dict[str, ColumnSpec]
    ) -> Tuple[str, str]:
        """Pick between several equally-unique candidate keys.

        Ordered weakest-evidence-last: an identifier-shaped name first, then a
        key-shaped dtype, then declaration order. All three are conventions
        rather than facts about the data -- which is precisely why they run
        only after the data has failed to decide.
        """
        for n in names:
            if _is_id_like(n):
                return n, "name"
        for n in names:
            c = by_name.get(n)
            if c is not None and c.dtype in ("int", "str"):
                return n, "dtype"
        return names[0], "order"
