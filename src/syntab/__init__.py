"""syntab — Spec-driven synthetic tabular/relational data generator."""
from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("syntab")
except PackageNotFoundError:  # pragma: no cover -- editable without installed metadata
    __version__ = "0.0.0+local"

from .spec import (
    Spec,
    SpecMetadata,
    Settings,
    TableSpec,
    ColumnSpec,
    RelationshipSpec,
    ColumnProfile,
    NumericProfile,
    CategoricalProfile,
    WhenClause,
    CURRENT_SPEC_VERSION,
    SpecVersionError,
)
from .loaders import from_file, from_dict, to_file
from .engine import GenerationEngine, GenerationResult, SpecError, RuleViolation
from .generators import (
    BaseGenerator,
    register_generator,
    get_registry,
    empirical_text_params,
)
from .infer import fit_text_params
from .profiler import DatasetProfiler, merge_preserving_edits, schema_drift
from .validator import validate, ValidationReport
from .quality import (
    quality_report,
    diagnostic_report,
    QualityReport,
    DiagnosticReport,
)
from .conformance import (
    validate_against_spec,
    SpecConformance,
    TableConformance,
    Check,
)

__all__ = [
    "Spec", "SpecMetadata", "Settings", "TableSpec", "ColumnSpec",
    "RelationshipSpec", "ColumnProfile", "NumericProfile", "CategoricalProfile", "WhenClause",
    "CURRENT_SPEC_VERSION", "SpecVersionError",
    "from_file", "from_dict", "to_file",
    "GenerationEngine", "GenerationResult", "SpecError", "RuleViolation",
    "BaseGenerator", "register_generator", "get_registry",
    "empirical_text_params", "fit_text_params",
    "DatasetProfiler",
    "merge_preserving_edits", "schema_drift",
    "validate", "ValidationReport",
    "quality_report", "diagnostic_report", "QualityReport", "DiagnosticReport",
    "validate_against_spec", "SpecConformance", "TableConformance", "Check",
]
