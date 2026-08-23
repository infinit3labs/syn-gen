"""syntab — Spec-driven synthetic tabular/relational data generator."""
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
from .profiler import DatasetProfiler
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
    "from_file", "from_dict", "to_file",
    "GenerationEngine", "GenerationResult", "SpecError", "RuleViolation",
    "BaseGenerator", "register_generator", "get_registry",
    "empirical_text_params", "fit_text_params",
    "DatasetProfiler",
    "validate", "ValidationReport",
    "quality_report", "diagnostic_report", "QualityReport", "DiagnosticReport",
    "validate_against_spec", "SpecConformance", "TableConformance", "Check",
]
