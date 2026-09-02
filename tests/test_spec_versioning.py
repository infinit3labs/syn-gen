"""spec_version compatibility policy (INF-232): normalization, precise
failure on unsupported versions, and the CLI validate/migrate path.

docs/spec-versioning.md is the human-readable version of these guarantees;
this file is the fixture-based check the policy calls for.
"""
import pytest
import yaml
from click.testing import CliRunner

from syntab.cli import cli
from syntab.loaders import from_dict, from_file, migrate_spec_dict
from syntab.spec import CURRENT_SPEC_VERSION, SUPPORTED_SPEC_VERSIONS, SpecVersionError

_MINIMAL_TABLES = [{
    "name": "people", "row_count": 3, "primary_key": "id",
    "columns": [{"name": "id", "dtype": "int", "generator": "sequence",
                 "constraints": {"unique": True}}],
}]


def _doc(spec_version):
    doc = {"metadata": {"name": "t"}, "tables": _MINIMAL_TABLES}
    if spec_version is not None:
        doc["spec_version"] = spec_version
    return doc


@pytest.mark.parametrize("legacy_version", [None, "1", "1.0.0"])
def test_legacy_version_aliases_normalize_to_current(legacy_version):
    normalized = migrate_spec_dict(_doc(legacy_version))
    assert normalized["spec_version"] == CURRENT_SPEC_VERSION


def test_current_version_document_is_unchanged():
    doc = _doc(CURRENT_SPEC_VERSION)
    assert migrate_spec_dict(doc)["spec_version"] == CURRENT_SPEC_VERSION


def test_unsupported_version_fails_precisely_not_silently():
    with pytest.raises(SpecVersionError, match=r"unsupported spec_version 'not-a-version'"):
        migrate_spec_dict(_doc("not-a-version"))


def test_unsupported_version_error_names_the_supported_set():
    with pytest.raises(SpecVersionError, match=r"supported versions: 1\.0"):
        migrate_spec_dict(_doc("2.0"))


def test_non_mapping_document_fails_precisely():
    with pytest.raises(SpecVersionError, match="must be a mapping"):
        migrate_spec_dict(["not", "a", "mapping"])  # type: ignore[arg-type]


def test_from_dict_loads_legacy_alias_as_current():
    spec = from_dict(_doc("1"))
    assert spec.spec_version == CURRENT_SPEC_VERSION


def test_from_dict_rejects_future_version():
    with pytest.raises(SpecVersionError):
        from_dict(_doc("9.9"))


def test_from_file_normalizes_a_legacy_document_on_disk(tmp_path):
    path = tmp_path / "legacy.yaml"
    path.write_text(yaml.safe_dump(_doc("1.0.0")), encoding="utf-8")
    spec = from_file(str(path))
    assert spec.spec_version == CURRENT_SPEC_VERSION


def test_migration_is_deterministic():
    doc = _doc("1")
    assert migrate_spec_dict(doc) == migrate_spec_dict(doc)


def test_cli_validate_reports_version_and_supported_range(tmp_path):
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(_doc(None)), encoding="utf-8")

    result = CliRunner().invoke(cli, ["validate", "--spec", str(spec_path)])

    assert result.exit_code == 0, result.output
    assert f"spec_version={CURRENT_SPEC_VERSION}" in result.output
    assert f"supported={', '.join(sorted(SUPPORTED_SPEC_VERSIONS))}" in result.output


def test_cli_validate_migrate_out_writes_normalized_spec_without_generating(tmp_path):
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(_doc("1")), encoding="utf-8")
    migrated_path = tmp_path / "migrated.yaml"

    result = CliRunner().invoke(cli, [
        "validate", "--spec", str(spec_path), "--migrate-out", str(migrated_path),
    ])

    assert result.exit_code == 0, result.output
    assert f"migrated to {migrated_path}" in result.output
    migrated = yaml.safe_load(migrated_path.read_text())
    assert migrated["spec_version"] == CURRENT_SPEC_VERSION
    # migrate-out validates only -- no generated rows anywhere near this path.
    assert not (tmp_path / "out").exists()


def test_cli_validate_reports_unsupported_version_as_invalid_not_a_crash(tmp_path):
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(_doc("2.0")), encoding="utf-8")

    result = CliRunner().invoke(cli, ["validate", "--spec", str(spec_path)])

    assert result.exit_code == 1
    assert "Invalid:" in result.output
    assert "unsupported spec_version" in result.output
