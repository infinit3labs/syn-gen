from pathlib import Path

from syntab.loaders import from_file, to_file
from syntab.spec import Spec


def test_load_yaml_spec():
    path = Path(__file__).parent.parent / "examples" / "spec.yaml"
    spec = from_file(path)
    assert isinstance(spec, Spec)
    assert {t.name for t in spec.tables} == {"users", "orders"}


def test_load_json_spec():
    path = Path(__file__).parent.parent / "examples" / "spec_profiled.json"
    spec = from_file(path)
    assert isinstance(spec, Spec)
    assert spec.tables[0].name == "orders"


def test_roundtrip_yaml(tmp_path):
    path = Path(__file__).parent.parent / "examples" / "spec.yaml"
    spec = from_file(path)
    out = tmp_path / "rt.yaml"
    to_file(spec, out, "yaml")
    reloaded = from_file(out)
    assert reloaded.metadata.name == spec.metadata.name
