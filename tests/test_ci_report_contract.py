import json

import pandas as pd
from click.testing import CliRunner

from syntab.cli import cli


def test_quality_report_includes_ci_context(tmp_path):
    real = tmp_path / "real.csv"
    synthetic = tmp_path / "synthetic.csv"
    out = tmp_path / "quality.json"
    frame = pd.DataFrame({"value": [1, 2, 3]})
    frame.to_csv(real, index=False)
    frame.to_csv(synthetic, index=False)

    result = CliRunner().invoke(cli, [
        "quality", "--real", str(real), "--synthetic", str(synthetic),
        "--min-score", "0.8", "--out", str(out), "--compact",
    ])

    assert result.exit_code == 0, result.output
    payload = json.loads(out.read_text())
    assert payload["tool_version"]
    assert payload["inputs"] == {"real": str(real), "synthetic": str(synthetic)}
    assert payload["thresholds"] == {"min_score": 0.8}


def test_check_report_includes_spec_identity(tmp_path):
    spec = tmp_path / "spec.yaml"
    data = tmp_path / "data"
    out = tmp_path / "conformance.json"
    spec.write_text(
        """metadata:\n  name: ci-contract\nsettings:\n  seed: 17\ntables:\n  - name: users\n    row_count: 2\n    primary_key: id\n    columns:\n    - name: id\n      dtype: int\n      generator: sequence\n      constraints:\n        unique: true\n"""
    )
    data.mkdir()
    pd.DataFrame({"id": [0, 1]}).to_csv(data / "users.csv", index=False)

    result = CliRunner().invoke(cli, [
        "check", "--spec", str(spec), "--data", str(data),
        "--out", str(out), "--compact",
    ])

    assert result.exit_code == 0, result.output
    payload = json.loads(out.read_text())
    assert payload["tool_version"]
    assert payload["inputs"] == {"spec": str(spec), "data": str(data)}
    assert payload["spec"] == {
        "name": "ci-contract", "version": "1.0", "seed": 17,
    }
