"""Load/Save Specs from YAML or JSON."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Union

import yaml

from .spec import Spec


def from_file(path: Union[str, Path]) -> Spec:
    p = Path(path)
    # make modules referenced by import-path generators importable
    d = str(p.resolve().parent)
    if d not in sys.path:
        sys.path.insert(0, d)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in (".yaml", ".yml"):
        data = yaml.safe_load(text)
    elif p.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        # Try YAML then JSON.
        try:
            data = yaml.safe_load(text)
        except Exception:
            data = json.loads(text)
    return Spec.model_validate(data)


def from_dict(data: dict) -> Spec:
    return Spec.model_validate(data)


def to_file(spec: Spec, path: Union[str, Path], fmt: str | None = None) -> None:
    p = Path(path)
    fmt = fmt or p.suffix.lower().lstrip(".")
    if fmt in ("yaml", "yml"):
        p.write_text(yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=False), encoding="utf-8")
    elif fmt == "json":
        p.write_text(json.dumps(spec.model_dump(mode="json"), indent=2), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported format: {fmt}")
