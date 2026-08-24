"""Load/Save Specs from YAML or JSON."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Union

import yaml

from .spec import CURRENT_SPEC_VERSION, SUPPORTED_SPEC_VERSIONS, Spec, SpecVersionError


def migrate_spec_dict(data: dict, *, target_version: str = CURRENT_SPEC_VERSION) -> dict:
    """Normalize a legacy document to the current Spec wire format.

    Version ``1`` and ``1.0.0`` were emitted by early callers and are
    losslessly equivalent to the current ``1.0`` format.  A missing version is
    treated as that same legacy format for backward compatibility.  Unknown
    versions fail explicitly instead of being silently interpreted as the
    current schema.
    """
    if not isinstance(data, dict):
        raise SpecVersionError("Spec document must be a mapping")
    if target_version not in SUPPORTED_SPEC_VERSIONS:
        raise SpecVersionError(f"unsupported target Spec version: {target_version}")
    normalized = dict(data)
    version = normalized.get("spec_version")
    if version in (None, "1", "1.0.0"):
        normalized["spec_version"] = target_version
        return normalized
    if version not in SUPPORTED_SPEC_VERSIONS:
        raise SpecVersionError(
            f"unsupported spec_version {version!r}; supported versions: "
            f"{', '.join(sorted(SUPPORTED_SPEC_VERSIONS))}"
        )
    return normalized


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
    return Spec.model_validate(migrate_spec_dict(data))


def from_dict(data: dict) -> Spec:
    return Spec.model_validate(migrate_spec_dict(data))


def spec_to_dict(spec: Spec) -> Dict[str, Any]:
    """Serialize a Spec to a plain dict, omitting nulls and unset defaults.

    A profiled Spec is mostly empty. Every ColumnSpec carries ``params: {}``,
    ``constraints`` partly filled, ``depends_on: []``, ``when: []``,
    ``pii: false``, ``pii_strategy: null``, ``description: null`` and a
    ColumnProfile whose four unused branches are all ``null``. Dumping that
    verbatim produced roughly 390 lines for an 18-column table, almost all of
    it noise, which made the emitted spec unreadable and impractical to edit
    by hand -- the exact opposite of the point of having a spec file.

    pydantic already provides the two flags that do the whole job, so there is
    no bespoke pruning here:

    * ``exclude_none``     drops every ``null``-valued key;
    * ``exclude_defaults`` drops every key still equal to its field default,
      which is what removes ``{}``, ``[]`` and ``false``.

    Both are lossless for this model: anything omitted is either ``None`` or a
    value pydantic re-applies from the field default at load time. The
    round-trip test in ``tests/test_spec_serialization.py`` asserts that.

    ``by_alias`` is set so the output uses the documented public key names --
    ``from:`` on a relationship and ``if:`` on a ``when`` clause -- rather than
    the internal field names ``from_`` and ``condition``. Loading accepts
    either, because both models set ``populate_by_name``.
    """
    data = spec.model_dump(
        mode="json", by_alias=True, exclude_none=True, exclude_defaults=True
    )
    # spec_version equals its default in practice and would be dropped, but a
    # spec file should always state the schema version it was written against.
    if "spec_version" not in data:
        data = {"spec_version": spec.spec_version, **data}
    return data


def to_file(spec: Spec, path: Union[str, Path], fmt: str | None = None) -> None:
    p = Path(path)
    fmt = fmt or p.suffix.lower().lstrip(".")
    data = spec_to_dict(spec)
    if fmt in ("yaml", "yml"):
        p.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    elif fmt == "json":
        p.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported format: {fmt}")
