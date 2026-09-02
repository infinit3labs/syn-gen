# Spec versioning and compatibility

A Spec carries two independent version fields, and it's easy to conflate
them:

- `spec_version` — the wire-format version: the shape and semantics of the
  Spec document itself (which fields exist, what they mean). This is what
  this document is about.
- `metadata.version` — the user's own revision label for *their* dataset or
  Spec content (e.g. `"2.3.0"`). syntab never reads or migrates this; it's
  yours.

`metadata.syntab_version` is a third, read-only field: the package version
that last profiled/emitted the Spec, recorded so a reviewer can tell which
behavior produced a given file. It plays no part in compatibility decisions.

## Current state

```python
from syntab.spec import CURRENT_SPEC_VERSION, SUPPORTED_SPEC_VERSIONS
```

`CURRENT_SPEC_VERSION` is `"1.0"`. `SUPPORTED_SPEC_VERSIONS` is the set of
`spec_version` values `from_file`/`from_dict` will load — today that's just
`{"1.0"}`, because the format hasn't shipped a breaking change yet. A
hand-authored Spec that omits `spec_version` entirely, or writes the early
aliases `1` or `1.0.0`, is accepted and normalized to `"1.0"` on load; those
three spellings are losslessly equivalent. Anything else fails loudly with
`SpecVersionError` naming the supported set, rather than being silently
parsed against the wrong schema.

## Checking and migrating a Spec

```bash
syntab validate --spec spec.yaml
# OK: shop (3 tables; spec_version=1.0, supported=1.0)

syntab validate --spec spec.yaml --migrate-out spec.normalized.yaml
# writes the current-version form without generating any data
```

`validate --migrate-out` is the CLI's validation/migration path: it loads,
normalizes, and re-serializes a Spec without touching a generator or writing
rows. Library callers get the same normalization for free — `from_file` and
`from_dict` both migrate before validating against the pydantic models, so a
loaded `Spec` object's `spec_version` is always the version it was loaded
*as*, not necessarily the version the file was authored as.

## The compatibility policy

1. **A field addition with a safe default is not a version bump.** New
   optional `Spec`/`TableSpec`/`ColumnSpec` fields that default to `None`,
   `False`, or `[]` don't change how an older document reads — they just go
   unset. This is how the format has evolved so far.
2. **Anything that changes how an existing field is interpreted, renames a
   field, or removes one is a version bump**, and ships with:
   - a bump to `CURRENT_SPEC_VERSION` and an addition to
     `SUPPORTED_SPEC_VERSIONS` (old versions stay supported until explicitly
     dropped — dropping one is itself a documented, deliberate decision, not
     a side effect of adding the next version);
   - a migration branch in `syntab.loaders.migrate_spec_dict` from the old
     shape to the new one, so old documents keep loading rather than erroring;
   - a fixture-based compatibility test: a Spec document written in the old
     format, asserted to migrate to a specific normalized-current-format
     result. `tests/test_streaming.py`/`tests/test_disclosure_*.py` show the
     fixture-as-code-object pattern already used elsewhere in this repo —
     spec-version fixtures follow the same shape, just loaded from a literal
     dict/YAML string instead of built with `Spec(...)`.
   - a note in this file's version history below.
3. **A migration is either lossless or it's an error, never silently lossy.**
   `migrate_spec_dict` raising `SpecVersionError` on an unrecognized
   `spec_version` is the enforcement of this: if a future migration can't
   preserve a document's meaning (a removed feature with no equivalent,
   say), loading fails with a precise message instead of guessing.
4. **Deterministic across the whole supported range.** The same input
   document, on the same syntab version, always normalizes to the same
   output — migration has no dependency on load order, environment, or
   randomness. That's what makes `--migrate-out` safe to run once and commit.

## Version history

- **1.0** (current) — first stable wire format. `1` and `1.0.0` are accepted
  as pre-stabilization aliases for this same format.
