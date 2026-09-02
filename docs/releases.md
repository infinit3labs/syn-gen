# Release checklist

Before publishing a release:

1. Update `[project].version` in `pyproject.toml` and record the user-visible
   changes under the matching version in [CHANGELOG.md](../CHANGELOG.md).
2. Keep `Spec.spec_version` at the current wire-format version unless a
   migration is implemented and compatibility fixtures are added, per the
   policy in [docs/spec-versioning.md](spec-versioning.md).
3. Run `python -m pytest -q` in a fresh environment with the core and `dev`
   extras.
4. Exercise `syntab validate`, streamed `syntab generate`, and `syntab check`
   using the committed demo Specs.
5. Verify optional extras independently. `parquet` requires PyArrow;
   `discovery` requires the platform-supported Desbordante wheel and is not
   required for core profiling.
6. Do not commit source datasets, profiled Specs derived from private data,
   generated output, or disclosure reports containing source-derived values.

## Versioning convention

syntab uses [Semantic Versioning](https://semver.org/): patch releases are
backward-compatible fixes, minor releases add backward-compatible features,
and major releases may remove or change public behavior. A release tag must
match `[project].version` (for example, `v0.1.0`). Keep the changelog's
`[Unreleased]` section at the top and move it under the new version when
tagging a release.
