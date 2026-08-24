# Release checklist

Before publishing a release:

1. Update `[project].version` in `pyproject.toml` and record the user-visible
   changes in the release notes.
2. Keep `Spec.spec_version` at the current wire-format version unless a
   migration is implemented and compatibility fixtures are added.
3. Run `python -m pytest -q` in a fresh environment with the core and `dev`
   extras.
4. Exercise `syntab validate`, streamed `syntab generate`, and `syntab check`
   using the committed demo Specs.
5. Verify optional extras independently. `parquet` requires PyArrow;
   `discovery` requires the platform-supported Desbordante wheel and is not
   required for core profiling.
6. Do not commit source datasets, profiled Specs derived from private data,
   generated output, or disclosure reports containing source-derived values.
