# Contributing to syntab

Thanks for helping improve syntab. Keep changes focused, explain the user
visible behavior they change, and include a regression test for bug fixes or
new behavior.

## Local setup

```bash
python -m pip install -e ".[parquet,dev]"
pytest
```

The optional `discovery` extra is only needed for tests or changes involving
Desbordante-backed discovery:

```bash
python -m pip install -e ".[discovery]"
```

## Before opening a change

- Run the relevant focused tests while developing.
- Run `pytest` from the repository root before submitting.
- Update the README or the relevant document when CLI behavior or public
  configuration changes.
- Do not commit real datasets, profiled Specs derived from private data,
  credentials, or generated output.
- Keep optional dependencies optional unless there is a clear compatibility
  and licensing reason to make them part of the core install.

There is no separate formatter or linter configured yet. Follow the existing
Python style and keep public APIs documented through examples and tests.

## Reporting security or privacy issues

Do not open a public issue containing private data or an undisclosed
vulnerability. Contact the project owners through a private channel first; if
you do not have one, open a minimal issue asking for a private contact without
including sensitive details.
