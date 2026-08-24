# CI and release checks

The CLI reports are intended to be automation boundaries. JSON reports carry a
`report_type` and `report_schema_version`; use the compact form when a pipeline
only needs a stable summary and the default full form when it needs diagnostics.

```bash
syntab quality --real real.csv --synthetic synthetic.csv \
  --min-score 0.80 --out quality.json --compact
syntab diagnose --real real.csv --synthetic synthetic.csv \
  --out diagnose.json --compact
syntab check --spec spec.yaml --data output/ \
  --out conformance.json --compact
```

Exit code `0` means the requested gate passed. Exit code `1` means a quality
threshold or structural/conformance check failed. Exit code `2` is reserved for
invalid command input or an unreadable dataset. `quality` without
`--min-score` is intentionally measurement-only and exits `0` even for a low
score.

Profile disclosure budgets are hard failures by default, so no Spec is written
when a budget is exceeded. For a review workflow that wants the artifact plus
an actionable warning, use `--disclosure-mode warning` and write the report.

```bash
syntab profile source.csv --out spec.yaml \
  --max-rare-values 0 --disclosure-mode warning \
  --disclosure-out disclosure.json
```

The report schema is deliberately additive: consumers should key on the
version/type and summary fields, and tolerate additional full-report fields.
