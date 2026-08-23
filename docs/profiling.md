# Profiling — reproducibility, scale, and merge semantics

A profiled spec is a *build artifact* derived from a source dataset. This
document is the contract for what that artifact is, and the levers a user
has to control it. The ticket is **INF-225**.

## What the contract is

A profiled spec must answer four questions without anyone re-running
anything:

1. **What did you read?** -- ``metadata.profiling.source.profiled_rows`` is
   the number of rows that ended up in memory; ``source.source_path``,
   ``source.source_size_bytes``, ``source.source_mtime`` are the file
   identity (path, size, mtime), present when the source was a file on
   disk and absent when the source was an in-memory frame.
2. **What knobs did you turn?** -- ``metadata.profiling.controls`` lists
   every profiling option that was set to a non-default value, and only
   those. The block is omitted when the user ran with default options
   everywhere, so default runs do not grow the spec.
3. **What version of syntab wrote this?** --
   ``metadata.syntab_version`` is ``syntab.__version__`` at the time
   ``profile()`` ran. ``None`` on hand-authored specs.
4. **What did the re-profile do?** -- ``metadata.merge`` is a summary
   (see *Re-profiling* below), present after any ``--merge-into`` run and
   ``None`` otherwise.

## Resource controls

The two new flags are the documented way to bound what ``profile`` reads
into memory:

- ``--max-rows N`` -- read at most ``N`` rows from each input. CSV, TSV,
  and JSONL stream in chunks of 50 000 rows; the row cap is applied as
  the chunks come in. Parquet, JSON, and Excel are loaded whole and then
  truncated, since the underlying readers do not expose ``chunksize``;
  documented in the help text and the function docstring.
- ``--columns id,val,score`` -- comma-separated column subset. Applied
  *before* ``--max-rows``, so ``--max-rows`` is a cap on the columns you
  asked for, not on the whole file.

Important: when ``--max-rows`` is set, the row cap replaces the previous
"full dataset" assumption for nullability and uniqueness. ``null_rate``,
``unique``, and the cell-count enumeration are then measured on the
capped subset, not the source. Pair ``--max-rows`` with ``--sample`` if
you want the contract to keep its previous meaning: ``--sample`` only
controls the per-column shape work (numeric min/max, string patterns);
``--max-rows`` controls the row cap on what is held in memory.

The previous default (``--max-rows`` unset) reads the file whole in one
call, so existing scripts and existing specs are unchanged.

## Reproducibility

The reproducibility contract is the same one ``test_profiler_reproducibility``
asserts, extended to file bytes:

> Same source data, same options, same seed -> the emitted YAML/JSON
> file is byte-identical across runs.

``test_same_input_produces_byte_equivalent_yaml`` and
``test_same_input_produces_byte_equivalent_json`` in
``tests/test_profiler_repro_inf225.py`` lock that in. It holds because:

- ``model_dump`` (pydantic v2) emits dicts in field-definition order,
- ``spec_to_dict`` always inserts ``spec_version`` first and never sorts
  the rest,
- ``yaml.safe_dump(data, sort_keys=False)`` and
  ``json.dumps(data, indent=2)`` both preserve insertion order.

If a future change adds a field to the dump whose order is not
deterministic (e.g. a set, a generator), the byte-equivalence test will
fail. The fix is to sort the affected dict, not to relax the test.

## Re-profiling (``--merge-into``)

The merge is the "re-profile without losing hand edits" loop. It runs
the new profile, then walks the merged result against the old spec and
decides what to keep:

| Case | Action |
|---|---|
| table in old + new | merge; primary key / relationships per the existing fingerprint logic |
| table in old, not new | **kept**, recorded in ``merge.tables_removed`` |
| table in new, not old | **added**, recorded in ``merge.tables_added`` |
| column in old + new, same shape | fresh column wins, recorded in ``merge.columns_preserved`` |
| column in old + new, **different** shape | fresh column wins, recorded in ``merge.columns_changed_reinferred`` |
| column in old, not in new | **kept**, recorded in ``merge.columns_removed`` |
| column in new, not in old | **added**, recorded in ``merge.columns_added`` |

The merge report is attached to the merged spec's
``metadata.merge`` and printed by the CLI. A column whose shape drifted
is *not* silently re-inferred away from a hand edit; the next re-profile
that finds the column stable will mark it ``columns_preserved`` instead
of ``columns_changed_reinferred``, so the report converges with a
second pass.

The drift signature is the data-facing shape:
``dtype``, ``generator``, ``params``, ``constraints``, and the column
profile (categorical vocabulary, numeric bounds, datetime range,
length). See ``_column_signature`` in ``profiler.py`` for the exact
fields.

## Caveats

- The merge assumes the fresh profile and the old spec are talking about
  the same schema. A column that was renamed in the source between two
  profiles is *removed* (under the old name) and *added* (under the new
  name); the merge report shows both, and a human review is the only
  way to decide whether to keep both.
- The bounded-memory read applies to the input file only. Foreign-key
  discovery (SPIDER) and conditional generation both build per-column
  structures whose memory is bounded by the number of distinct values,
  not by ``--max-rows``; for those, ``--discover`` / ``--discover-fds``
  have their own cost model and their own documented controls.
- ``max_rows`` defaults to ``None`` (whole-file read). The first
  profiling of a multi-GB file should be on a sample; a re-profile on
  a bounded row cap is fine when the cap is large enough for the
  nullability / uniqueness claims to remain accurate.
