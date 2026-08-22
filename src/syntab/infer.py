"""Auto-resolve generators from profile statistics.

When a column's ``generator`` is ``None``/``"auto"`` but it carries a
``profile``, we infer a generator + params so a profiled dataset can be
re-synthesized compatibly.
"""
from __future__ import annotations

from typing import Optional, Tuple

from .spec import ColumnProfile, ColumnSpec


def resolve_generator(col: ColumnSpec) -> Tuple[Optional[str], dict]:
    """Return (generator_str, params) for a column, inferring from profile."""
    gen = col.generator
    params = dict(col.params)

    if gen not in (None, "auto", ""):
        return gen, params

    prof: Optional[ColumnProfile] = col.profile
    if prof is None:
        # No profile: fall back to a dtype-based sensible default.
        default = {
            "int": "int",
            "float": "float",
            "str": "string",
            "bool": "bool",
            "datetime": "datetime",
            "date": "date",
            "uuid": "uuid",
        }.get(col.dtype, "string")
        return default, params

    if prof.categorical is not None and prof.categorical.values:
        params.setdefault("values", list(prof.categorical.values.keys()))
        params.setdefault("weights", list(prof.categorical.values.values()))
        if prof.categorical.null_rate:
            params.setdefault("null_rate", prof.categorical.null_rate)
        return "choice", params

    if prof.numeric is not None:
        n = prof.numeric
        if n.min is not None:
            params.setdefault("min", n.min)
        if n.max is not None:
            params.setdefault("max", n.max)
        if n.null_rate:
            params.setdefault("null_rate", n.null_rate)

        dist = n.distribution
        if dist == "empirical" and n.hist_edges and n.hist_counts:
            params.setdefault("distribution", "empirical")
            params.setdefault("hist_edges", n.hist_edges)
            params.setdefault("hist_counts", n.hist_counts)
        elif dist == "lognormal" and n.log_mu is not None:
            params.setdefault("distribution", "lognormal")
            params.setdefault("log_mu", n.log_mu)
            params.setdefault("log_sigma", n.log_sigma)
        elif dist == "exponential" and n.scale is not None:
            params.setdefault("distribution", "exponential")
            params.setdefault("scale", n.scale)
        elif dist == "pareto" and n.pareto_b is not None:
            params.setdefault("distribution", "pareto")
            params.setdefault("pareto_b", n.pareto_b)
            params.setdefault("pareto_xmin", n.pareto_xmin)
        elif dist == "normal" and n.mean is not None and n.std is not None:
            # treat as normal via min/max guard rails from mean +/- 3 std
            lo = n.mean - 3 * n.std
            hi = n.mean + 3 * n.std
            params.setdefault("min", max(lo, n.min if n.min is not None else lo))
            params.setdefault("max", min(hi, n.max if n.max is not None else hi))
            params.setdefault("mean", n.mean)
            params.setdefault("std", n.std)
            params.setdefault("distribution", "normal")

        gtype = "int" if col.dtype == "int" else "float"
        return gtype, params

    if prof.datetime_range is not None:
        start, end = prof.datetime_range
        params.setdefault("start", start.isoformat() if hasattr(start, "isoformat") else start)
        params.setdefault("end", end.isoformat() if hasattr(end, "isoformat") else end)
        return "datetime", params

    if prof.string_pattern is not None:
        params.setdefault("pattern", prof.string_pattern)
        return "regex", params

    if prof.length is not None:
        # The observed RANGE, not the maximum. Passing prof.length[1] as a
        # scalar `length` pinned every generated value to the longest length
        # in the source; `min_length`/`max_length` let the string generator
        # sample across the range instead.
        #
        # This is still only the range. The full observed length distribution
        # and the source character distribution are what text generation
        # really wants, and neither is carried on ColumnProfile today --
        # generators.empirical_text_params() computes both in the form the
        # string generator consumes, and anything already present in
        # col.params is passed straight through, so a profiler can supply
        # them without a spec schema change (params is free-form).
        lo, hi = prof.length
        params.setdefault("min_length", lo)
        params.setdefault("max_length", hi)
        return "string", params

    # Last resort: dtype default.
    default = {
        "int": "int",
        "float": "float",
        "str": "string",
        "bool": "bool",
        "datetime": "datetime",
        "date": "date",
        "uuid": "uuid",
    }.get(col.dtype, "string")
    return default, params
