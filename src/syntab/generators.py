"""Generators: built-ins, the custom-generator registry, and BaseGenerator.

A generator is internally a callable ``GenFn(ctx) -> Any`` where ``ctx`` is a
:class:`GenContext`. User-supplied generators (via ``@register_generator`` or a
``BaseGenerator`` subclass) keep the simple public signature
``(row, rng, faker, params) -> Any``; the engine wraps them.
"""
from __future__ import annotations

import bisect
import importlib
import math
import random
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from faker import Faker

Registry = Dict[str, Callable]


@dataclass
class GenContext:
    row: Dict[str, Any]
    parents: Dict[str, Dict[str, Any]]
    rng: random.Random
    faker: Faker
    params: Dict[str, Any] = field(default_factory=dict)
    table: Optional[str] = None
    col: Optional[str] = None
    # per-column mutable state for sequence / uniqueness
    seq: int = 0
    unique_set: Optional[set] = None
    # "table.col" -> list of generated parent key values
    fk_index: Dict[str, List[Any]] = field(default_factory=dict)


GenFn = Callable[[GenContext], Any]


class BaseGenerator:
    """Subclass for stateful custom generators.

    def generate(self, row, rng, faker, params):
        ...
    """

    def generate(self, row: Dict[str, Any], rng: random.Random, faker: Faker, params: Dict[str, Any]) -> Any:
        raise NotImplementedError


_REGISTRY: Registry = {}


def register_generator(name: str):
    """Decorator to register a custom generator function or BaseGenerator instance."""

    def deco(obj):
        _REGISTRY[name] = obj
        return obj

    return deco


def get_registry() -> Registry:
    return dict(_REGISTRY)


# ---------------------------------------------------------------------------
# Built-in generators
# ---------------------------------------------------------------------------

def _sample_empirical(rng: random.Random, edges: List[float], counts: List[float]) -> float:
    """Inverse-CDF sampling from a stored histogram."""
    total = sum(counts)
    if total <= 0 or len(edges) < 2 or len(counts) == 0:
        return edges[0] if edges else 0.0
    cdf: List[float] = []
    acc = 0.0
    for c in counts:
        acc += c
        cdf.append(acc)
    r = rng.random() * total
    idx = bisect.bisect_left(cdf, r)
    if idx >= len(counts):
        idx = len(counts) - 1
    lo, hi = edges[idx], edges[idx + 1]
    if hi <= lo:
        return lo
    return lo + rng.random() * (hi - lo)


def _sample_numeric(ctx: GenContext) -> float:
    """Sample a raw float from the configured numeric distribution."""
    p = ctx.params
    dist = p.get("distribution")
    lo = float(p.get("min", 0)) if p.get("min") is not None else None
    hi = float(p.get("max", 1)) if p.get("max") is not None else None

    if dist == "empirical" and p.get("hist_edges") and p.get("hist_counts"):
        return _sample_empirical(ctx.rng, p["hist_edges"], p["hist_counts"])

    if dist == "lognormal" and p.get("log_mu") is not None and p.get("log_sigma") not in (None, "", 0):
        v = math.exp(ctx.rng.gauss(float(p["log_mu"]), float(p["log_sigma"])))
        if lo is not None:
            v = min(max(v, lo), hi if hi is not None else v)
        return v

    if dist == "exponential":
        scale = float(p.get("scale", 1))
        v = ctx.rng.expovariate(1.0 / scale)
        if lo is not None:
            v = max(v, lo)
        if hi is not None:
            v = min(v, hi)
        return v

    if dist == "pareto":
        b = float(p.get("pareto_b", 1.0))
        xmin = float(p.get("pareto_xmin", lo if (lo is not None and lo > 0) else 1.0))
        v = xmin * ctx.rng.paretovariate(b)
        if hi is not None:
            v = min(v, hi)
        return v

    if dist == "normal":
        mean = p.get("mean")
        std = p.get("std")
        if mean is not None and std not in (None, "", 0):
            v = ctx.rng.gauss(float(mean), float(std))
            if lo is not None:
                v = min(max(v, lo), hi if hi is not None else v)
            return v

    # default: uniform
    a = lo if lo is not None else 0.0
    b = hi if hi is not None else a + 1.0
    return ctx.rng.uniform(a, b)


def _g_int(ctx: GenContext) -> Any:
    p = ctx.params
    lo = int(p.get("min", 0))
    hi = int(p.get("max", 100)) if p.get("max") is not None else (lo + 100)
    step = int(p.get("step", 1))
    dist = p.get("distribution")

    if dist in ("empirical", "lognormal", "exponential", "pareto", "normal"):
        val = int(round(_sample_numeric(ctx)))
        val = min(max(val, lo), hi)
        if step != 1:
            val = lo + ((val - lo) // step) * step
            val = min(val, hi)
        return val

    if step != 1:
        span = max(0, (hi - lo) // step)
        return lo + ctx.rng.randint(0, span) * step
    return ctx.rng.randint(lo, hi)


def _g_float(ctx: GenContext) -> Any:
    p = ctx.params
    lo = float(p.get("min", 0.0)) if p.get("min") is not None else None
    hi = float(p.get("max", 1.0)) if p.get("max") is not None else None
    dist = p.get("distribution")

    if dist in ("empirical", "lognormal", "exponential", "pareto", "normal"):
        v = _sample_numeric(ctx)
        if lo is not None:
            v = max(v, lo)
        if hi is not None:
            v = min(v, hi)
    else:
        a = lo if lo is not None else 0.0
        b = hi if hi is not None else a + 1.0
        v = ctx.rng.uniform(a, b)

    decimals = p.get("decimals")
    if decimals is not None:
        v = round(v, int(decimals))
    return v


def _g_bool(ctx: GenContext) -> Any:
    p = ctx.params
    if "weights" in p:
        return ctx.rng.choices([True, False], weights=p["weights"])[0]
    return ctx.rng.random() < float(p.get("p", 0.5))


def _g_uuid(ctx: GenContext) -> Any:
    return str(uuid.uuid4())


def _g_datetime(ctx: GenContext) -> Any:
    p = ctx.params
    start = p.get("start")
    end = p.get("end")
    if start is None or end is None:
        start_d = datetime(2000, 1, 1)
        end_d = datetime(2030, 1, 1)
    else:
        start_d = _parse_dt(start)
        end_d = _parse_dt(end)
    delta = (end_d - start_d).total_seconds()
    return start_d + timedelta(seconds=ctx.rng.uniform(0, max(0, delta)))


def _g_date(ctx: GenContext) -> Any:
    val = _g_datetime(ctx)
    if isinstance(val, datetime):
        return val.date()
    return val


DEFAULT_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def _sample_text_length(ctx: GenContext) -> int:
    """Pick a value length from whichever length spec the params carry.

    In precedence order:

      * ``length_edges`` + ``length_counts`` -- a histogram of observed
        lengths, sampled by inverse CDF (same scheme as the numeric
        ``empirical`` distribution);
      * ``length_values`` + ``length_weights`` -- an exact discrete
        distribution, for columns with few distinct lengths;
      * ``min_length`` / ``max_length`` -- uniform over the observed range;
      * ``length`` -- a single fixed width, for hand-authored specs.

    A profiled column used to arrive here with ``length`` set to the MAXIMUM
    observed length, so every generated value came out at the longest width
    in the source. The first three forms exist so that no longer happens.
    """
    p = ctx.params
    lo = p.get("min_length")
    hi = p.get("max_length")

    if p.get("length_edges") and p.get("length_counts"):
        n = int(round(_sample_empirical(ctx.rng, p["length_edges"], p["length_counts"])))
    elif p.get("length_values"):
        values = p["length_values"]
        weights = p.get("length_weights")
        n = int(ctx.rng.choices(values, weights=weights)[0] if weights
                else ctx.rng.choice(values))
    elif lo is not None or hi is not None:
        a = int(lo if lo is not None else 0)
        b = int(hi if hi is not None else a)
        n = ctx.rng.randint(min(a, b), max(a, b))
    elif p.get("length") is not None:
        n = int(p["length"])
    else:
        n = 8

    if lo is not None:
        n = max(n, int(lo))
    if hi is not None:
        n = min(n, int(hi))
    return max(0, n)


def _g_string(ctx: GenContext) -> Any:
    """Generate a text value of a sampled length from a sampled alphabet.

    ``char_values`` + ``char_weights`` supply an observed character
    distribution. Without them the alphabet falls back to a uniform
    ``[a-zA-Z0-9]``, which is fine for an opaque identifier and badly wrong
    for free text: it contains no spaces or punctuation and makes digits as
    common as vowels.
    """
    p = ctx.params
    n = _sample_text_length(ctx)
    if n == 0:
        return ""

    chars = p.get("char_values")
    if chars:
        weights = p.get("char_weights")
        picked = (ctx.rng.choices(chars, weights=weights, k=n) if weights
                  else ctx.rng.choices(chars, k=n))
        return "".join(picked)

    alphabet = p.get("alphabet", DEFAULT_ALPHABET)
    return "".join(ctx.rng.choices(alphabet, k=n))


def empirical_text_params(
    values: Iterable[Any],
    length_bins: int = 32,
    max_characters: int = 256,
) -> Dict[str, Any]:
    """Derive ``string`` generator params from observed text.

    Returns the length distribution and character distribution of ``values``
    in exactly the form :func:`_g_string` consumes, so that

        params = empirical_text_params(source_column)

    produces text with the source's length profile and character mix instead
    of fixed-width uniform alphanumeric noise.

    Few distinct lengths are stored exactly (``length_values`` /
    ``length_weights``); more than ``length_bins`` of them are stored as a
    histogram (``length_edges`` / ``length_counts``). The character set is
    truncated to the ``max_characters`` most frequent characters, which for
    ordinary text is lossless and for something like CJK keeps the params
    bounded.

    DISCLOSURE NOTE
    ---------------
    The output is derived from the source data and embedding it in a spec
    makes that spec carry source-derived statistics, exactly as a categorical
    frequency vector does -- see ``docs/disclosure.md``. Character unigram
    frequencies over a corpus are weak evidence about any individual record,
    but a length histogram over a very small group is not nothing. This is
    why the helper is opt-in rather than something profiling does on its own.
    """
    lengths: List[int] = []
    chars: Counter = Counter()
    for v in values:
        if v is None:
            continue
        if isinstance(v, float) and v != v:  # NaN
            continue
        s = str(v)
        lengths.append(len(s))
        chars.update(s)

    params: Dict[str, Any] = {}
    if not lengths:
        return params

    lo, hi = min(lengths), max(lengths)
    params["min_length"] = lo
    params["max_length"] = hi

    counts = Counter(lengths)
    if len(counts) <= length_bins:
        ordered = sorted(counts)
        total = float(sum(counts.values()))
        params["length_values"] = ordered
        params["length_weights"] = [counts[n] / total for n in ordered]
    else:
        edges, hist = _histogram(lengths, lo, hi, length_bins)
        params["length_edges"] = edges
        params["length_counts"] = hist

    if chars:
        common = chars.most_common(max_characters)
        total = float(sum(c for _, c in common))
        params["char_values"] = [ch for ch, _ in common]
        params["char_weights"] = [c / total for _, c in common]

    return params


def _histogram(values: List[int], lo: int, hi: int,
               bins: int) -> Tuple[List[float], List[float]]:
    """Equal-width histogram over [lo, hi], without requiring numpy."""
    if hi <= lo:
        return [float(lo), float(lo + 1)], [float(len(values))]
    width = (hi - lo) / bins
    counts = [0.0] * bins
    for v in values:
        idx = int((v - lo) / width)
        if idx >= bins:
            idx = bins - 1
        elif idx < 0:
            idx = 0
        counts[idx] += 1.0
    edges = [lo + i * width for i in range(bins + 1)]
    return edges, counts


def _g_regex(ctx: GenContext) -> Any:
    p = ctx.params
    pattern = p.get("pattern", "????##")
    return ctx.faker.bothify(pattern)


def _g_choice(ctx: GenContext) -> Any:
    p = ctx.params
    values = p.get("values")
    if not values:
        raise ValueError("choice generator requires params.values")
    weights = p.get("weights")
    if weights:
        return ctx.rng.choices(values, weights=weights)[0]
    return ctx.rng.choice(values)


def _g_sequence(ctx: GenContext) -> Any:
    return ctx.seq


def _g_const(ctx: GenContext) -> Any:
    return ctx.params.get("value")


def _g_fk(ctx: GenContext) -> Any:
    p = ctx.params
    ref = p.get("ref")
    if not ref:
        raise ValueError("fk generator requires params.ref (e.g. 'users.id')")
    key = ref.replace(".", ".")
    choices = ctx.fk_index.get(ref)
    if not choices:
        raise ValueError(f"No parent key values found for fk ref '{ref}'")
    return ctx.rng.choice(choices)


BUILTINS: Dict[str, GenFn] = {
    "int": _g_int,
    "float": _g_float,
    "bool": _g_bool,
    "uuid": _g_uuid,
    "datetime": _g_datetime,
    "date": _g_date,
    "string": _g_string,
    "regex": _g_regex,
    "choice": _g_choice,
    "sequence": _g_sequence,
    "const": _g_const,
    "fk": _g_fk,
}


def _parse_dt(v: Any) -> datetime:
    if isinstance(v, datetime):
        return v
    if isinstance(v, date):
        return datetime(v.year, v.month, v.day)
    return datetime.fromisoformat(str(v))


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def _wrap_user(fn: Callable) -> GenFn:
    def _wrapped(ctx: GenContext) -> Any:
        return fn(ctx.row, ctx.rng, ctx.faker, ctx.params)
    return _wrapped


def _wrap_class(inst: BaseGenerator) -> GenFn:
    def _wrapped(ctx: GenContext) -> Any:
        return inst.generate(ctx.row, ctx.rng, ctx.faker, ctx.params)
    return _wrapped


def build_generator(spec_str: Optional[str]) -> GenFn:
    """Resolve a generator reference string into a GenFn."""
    if spec_str is None:
        raise ValueError("generator is required (or set to 'auto' with a profile)")
    spec_str = spec_str.strip()

    if spec_str.startswith("faker."):
        provider = spec_str[len("faker."):]
        def _faker_gen(ctx: GenContext) -> Any:
            method = getattr(ctx.faker, provider, None)
            if method is None:
                raise ValueError(f"Unknown faker provider: {provider}")
            return method()
        return _faker_gen

    if spec_str.startswith("custom:"):
        name = spec_str[len("custom:"):]
        obj = _REGISTRY.get(name)
        if obj is None:
            raise ValueError(f"Unknown custom generator: {name}")
        if isinstance(obj, BaseGenerator):
            return _wrap_class(obj)
        return _wrap_user(obj)

    if ":" in spec_str:
        module_path, attr = spec_str.split(":", 1)
        module = importlib.import_module(module_path)
        obj = getattr(module, attr)
        if isinstance(obj, type) and issubclass(obj, BaseGenerator):
            return _wrap_class(obj())
        if isinstance(obj, BaseGenerator):
            return _wrap_class(obj)
        if callable(obj):
            return _wrap_user(obj)
        raise ValueError(f"{spec_str} is not callable")

    if spec_str in BUILTINS:
        return BUILTINS[spec_str]

    raise ValueError(f"Unknown generator: {spec_str}")
