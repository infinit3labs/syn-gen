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
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

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


def _g_string(ctx: GenContext) -> Any:
    p = ctx.params
    length = int(p.get("length", 8))
    alphabet = p.get("alphabet", "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    return "".join(ctx.rng.choice(alphabet) for _ in range(length))


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
