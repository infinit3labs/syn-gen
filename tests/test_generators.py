import random

from faker import Faker

from syntab import generators
from syntab.generators import GenContext, BaseGenerator, build_generator, register_generator


def _ctx(row=None, parents=None, params=None, **kw):
    return GenContext(
        row=row or {},
        parents=parents or {},
        rng=random.Random(1),
        faker=Faker(),
        params=params or {},
        **kw,
    )


def test_builtin_int():
    g = build_generator("int")
    out = [g(_ctx(params={"min": 5, "max": 5})) for _ in range(10)]
    assert all(v == 5 for v in out)


def test_builtin_float_decimals():
    g = build_generator("float")
    v = g(_ctx(params={"min": 0, "max": 1, "decimals": 2}))
    assert round(v, 2) == v


def test_builtin_choice_weights():
    g = build_generator("choice")
    vals = [g(_ctx(params={"values": ["a", "b"], "weights": [1, 0]})) for _ in range(20)]
    assert set(vals) == {"a"}


def test_builtin_sequence():
    g = build_generator("sequence")
    c = _ctx(seq=3)
    assert g(c) == 3


def test_faker_generator():
    g = build_generator("faker.email")
    v = g(_ctx())
    assert "@" in v


def test_custom_decorator():
    @register_generator("double_age")
    def double_age(row, rng, faker, params):
        return row["age"] * 2

    g = build_generator("custom:double_age")
    assert g(_ctx(row={"age": 21})) == 42


def test_custom_class():
    class Adder(BaseGenerator):
        def generate(self, row, rng, faker, params):
            return row["x"] + params.get("k", 0)

    generators._REGISTRY["adder"] = Adder()
    g = build_generator("custom:adder")
    assert g(_ctx(row={"x": 10}, params={"k": 5})) == 15


def test_custom_import_path(tmp_path):
    mod = tmp_path / "tmpmod.py"
    mod.write_text("def triple(row, rng, faker, params):\n    return params['v'] * 3\n")
    import sys
    sys.path.insert(0, str(tmp_path))
    try:
        g = build_generator("tmpmod:triple")
        assert g(_ctx(params={"v": 4})) == 12
    finally:
        sys.path.remove(str(tmp_path))


def test_unknown_generator():
    import pytest
    with pytest.raises(ValueError):
        build_generator("nope")
