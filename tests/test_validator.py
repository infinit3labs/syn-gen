import numpy as np
import pandas as pd

from syntab.validator import validate


def _real(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    num = rng.normal(50, 10, n)
    num = np.clip(num, 0, 100)
    cat = rng.choice(["a", "b", "c"], n, p=[0.5, 0.3, 0.2])
    flag = rng.random(n) < 0.3
    days = rng.integers(0, 365, n)
    base = pd.Timestamp("2020-01-01")
    dt = [base + pd.Timedelta(days=int(d), hours=int(h)) for d, h in zip(days, rng.integers(0, 24, n))]
    txt = [("".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz"), size=int(rng.integers(5, 15))))) for _ in range(n)]
    return pd.DataFrame({"num": num, "cat": cat, "flag": flag, "dt": dt, "txt": txt})


def test_matching_synthetic_passes():
    real = _real(seed=1)
    synth = _real(seed=2)  # same distributions, different sample
    report = validate(real, synth)
    assert report.overall_pass is True
    assert report.n_fail == 0
    # every distance should be below its pass threshold
    for c in report.columns:
        assert c.distance <= c.threshold, c


def test_mismatched_synthetic_fails():
    real = _real(seed=1)
    rng = np.random.default_rng(99)
    n = len(real)
    synth = real.copy()
    # break numeric: uniform instead of normal
    synth["num"] = rng.uniform(0, 100, n)
    # break categorical: flip to a different distribution
    synth["cat"] = rng.choice(["a", "b", "c"], n, p=[0.1, 0.1, 0.8])
    report = validate(real, synth)
    assert report.overall_pass is False
    assert report.n_fail >= 1


def test_report_serialization():
    real = _real(seed=3)
    synth = _real(seed=4)
    report = validate(real, synth)
    d = report.to_dict()
    assert d["overall_pass"] is True
    assert "columns" in d
    assert report.to_text()  # renders without error
