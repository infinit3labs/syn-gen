"""Tests for spec-conformance checking (syntab.check)."""
import os
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from syntab.conformance import validate_against_spec
from syntab.engine import GenerationEngine
from syntab.loaders import from_file

MOCK = os.path.join(REPO_ROOT, "examples", "mock_specs")


def _frames(name):
    spec = from_file(os.path.join(MOCK, name))
    return GenerationEngine(spec).run().to_frames()


def test_conformance_passes_for_generated_ecommerce():
    frames = _frames("ecommerce.yaml")
    spec = from_file(os.path.join(MOCK, "ecommerce.yaml"))
    report = validate_against_spec(frames, spec)
    assert report.overall_ok, report.to_text()
    # spot-check that FK + rule checks were performed
    checks = [c for t in report.tables for c in t.checks]
    assert any(c.name.startswith("FK") for c in checks)
    assert any("business rule" in c.name for c in checks)


def test_conformance_passes_for_generated_accounts():
    frames = _frames("accounts.yaml")
    spec = from_file(os.path.join(MOCK, "accounts.yaml"))
    report = validate_against_spec(frames, spec)
    assert report.overall_ok, report.to_text()


def test_conformance_detects_orphan_fk():
    frames = _frames("ecommerce.yaml")
    orders = frames["orders"].copy()
    # break a foreign key
    orders.loc[0, "customer_id"] = -999
    frames["orders"] = orders
    spec = from_file(os.path.join(MOCK, "ecommerce.yaml"))
    report = validate_against_spec(frames, spec)
    assert not report.overall_ok
    orders_tbl = next(t for t in report.tables if t.table == "orders")
    fk = next(c for c in orders_tbl.checks if c.name.startswith("FK"))
    assert fk.status == "fail"


def test_conformance_detects_duplicate_pk():
    frames = _frames("ecommerce.yaml")
    customers = frames["customers"].copy()
    customers.loc[0, "id"] = customers.loc[1, "id"]  # duplicate PK
    frames["customers"] = customers
    spec = from_file(os.path.join(MOCK, "ecommerce.yaml"))
    report = validate_against_spec(frames, spec)
    assert not report.overall_ok
    cust_tbl = next(t for t in report.tables if t.table == "customers")
    pk = next(c for c in cust_tbl.checks if c.name == "primary key unique")
    assert pk.status == "fail"


def test_conformance_detects_rule_violation():
    frames = _frames("ecommerce.yaml")
    orders = frames["orders"].copy()
    # violate discount <= total * 0.3 for one row
    orders.loc[0, "discount"] = orders.loc[0, "total"] * 0.9
    frames["orders"] = orders
    spec = from_file(os.path.join(MOCK, "ecommerce.yaml"))
    report = validate_against_spec(frames, spec)
    assert not report.overall_ok
    orders_tbl = next(t for t in report.tables if t.table == "orders")
    rule = next(c for c in orders_tbl.checks if "business rule" in c.name)
    assert rule.status == "fail"


def test_conformance_reports_missing_columns_without_crashing():
    frames = _frames("ecommerce.yaml")
    # Drop columns the spec declares, mimicking a partial/renamed export.
    customers = frames["customers"].drop(columns=["region", "signup_date"])
    frames["customers"] = customers
    spec = from_file(os.path.join(MOCK, "ecommerce.yaml"))
    report = validate_against_spec(frames, spec)
    assert not report.overall_ok
    cust_tbl = next(t for t in report.tables if t.table == "customers")
    col_check = next(c for c in cust_tbl.checks if c.name == "all spec columns present")
    assert col_check.status == "fail"
    assert "region" in col_check.detail and "signup_date" in col_check.detail

    # A totally absent table is also reported as a failure, not a traceback.
    frames2 = {"orders": frames["orders"]}
    report2 = validate_against_spec(frames2, spec)
    assert not report2.overall_ok
    cust_tbl2 = next(t for t in report2.tables if t.table == "customers")
    present = next(c for c in cust_tbl2.checks if c.name == "table 'customers' present")
    assert present.status == "fail"
