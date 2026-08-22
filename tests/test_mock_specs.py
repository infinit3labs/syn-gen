"""Regression tests for the mock_specs examples.

These assert the guarantees the mock specs are meant to demonstrate:
custom business rules (in-table + cross-join), in-table constraints, and
consistent keys across tables.
"""
import os
import sys

import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from syntab.engine import GenerationEngine
from syntab.loaders import from_file

MOCK = os.path.join(REPO_ROOT, "examples", "mock_specs")


def _frames(name):
    spec = from_file(os.path.join(MOCK, name))
    return GenerationEngine(spec).run().to_frames()


def test_ecommerce_keys_and_rules():
    f = _frames("ecommerce.yaml")
    customers, orders, items = f["customers"], f["orders"], f["order_items"]

    cust_ids = set(customers["id"])
    order_ids = set(orders["id"])

    # primary key + uniqueness
    assert customers["id"].is_unique
    assert orders["id"].is_unique
    assert items["id"].is_unique

    # FK referential integrity
    assert set(orders["customer_id"]).issubset(cust_ids)
    assert set(items["order_id"]).issubset(order_ids)

    # cross-join rules
    reg = orders.merge(customers[["id", "region"]], left_on="customer_id",
                       right_on="id", suffixes=("", "_cust"))
    assert (reg["region"] == reg["region_cust"]).all()
    lim = orders.merge(customers[["id", "credit_limit"]], left_on="customer_id",
                       right_on="id")
    assert (lim["total"] <= lim["credit_limit"]).all()
    assert (orders["discount"] <= orders["total"] * 0.3 + 1e-9).all()

    # keys consistent across three tables: item's customer == order's customer
    ord_cust = orders.set_index("id")["customer_id"].to_dict()
    items = items.copy()
    items["order_customer"] = items["order_id"].map(ord_cust)
    assert (items["customer_id"] == items["order_customer"]).all()
    assert set(items["customer_id"]).issubset(cust_ids)

    # in-table computed column
    assert (abs(items["line_total"]
                - items["unit_price"] * items["quantity"]) < 1e-6).all()


def test_accounts_business_rules():
    f = _frames("accounts.yaml")
    a = f["accounts"]

    assert a["id"].is_unique

    od = pd.to_datetime(a["opened_date"])
    cd = pd.to_datetime(a["closed_date"])
    assert (cd >= od).all()

    sav = a[a["account_type"] == "savings"]
    assert (sav["overdraft_limit"] == 0).all()

    assert ((a["balance"] >= 0) | (a["status"] == "frozen")).all()
    assert not ((a["status"] == "open") & (a["balance"] < 0)).any()
    assert (a["balance"].fillna(0) >= -1000).all()
    assert (a["balance"].abs() >= 0).all()
    assert all(min(v, 0) <= 0 for v in a["balance"])
