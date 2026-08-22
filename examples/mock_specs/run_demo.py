"""Run the mock_specs examples and verify every constraint they declare.

Covers:
  * custom business rules (DSL) -- re-checked directly on the generated data;
  * in-table constraints (primary keys, uniqueness, column dependencies);
  * cross-table constraints (foreign keys + cross-join rules via aliases);
  * consistent application of keys across all three ecommerce tables.

Run from the repo root:  python examples/mock_specs/run_demo.py
"""
from __future__ import annotations

import os
import sys
from datetime import datetime

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from syntab.loaders import from_file
from syntab.engine import GenerationEngine

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    mark = "PASS" if ok else "FAIL"
    line = f"  [{mark}] {name}"
    if detail:
        line += f" -- {detail}"
    print(line)


def generate(spec_path: str) -> dict[str, pd.DataFrame]:
    spec = from_file(spec_path)
    result = GenerationEngine(spec).run()
    frames = result.to_frames()
    base = os.path.splitext(os.path.basename(spec_path))[0]
    for tbl, df in frames.items():
        df.to_csv(os.path.join(OUT, f"{base}.{tbl}.csv"), index=False)
    return frames


def verify_ecommerce(f: dict[str, pd.DataFrame]) -> None:
    customers = f["customers"]
    orders = f["orders"]
    items = f["order_items"]

    cust_ids = set(customers["id"])
    order_ids = set(orders["id"])

    check("customers.id unique", customers["id"].is_unique, f"n={len(customers)}")
    check("customers.id non-null", customers["id"].notna().all())

    # FK referential integrity: every order points at a real customer
    check(
        "orders.customer_id -> customers.id (FK integrity)",
        set(orders["customer_id"]).issubset(cust_ids),
    )

    # Cross-join rule: order.region must equal its customer's region
    reg = orders.merge(customers[["id", "region"]], left_on="customer_id",
                       right_on="id", suffixes=("", "_cust"))
    check("orders: region == customer.region (cross-table rule)",
          bool((reg["region"] == reg["region_cust"]).all()))

    # Cross-join rule: order.total must not exceed the customer's credit_limit
    lim = orders.merge(customers[["id", "credit_limit"]], left_on="customer_id",
                       right_on="id")
    check("orders: total <= customer.credit_limit (cross-table rule)",
          bool((lim["total"] <= lim["credit_limit"]).all()))

    # In-table rule: discount <= total * 0.3
    check("orders: discount <= total * 0.3 (in-table rule)",
          bool((orders["discount"] <= orders["total"] * 0.3 + 1e-9).all()))
    check("orders: total > 0 (in-table rule)",
          bool((orders["total"] > 0).all()))

    # FK referential integrity: every item points at a real order
    check("order_items.order_id -> orders.id (FK integrity)",
          set(items["order_id"]).issubset(order_ids))

    # Keys consistent across three tables: the item's customer must equal the
    # customer of the order it belongs to.
    ord_cust = orders.set_index("id")["customer_id"].to_dict()
    items = items.copy()
    items["order_customer"] = items["order_id"].map(ord_cust)
    check("order_items: customer_id == order.customer_id (cross-table key consistency)",
          bool((items["customer_id"] == items["order_customer"]).all()))
    check("order_items.customer_id -> customers.id (consistent FK)",
          set(items["customer_id"]).issubset(cust_ids))

    # In-table computed column + rules
    check("order_items: line_total == unit_price * quantity (in-table rule)",
          bool((abs(items["line_total"]
                    - (items["unit_price"] * items["quantity"])) < 1e-6).all()))
    check("order_items: quantity > 0 (in-table rule)",
          bool((items["quantity"] > 0).all()))
    check("order_items: unit_price >= 0 (in-table rule)",
          bool((items["unit_price"] >= 0).all()))


def verify_accounts(f: dict[str, pd.DataFrame]) -> None:
    a = f["accounts"]

    check("accounts.id unique", a["id"].is_unique, f"n={len(a)}")
    check("accounts.id non-null", a["id"].notna().all())

    od = pd.to_datetime(a["opened_date"])
    cd = pd.to_datetime(a["closed_date"])
    check("accounts: closed_date >= opened_date (in-table rule)",
          bool((cd >= od).all()))
    check("accounts: days_between(closed, opened) >= 0 (func rule)",
          bool(((cd - od).dt.days >= 0).all()))

    sav = a[a["account_type"] == "savings"]
    check("accounts: savings => overdraft_limit == 0 (if/then rule)",
          bool((sav["overdraft_limit"] == 0).all()), f"n_savings={len(sav)}")

    check("accounts: balance >= 0 or status == 'frozen' (or rule)",
          bool(((a["balance"] >= 0) | (a["status"] == "frozen")).all()))

    bad = a[(a["status"] == "open") & (a["balance"] < 0)]
    check("accounts: not (open and balance < 0) (not/and rule)",
          len(bad) == 0, f"violations={len(bad)}")

    check("accounts: coalesce(balance,0) >= -1000 (func rule)",
          bool((a["balance"].fillna(0) >= -1000).all()))
    check("accounts: abs(balance) >= 0 (func rule)",
          bool((a["balance"].abs() >= 0).all()))
    check("accounts: min(balance, 0) <= 0 (func rule)",
          all(min(v, 0) <= 0 for v in a["balance"]))


def main() -> int:
    print("=" * 72)
    print("MOCK SPEC 1: ecommerce (cross-table + keys + custom generators)")
    print("=" * 72)
    verify_ecommerce(generate(os.path.join(HERE, "ecommerce.yaml")))

    print()
    print("=" * 72)
    print("MOCK SPEC 2: accounts (in-table DSL business rules)")
    print("=" * 72)
    verify_accounts(generate(os.path.join(HERE, "accounts.yaml")))

    print()
    print("-" * 72)
    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"TOTAL: {len(results)} checks, {len(results) - n_fail} passed, {n_fail} failed")
    print(f"Outputs written to: {OUT}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
