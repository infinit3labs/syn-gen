"""Custom generators used by the mock_specs examples.

Demonstrates both extension styles supported by syntab:
  * a :class:`BaseGenerator` subclass (stateful) for a derived/computed column;
  * a plain generator function, used here to propagate a foreign key value so
    that keys stay consistent across tables by construction.
"""
from syntab.generators import BaseGenerator


class LineTotalGenerator(BaseGenerator):
    """Compute ``line_total = unit_price * quantity``.

    Marking it as a generator (rather than a post-step) keeps the value
    consistent with the row it lives in and lets business rules reference it.
    """

    def generate(self, row, rng, faker, params):
        unit_price = row.get("unit_price") or 0
        quantity = row.get("quantity") or 0
        decimals = int(params.get("decimals", 2))
        return round(unit_price * quantity, decimals)


def order_customer_id(row, rng, faker, params):
    """Derive an order item's ``customer_id`` from its parent order.

    Because the chosen ``order`` already carries a valid ``customer_id`` (itself
    a foreign key into ``customers``), this guarantees the item's customer is
    identical to the order's customer. Referential integrity hence holds across
    all three tables without relying on rejection sampling.
    """
    order = (row.get("__parents__") or {}).get("order", {})
    return order.get("customer_id")


def overdraft_limit_for(row, rng, faker, params):
    """Savings accounts cannot have an overdraft; checking accounts can.

    Returning 0 for savings makes the business rule
    ``if account_type == 'savings' then overdraft_limit == 0`` hold by
    construction instead of relying on rejection sampling hitting an exact
    float value.
    """
    if row.get("account_type") == "savings":
        return 0
    return round(rng.uniform(0, float(params.get("max", 1000))), 2)

