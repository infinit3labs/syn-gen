"""Example custom generators (referenced by examples/spec.yaml).

Register with @register_generator OR reference by import path `custom_gen:order_region`
(the Spec file's directory is added to sys.path, so modules next to the Spec import).
"""
from syntab import register_generator


@register_generator("order_region")
def order_region(row, rng, faker, params):
    # inherit the parent user's region (parent rows exposed as row["__parents__"])
    return row["__parents__"]["user"]["region"]


def eu_tax(row, rng, faker, params):
    # tax rate depends on region via the parent user
    region = row["__parents__"]["user"]["region"]
    return 0.2 if region == "EU" else 0.0
